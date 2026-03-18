import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from config import config

# === Dataset class ===
class DepartureDataset(torch.utils.data.Dataset):
    def __init__(self, x_context, abs_time, dow_idx, event_time, mask, user_ids=None):
        self.x_context = x_context
        self.abs_time = abs_time / (x_context.shape[1] - 1)
        self.dow_idx = dow_idx
        self.event_time = event_time
        self.mask = mask
        self.user_ids = user_ids

    def __len__(self):
        return len(self.x_context)

    def __getitem__(self, idx):
        item = {
            'x_context': self.x_context[idx],
            'abs_time': self.abs_time[idx],
            'dow_idx': self.dow_idx[idx],
            'event_time': self.event_time[idx],
            'mask': self.mask[idx]
        }
        if self.user_ids is not None:
            item['user_id'] = self.user_ids[idx]
        return item

# === Time feature extraction ===
def extract_hour_decimal(timestamps):
    return np.array([ts.hour + ts.minute / 60 for ts in timestamps])

# === Main data loader ===
def load_preprocessed_data(config):
    base_path = config["data_root"]

    # Load DoW mapping
    dow_file = os.path.join(base_path, "dow_labeled.csv")
    dow_table = pd.read_csv(dow_file, parse_dates=["date"])
    dow_map = dict(zip(dow_table["date"].dt.date, dow_table["is_weekend"]))

    # List all user folders
    all_users = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    user_data = {}

    for user_id in all_users:
        user_data[user_id] = process_user(user_dir=os.path.join(base_path, user_id), dow_map=dow_map, config=config)

    return user_data

# === Process one user's folder ===
def process_user(user_dir, dow_map, config):
    sensor_channels = config["sensor_channels"]
    full_data = []

    # Load all sensor CSV files
    for sensor in sensor_channels:
        sensor_path = os.path.join(user_dir, f"{sensor}.csv")
        df = pd.read_csv(sensor_path)
        full_data.append(df)

    merged = pd.concat(full_data, axis=1)

    # Load meta
    meta_path = os.path.join(user_dir, "meta.csv")
    meta = pd.read_csv(meta_path, parse_dates=["window_end", "date", "departure_time"])

    timestamps = pd.to_datetime(meta["window_end"])
    days = meta["date"].unique()
    num_days = len(days)

    # Reshape features: (num_days, seq_len, D_context)
    features = merged.values.reshape(num_days, config["seq_len"], -1)

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)

    # Absolute time: normalize from timestamps
    abs_time = extract_hour_decimal(timestamps).reshape(num_days, config["seq_len"], 1)

    # Day-of-week mapping (weekday/weekend)
    dow_idx = np.array([dow_map[pd.to_datetime(d).date()] for d in days])

    # Labels & event times
    event_labels = meta["label"].values.reshape(num_days, config["seq_len"])
    event_times = event_labels.argmax(axis=1)

    mask = np.ones(num_days) 

    return {
        'x_context': features,
        'abs_time': abs_time,
        'dow_idx': dow_idx,
        'event_time': event_times,
        'mask': mask
    }

# === Dataset splitters ===
def split_generalized(user_data, config):
    users = list(user_data.keys())
    np.random.seed(config.get("split_seed", 42))
    np.random.shuffle(users)

    # First 75 users for CV, last 18 for test
    train_val_users, test_users = np.split(users, [75])
    print(f"Train+Val Pool: {len(train_val_users)}, Test: {len(test_users)}")

    # Create 5 folds on train_val_users
    kf = KFold(n_splits=5, shuffle=True, random_state=config.get("split_seed", 42))
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_val_users)):
        fold_train = [train_val_users[i] for i in train_idx]
        fold_val = [train_val_users[i] for i in val_idx]
        folds.append((fold_train, fold_val))
        print(f"Fold {fold_idx+1}: Train={len(fold_train)}, Val={len(fold_val)}")

    return folds, test_users

def split_personalized_single(user_id, user_data):
    """
    Split a single user's data into train, val, and test sets (3-1-2 weeks).
    Assumes 42 days (6 weeks) of data per user.
    """
    user_seq = user_data[user_id]
    total_samples = len(user_seq['event_time'])  # Expected: 42
    if total_samples < 42:
        raise ValueError(f"User {user_id} has only {total_samples} samples, expected 42.")

    train_idx = np.arange(0, 21)  # First 3 weeks
    val_idx   = np.arange(21, 28) # Next 1 week
    test_idx  = np.arange(28, 42) # Last 2 weeks

    # Add user_ids for personalized split
    user_ids_train = [user_id] * len(train_idx)
    user_ids_val = [user_id] * len(val_idx)
    user_ids_test = [user_id] * len(test_idx)

    return {
        'train': {
            k: torch.tensor(user_seq[k][train_idx], dtype=torch.float32 if k != 'dow_idx' else torch.int64)
            for k in user_seq
        } | {'user_ids': user_ids_train}, # Use dictionary union for Python 3.9+
        'val': {
            k: torch.tensor(user_seq[k][val_idx], dtype=torch.float32 if k != 'dow_idx' else torch.int64)
            for k in user_seq
        } | {'user_ids': user_ids_val},
        'test': {
            k: torch.tensor(user_seq[k][test_idx], dtype=torch.float32 if k != 'dow_idx' else torch.int64)
            for k in user_seq
        } | {'user_ids': user_ids_test}
    }
    
# === Build combined tensors for a split ===
def build_split(user_data, train_users, val_users, test_users, train_idx=None, val_idx=None, test_idx=None):
    def collect(users, idx_range=None):
        x, abs_, dow, evt, mask, uids = [], [], [], [], [], []
        for uid in users:
            d = user_data[uid]
            indices = idx_range if idx_range is not None else np.arange(len(d['event_time']))
            x.append(d['x_context'][indices])
            abs_.append(d['abs_time'][indices])
            dow.append(d['dow_idx'][indices])
            evt.append(d['event_time'][indices])
            mask.append(d['mask'][indices])
            uids.extend([uid] * len(indices))

        return {
            'x_context': torch.tensor(np.concatenate(x), dtype=torch.float32),
            'abs_time': torch.tensor(np.concatenate(abs_), dtype=torch.float32),
            'dow_idx': torch.tensor(np.concatenate(dow), dtype=torch.int64),
            'event_time': torch.tensor(np.concatenate(evt), dtype=torch.int64),
            'mask': torch.tensor(np.concatenate(mask), dtype=torch.float32),
            'user_ids': uids
        }

    result = {}
    if train_users:
        result['train'] = collect(train_users, train_idx)
    if val_users:
        result['val'] = collect(val_users, val_idx)
    if len(test_users)>0:
        result['test'] = collect(test_users, test_idx)
    return result