import os
import json
import numpy as np
import copy
import random
import pandas as pd
from datetime import datetime

import torch
from itertools import product
from torch.utils.data import DataLoader

from config import config
from data import load_preprocessed_data, build_split, DepartureDataset, split_generalized
from tte_transformer import TransformerTTE
from train import train_model

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

RESULTS_DIR = "results/sensitivity"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Force mode
config["mode"] = "generalized"

# Load best hyperparameters
# BEST_PARAMS_PATH = "./results/tuning/best_params.json"
# if os.path.exists(BEST_PARAMS_PATH):
#     with open(BEST_PARAMS_PATH, "r") as f:
#         best_params = json.load(f)
# else:
#     best_params = {}

# Sensitivity ranges
EVENT_WEIGHTS = [0.5, 1.0, 1.5, 2.0]
WEEKEND_WEIGHTS = [0.5, 1.0, 1.5, 2.0]
UNCERTAINTY_THRESHOLDS = [0.05, 0.1, 0.15, 0.2]

results = []

def run_sensitivity_multi(ew, ww, th, seed=None):
    if seed is not None:
        set_seed(seed)

    exp_config = copy.deepcopy(config)
    exp_config["event_weight"] = ew
    exp_config["weekend_weight"] = ww
    exp_config["uncertainty_threshold"] = th
    exp_config["exp_name"] = f"sens_ew{ew}_ww{ww}_th{th}_{datetime.now().strftime('%H%M%S')}"

    # Load data
    user_data = load_preprocessed_data(exp_config)
    folds, _ = split_generalized(user_data, exp_config)
    all_train_users = sorted(list(set(u for fold in folds for u in fold[0] + fold[1])))

    random.seed(exp_config["split_seed"])
    random.shuffle(all_train_users)
    val_size = max(1, int(0.2 * len(all_train_users)))
    val_users = all_train_users[:val_size]
    train_users = all_train_users[val_size:]

    split_data = build_split(user_data, train_users, val_users, [])

    train_loader = DataLoader(DepartureDataset(**split_data["train"]),
                              batch_size=exp_config["batch_size"], shuffle=True)
    val_loader = DataLoader(DepartureDataset(**split_data["val"]),
                            batch_size=exp_config["batch_size"], shuffle=False)

    device = torch.device(exp_config["device"])

    # Model
    model = TransformerTTE(
        input_dim_context=exp_config["input_dim"],
        d_model=exp_config["d_model"],
        nhead=exp_config["nhead"],
        num_layers=exp_config["num_layers"],
        dropout=exp_config["dropout"],
        dropout_time=exp_config["dropout_time"],
        max_seq_len=exp_config["seq_len"],
        dow_embedding_dim=2,
        num_dow=2,
        use_positional_encoding=exp_config.get("use_positional_encoding", True),
        use_alpha_fusion=exp_config.get("use_alpha_fusion", True),
        use_abs_time_scale=exp_config.get("use_abs_time_scale", True),
        use_time_features=exp_config.get("use_time_features", True),
        use_context_features=exp_config.get("use_context_features", True),
        use_DoW=exp_config.get("use_DoW", True), 
    )

    print(f"\n▶ Running: EW={ew}, WW={ww}, TH={th}")

    ckpt_name = f"ew{ew}_ww{ww}_th{th}.pt"
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    best_val_loss = train_model(model, train_loader, val_loader, device, exp_config, save_path=ckpt_path)
    print(f"✅ Training completed for EW={ew}, WW={ww}, TH={th}. Best model saved at {ckpt_path}")

    # Record result (CSV row)
    results.append({
        "event_weight": ew,
        "weekend_weight": ww,
        "uncertainty_threshold": th,
        "val_loss": best_val_loss,
        "checkpoint": ckpt_path
    })
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "sensitivity_results.csv"), index=False)

def main():
    for ew, ww, th in product(EVENT_WEIGHTS, WEEKEND_WEIGHTS, UNCERTAINTY_THRESHOLDS):
        run_sensitivity_multi(ew, ww, th, seed=44)

    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()