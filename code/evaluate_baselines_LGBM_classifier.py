import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import warnings
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
import optuna
import random

# --- Hide warning messages ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Basic settings ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# =======================================
# 1. Data Loading and Preparation Function
# =======================================
def load_and_prepare_single_user_data(base_dir, user_id, stride):
    user_folder_path = os.path.join(base_dir, f"stride#{stride}", str(user_id))
    
    if not os.path.isdir(user_folder_path):
        return pd.DataFrame()

    meta_file = os.path.join(user_folder_path, "meta.csv")
    if not os.path.exists(meta_file):
        return pd.DataFrame()

    try:
        # 1. Read meta.csv as the base DataFrame
        base_df = pd.read_csv(meta_file)

        # 2. Read and horizontally merge the remaining feature CSV files
        feature_files = glob(os.path.join(user_folder_path, "*.csv"))
        feature_files = [f for f in feature_files if 'meta.csv' not in f]
        
        if not feature_files:
            return pd.DataFrame()

        feature_dfs = [pd.read_csv(f) for f in feature_files]
        features_df = pd.concat(feature_dfs, axis=1)

        # 3. Combine meta info and feature info
        if len(base_df) != len(features_df):
            print(f"Warning: Row count mismatch for user {user_id}. Meta: {len(base_df)}, Features: {len(features_df)}")
            return pd.DataFrame()
            
        merged_df = pd.concat([base_df, features_df], axis=1)
        
        # Feature Engineering
        for col in ['window_start', 'window_end', 'departure_time']:
            merged_df[col] = pd.to_datetime(merged_df[col])
        merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date
        
        merged_df['hour'] = merged_df['window_end'].dt.hour + merged_df['window_end'].dt.minute / 60
        merged_df['hour'] = merged_df['hour'].replace(0, 24)

        # Add weekend info (use dow_labeled.csv if it exists, otherwise calculate)
        weekend_file = os.path.join(base_dir, f"stride#{stride}", "dow_labeled.csv")
        weekend_df = pd.read_csv(weekend_file)
        weekend_df['date'] = pd.to_datetime(weekend_df['date']).dt.date
        merged_df = pd.merge(merged_df, weekend_df, on='date', how='left')

        # Use only 42 days of data for each user
        unique_dates = sorted(merged_df['date'].unique())
        if len(unique_dates) >= 42:
            dates_to_use = unique_dates[:42]
            return merged_df[merged_df['date'].isin(dates_to_use)].reset_index(drop=True)
        else:
            return pd.DataFrame()

    except Exception as e:
        print(f"Warning: Could not process data for user {user_id}. Error: {e}")
        return pd.DataFrame()

def load_generalized_data(user_list, base_dir, stride):
    """Loads data for a given list of users and merges it into a single DataFrame"""
    all_user_dfs = []
    for user_id in tqdm(user_list, desc=f"Loading data for {len(user_list)} users", leave=False):
        user_df = load_and_prepare_single_user_data(base_dir, user_id, stride)
        if not user_df.empty:
            all_user_dfs.append(user_df)
    
    if not all_user_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_user_dfs, ignore_index=True)

# =======================================
# 2. Feature Selection Function
# =======================================
def get_global_features(base_dir, stride, p_value_threshold=0.05):
    """Creates a global feature set based on p-values from the significant features file"""
    print(f"\n[LOG] Step 2: Getting global features for stride {stride}...")
    sig_file = os.path.join(base_dir, f"stride#{stride}", f"sig#120#{stride}.csv")
    sig_df = pd.read_csv(sig_file)
    
    significant_features = sig_df[sig_df['p_value'] < p_value_threshold]['feature'].tolist()
    print(f"   [LOG] Found {len(significant_features)} features with p-value < {p_value_threshold}.")

    # Add base features
    base_features = ['hour', 'is_weekend']
    feature_cols = sorted(list(set(significant_features + base_features)))
    print(f"   [LOG] Total features to be used: {len(feature_cols)}")
    
    return feature_cols

# =======================================
# 3. Model Training Function
# =======================================
def _get_lgbm_with_params(trial):
    """Returns a LightGBM model with hyperparameters suggested by Optuna trial"""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),  # L1
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), # L2
    }
    return LGBMClassifier(**params)

def train_generalized_lgbm_model(dev_users, feature_cols, stride, base_data_dir, n_trials=50):
    """Trains and tunes a generalized LightGBM model using Optuna and 5-Fold CV (user-based)"""
    print(f"\n[LOG] Step 3: Starting LightGBM model training with 5-Fold CV...")

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    categorical_features = [col for col in ['is_weekend'] if col in feature_cols]
    numerical_features = [col for col in feature_cols if col not in categorical_features]

    print("   [LOG] Pre-loading all development data for Scikit-learn CV...")
    dev_df = load_generalized_data(dev_users, base_data_dir, stride)
    if dev_df.empty:
        print("[ERROR] Failed to load development data. Exiting.")
        return None
        
    X_dev, y_dev = dev_df[feature_cols], dev_df['label']
    user_ids_dev = dev_df['user_id']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ], remainder='passthrough')

    def objective(trial):
        fold_aps = []
        for fold, (train_user_idx, val_user_idx) in enumerate(kf.split(dev_users)):
            train_users_fold = [dev_users[i] for i in train_user_idx]
            val_users_fold = [dev_users[i] for i in val_user_idx]
            
            print(f"       [TRIAL {trial.number} | FOLD {fold+1}/5] Train users: {len(train_users_fold)}, Val users: {len(val_users_fold)}")
            
            train_indices = user_ids_dev.isin(train_users_fold)
            val_indices = user_ids_dev.isin(val_users_fold)
            X_train, X_val = X_dev[train_indices], X_dev[val_indices]
            y_train, y_val = y_dev[train_indices], y_dev[val_indices]
            
            if X_train.empty or len(y_train.unique()) < 2: continue

            model = _get_lgbm_with_params(trial)
            pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
            
            pipeline.fit(X_train, y_train)
            y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
            fold_aps.append(average_precision_score(y_val, y_pred_proba))

        return np.mean(fold_aps) if fold_aps else 0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    
    print(f"   [LOG] Optuna study complete. Best Mean PR AUC: {study.best_value:.4f}")
    
    print(f"   [LOG] Training final LightGBM model on {len(dev_users)} development users...")
    final_model = _get_lgbm_with_params(optuna.trial.FixedTrial(study.best_params))
    final_pipeline = Pipeline([('preprocessor', preprocessor), ('model', final_model)])
    final_pipeline.fit(X_dev, y_dev) # Train final model on the entire development set
    
    print(f"[LOG] Step 3.2: Final LightGBM model training complete.")
    return final_pipeline

# =======================================
# 4. Probability Prediction and Saving Function
# =======================================
def predict_and_save_probabilities(model, test_users, feature_cols, base_data_dir, stride, model_name, save_dir):
    """Predicts probabilities on the test set with the trained model and saves them"""
    print("\n[LOG] Step 4: Predicting probabilities for test users...")
    test_df = load_generalized_data(test_users, base_data_dir, stride)
    if test_df.empty:
        print("   [WARN] Test data is empty. Skipping prediction.")
        return

    X_test = test_df[feature_cols]
    probs = model.predict_proba(X_test)[:, 1]
    
    prob_df = test_df[['user_id', 'date', 'window_end', 'departure_time', 'label']].copy()
    prob_df['prob'] = probs
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"probs_generalized_{model_name}_stride_{stride}.csv")
    prob_df.to_csv(save_path, index=False)
    print(f"   [LOG] Test probabilities saved to {save_path}")

# =======================================
# 5. Full Pipeline Execution
# =======================================
def run_full_pipeline(stride, n_trials=30):
    BASE_DATA_DIR = "/home/yglee/Dataset/STDD_Data"
    PROB_RESULTS_DIR = "/home/yglee/evdp/results/baseline/passive"
    MODEL_NAME = 'LightGBM'
    
    print("="*60)
    print(f"STARTING GENERALIZED PIPELINE: MODEL={MODEL_NAME}, STRIDE={stride}")
    print("="*60)

    # --- Step 1: Get user list and split into dev/test sets ---
    print("\n[LOG] Step 1: Getting user list and splitting into dev/test sets...")
    
    # Scan the data folder directly to get the full list of users
    stride_path = os.path.join(BASE_DATA_DIR, f"stride#{stride}")
    if not os.path.isdir(stride_path):
        print(f"[ERROR] Stride directory not found at {stride_path}. Exiting.")
        return
        
    all_users_str = [d for d in os.listdir(stride_path) if os.path.isdir(os.path.join(stride_path, d)) and d.isdigit()]
    all_users_str_sorted = sorted(all_users_str)
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(all_users_str_sorted)

    num_dev_users = 75
    num_test_users = 18
    
    dev_users_str = all_users_str_sorted[:num_dev_users]
    test_users_str = all_users_str_sorted[num_dev_users : num_dev_users + num_test_users]

    # Convert final user_id lists to int type
    dev_users = [int(uid) for uid in dev_users_str]
    test_users = [int(uid) for uid in test_users_str]
    print(f"Test users: {test_users}")

    print(f"   [LOG] Total users found: {len(all_users_str)}")
    print(f"   [LOG] Splitting into -> Dev users: {len(dev_users)}, Test users: {len(test_users)}") 

    # --- Step 2: Select global features ---
    feature_cols = get_global_features(BASE_DATA_DIR, stride)
    if not feature_cols:
        print("[ERROR] No features selected. Exiting pipeline.")
        return

    # --- Step 3: Train model ---
    final_model = train_generalized_lgbm_model(dev_users, feature_cols, stride, BASE_DATA_DIR, n_trials=n_trials)
    if final_model is None:
        print("[ERROR] Model training failed. Exiting pipeline.")
        return

    # --- Step 4: Predict probabilities and save ---
    save_dir = os.path.join(PROB_RESULTS_DIR, MODEL_NAME, f"stride_{stride}")
    predict_and_save_probabilities(final_model, test_users, feature_cols, BASE_DATA_DIR, stride, MODEL_NAME, save_dir)

    print("\n" + "="*50); print("✅ Generalized pipeline complete!"); print("="*50)

if __name__ == '__main__':
    # --- Execution settings ---
    STRIDE_TO_RUN = 5
    N_TRIALS = 30

    run_full_pipeline(
        stride=STRIDE_TO_RUN,
        n_trials=N_TRIALS
    )