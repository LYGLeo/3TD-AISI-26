#Baseline 0
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from tqdm import tqdm
import os
import warnings
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import rtdl

# Basic configuration
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 1. Data Loading and Feature Generation Functions
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'].str.split(' ').str[0])
    df['departure_time'] = pd.to_datetime(df['departure_time'])
    df['departure_hour'] = df['departure_time'].dt.hour + df['departure_time'].dt.minute / 60
    df = df.sort_values(['user_id', 'date']).reset_index(drop=True)
    return df

#Generates historical features for a single user's data.
def generate_all_features_for_user(user_df):
    user_df = user_df.sort_values('date').reset_index(drop=True)
    features_list = []
    unique_dates = user_df['date'].unique()

    # Iterate through dates to create features for each target day
    for i in range(7, len(unique_dates)):
        target_date = unique_dates[i]
        history_dates = unique_dates[i-7:i]
        past_df = user_df[user_df['date'].isin(history_dates)]
        target_row = user_df[user_df['date'] == target_date]

        feature = {'date': target_date}
        dep_hour = past_df['departure_hour']
        stats = {'mean': 'mean', 'std': 'std', 'min': 'min', 'max': 'max', 'skew': 'skew', 'kurt': 'kurt'}
        for stat_name, method in stats.items():
            feature[f'all_{stat_name}'] = getattr(dep_hour, method)()
        for is_weekend_flag, group_name in zip([0, 1], ['weekday', 'weekend']):
            group_dep = past_df[past_df['is_weekend'] == is_weekend_flag]['departure_hour']
            for stat_name, method in stats.items():
                feature[f'{group_name}_{stat_name}'] = getattr(group_dep, method)() if not group_dep.empty else np.nan

        feature['target'] = target_row['departure_hour'].iloc[0]
        feature['is_weekend'] = target_row['is_weekend'].iloc[0]
        feature['departure_time'] = target_row['departure_time'].iloc[0]
        features_list.append(feature)
    return pd.DataFrame(features_list).fillna(0)

# 2. Objective Functions for Hyperparameter Optimization (HPO)
# --- Optuna HPO for LGBM (quantile regression) ---
def objective_lgbm(trial, train_feature_df, train_user_ids):
    params = {
        'objective': 'quantile', 'metric': 'l1', 'alpha': 0.5, 'random_state': 42, 'verbose': -1,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-2, 3e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    losses = []

    for _, (train_user_idx, val_user_idx) in enumerate(kf.split(train_user_ids)):
        train_users, val_users = np.array(train_user_ids)[train_user_idx], np.array(train_user_ids)[val_user_idx]
        train_data, val_data = train_feature_df[train_feature_df['user_id'].isin(train_users)], train_feature_df[train_feature_df['user_id'].isin(val_users)]
        X_train, y_train = train_data.drop(columns=['date', 'target', 'departure_time', 'user_id']), train_data['target']
        X_val, y_val = val_data.drop(columns=['date', 'target', 'departure_time', 'user_id']), val_data['target']
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[optuna.integration.LightGBMPruningCallback(trial, 'l1')], categorical_feature=['is_weekend'])
        preds = model.predict(X_val)
        losses.append(mean_squared_error(y_val, preds))
    return np.mean(losses)

# --- Optuna HPO for SVR ---
def objective_svr(trial, train_feature_df, train_user_ids):
    params = {
        'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
        'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True)
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    losses = []
    for _, (train_user_idx, val_user_idx) in enumerate(kf.split(train_user_ids)):
        train_users, val_users = np.array(train_user_ids)[train_user_idx], np.array(train_user_ids)[val_user_idx]
        train_data, val_data = train_feature_df[train_feature_df['user_id'].isin(train_users)], train_feature_df[train_feature_df['user_id'].isin(val_users)]
        X_train, y_train = train_data.drop(columns=['date', 'target', 'departure_time', 'user_id']), train_data['target']
        X_val, y_val = val_data.drop(columns=['date', 'target', 'departure_time', 'user_id']), val_data['target']
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model = SVR(**params)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)
        losses.append(mean_squared_error(y_val, preds))
    return np.mean(losses)
    
# --- Optuna HPO for FT-Transformer ---
def objective_ftt(trial, all_train_feature_df, train_user_ids):
    params = {
        'd_token': trial.suggest_categorical('d_token', [64, 128, 256]),
        'n_blocks': trial.suggest_int('n_blocks', 1, 4),
        'attention_dropout': trial.suggest_float('attention_dropout', 0.1, 0.5),
        'ffn_d_hidden_factor': trial.suggest_float('ffn_d_hidden_factor', 0.5, 2.0),
        'ffn_dropout': trial.suggest_float('ffn_dropout', 0.1, 0.5),
        'residual_dropout': trial.suggest_float('residual_dropout', 0.0, 0.2),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_val_losses, fold_best_epochs = [], []
    for _, (train_user_idx, val_user_idx) in enumerate(kf.split(train_user_ids)):
        # Prepare data
        train_users, val_users = np.array(train_user_ids)[train_user_idx], np.array(train_user_ids)[val_user_idx]
        train_df, val_df = all_train_feature_df[all_train_feature_df['user_id'].isin(train_users)], all_train_feature_df[all_train_feature_df['user_id'].isin(val_users)]
        categorical_features, numerical_features = ['is_weekend'], [c for c in train_df.columns if c not in ['date', 'target', 'departure_time', 'user_id', 'is_weekend']]
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(train_df[numerical_features].values)
        X_val_num = scaler.transform(val_df[numerical_features].values)
        X_train_num_t = torch.tensor(X_train_num, dtype=torch.float32)
        X_train_cat_t = torch.tensor(train_df[categorical_features].values, dtype=torch.int64)
        y_train_t = torch.tensor(train_df['target'].values, dtype=torch.float32).view(-1, 1)
        X_val_num_t, X_val_cat_t, y_val_t = torch.tensor(X_val_num, dtype=torch.float32).to(DEVICE), torch.tensor(val_df[categorical_features].values, dtype=torch.int64).to(DEVICE), torch.tensor(val_df['target'].values, dtype=torch.float32).view(-1, 1).to(DEVICE)
        
        model = rtdl.FTTransformer.make_baseline(
            n_num_features=X_train_num.shape[1],
            cat_cardinalities=[int(all_train_feature_df[col].max()) + 1 for col in categorical_features],
            d_token=params['d_token'], n_blocks=params['n_blocks'], attention_dropout=params['attention_dropout'],
            ffn_d_hidden=int(params['d_token'] * params['ffn_d_hidden_factor']),
            ffn_dropout=params['ffn_dropout'], residual_dropout=params['residual_dropout'], d_out=1
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        loss_fn = nn.L1Loss()
        train_loader = DataLoader(TensorDataset(X_train_num_t, X_train_cat_t, y_train_t), batch_size=64, shuffle=True)
        
        best_val_loss, best_epoch, patience_counter = float('inf'), 0, 0
        for epoch in range(100):
            model.train()
            for x_num, x_cat, y_batch in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(x_num.to(DEVICE), x_cat.to(DEVICE)), y_batch.to(DEVICE))
                loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad(): current_val_loss = loss_fn(model(X_val_num_t, X_val_cat_t), y_val_t).item()
            if current_val_loss < best_val_loss:
                best_val_loss, best_epoch, patience_counter = current_val_loss, epoch + 1, 0
            else:
                patience_counter += 1
                if patience_counter >= 10: break
        fold_val_losses.append(best_val_loss); fold_best_epochs.append(best_epoch)
    
    avg_val_loss, avg_best_epoch = np.mean(fold_val_losses), int(np.mean(fold_best_epochs))
    trial.set_user_attr("best_epoch", avg_best_epoch)
    return avg_val_loss

# 3. Final Training and Prediction Pipeline Functions
# --- MLR, SVR, LGBM Pipelines ---
def run_sklearn_pipeline(all_feature_df, train_user_ids, test_user_ids, best_lgbm_params, best_svr_params):
    train_df, test_df = all_feature_df[all_feature_df['user_id'].isin(train_user_ids)], all_feature_df[all_feature_df['user_id'].isin(test_user_ids)]
    feature_cols = [c for c in train_df.columns if c not in ['date', 'target', 'departure_time', 'user_id']]
    X_train, y_train, X_test = train_df[feature_cols], train_df['target'], test_df[feature_cols]
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
    models = { "MLR": LinearRegression(), "SVR": SVR(**best_svr_params) }
    models['LGBM_Regressor'] = LGBMRegressor(objective='quantile', alpha=0.5, random_state=42, **best_lgbm_params)
    for alpha in np.arange(0.05, 1.0, 0.05): models[f'LGBM_Q{int(alpha*100)}'] = LGBMRegressor(objective='quantile', alpha=round(alpha, 2), random_state=42, **best_lgbm_params)
    
    print("\n--- Training sklearn/LGBM models and making predictions ---")
    predictions_df = test_df[['user_id', 'date', 'departure_time', 'target']].rename(columns={'target': 'departure_hour'})
    for name, model in tqdm(models.items(), desc="Training and Predicting"):
        if name == "SVR": model.fit(X_train_scaled, y_train); preds = model.predict(X_test_scaled)
        elif 'LGBM' in name: 
            model.fit(X_train, y_train, categorical_feature=['is_weekend'])
            preds = model.predict(X_test)
        else:
            model.fit(X_train,y_train)
            preds = model.predict(X_test)

        pred_col_name = f'pred_{name}' if name in ["MLR", "SVR"] else f'pred_hour_{name}'
        predictions_df[pred_col_name] = preds
    return predictions_df

# --- FT-Transformer Pipeline ---
def run_ftt_pipeline(all_feature_df, train_user_ids, test_user_ids, best_params, best_epoch):
    train_df, test_df = all_feature_df[all_feature_df['user_id'].isin(train_user_ids)], all_feature_df[all_feature_df['user_id'].isin(test_user_ids)]
    categorical_features, numerical_features = ['is_weekend'], [c for c in train_df.columns if c not in ['date', 'target', 'departure_time', 'user_id', 'is_weekend']]
    scaler = StandardScaler()
    X_train_num, X_test_num = scaler.fit_transform(train_df[numerical_features].values), scaler.transform(test_df[numerical_features].values)
    X_train_num_t, X_train_cat_t, y_train_t = torch.tensor(X_train_num, dtype=torch.float32), torch.tensor(train_df[categorical_features].values, dtype=torch.int64), torch.tensor(train_df['target'].values, dtype=torch.float32).view(-1, 1)
    X_test_num_t, X_test_cat_t = torch.tensor(X_test_num, dtype=torch.float32).to(DEVICE), torch.tensor(test_df[categorical_features].values, dtype=torch.int64).to(DEVICE)
    
    model = rtdl.FTTransformer.make_baseline(
        n_num_features=X_train_num.shape[1],
        cat_cardinalities=[int(all_feature_df[col].max()) + 1 for col in categorical_features],
        d_token=best_params['d_token'], n_blocks=best_params['n_blocks'], attention_dropout=best_params['attention_dropout'],
        ffn_d_hidden=int(best_params['d_token'] * best_params['ffn_d_hidden_factor']),
        ffn_dropout=best_params['ffn_dropout'], residual_dropout=best_params['residual_dropout'], d_out=1
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    loss_fn = nn.L1Loss()
    train_loader = DataLoader(TensorDataset(X_train_num_t, X_train_cat_t, y_train_t), batch_size=64, shuffle=True)
    
    print(f"\n--- Training final FTT model for {best_epoch} epochs ---")
    model.train()
    for _ in tqdm(range(best_epoch), desc="Final FTT Training"):
        for x_num, x_cat, y_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_num.to(DEVICE), x_cat.to(DEVICE)), y_batch.to(DEVICE))
            loss.backward(); optimizer.step()
            
    print("\n--- Predicting with FTT on the test set ---")
    model.eval()
    with torch.no_grad(): preds = model(X_test_num_t, X_test_cat_t).cpu().numpy().flatten()
    
    results_df = test_df[['user_id', 'date', 'departure_time', 'target']].rename(columns={'target': 'departure_hour'})
    results_df['pred_hour_FTT'] = preds
    return results_df

# =====================================================================================
# 4. Main Execution Logic
# =====================================================================================
def main():
    csv_path = "/home/yglee/Dataset/STDD_Data/baseline_historical_dataset.csv"
    save_dir = "/home/yglee/evdp/results/baseline/historical/"
    os.makedirs(save_dir, exist_ok=True)
    
    df = load_and_preprocess_data(csv_path)
    
    all_features_list = []
    for user_id in tqdm(df['user_id'].unique(), desc="Generating Features"):
        user_df = df[df['user_id'] == user_id].copy()
        if len(user_df['date'].unique()) < 42: continue
        feature_df = generate_all_features_for_user(user_df)
        if len(feature_df) < 35: continue
        feature_df, feature_df['user_id'] = feature_df.head(35), user_id
        all_features_list.append(feature_df)
    
    all_feature_df = pd.concat(all_features_list, ignore_index=True)
    
    # --- Correct user splitting logic ---
    qualified_user_ids_int = list(all_feature_df['user_id'].unique())
    qualified_user_ids_str_sorted = sorted([str(uid) for uid in qualified_user_ids_int])
    np.random.seed(42); np.random.shuffle(qualified_user_ids_str_sorted)
    train_users_str, test_users_str = np.split(qualified_user_ids_str_sorted, [75])
    train_user_ids = [int(uid) for uid in train_users_str]
    test_user_ids = [int(uid) for uid in test_users_str if int(uid) in df['user_id'].unique()][:18]
    print(f"Training with {len(train_user_ids)} users. Testing with {len(test_user_ids)} users.")
    
    train_feature_df = all_feature_df[all_feature_df['user_id'].isin(train_user_ids)]
    
    # --- Sequential HPO execution ---
    print("\n--- Tuning LightGBM ---"); study_lgbm = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner()); study_lgbm.optimize(lambda t: objective_lgbm(t, train_feature_df, train_user_ids), n_trials=30); best_lgbm_params = {k: v for k, v in study_lgbm.best_params.items() if k not in ['objective', 'metric', 'alpha']}
    print("\n--- Tuning SVR ---"); study_svr = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42)); study_svr.optimize(lambda t: objective_svr(t, train_feature_df, train_user_ids), n_trials=30); best_svr_params = study_svr.best_params
    print("\n--- Tuning FT-Transformer ---"); study_ftt = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42)); study_ftt.optimize(lambda t: objective_ftt(t, train_feature_df, train_user_ids), n_trials=30); best_ftt_params, best_ftt_epoch = study_ftt.best_trial.params, study_ftt.best_trial.user_attrs["best_epoch"]

    print(f"\n--- Best LGBM Params: {best_lgbm_params} ---")
    print(f"--- Best SVR Params: {best_svr_params} ---")
    print(f"--- Best FTT Params: {best_ftt_params} (Best Epoch: {best_ftt_epoch}) ---")

    # --- Execute each pipeline to collect prediction results ---
    sklearn_preds_df = run_sklearn_pipeline(all_feature_df, train_user_ids, test_user_ids, best_lgbm_params, best_svr_params)
    ftt_preds_df = run_ftt_pipeline(all_feature_df, train_user_ids, test_user_ids, best_ftt_params, best_ftt_epoch)

    # --- Merge results ---
    final_df = pd.merge(sklearn_preds_df, ftt_preds_df[['user_id', 'date', 'pred_hour_FTT']], on=['user_id', 'date'])
    final_df = pd.merge(final_df, all_feature_df[['user_id', 'date', 'is_weekend']], on=['user_id', 'date'])

    # --- Comprehensive performance evaluation and saving ---
    summary_rows = []
    model_pred_cols = {
        'MLR': 'pred_MLR', 'SVR': 'pred_SVR', 'FTT': 'pred_hour_FTT',
        **{f'LGBM_Q{int(a*100)}': f'pred_hour_LGBM_Q{int(a*100)}' for a in np.arange(0.05, 1.0, 0.05)},
        'LGBM_Regressor': 'pred_hour_LGBM_Regressor'
    }
    for uid, user_group in final_df.groupby('user_id'):
        user_group.to_csv(os.path.join(save_dir, f"{uid}_test_predictions_ALL_MODELS.csv"), index=False)
        weekday_df, weekend_df = user_group[user_group['is_weekend'] == 0], user_group[user_group['is_weekend'] == 1]
        for model_name, pred_col in model_pred_cols.items():
            if pred_col not in user_group.columns: continue
            summary_rows.append({
                'user_id': uid, 'model': model_name,
                'Overall_MAE': mean_absolute_error(user_group['departure_hour'], user_group[pred_col]),
                'Weekday_MAE': mean_absolute_error(weekday_df['departure_hour'], weekday_df[pred_col]) if not weekday_df.empty else np.nan,
                'Weekend_MAE': mean_absolute_error(weekend_df['departure_hour'], weekend_df[pred_col]) if not weekend_df.empty else np.nan
            })
    
    summary_df = pd.DataFrame(summary_rows)
    average_summary = summary_df.groupby('model')[['Overall_MAE', 'Weekday_MAE', 'Weekend_MAE']].mean()
    print("\n--- Average MAE Results Across All Users ---\n", average_summary)

    average_summary.to_csv(os.path.join(save_dir, "average_mae_summary.csv"))

if __name__ == "__main__":
    main()