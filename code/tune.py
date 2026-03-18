import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import ParameterGrid
import torch
from torch.utils.data import DataLoader

from config import config
from train import train_model
from data import load_preprocessed_data, DepartureDataset, build_split, split_generalized
from tte_transformer import TransformerTTE

def evaluate_mae(model, val_loader, config):
    model.eval()
    device = config["device"]
    threshold = config["uncertainty_threshold"]
    total_mae, count = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            x_context, abs_time, dow_idx = batch['x_context'].to(device), batch['abs_time'].to(device), batch['dow_idx'].to(device)
            event_time = batch['event_time'].cpu().numpy()
            pred_q = model(x_context, abs_time, dow_idx).cpu().numpy()
            for i in range(pred_q.shape[0]):
                q_seq = pred_q[i]
                s_seq = np.cumprod(np.clip(q_seq, 1e-6, 1 - 1e-6))
                true_time = event_time[i]
                detected_time = None
                for t in range(len(s_seq)):
                    if s_seq[t] < threshold:
                        detected_time = t
                        break
                if detected_time is not None:
                    penalty = 1
                    error = true_time - detected_time if detected_time <= true_time else (detected_time - true_time) * penalty
                    total_mae += error
                    count += 1
    return total_mae / count if count > 0 else float('inf')

global_trial_counter = 0

def evaluate_combination(arch_params, hyperparam_params, user_data, folds, device):
    global global_trial_counter
    global_trial_counter += 1
    
    full_params = {**arch_params, **hyperparam_params}
    config.update(full_params)
    
    print(f"\n🚀 Grid Search Trial {global_trial_counter} | Arch: {arch_params}, Params: {hyperparam_params}")

    torch.manual_seed(config["split_seed"])
    np.random.seed(config["split_seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fold_maes, fold_val_losses = [], []

    for fold_idx, (train_u, val_u) in enumerate(folds):
        print(f"  Fold {fold_idx+1}/5...")
        fold_data = build_split(user_data, train_u, val_u, [])
        train_loader = DataLoader(DepartureDataset(**fold_data["train"]), batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(DepartureDataset(**fold_data["val"]), batch_size=config["batch_size"], shuffle=False)

        model = TransformerTTE(
            input_dim_context=config["input_dim"], d_model=config["d_model"], nhead=config["nhead"],
            num_layers=config["num_layers"], dropout=config["dropout"], dropout_time=config["dropout_time"],
            max_seq_len=config["seq_len"], dow_embedding_dim=2, num_dow=2,

            use_positional_encoding=config["use_positional_encoding"],
            use_alpha_fusion=config["use_alpha_fusion"],
            use_abs_time_scale=config["use_abs_time_scale"],
            use_time_features=config["use_time_features"],
            use_context_features=config["use_context_features"],
            use_DoW=config["use_DoW"]
        )

        val_loss = train_model(model, train_loader, val_loader, device, config, use_weight_decay=True)
        mae = evaluate_mae(model, val_loader, config)

        fold_maes.append(mae)
        fold_val_losses.append(val_loss)
        
        del model
        torch.cuda.empty_cache()

    avg_mae = np.mean(fold_maes)
    avg_val_loss = np.mean(fold_val_losses)
    
    print(f"  📊 Validation Results | Avg MAE: {avg_mae:.4f} | Avg Loss: {avg_val_loss:.4f}")

    return avg_mae, avg_val_loss

def tune():
    print("Loading data for tuning...")
    user_data = load_preprocessed_data(config)
    folds, _ = split_generalized(user_data, config)
    device = torch.device(config["device"])

    all_results = []
    summary_file_path = "./results/tuning/grid_search_tuning_summary.csv"

    arch_grid = {"d_model": [32, 64], "nhead": [1, 2], "num_layers": [3, 4]}
    hyperparam_grid = {
        "lr": [1e-4, 1e-3], "dropout": [0.1],
        "dropout_time": [0.1, 0.2], "weight_decay": [5e-4, 1e-3]
    }

    for arch_params in ParameterGrid(arch_grid):
        for hyperparam_params in ParameterGrid(hyperparam_grid):
            avg_mae, avg_val_loss = evaluate_combination(
                arch_params, hyperparam_params, user_data, folds, device
            )
            all_results.append({
                "architecture": str(arch_params),
                "hyperparameters": str(hyperparam_params),
                "avg_val_mae": avg_mae,
                "avg_val_loss": avg_val_loss
            })
            
            temp_summary_df = pd.DataFrame(all_results)
            temp_summary_df.to_csv(summary_file_path, index=False)

    print(f"\n✅ Grid Search complete. Summary saved to {summary_file_path}.")
    print("\n🔍 Analyzing tuning results to find the best combination...")

    results_df = pd.read_csv(summary_file_path)
    best_trial = results_df.loc[results_df["avg_val_loss"].idxmin()]

    print("\n--- Best Hyperparameter Combination Found (based on Validation Loss) ---")
    print(f"  - Best Architecture: {best_trial['architecture']}")
    print(f"  - Best Hyperparameters: {best_trial['hyperparameters']}")
    print(f"  - Best Average Validation Loss: {best_trial['avg_val_loss']:.4f}")
    print("\nNOTE: Final model retraining and testing has been removed as requested.")


if __name__ == "__main__":
    tune()