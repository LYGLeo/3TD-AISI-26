import os
import torch
import pandas as pd
import numpy as np
import json
import copy
from torch.utils.data import DataLoader
import argparse
import re
from config import config
import matplotlib.pyplot as plt
from tte_transformer import TransformerTTE
from data import load_preprocessed_data, DepartureDataset, split_generalized, split_personalized_single, build_split

# === Compute survival function from q(t|x_t) ===
def compute_survival(pred_q):
    q = pred_q.clamp(min=1e-6, max=1 - 1e-6)
    S = torch.cumprod(q, dim=1)
    return S

# === Compute expected time from survival curve ===
def compute_expected_time(survival):
    time_grid = torch.arange(survival.size(1), dtype=torch.float32, device=survival.device)
    return torch.sum(survival * time_grid, dim=1)

# === Inference for one user or generalized ===
def run_final_test(model, test_loader, config):
    model.eval()
    device = config["device"]
    threshold = config["uncertainty_threshold"]

    detected_times, expected_times, actual_times, real_times, dow_labels = [], [], [], [], []

    # Convert entire dataset into one tensor for easier management
    all_batches = list(test_loader)  # load everything in memory (safe for test set)
    x_all = torch.cat([b['x_context'] for b in all_batches], dim=0).to(device)
    abs_all = torch.cat([b['abs_time'] for b in all_batches], dim=0).to(device)
    dow_all = torch.cat([b['dow_idx'] for b in all_batches], dim=0)
    event_all = np.concatenate([b['event_time'].cpu().numpy() for b in all_batches], axis=0)

    N, T, _ = x_all.shape
    active_idx = list(range(N))  # indices of active sequences
    detected_map = {i: None for i in range(N)}  # store detection time
    expected_map = {i: None for i in range(N)}  # store expected time

    with torch.no_grad():
        for t in range(T):
            print(f'inferencing time grid at {t}')
            if not active_idx:  # all sequences finished
                break

            # Prepare slices for active sequences
            x_batch = torch.stack([x_all[i, :t+1, :] for i in active_idx]).to(device)
            abs_batch = torch.stack([abs_all[i, :t+1] for i in active_idx]).to(device)
            dow_batch = dow_all[active_idx].to(device)

            # Forward pass for all active sequences at timestep t
            q = model(x_batch, abs_batch, dow_batch)  # shape: [len(active_idx), t+1, ?]
            S = compute_survival(q)  # cumulative survival curve

            s_t = S[:, -1].cpu().numpy()

            # Check threshold for each active sequence
            finished_indices = []
            for i, idx in enumerate(active_idx):
                if s_t[i] < threshold:
                    detected_map[idx] = t
                    expected_map[idx] = compute_expected_time(S[i:i+1])[0].item()
                    finished_indices.append(idx)

            # Remove finished sequences
            active_idx = [idx for idx in active_idx if idx not in finished_indices]

        # For remaining sequences (never crossed threshold)
        for idx in active_idx:
            # Full survival curve for entire sequence
            x_full = x_all[idx:idx+1, :, :]
            abs_full = abs_all[idx:idx+1, :]
            dow_full = dow_all[idx].view(1).to(device)
            q = model(x_full, abs_full, dow_full)
            S = compute_survival(q)
            expected_map[idx] = compute_expected_time(S)[0].item()

    # Collect results
    for i in range(N):
        #detected_times.append(detected_map[i] if detected_map[i] is not None else np.nan)
        detected_times.append(detected_map[i] if detected_map[i] is not None else 264)
        
        expected_times.append(expected_map[i])
        actual_times.append(event_all[i])
        real_times.append(abs_all[i].cpu().numpy())
        dow_labels.append(int(dow_all[i].item()))

    # Build DataFrame
    df = pd.DataFrame({
        'detected_time': detected_times,
        'expected_time': expected_times,
        'actual_time': actual_times,
        'DoW': dow_labels,
        'abs_time_sequence': real_times
    })

    return df

# === Compute MAE by category ===
def compute_mae_by_dow(df):
    # Filter weekday (DoW=0) and weekend (DoW=1)
    weekday_df = df[df['DoW'] == 0]
    weekend_df = df[df['DoW'] == 1]

    # Compute MAEs
    overall_mae = np.nanmean(np.abs(df['detected_time'] - df['actual_time']))
    weekday_mae = np.nanmean(np.abs(weekday_df['detected_time'] - weekday_df['actual_time'])) if not weekday_df.empty else np.nan
    weekend_mae = np.nanmean(np.abs(weekend_df['detected_time'] - weekend_df['actual_time'])) if not weekend_df.empty else np.nan

    return overall_mae, weekday_mae, weekend_mae

# === Argument Parser ===
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory of checkpoints for batch evaluation")
    parser.add_argument("--variant_name", type=str, default=None, help="Optional tag for the experiment")
    return parser.parse_args()

# === Evaluate One Checkpoint ===
def evaluate_checkpoint(ckpt_path, variant, test_loader):
    flags = parse_variant_flags(variant)
    kv = parse_variant_values(variant)

    cfg = copy.deepcopy(config)
    cfg.update(kv)  # ew/ww/th 적용

    print(f"[INFO] Evaluating {variant} with cfg updates: {kv}")
    
    model = TransformerTTE(
        input_dim_context=cfg["input_dim"],
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"], dropout_time=cfg.get("dropout_time", 0.0),
        max_seq_len=cfg["seq_len"], dow_embedding_dim=2, num_dow=2,
        use_positional_encoding=flags["use_positional_encoding"],
        use_alpha_fusion=flags["use_alpha_fusion"],
        use_abs_time_scale=flags["use_abs_time_scale"],
        use_DoW=flags["use_DoW"],
        use_time_features=flags["use_time_features"],
        use_context_features=flags["use_context_features"]
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg["device"]))
    model.to(cfg["device"])

    df = run_final_test(model, test_loader, cfg)
    overall_mae, weekday_mae, weekend_mae = compute_mae_by_dow(df)
    
    df.to_csv(f"./results/test_eval/{variant}_predictions.csv", index=False)

    return {
        "variant": variant,
        "checkpoint": ckpt_path,
        "overall_MAE": overall_mae,
        "weekday_MAE": weekday_mae,
        "weekend_MAE": weekend_mae
    }


# === Extract ablation type ===
def parse_variant_flags(variant_name):
    v = variant_name.lower()
    return {
        # Feature-level
        "use_time_features": not ("context_only" in v),
        "use_context_features": not ("time_only" in v),

        # Architectural
        "use_DoW": not ("no_dow" in v),
        "use_positional_encoding": not ("no_pos_enc" in v),
        "use_alpha_fusion": not ("no_alpha" in v),
        "use_abs_time_scale": not ("no_abs_time_scale" in v),

        # Loss
        "use_gaussian_smoothing": not ("no_gaussian" in v)
    }

# === Extract numeric ew/ww/th values from variant or filename ===
def parse_variant_values(variant_name):
    base = os.path.basename(variant_name).replace(".pt", "")
    m = re.search(r'ew(?P<ew>[\d\.]+)_ww(?P<ww>[\d\.]+)_th(?P<th>[\d\.]+)', base)
    if not m:
        return {}
    return {
        "event_weight": float(m.group("ew")),
        "weekend_weight": float(m.group("ww")),
        "uncertainty_threshold": float(m.group("th"))
    }

# === Main ===
if __name__ == '__main__':
    args = parse_args()
    user_data = load_preprocessed_data(config)
    os.makedirs("./results/test_eval", exist_ok=True)

    if config['mode'] == "personalized":
        folds, test_users = split_generalized(user_data, config)
        summary_records = []
        
        # Check if feature selection should be applied
        apply_feature_selection = config.get("apply_feature_selection", False)
        if apply_feature_selection:
            print("🔍 Feature selection enabled: Using only p<0.05 features for personalized models")
        
        for user_id in test_users:
            model_path = f"./results/checkpoints/fine_tuned_{user_id}.pt"
            if not os.path.exists(model_path):
                print(f"⚠ Fine-tuned model for {user_id} not found. Skipping...")
                continue
            user_split = split_personalized_single(user_id, user_data, config, apply_feature_selection=apply_feature_selection)

            # Create DataLoader with the new dataset structure
            test_loader = DataLoader(user_split['test'], batch_size=1, shuffle=False)
            summary_records.append(evaluate_checkpoint(model_path, f"user_{user_id}", test_loader))
        pd.DataFrame(summary_records).to_csv("./results/test_eval/personalized_summary.csv", index=False)
        overall_avg = np.nanmean([r['overall_MAE'] for r in summary_records])
        weekday_avg = np.nanmean([r['weekday_MAE'] for r in summary_records])
        weekend_avg = np.nanmean([r['weekend_MAE'] for r in summary_records])

        print(f"MAE (전체 평균): Overall={overall_avg:.2f}, Weekday={weekday_avg:.2f}, Weekend={weekend_avg:.2f}")

    else:
        folds, test_users = split_generalized(user_data, config)
        test_data = build_split(user_data, [], [], test_users)
        test_loader = DataLoader(DepartureDataset(**test_data["test"]), batch_size=1, shuffle=False)

        summary_records = []
        if args.checkpoint:
            summary_records.append(evaluate_checkpoint(args.checkpoint, args.variant_name or "custom", test_loader))
        elif args.checkpoint_dir:
            for fname in os.listdir(args.checkpoint_dir):
                if fname.endswith(".pt"):
                    variant = fname.replace(".pt", "")
                    ckpt_path = os.path.join(args.checkpoint_dir, fname)

                    flags = parse_variant_flags(variant)
                    # === 여기서 zeroing 처리 ===
                    data = copy.deepcopy(test_data["test"])  # 원본 손상 방지
                    if not flags["use_context_features"]:
                        data["x_context"] = torch.zeros_like(data["x_context"])
                    if not flags["use_time_features"]:
                        data["abs_time"] = torch.zeros_like(data["abs_time"])
                    # if not flags["use_DoW"]:
                    #     data["dow_idx"] = torch.zeros_like(data["dow_idx"]) 

                    test_loader = DataLoader(
                        DepartureDataset(**data),
                        batch_size=1,
                        shuffle=False
                    )

                    summary_records.append(evaluate_checkpoint(ckpt_path, variant, test_loader))

        else:
            summary_records.append(evaluate_checkpoint(config["save_model_path"], "global_model", test_loader))

        pd.DataFrame(summary_records).to_csv("./results/test_eval/generalized_summary.csv", index=False)
        record = summary_records[0]
        print(f"MAE: Overall={record['overall_MAE']:.2f}, Weekday={record['weekday_MAE']:.2f}, Weekend={record['weekend_MAE']:.2f}")
        print("\n✅ Generalized evaluation complete.")