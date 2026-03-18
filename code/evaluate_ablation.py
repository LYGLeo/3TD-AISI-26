import os
import json
import copy
import random
import pandas as pd
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from config import config
from data import load_preprocessed_data, build_split, DepartureDataset, split_generalized
from tte_transformer import TransformerTTE
from train import train_model

RESULTS_DIR = "results/ablation"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

# === Force global mode ===
config["mode"] = "generalized"

# === Load best hyperparameters (from tuning) ===
# BEST_PARAMS_PATH = "./results/tuning/best_params.json"
# if os.path.exists(BEST_PARAMS_PATH):
#     with open(BEST_PARAMS_PATH, "r") as f:
#         best_params = json.load(f)
# else:
#     best_params = {}

# === Define Ablation Variants ===
ABLATION_VARIANTS = {
    # Feature-Level
    "context_only": {"use_time_features": False, "use_context_features": True},
    "time_only": {"use_time_features": True, "use_context_features": False},
    "full": {"use_time_features": True, "use_context_features": True},
    "no_DoW": {"use_DoW": False},

    # Architectural / Loss
    "no_pos_enc": {"use_positional_encoding": False},
    "no_alpha": {"use_alpha_fusion": False},
    "no_gaussian_and_abs_time": {
        "use_gaussian_smoothing": False,
        "use_abs_time_scale": False
    }
}

results = []

def run_ablation(variant_name, overrides, seed=None):

    if seed is not None:
        set_seed(seed)

    exp_config = copy.deepcopy(config)

    exp_config.update(overrides)
    exp_config["exp_name"] = f"ablation_{variant_name}_{datetime.now().strftime('%H%M%S')}"

    # === Data Loading ===
    user_data = load_preprocessed_data(exp_config)
    folds, _ = split_generalized(user_data, exp_config)
    all_train_users = sorted(list(set(u for fold in folds for u in fold[0] + fold[1])))

    # Simple split: 80% train, 20% val
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

    # === Model ===
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

        # Architectural
        use_positional_encoding=exp_config.get("use_positional_encoding", True),
        use_alpha_fusion=exp_config.get("use_alpha_fusion", True),
        use_abs_time_scale=exp_config.get("use_abs_time_scale", True),

        # Feature-level
        use_time_features=exp_config.get("use_time_features", True),
        use_context_features=exp_config.get("use_context_features", True),
        use_DoW=exp_config.get("use_DoW", True),    
    )

    print(f"\nRunning Ablation: {variant_name} | Overrides: {overrides}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{variant_name}.pt")
    best_val_loss = train_model(model, train_loader, val_loader, device, exp_config, save_path=ckpt_path)

    # === Save checkpoint for test evaluation ===
    print(f"✅ Saved model checkpoint for {variant_name} at {ckpt_path}")

    results.append({"Variant": variant_name, "Val_Loss": best_val_loss, "Checkpoint": ckpt_path})
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)


def main():
    for variant_name, overrides in ABLATION_VARIANTS.items():
        run_ablation(variant_name, overrides, seed=42)

    print("\n✅ Ablation Study Completed.")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()