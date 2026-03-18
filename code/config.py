import torch
import os

# === Shared Settings ===
shared_config = {
    "input_dim": 94,  # Number of features per timestep
    "seq_len": 265,   # Max sequence length
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),

    "data_root": "/home/yglee/Dataset/STDD_Data/stride#5",
    "sensor_channels": ["ACTIVITY", "APP_USAGE", "CALLS", "LIGHT", "MOTION", "SCREEN", "SOUND", "STEP", "UNLOCK"],

    # Model params
    "d_model": 32,
    "nhead": 1,
    "num_layers": 3,

    # Paths for saving models and tuning results
    "save_hp_path": "./results/tuning",
    "checkpoint_dir": "./results/checkpoints",
    "split_seed": 42, 

    # Early stopping
    "early_stopping_patience": 3,

    # Loss-related
    "weekend_weight": 1.5,
    "event_weight": 1.5,
 
    # Detection threshold for inference
    "uncertainty_threshold": 0.1,

    # Mode: "generalized" or "personalized"
    "mode": "generalized",
    #"mode": "personalized",

    # === Ablation / Architecture Flags ===
    # Feature-level
    "use_time_features": True,       # Control inclusion of time-related features
    "use_context_features": True,    # Control inclusion of contextual features
    "use_DoW": True,                 # Day-of-week embedding on/off

    # Architectural choices
    "use_positional_encoding": True, # Positional encoding on/off
    "use_alpha_fusion": True,        # Alpha parameter fusion on/off
    "use_abs_time_scale": True,      # Learnable abs time scaling on/off

    # Loss design
    "use_gaussian_smoothing": True  # Enable/disable Gaussian label smoothing
}

# === Default Hyperparameters for Generalized Mode ===
generalized_config = {
    "dropout": 0.1,
    "dropout_time": 0.2,

    # Optimization
    "lr": 1e-3,
    "batch_size": 64,
    "epochs": 100,

    # Regularization
    "weight_decay": 5e-4,  # L2 regularization for global model
    "l1_lambda": 0.0,      # No L1 in generalized model
    # Paths
    "save_model_path": os.path.join(shared_config["checkpoint_dir"], "global_model.pt")
}

# === Default Hyperparameters for Personalized Mode ===
personalized_config = {
    "dropout": 0.0,
    "dropout_time": 0.2,

    # Optimization
    "fine_tune_strategy": "last_k_layers",  # options: last_k_layers
    "fine_tune_k": 1,  # number of Transformer layers to unfreeze
    "lr": 5e-4,
    "batch_size": 4,
    "epochs": 50,

    # Regularization
    "weight_decay": 0,  # No L2 for fine-tuning
    "l1_lambda": 1e-3,
    # Paths
    "save_model_path": os.path.join(shared_config["checkpoint_dir"], "global_model.pt")
}

# === Merge Config ===
mode = shared_config["mode"]
if mode == "generalized":
    config = {**shared_config, **generalized_config}
elif mode == "personalized":
    config = {**shared_config, **personalized_config}
else:
    raise ValueError(f"Unknown mode: {mode}")
