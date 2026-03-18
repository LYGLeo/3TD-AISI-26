import os
import json
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np

from config import config
from tte_transformer import TransformerTTE
from loss import ordinal_regression_loss
from data import (
    load_preprocessed_data, DepartureDataset,
    build_split, split_generalized, split_personalized_single
)

# torch.manual_seed(42)
# np.random.seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def set_seed(seed):
    """재현성을 위해 랜덤 시드를 설정하는 함수"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Seed set to {seed}")

# === Training Function ===
def train_model(model, train_loader, val_loader, device, config, save_path=None,
                user_id=None, l1_lambda=0.0, use_weight_decay=True):
    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4) if use_weight_decay else 0.0
    )

    best_val_loss, best_epoch = float('inf'), -1
    patience, max_epochs = config["early_stopping_patience"], config["epochs"]
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x_context = batch['x_context'].to(device)
            abs_time = batch['abs_time'].to(device)
            dow_idx = batch['dow_idx'].to(device)
            event_time = batch['event_time'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()
            pred_q = model(x_context, abs_time, dow_idx)

            # Base task loss
            loss = ordinal_regression_loss(
                pred_q=pred_q, event_time=event_time, mask=mask,
                is_weekend=dow_idx,
                weekend_weight=config["weekend_weight"], event_weight=config["event_weight"],
                soft=config.get("use_gaussian_smoothing", True), sigma=config.get("sigma", 2.0)
            )

            # L1 regularization (optional for fine-tuning)
            if l1_lambda > 0:
                l1_penalty = sum(torch.sum(torch.abs(p)) for p in model.parameters() if p.requires_grad)
                loss += l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_loss = validate_model(model, val_loader, device, config)
        train_losses.append(avg_train_loss); val_losses.append(val_loss)

        print(f"[Epoch {epoch+1}/{max_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            epochs_no_improve=0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"🔥 Best model saved at epoch {epoch+1} ({save_path})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("⏹ Early stopping triggered.")
                break

    print(f"Learned abs_time_scale: {model.abs_time_scale.item():.4f}")
    # # Save loss plot
    # if save_path:
    #     plt.figure()
    #     plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val")
    #     plt.title(f"Loss Curve {'(User ' + str(user_id) + ')' if user_id else '(Generalized)'}")
    #     plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    #     plt.savefig(os.path.join(save_path, f"loss_plot_trial_{user_id}.png"))
    #     plt.close()

    return best_val_loss


# === Validation Function ===
@torch.no_grad()
def validate_model(model, val_loader, device, config):
    model.eval()
    val_loss = 0.0
    for batch in val_loader:
        pred_q = model(batch['x_context'].to(device), batch['abs_time'].to(device), batch['dow_idx'].to(device))
        loss = ordinal_regression_loss(
            pred_q=pred_q, event_time=batch['event_time'].to(device),
            mask=batch['mask'].to(device), is_weekend=batch['dow_idx'].to(device),
            weekend_weight=config["weekend_weight"], event_weight=config["event_weight"],
            soft=config.get("use_gaussian_smoothing", True), sigma=config.get("sigma", 2.0)
        )
        val_loss += loss.item()
    return val_loss / len(val_loader)


# === Fine-Tuning Strategy ===
def fine_tuning(model, strategy="last_k_layers", k=1):
    """
    strategy:
      - 'last_layer': Only output layer + time-related scalars
      - 'last_k_layers': Last k Transformer layers + embeddings + output layer
      - 'all': Full fine-tuning
    """
    if strategy == "last_layer":
        for name, param in model.named_parameters():
            if not (name.startswith("output_layer") or "abs_time_scale" in name or "alpha_param" in name):
                param.requires_grad = False

    elif strategy == "last_k_layers":
        total_layers = len(model.transformer_encoder.layers)
        for name, param in model.named_parameters():
            if (
                name.startswith("output_layer")
                or "abs_time_scale" in name
                or "alpha_param" in name
                or name.startswith("dow_embedding")
                or name.startswith("abs_time_proj")
            ):
                param.requires_grad = True
            elif name.startswith("transformer_encoder.layers"):
                layer_idx = int(name.split(".")[2])
                param.requires_grad = layer_idx >= total_layers - k
            else:
                param.requires_grad = False

    elif strategy == "all":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown fine-tune strategy: {strategy}")


# === Main Runner ===
def run(seed=None):

    if seed is not None:
        set_seed(seed)

    user_data = load_preprocessed_data(config)
    device = torch.device(config["device"])
    #global_model_path = "./results/checkpoints/global_model.pt"
    global_model_path = config.get("save_model_path", "./results/checkpoints/global_model.pt")

    if config["mode"] == "generalized":
        # === Global Model Training ===
        #with open("./results/tuning/best_params.json", "r") as f:
        #    best_params = json.load(f)
        #config.update(best_params)

        folds, _ = split_generalized(user_data, config)
        # Merge all folds but remove duplicates
        all_train_users = sorted(list(set(u for fold in folds for u in fold[0] + fold[1])))
        
        # ✅ Reserve 20% of users for validation
        random.seed(config['split_seed'])
        random.shuffle(all_train_users)

        val_size = max(1, int(0.2 * len(all_train_users)))  
        val_users = all_train_users[:val_size]
        train_users = all_train_users[val_size:]

        print(f"Global Training: Train users={len(train_users)}, Val users={len(val_users)}")

        # Build datasets
        global_data = build_split(user_data, train_users, val_users, [])
        train_loader = DataLoader(DepartureDataset(**global_data["train"]),
                                  batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(DepartureDataset(**global_data["val"]),
                                batch_size=config["batch_size"], shuffle=False)

        # Model
        model = TransformerTTE(
            input_dim_context=config["input_dim"], d_model=config["d_model"],
            nhead=config["nhead"], num_layers=config["num_layers"],
            dropout=config["dropout"], dropout_time=config["dropout_time"],
            max_seq_len=config["seq_len"], dow_embedding_dim=2, num_dow=2,
            # ablation
            use_positional_encoding=config.get("use_positional_encoding", True),
            use_alpha_fusion=config.get("use_alpha_fusion", True),
            use_abs_time_scale=config.get("use_abs_time_scale", True),
            use_time_features=config.get("use_time_features", True),
            use_context_features=config.get("use_context_features", True),
            use_DoW=config.get("use_DoW", True)
        )

        print("Training global model with best hyperparameters (with internal validation)...")
        # train_model(model, train_loader, val_loader, device, config,
        #             save_path=global_model_path, use_weight_decay=True)
        train_model(model, train_loader, val_loader, device, config,
                    save_path=global_model_path, use_weight_decay=True)

    elif config["mode"] == "personalized":
        # === Personalized Fine-Tuning ===
        print(global_model_path)
        if not os.path.exists(global_model_path):
            raise FileNotFoundError("Run generalized mode first to create global model.")

        # Check if feature selection should be applied
        apply_feature_selection = config.get("apply_feature_selection", False)
        if apply_feature_selection:
            print("🔍 Feature selection enabled: Using only p<0.05 features for personalized models")

        folds, test_users = split_generalized(user_data, config)
        for user_id in test_users:
            print(f"\nFine-tuning for user {user_id}...")
            user_split = split_personalized_single(
                user_id, 
                user_data, 
                config, 
                apply_feature_selection=apply_feature_selection
            )
            
            # Create DataLoaders with the new dataset structure
            train_loader = DataLoader(user_split["train"],
                                      batch_size=config["batch_size"], shuffle=True)
            
            val_loader = DataLoader(user_split["val"],
                        batch_size=config["batch_size"], shuffle=False)

            model = TransformerTTE(
                input_dim_context=config["input_dim"], d_model=config["d_model"],
                nhead=config["nhead"], num_layers=config["num_layers"],
                dropout=config["dropout"], dropout_time=config["dropout_time"],
                max_seq_len=config["seq_len"], dow_embedding_dim=2, num_dow=2,

                use_positional_encoding=config["use_positional_encoding"],
                use_alpha_fusion=config["use_alpha_fusion"],
                use_abs_time_scale=config["use_abs_time_scale"],

                # Feature-level
                use_time_features=config["use_time_features"],
                use_context_features=config["use_context_features"],
                use_DoW=config["use_DoW"],  
            )
            model.load_state_dict(torch.load(global_model_path))

            # Apply fine-tuning strategy
            fine_tuning(model, strategy=config.get("fine_tune_strategy", "last_k_layers"),
                        k=config.get("fine_tune_k", 1))

            fine_tune_path = f"./results/checkpoints/fine_tuned_{user_id}.pt"
            train_model(model, train_loader, val_loader, device, config,
                        fine_tune_path, user_id=user_id,
                        l1_lambda=config.get("l1_lambda", 0.0), use_weight_decay=False)

    else:
        raise ValueError("Invalid mode in config. Use 'generalized' or 'personalized'.")


if __name__ == "__main__":
    run(seed=42)