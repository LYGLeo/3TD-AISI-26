# (AAAI'26) Enabling Delayed-Full Charging through Transformer-Based Real-Time-to-Departure  Modeling for EV Battery Longevity

This repository provides an implementation of a Transformer-based Time-to-Event (TTE) model for predicting EV departure times using smartphone sensor data.

The model is designed to capture temporal behavioral patterns and contextual signals to enable accurate and real-time departure prediction.

⚠️ Note
This repository is currently under active development. Improved versions and additional features will be released in future updates.

📊 Dataset
For evaluation, we collected a dataset using our self-developed mobile application, EVA (EV Analyzer. See https://eva.hanyang.ac.kr).
This dataset is an in-the-wild human behavioral dataset gathered from real-world EV users via smartphone sensing.

Due to the involvement of human subjects, the dataset is not publicly available and is regulated under IRB (Institutional Review Board) protocols.

📄 More Information
For detailed descriptions of the dataset, data collection process, and experimental setup, please refer to the paper and supplementary materials available on arXiv:
https://arxiv.org/pdf/2512.07723

📧 Contact
For inquiries regarding the dataset or collaboration opportunities, please contact:
yonggeonlee@hanyang.ac.kr


## Overview

Our main contribution is a **TTE Transformer** that uses ordinal regression loss to predict departure times from sequential sensor data. Unlike previous methods primarily dependent on temporal dependency from historical patterns, our method leverages streaming contextual behavioral and environmental information to predict departures. The model can operate in two modes:
- **Generalized**: Single model trained on multiple users
- **Personalized**: Fine-tuned models for individual users

## Project Structure

### Core TTE Transformer Implementation
- `tte_transformer.py` - Main transformer architecture
- `config.py` - Configuration settings for both modes
- `data.py` - Data loading and preprocessing
- `train.py` - Training pipeline for both generalized and personalized modes
- `loss.py` - Ordinal regression loss function
- `test.py` - Evaluation and inference

### Hyperparameter Optimization & Analysis
- `tune.py` - Hyperparameter tuning using grid search
- `evaluate_ablation.py` - Ablation studies for model components
- `evaluate_sensitivity.py` - Sensitivity analysis for loss weights and thresholds

### Baseline Comparisons
- `evaluate_baseline_regressor.py` - Regression baselines (MLR, SVR, LGBM, FT-Transformer)
- `evaluate_baselines_LGBM_classifier.py` - LightGBM classifier baseline
- `evaluate_baseline_dbscan.py` - DBSCAN-based departure detection (post-processing of LightGBM (classifier) probabilities)

## Quick Start

### 1. Training Generalized Model
```bash
# First ensure config.py has mode="personalized"
python train.py
```

### 2. Training Personalized Models
```bash
# First ensure config.py has mode="personalized"
python train.py
```

### 3. Evaluation
```bash
python test.py --checkpoint path/to/model.pt
```

### 4. Hyperparameter Tuning
```bash
python tune.py
```

### 5. Ablation Studies
```bash
python evaluate_ablation.py
python test.py --checkpoint_dir ./results/ablation/checkpoints
```

### 6. Sensitivity Analysis
```bash
python evaluate_sensitivity.py
python test.py --checkpoint_dir ./results/sensitivity/checkpoints
```

## Configuration

Edit `config.py` to adjust:
- Model architecture (d_model, nhead, num_layers)
- Training parameters (lr, batch_size, epochs, early stopping patience)
- Loss weights (weekend_weight, event_weight)
- Uncertainty threshold for inference
- Feature usage flags (use_time_features, use_context_features, etc.)

## Data Format

The model uses passive smartphone sensor data including:
- **ACTIVITY**: Physical activity patterns
- **APP_USAGE**: Application usage patterns  
- **CALLS**: Phone call patterns
- **LIGHT**: Ambient light levels
- **MOTION**: Device motion patterns
- **SCREEN**: Screen on/off patterns
- **SOUND**: Audio environment patterns
- **STEP**: Step count patterns
- **UNLOCK**: Device unlock patterns

Expected data structure:
```
Dataset/
├── DATASET/
│   ├── user_1/
│   │   ├── meta.csv          # Departure times and labels
│   │   ├── ACTIVITY.csv      # Activity sensor data
│   │   ├── APP_USAGE.csv     # App usage data
│   │   └── ... (other sensor files)
│   ├── user_2/
│   │   └── ...
│   ├── dow_labeled.csv       # Day-of-week labels
│   └── sig#120#5.csv         # Significant features for classifier
└── baseline_historical_dataset.csv  # Aggregated departure times(in meta.csv) for baseline models
```

**meta.csv format:**
- `user_id`: User identifier
- `date`: Date of the day
- `departure_time`: Actual departure time (target)
- `window_start`: Start time of the time window
- `window_end`: End time of the time window  
- `label`: Binary label (1 if departure occurs, 0 otherwise)

## Results

Results are saved in:
- `./results/checkpoints/` - Model checkpoints
- `./results/tuning/` - Hyperparameter tuning results
- `./results/ablation/` - Ablation study results
- `./results/test_eval/` - Test evaluation results
