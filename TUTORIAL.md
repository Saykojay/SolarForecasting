# Tutorial: Solar PV Forecasting Pipeline

## Table of Contents
1. [Getting Started](#1-getting-started)
2. [Project Structure](#2-project-structure)
3. [Configuration Guide](#3-configuration-guide)
4. [Running the Pipeline](#4-running-the-pipeline)
5. [Web Dashboard (Streamlit)](#5-web-dashboard-streamlit)
6. [Terminal Mode (TUI)](#6-terminal-mode-tui)
7. [Command Line (CLI)](#7-command-line-cli)
8. [Changing Model Architecture](#8-changing-model-architecture)
9. [Hyperparameter Tuning (Optuna)](#9-hyperparameter-tuning-optuna)
10. [Batch Experiments](#10-batch-experiments)
11. [Time Series Cross-Validation (TSCV)](#11-time-series-cross-validation-tscv)
12. [Target Domain Testing](#12-target-domain-testing)
13. [Model Comparison](#13-model-comparison)
14. [Data Versioning & Model Management](#14-data-versioning--model-management)
15. [Key Features](#15-key-features)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Getting Started

### 1.1 Install Dependencies
```bash
# Create environment (recommended)
conda create -n solar python=3.10
conda activate solar

# Install requirements
pip install -r requirements.txt
```

> **Note**: For GPU acceleration, install TensorFlow with CUDA support and PyTorch with CUDA:
> ```bash
> pip install tensorflow[and-cuda]
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 1.2 Prepare Your Data
Place your PV system CSV file in `data/raw/`:
```
data/raw/your_pv_data.csv
```

**Required CSV columns** (names are configurable in `config.yaml`):
| Column | Description | Example |
|:-------|:-----------|:--------|
| `timestamp` | Date/time column | `01/01/2024 00:00` |
| `pv_output_kw` | PV power output (kW) | `5.32` |
| `ghi_wm2` | Global Horizontal Irradiance (W/m²) | `450.0` |
| `ambient_temp_c` | Ambient temperature (°C) | `28.5` |
| `relative_humidity_pct` | Relative humidity (%) | `65.0` |
| `wind_speed_ms` | Wind speed (m/s) | `3.2` |

### 1.3 Launch the Dashboard
```bash
streamlit run app.py
```
Dashboard opens at **http://localhost:8501**

**Windows shortcut**: Double-click `launch_dashboard.bat` — auto-detects your Conda environment.

---

## 2. Project Structure

```
├── app.py                    # Web Dashboard (Streamlit) — main interface
├── main.py                   # CLI / TUI entry point
├── config.yaml               # Global configuration file
├── requirements.txt          # Python dependencies
├── launch_dashboard.bat      # Windows launcher (auto-detect Conda)
│
├── src/                      # Core pipeline modules
│   ├── data_prep.py          # Data cleaning, feature engineering, sequencing
│   ├── model_factory.py      # All Keras model architectures
│   ├── model_hf.py           # HuggingFace/PyTorch model wrappers
│   ├── trainer.py            # Training, Optuna, TSCV, batch experiments
│   ├── predictor.py          # Evaluation, inverse transforms, target testing
│   ├── baseline_models.py    # Classical ML & physics-based baselines
│   ├── config_loader.py      # Configuration management
│   └── lang.py               # UI label definitions
│
├── data/
│   ├── raw/                  # Place raw CSV data here
│   ├── processed/            # Auto-generated preprocessing artifacts
│   └── target/               # Place cross-domain test data here
│
├── models/                   # Saved models with metadata and scalers
├── configs/presets/           # Feature engineering presets
├── scripts/                  # Utility scripts (data merging, comparison)
├── docs/                     # Documentation & tuning strategy guides
└── logs/
    └── session/              # Persistent data (training history, eval results)
```

**Key Principle**: You primarily work with two things:
1. `config.yaml` — to change parameters
2. `app.py` (Dashboard) or `main.py` (CLI) — to run the pipeline

---

## 3. Configuration Guide

All pipeline behavior is controlled through `config.yaml`.

### 3.1 Data Source
```yaml
data:
  csv_path: "data/raw/your_pv_data.csv"
  csv_separator: ";"
  target_col: "pv_output_kw"
  time_col: "timestamp"
  time_format: '%d/%m/%Y %H:%M'
  ghi_col: "ghi_wm2"
```

### 3.2 PV System Parameters
```yaml
pv_system:
  nameplate_capacity_kw: 10.5   # Panel capacity (kW)
  temp_coeff: -0.0045           # Temperature coefficient
  system_efficiency: 0.86       # System efficiency
```

### 3.3 Data Splitting
```yaml
splitting:
  split_mode: "standard"        # "standard" or "seasonal"
  train_ratio: 0.8              # 80% training
  test_ratio: 0.2               # 20% testing
```

### 3.4 Model Architecture & Hyperparameters
```yaml
model:
  architecture: "patchtst"      # See supported architectures below
  hyperparameters:
    lookback: 72                # Input window (hours)
    patch_len: 24               # Patch length (Transformer only)
    stride: 12                  # Patch stride (Transformer only)
    d_model: 64                 # Model dimension
    n_heads: 8                  # Number of attention heads
    n_layers: 3                 # Number of encoder layers
    dropout: 0.2                # Dropout rate
    learning_rate: 0.001        # Learning rate
    batch_size: 32              # Batch size
    loss_fn: "huber"            # Loss function (mse, mae, huber)
```

**Supported architectures:**
`patchtst`, `patchtst_hf`, `autoformer`, `autoformer_hf`, `timetracker`, `timeperceiver`, `gru`, `lstm`, `rnn`, `causal_transformer_hf`, `linear`, `mlp`

### 3.5 Feature Engineering
```yaml
features:
  selection_mode: "auto"        # "auto" or "manual"
  corr_threshold: 0.1           # Minimum correlation with target
  multicol_threshold: 0.95      # Maximum inter-feature correlation
  scaler_type: "minmax"         # "minmax" or "standard"
  groups:
    weather: true               # Weather features (GHI, Temp, RH, Wind)
    lags: true                  # Lag features (1h, 12h, 24h, 48h, 168h)
    physics: true               # Physics features (Clear Sky, CSI)
    rolling: true               # Moving average & std deviation
    time_hour: true             # Cyclical hour encoding (sin/cos)
    time_month: true            # Cyclical month encoding
    time_doy: true              # Cyclical day-of-year encoding
```

### 3.6 Preprocessing (Data Cleaning)
```yaml
preprocessing:
  resample_1h: true             # Resample data to 1-hour intervals
  remove_outliers: true         # Enable outlier detection
  ghi_high_pv_zero: true        # Remove PV=0 when GHI is bright
  ghi_dark_pv_high: true        # Remove high PV when GHI is dark
  fix_ghi_dhi: true             # Fix physical inconsistency (GHI < DHI)
  clip_precipitation: 100.0     # Cap extreme rainfall values (mm/h)
  impute_missing_pv: false      # Impute missing PV using CSI correlation
```

### 3.7 Target Variable
```yaml
target:
  use_csi: true                 # Predict Clear-Sky Index (recommended)
  csi_ghi_threshold: 50         # GHI threshold for CSI calculation
  csi_max: 1.2                  # Maximum CSI clipping value
```

### 3.8 Training Configuration
```yaml
training:
  epochs: 100                   # Maximum training epochs
  patience: 15                  # Early stopping patience
  lr_patience: 7                # LR scheduler patience
  lr_factor: 0.2                # LR reduction factor
  min_lr: 0.000001              # Minimum learning rate

forecasting:
  horizon: 24                   # Forecast horizon (hours ahead)
```

### 3.9 Optuna Tuning & TSCV
```yaml
tuning:
  enabled: false                # Enable Optuna tuning
  n_trials: 50                  # Number of trials
  search_space:
    d_model: [32, 64, 128, 256]
    n_layers: [2, 6]
    dropout: [0.1, 0.3]
    learning_rate: [0.00005, 0.0005]
    batch_size: [32]
    lookback: [72, 336, 24]     # [min, max, step]

tscv:
  enabled: false                # Enable Time Series Cross-Validation
  n_splits: 5                   # Number of CV folds
```

---

## 4. Running the Pipeline

There are **3 ways** to run the pipeline:

| Mode | Best For | Command |
|:-----|:---------|:--------|
| **Web Dashboard** | Visual workflow, monitoring | `streamlit run app.py` |
| **TUI** (Interactive menu) | Terminal exploration | `python main.py` |
| **CLI** (Direct commands) | Automation, scripting | `python main.py <command>` |

---

## 5. Web Dashboard (Streamlit)

> **Recommended** — full visual interface with real-time monitoring.

### 5.1 Sidebar Controls

The sidebar contains:
- **Device Acceleration** — Toggle GPU / CPU
- **Model Manager** — Select which saved model to use for evaluation
- **Save Config** — Persist changes to `config.yaml`
- **Stop All Processes** — Emergency stop for runaway processes

### 5.2 Dashboard Tabs

The dashboard follows a logical ML pipeline workflow:

| # | Tab | Purpose | Key Actions |
|:-:|:----|:--------|:------------|
| 1 | **Data Prep & Features** | Data preprocessing | Select dataset, configure features, cleaning rules, run preprocessing |
| 2 | **Data Insights** | Data verification | View cleaning stats, correlation matrix, rolling correlation analysis, sequence preview |
| 3 | **Baseline & Physics** | Benchmark models | Run Linear Regression, Random Forest, PVWatts, Single Diode models |
| 4 | **Training Center** | Model training | Select data version, configure architecture, run training with live loss curves |
| 5 | **Batch Experiments** | Multi-model queue | Queue experiments with different architectures and hyperparameters, run all automatically |
| 6 | **Optuna Tuning** | HP optimization | Configure search space, run Bayesian optimization, view trial history |
| 7 | **Prediction / Eval** | Evaluation | Select model, run evaluation, view metrics (MAE, RMSE, R², MAPE), prediction charts |
| 8 | **Model Comparison** | Leaderboard | Select multiple models, compare metrics side-by-side, visualize relative performance |
| 9 | **Target Testing** | Cross-domain test | Load external data, test trained models on different locations |

### 5.3 Model Manager

Every training run saves a model in the `models/` folder with a unique name (e.g., `patchtst_20260212_1330`).

The **Model Manager** in the sidebar lets you:
- View all saved models
- Switch between models for evaluation
- The latest model is automatically selected after training

### 5.4 Live Training Monitor

During training, you see in real-time:
- **Progress bar** with percentage and ETA
- **Loss curve chart** updated every epoch
- **Metric cards** showing Train Loss, Val Loss
- **Epoch log** with detailed per-epoch information

### 5.5 Data Persistence

Your data **survives page refreshes**:

| Data | Storage Location |
|:-----|:-----------------|
| Training history (loss curves) | `logs/session/last_training_history.json` |
| Evaluation metrics | `logs/session/last_eval_results.json` |
| Selected model | `logs/session/selected_model.txt` |
| Model files & metadata | `models/` |

---

## 6. Terminal Mode (TUI)

> Requires Anaconda Prompt, CMD, or PowerShell (not VS Code terminal).

```bash
python main.py
```

Interactive menu:
```
? What would you like to do?
> 1. Preprocessing (Data → Artifacts)
  2. Training (Train Model)
  3. Hyperparameter Tuning (Optuna)
  4. TSCV (Cross-Validation)
  5. Evaluate (Metrics & Analysis)
  6. Target Testing (External Data)
  7. Full Pipeline (All Automatic)
  8. Edit Configuration
  9. Exit
```

---

## 7. Command Line (CLI)

```bash
python main.py preprocess     # Preprocess CSV → .npy tensors
python main.py train          # Train model
python main.py tune           # Run Optuna hyperparameter tuning
python main.py tscv           # Time Series Cross-Validation
python main.py evaluate       # Evaluate trained model
python main.py target         # Test on external data
python main.py full           # Full pipeline (Preprocess → Train → Evaluate)
```

### Example: Full Pipeline
```bash
python main.py full
```

Expected output:
```
[1/4] Preprocessing...
  Raw Split: Train=39463, Test=4385
  19 features selected
  Train: (39176, 96, 19), Test: (4266, 96, 19)

[2/4] Tuning skipped (disabled)

[3/4] Training...
  Epoch 1/100 - loss: 0.0800 - val_loss: 0.0500
  ...
  Model saved: models/patchtst_20260212_1130

[4/4] Evaluating...
  TRAIN - MAE=0.25, RMSE=0.45, R²=0.85
  TEST  - MAE=0.26, RMSE=0.46, R²=0.84

FULL PIPELINE COMPLETE!
```

---

## 8. Changing Model Architecture

### Available Architectures

| Architecture | Type | config.yaml value |
|:-------------|:-----|:------------------|
| PatchTST (Keras) | Transformer | `patchtst` |
| PatchTST (HuggingFace) | Transformer/PyTorch | `patchtst_hf` |
| Autoformer (Keras) | Transformer | `autoformer` |
| Autoformer (HuggingFace) | Transformer/PyTorch | `autoformer_hf` |
| Causal Transformer (HF) | Decoder-only/PyTorch | `causal_transformer_hf` |
| TimeTracker | Transformer + MoE | `timetracker` |
| TimePerceiver | Latent Bottleneck | `timeperceiver` |
| GRU | Recurrent | `gru` |
| LSTM | Recurrent | `lstm` |
| SimpleRNN | Recurrent | `rnn` |
| Linear | Baseline | `linear` |
| MLP | Baseline | `mlp` |

### How to Switch

**Option 1 — Edit `config.yaml`:**
```yaml
model:
  architecture: "gru"    # Change to desired architecture
```

**Option 2 — Via TUI:**
`python main.py` → "Edit Configuration" → "Architecture"

**Option 3 — Via Dashboard:**
Set architecture in the Training Center tab → Save Config

> **Note**: After changing architecture, re-run Training. Preprocessing does not need to be repeated if data and features haven't changed.

---

## 9. Hyperparameter Tuning (Optuna)

### Enable Tuning
In `config.yaml`:
```yaml
tuning:
  enabled: true
  n_trials: 50
```

Or via Dashboard: Optuna Tuning tab → configure and run.

### Search Space Configuration
```yaml
tuning:
  search_space:
    patch_len: [12, 24, 4]              # [min, max, step]
    stride: [4, 12, 4]
    d_model: [32, 64, 128, 256]         # Categorical choices
    n_heads: [4, 8, 12]
    n_layers: [2, 6]                    # [min, max] range
    dropout: [0.1, 0.3]                 # [min, max] float range
    learning_rate: [0.00005, 0.0005]    # [min, max] log scale
    batch_size: [32]                    # Fixed value
    lookback: [72, 336, 24]             # [min, max, step]
```

### Run Tuning
```bash
python main.py tune
```
Or use the Optuna Tuning tab in the dashboard for visual trial monitoring.

---

## 10. Batch Experiments

The **Batch Experiments** tab lets you queue multiple training configurations:

1. Open the **Batch Experiments** tab
2. Configure each experiment:
   - Architecture (e.g., PatchTST, GRU, LSTM)
   - Hyperparameters
   - Data version
3. Click **"Add to Queue"**
4. Repeat for all desired configurations
5. Click **"Run Batch Now"** to train all models sequentially

Results are collected and can be compared in the **Model Comparison** tab.

---

## 11. Time Series Cross-Validation (TSCV)

### Enable TSCV
```yaml
tscv:
  enabled: true
  n_splits: 5
```

### Run TSCV
```bash
python main.py tscv
```

Output:
```
--- Fold 1/5 ---
  Train: 6500, Val: 6500
  MAE=0.25, RMSE=0.42, R²=0.84

--- Fold 2/5 ---
  ...

TSCV SUMMARY
  Avg MAE:  0.26
  Avg RMSE: 0.44
  Avg R²:   0.83
```

---

## 12. Target Domain Testing

Test your trained model on data from a different location or system:

1. Place the external CSV in `data/target/`
2. Ensure the CSV has the same column format as training data
3. In the dashboard, go to **Target Testing** tab
4. Select the model and target CSV file
5. Run the test

The system will:
- Apply the same preprocessing and scaling as the training data
- Generate predictions using the trained model
- Display performance metrics on the external data

---

## 13. Model Comparison

Compare multiple trained models side-by-side:

1. Go to **Model Comparison** tab
2. Select 2+ models from the list
3. Choose evaluation method:
   - **Original Dataset** — use each model's training data
   - **Cross-Domain** — test all models on active target data
4. Click **"Run Comparison Analysis"**

The comparison shows:
- Leaderboard table (MAE, RMSE, R², MAPE)
- Bar charts comparing key metrics
- Overfitting analysis (Train R² vs Test R²)
- Inference time comparison

---

## 14. Data Versioning & Model Management

### Preprocessed Data Versions
Each preprocessing run creates a uniquely-named folder in `data/processed/` (e.g., `v_0307_1430_dataset`).

**To use a specific version for training:**
1. Open the **Training Center** tab
2. Expand **"Select Preprocessed Data Version"**
3. Choose the desired version (default: Latest)

### Model Registry
Models are saved in `models/` with:
- Model weights (`.keras` or PyTorch files)
- `meta.json` — training metadata, hyperparameters, metrics
- `scaler_X.pkl` / `scaler_y.pkl` — fitted scalers for inference

Switch between models using the **Model Manager** in the sidebar.

---

## 15. Key Features

### Cyclical Time Encoding (Sin/Cos)
Instead of using raw numbers (hour 0-23, month 1-12), the pipeline uses trigonometric transformations:
- **Hourly**: Connects hour 23 back to hour 0 smoothly
- **Monthly**: Connects December to January for annual cycle capture
- **Day of Year**: High-resolution 1-365 seasonal encoding

### Physics-Based Preprocessing (Algorithm 1)
- Physical extreme detection (GHI > 2000, Temp < -30, etc.)
- PV-GHI inconsistency detection (PV=0 when sun is bright)
- Frozen sensor detection (stagnant readings > 10 hours)
- Precipitation clipping for extreme rainfall values
- Optional PV imputation using CSI-based correlation

### Clear-Sky Index (CSI) Normalization
Predicting CSI (radiation ratio) instead of raw power (kW) improves model accuracy by removing the daily solar cycle from the prediction task.

### RevIN (Reversible Instance Normalization)
Applied in Transformer and optionally in RNN models to normalize input distributions per-instance, improving generalization across different weather patterns.

---

## 16. Troubleshooting

### `No module named 'tensorflow'`
**Cause**: Wrong Python environment.
**Fix**: Activate your conda environment:
```bash
conda activate solar
```

### `NoConsoleScreenBufferError`
**Cause**: Terminal doesn't support TUI (e.g., VS Code terminal).
**Fix**: Use CLI mode or the web dashboard:
```bash
python main.py full          # CLI mode
streamlit run app.py         # Web dashboard
```

### `FileNotFoundError: Config file not found`
**Cause**: Running from wrong directory.
**Fix**: Navigate to the project folder:
```bash
cd "path/to/Modular Pipeline v1"
```

### GPU Out of Memory (OOM)
**Cause**: GPU memory not freed after previous run.
**Fix**:
1. Restart Streamlit (`Ctrl+C` then `streamlit run app.py`)
2. Reduce `batch_size` in config.yaml
3. Use a smaller `d_model` or fewer `n_layers`

### Shape Mismatch during Evaluation
**Cause**: Horizon/lookback settings don't match the loaded model.
**Fix**: The system auto-detects these from the model's metadata. If issues persist, ensure the correct model is selected in Model Manager.

### Preprocessing: Data reduced drastically
**Cause**: Too many temporal gaps in raw data.
**Fix**: Adjust split ratio or relax cleaning rules:
```yaml
splitting:
  train_ratio: 0.8
preprocessing:
  remove_outliers: false    # Temporarily disable
```

### Port 8501 already in use
**Cause**: Previous Streamlit instance still running.
**Fix**:
```bash
# Windows: Kill existing processes
taskkill /F /IM python.exe /T

# Then restart
streamlit run app.py
```

---

## Daily Workflow (Recommended)

### Using the Dashboard:
```
1. Launch: streamlit run app.py
2. Configure preprocessing in "Data Prep & Features" tab
3. Run preprocessing → verify in "Data Insights" tab
4. Configure model in "Training Center" tab → Start Training
5. Monitor training progress in real-time
6. View results in "Prediction / Eval" tab
7. Compare models in "Model Comparison" tab
```

### Using CLI (Quick):
```
1. Edit config.yaml with desired parameters
2. Run: python main.py full
3. View metrics in terminal output
4. Repeat with different configurations
```
