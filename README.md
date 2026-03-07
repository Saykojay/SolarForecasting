# ☀️ Solar PV Forecasting Pipeline

An end-to-end deep learning pipeline for photovoltaic (PV) power output forecasting. Built for researchers and engineers who need accurate solar energy predictions using state-of-the-art time series models.

## Features

- **9 Model Architectures** — PatchTST, Autoformer, TimeTracker (MoE), TimePerceiver, GRU, LSTM, SimpleRNN, Linear, MLP
- **HuggingFace Integration** — PatchTST and Autoformer via 🤗 Transformers (PyTorch)
- **Automated Preprocessing** — Physics-based data cleaning, cyclical encoding, feature engineering
- **Optuna Hyperparameter Tuning** — Bayesian optimization with pruning
- **Batch Experiments** — Queue and run multiple model configurations automatically
- **Interactive Dashboard** — Full Streamlit web interface with live training monitoring
- **Model Comparison** — Side-by-side leaderboard with comprehensive metrics
- **Target Domain Testing** — Cross-location transfer learning evaluation
- **Baseline Benchmarks** — Classical ML and physics models (PVWatts, Single Diode) for comparison

## Quick Start

### 1. Install Dependencies
```bash
# Create and activate a conda environment (recommended)
conda create -n solar python=3.10
conda activate solar

# Install requirements
pip install -r requirements.txt
```

### 2. Prepare Data
Place your PV system CSV file in the `data/raw/` folder:
```
data/raw/your_pv_data.csv
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

**Or use the launcher (Windows):**
Double-click `launch_dashboard.bat` — it auto-detects your Conda environment.

**Or use CLI:**
```bash
python main.py full    # Run full pipeline (preprocess → train → evaluate)
```

## Project Structure

```
├── app.py                    # Streamlit Web Dashboard (main interface)
├── main.py                   # CLI / TUI entry point
├── config.yaml               # Global configuration
├── requirements.txt          # Python dependencies
├── launch_dashboard.bat      # Windows launcher (auto-detect Conda)
│
├── src/                      # Core pipeline modules
│   ├── data_prep.py          # Data cleaning, feature engineering, sequencing
│   ├── model_factory.py      # All model architectures (Keras)
│   ├── model_hf.py           # HuggingFace model wrappers (PyTorch)
│   ├── trainer.py            # Training, Optuna tuning, batch experiments
│   ├── predictor.py          # Evaluation, inverse transforms, target testing
│   ├── baseline_models.py    # Classical ML & physics baselines
│   ├── config_loader.py      # Configuration management
│   └── lang.py               # UI label definitions
│
├── data/
│   ├── raw/                  # Place raw CSV files here
│   ├── processed/            # Auto-generated preprocessing artifacts (.npy)
│   └── target/               # Place cross-domain test CSV files here
│
├── models/                   # Saved models with metadata and scalers
├── configs/presets/           # Feature engineering presets
├── scripts/                  # Data utility scripts
├── docs/                     # Technical documentation & tuning strategies
├── setup_runpod.sh           # Cloud GPU deployment script (RunPod)
└── start.sh                  # Linux/RunPod quick-start
```

## Dashboard Tabs

| Tab | Purpose |
|:----|:--------|
| **Data Prep & Features** | Configure preprocessing, feature selection, data splitting |
| **Data Insights** | Visualize data pipeline flow, correlation matrices, feature analysis |
| **Baseline & Physics** | Run classical ML and physics-based benchmark models |
| **Training Center** | Configure architecture, hyperparameters, and run training with live monitoring |
| **Batch Experiments** | Queue multiple experiments and run them sequentially |
| **Optuna Tuning** | Automated hyperparameter optimization with visualization |
| **Prediction / Eval** | Evaluate model performance with comprehensive metrics and charts |
| **Model Comparison** | Side-by-side model leaderboard and analysis |
| **Target Testing** | Test trained models on data from different locations |

## Supported Architectures

| Architecture | Type | Key Feature |
|:-------------|:-----|:------------|
| **PatchTST** | Transformer | Channel-independent patching with RevIN normalization |
| **PatchTST (HF)** | Transformer (PyTorch) | HuggingFace Transformers implementation |
| **Autoformer** | Transformer | Series decomposition with auto-correlation |
| **Autoformer (HF)** | Transformer (PyTorch) | HuggingFace Transformers implementation |
| **TimeTracker** | Transformer + MoE | Mixture-of-Experts with causal attention |
| **TimePerceiver** | Transformer | Latent bottleneck encoder with query decoder |
| **GRU** | Recurrent | Bidirectional GRU with optional RevIN |
| **LSTM** | Recurrent | Bidirectional LSTM with optional RevIN |
| **SimpleRNN** | Recurrent | Bidirectional SimpleRNN baseline |
| **Causal Transformer (HF)** | Transformer (PyTorch) | Decoder-only causal language model style |

## Configuration

All parameters are managed through `config.yaml` or the dashboard sidebar:

```yaml
model:
  architecture: "patchtst"        # Model architecture to use
  hyperparameters:
    lookback: 72                  # Input window (hours)
    d_model: 64                   # Model dimension
    n_heads: 8                    # Attention heads
    n_layers: 3                   # Encoder layers
    dropout: 0.2                  # Dropout rate
    learning_rate: 0.001          # Learning rate
    batch_size: 32                # Batch size

forecasting:
  horizon: 24                     # Forecast horizon (hours)

target:
  use_csi: true                   # Predict Clear-Sky Index (recommended)
```

## Deployment

### Local (Windows)
```bash
streamlit run app.py
```

### Cloud GPU (RunPod)
```bash
bash setup_runpod.sh
bash start.sh
```

See `docs/runpod_deployment_guide.md` for detailed instructions.

## Documentation

- [TUTORIAL.md](TUTORIAL.md) — Comprehensive usage guide
- [docs/](docs/) — Technical documentation, tuning strategies, and deployment guides

## License

This project is developed for academic research purposes.

---

*Built with TensorFlow, PyTorch, Streamlit, and Optuna.*
