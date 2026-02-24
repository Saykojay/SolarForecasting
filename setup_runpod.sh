#!/bin/bash
# ============================================================
# SETUP SCRIPT FOR RUNPOD (Clean Install)
# Template: RunPod PyTorch / TensorFlow (Python 3.11)
# GPU: RTX A5000 / RTX 4090 / RTX 3090
# ============================================================
# USAGE:
#   1. Deploy Pod baru di RunPod
#   2. Buka Jupyter Lab > Terminal
#   3. cd /workspace
#   4. git clone https://github.com/Saykojay/SolarForecasting.git
#   5. cd SolarForecasting
#   6. bash setup_runpod.sh
#   7. Tunggu sampai selesai (~3-5 menit)
#   8. Jalankan: bash start.sh
# ============================================================

set -e  # Stop on first error

echo "============================================"
echo " [1/5] Cleaning pip cache..."
echo "============================================"
pip cache purge 2>/dev/null || true
rm -rf /root/.cache/pip /tmp/pip-* 2>/dev/null || true

echo "============================================"
echo " [2/5] Installing PyTorch (CUDA 12.1)..."
echo "============================================"
# Install PyTorch FIRST with correct CUDA version
# Using --no-cache-dir to save storage
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-cache-dir 2>&1 | tail -5

echo "============================================"
echo " [3/5] Pinning numpy & core deps for TF..."
echo "============================================"
# Pin numpy < 2.0 BEFORE installing anything else
# This prevents TensorFlow/scipy/sklearn breakage
pip install "numpy==1.26.4" "protobuf==4.25.3" "ml-dtypes==0.2.0" \
    --force-reinstall --no-deps --no-cache-dir 2>&1 | tail -3

echo "============================================"
echo " [4/5] Installing HuggingFace & project deps..."
echo "============================================"
# Install transformers ecosystem
pip install transformers accelerate evaluate datasets --no-cache-dir 2>&1 | tail -5

# Install remaining project dependencies (skip torch, numpy, protobuf — already done)
pip install \
    scikit-learn \
    joblib \
    optuna \
    optuna-integration \
    mlflow \
    rich \
    questionary \
    pyyaml \
    python-dotenv \
    matplotlib \
    seaborn \
    streamlit \
    plotly \
    scipy \
    pandas==2.1.4 \
    openpyxl \
    --no-cache-dir \
    --ignore-installed blinker 2>&1 | tail -5

echo "============================================"
echo " [5/5] Re-pin numpy (safety check)..."
echo "============================================"
# Some packages may have overwritten numpy, force it back
pip install "numpy==1.26.4" --force-reinstall --no-deps --no-cache-dir 2>&1 | tail -1

echo ""
echo "============================================"
echo " VERIFYING INSTALLATION..."
echo "============================================"
python -c "
import numpy; print(f'  numpy:        {numpy.__version__}')
import tensorflow as tf; print(f'  TensorFlow:   {tf.__version__}')
import torch; print(f'  PyTorch:      {torch.__version__} | CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
import transformers; print(f'  Transformers: {transformers.__version__}')
import sklearn; print(f'  Sklearn:      {sklearn.__version__}')
import streamlit; print(f'  Streamlit:    {streamlit.__version__}')
import optuna; print(f'  Optuna:       {optuna.__version__}')
print()
print('  ✅ ALL LIBRARIES OK!')
print()
print('  Jalankan dashboard dengan:')
print('    bash start.sh')
print('  Atau:')
print('    streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.enableCORS=false --server.enableXsrfProtection=false')
"

echo "============================================"
echo " SETUP COMPLETE!"  
echo "============================================"
