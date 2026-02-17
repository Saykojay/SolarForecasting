# PV Forecasting Pipeline - Modular v1

Pipeline modular untuk prediksi PV output menggunakan model Deep Learning (PatchTST, GRU, dll).

## Quick Start

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Letakkan CSV data di folder data/raw/

# 3. Jalankan controller
python main.py
```

## Struktur
```
├── config.yaml           # Konfigurasi terpusat
├── main.py               # Universal TUI Controller
├── src/
│   ├── config_loader.py  # Pembaca config
│   ├── data_prep.py      # Preprocessing & Feature Engineering
│   ├── model_factory.py  # PatchTST, GRU, dll.
│   ├── trainer.py        # Training, Optuna, TSCV
│   └── predictor.py      # Evaluasi & Target Testing
├── data/
│   ├── raw/              # CSV mentah
│   ├── processed/        # Artefak .npy
│   └── target/           # Data Indonesia
└── models/               # Model tersimpan
```

## Cara Pakai
- **TUI Mode**: Jalankan `python main.py`, pilih opsi dari menu
- **CLI Mode**: `python main.py preprocess`, `python main.py train`, dll.
- **Edit Config**: Edit `config.yaml` atau lewat menu "Edit Konfigurasi" di TUI
