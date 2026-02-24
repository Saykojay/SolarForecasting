# 🚀 Panduan Setup RunPod — Solar Forecasting Dashboard

Panduan singkat dan lengkap untuk men-deploy sistem Solar Forecasting di RunPod, mulai dari membuat Pod hingga menjalankan Hyperparameter Tuning.

---

## 📋 Prasyarat

- Akun RunPod aktif dengan saldo/kredit
- Repository GitHub: `https://github.com/Saykojay/SolarForecasting.git`

---

## Step 1: Buat Pod Baru

1. Login ke [RunPod.io](https://runpod.io)
2. Klik **"+ Deploy"** atau **"New Pod"**
3. Pilih konfigurasi:

| Setting | Rekomendasi |
|---|---|
| **Template** | `RunPod PyTorch 2.x` (Python 3.11) |
| **GPU** | RTX A4500 (20GB) / RTX A5000 (24GB) / RTX 4090 (24GB) |
| **Volume** | Minimal 20 GB |
| **Expose Port** | Tambahkan port **`8501`** (untuk Streamlit) |

4. Klik **Deploy** dan tunggu Pod berstatus **Running**

---

## Step 2: Buka Terminal

1. Klik **"Connect"** pada Pod yang sudah running
2. Pilih **"Jupyter Lab"** (bukan "SSH")
3. Di Jupyter Lab, klik **"Terminal"** untuk membuka terminal baru

---

## Step 3: Clone Repository

```bash
cd /workspace
git clone https://github.com/Saykojay/SolarForecasting.git
cd SolarForecasting
```

---

## Step 4: Install Semua Library

Jalankan script setup otomatis (estimasi ~3-5 menit):

```bash
bash setup_runpod.sh
```

Script ini secara otomatis akan:
- Membersihkan pip cache
- Install PyTorch dengan CUDA 12.1
- Pin NumPy ke versi 1.26.4 (mencegah konflik TensorFlow)
- Install HuggingFace Transformers, Optuna, Streamlit, dan semua dependensi
- Verifikasi semua library terinstall benar

**⚠️ Jika ada error**, coba jalankan ulang `bash setup_runpod.sh`. Biasanya error pertama disebabkan cache yang conflict.

### Verifikasi Manual (Opsional)

```bash
python -c "
import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')
import tensorflow as tf; print(f'TF: {tf.__version__}')
import streamlit; print(f'Streamlit: {streamlit.__version__}')
import optuna; print(f'Optuna: {optuna.__version__}')
print('✅ ALL OK')
"
```

---

## Step 5: Jalankan Streamlit Dashboard

### Cara Cepat (Foreground)
```bash
bash start.sh
```

### Cara Aman (Background — Tidak Mati Saat Tab Ditutup)
```bash
nohup bash start.sh > streamlit.log 2>&1 &
```

Pantau log-nya:
```bash
tail -f streamlit.log
```

---

## Step 6: Akses Dashboard

1. Kembali ke halaman Pod di RunPod
2. Klik **"Connect"** → cari tombol **port 8501**
3. Atau buka URL: `https://<POD_ID>-8501.proxy.runpod.net`
4. Dashboard Streamlit akan muncul 🎉

---

## 🔧 Menjalankan Tuning via CLI (Terminal)

Selain dari dashboard Streamlit, Anda bisa menjalankan Hyperparameter Tuning langsung dari terminal. Ini **lebih ringan dan stabil** untuk tuning jangka panjang.

### Contoh Perintah

```bash
# GRU (paling cepat untuk RNN)
python run_tuning_cli.py gru --trials 50 --subsample

# LSTM
python run_tuning_cli.py lstm --trials 50 --subsample

# Autoformer
python run_tuning_cli.py autoformer --trials 100 --subsample

# PatchTST
python run_tuning_cli.py patchtst_hf --trials 100 --subsample

# Causal Transformer 
python run_tuning_cli.py causal_transformer_hf --trials 100 --subsample
```

### Opsi Tambahan
| Flag | Fungsi |
|---|---|
| `--trials N` | Jumlah trial Optuna |
| `--subsample` | Gunakan 20% data (lebih cepat) |
| `--ratio 0.3` | Ganti rasio subsample (default 0.2) |

---

## 📌 Tips Penting: Gunakan `tmux`

`tmux` menjaga proses tetap hidup meskipun tab browser/koneksi putus. **Wajib digunakan untuk tuning jangka panjang!**

### Cheat Sheet tmux

```bash
# Buat sesi baru
tmux new -s tuning

# Detach (keluar tanpa mematikan proses): tekan Ctrl+B, lalu D

# Kembali ke sesi yang sudah ada
tmux attach -t tuning

# List semua sesi
tmux ls

# Buat window baru dalam sesi: tekan Ctrl+B, lalu C
# Pindah antar window: tekan Ctrl+B, lalu 0 / 1 / 2
```

### Contoh: Jalankan 2 Tuning Paralel

```bash
# Buat sesi tmux
tmux new -s tuning

# Window 0: Jalankan Autoformer
python run_tuning_cli.py autoformer --trials 100 --subsample

# Tekan Ctrl+B, lalu C (buat window baru)

# Window 1: Jalankan LSTM
python run_tuning_cli.py lstm --trials 50 --subsample

# Tekan Ctrl+B, lalu D untuk detach (aman tutup browser)
```

---

## 📊 Cek Progress Tuning

### Dari Terminal
```bash
python -c "
import optuna
studies = optuna.get_all_study_summaries(storage='sqlite:///optuna_history.db')
for s in studies:
    best = f'{s.best_trial.value:.6f}' if s.best_trial else 'N/A'
    print(f'  {s.study_name}: {s.n_trials} trials | best: {best}')
"
```

### Monitor GPU
```bash
watch -n 2 nvidia-smi
```

### Cek Proses Berjalan
```bash
ps aux | grep python | grep -v grep
```

---

## ⚠️ Troubleshooting

### Streamlit tidak bisa diakses
- Pastikan port **8501** sudah di-expose saat membuat Pod
- Gunakan perintah lengkap:
  ```bash
  python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.enableCORS=false --server.enableXsrfProtection=false
  ```

### `ModuleNotFoundError`
- Jalankan ulang: `bash setup_runpod.sh`

### CUDA Out of Memory (OOM)
- Kurangi `batch_size` atau `d_model` di search space
- Gunakan flag `--subsample` untuk memperkecil data
- Pastikan tidak ada proses zombie: `ps aux | grep python`

### Tuning Terhenti / Crash
- Proses bisa di-**resume** otomatis (database Optuna disimpan di `optuna_history.db`)
- Jalankan ulang perintah CLI yang sama, Optuna akan melanjutkan dari trial terakhir

### Konflik NumPy / TensorFlow
```bash
pip install "numpy==1.26.4" --force-reinstall --no-deps --no-cache-dir
```

---

## 🧹 Sebelum Terminate Pod

**Jangan lupa backup hasil tuning!**

```bash
# Download database Optuna via Jupyter Lab file browser
# Lokasi: /workspace/SolarForecasting/optuna_history.db

# Atau push ke GitHub (jika database tidak terlalu besar)
# Atau copy ke persistent volume
cp optuna_history.db /workspace/optuna_backup.db
```

---

*Dokumen ini terakhir diperbarui: 25 Februari 2026*
