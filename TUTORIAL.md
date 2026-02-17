# Tutorial Lengkap: PV Forecasting Modular Pipeline v1

## Daftar Isi
1. [Persiapan Awal](#1-persiapan-awal)
2. [Memahami Struktur Folder](#2-memahami-struktur-folder)
3. [Memahami File Konfigurasi](#3-memahami-file-konfigurasi)
4. [Menjalankan Pipeline](#4-menjalankan-pipeline)
5. [Mode Web Dashboard (Streamlit)](#5-mode-web-dashboard-streamlit)
6. [Mode TUI (Terminal Interaktif)](#6-mode-tui-terminal-interaktif)
7. [Mode CLI (Command Line)](#7-mode-cli-command-line)
8. [Mengedit Konfigurasi](#8-mengedit-konfigurasi)
9. [Mengganti Arsitektur Model](#9-mengganti-arsitektur-model)
10. [Hyperparameter Tuning (Optuna)](#10-hyperparameter-tuning-optuna)
11. [Time Series Cross-Validation (TSCV)](#11-time-series-cross-validation-tscv)
12. [Target Domain Testing (Data Indonesia)](#12-target-domain-testing-data-indonesia)
13. [Fitur-Fitur Terbaru (v1)](#13-fitur-fitur-terbaru-v1)
14. [Manajemen Versi Data & Model](#14-manajemen-versi-data--model)
15. [Version Control (Git & GitHub)](#15-version-control-git--github)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Persiapan Awal

### 1.1 Install Dependencies
Buka **Anaconda Prompt**, lalu jalankan:
```bash
conda activate tf-gpu
cd "c:\Users\Lenovo\OneDrive\Pretrain GRU\Pre-train model PatchTST\Modular Pipeline v1"
pip install -r requirements.txt
```

### 1.2 Letakkan Data CSV
Pastikan file CSV data PV ada di folder `data/raw/`.
Contoh: `data/raw/dkasc_with_dc_power_10_5kw.csv`

### 1.3 Verifikasi Setup
```bash
python _verify.py
```
Jika semua modul menampilkan `[OK]`, pipeline siap digunakan.

---

## 2. Memahami Struktur Folder

```
Modular Pipeline v1/
│
├── config.yaml              <<< PUSAT KENDALI (semua parameter di sini)
├── main.py                  <<< Controller utama (TUI + CLI)
├── app.py                   <<< Web Dashboard (Streamlit)
├── requirements.txt         <<< Daftar library Python
├── .gitignore               <<< File yang diabaikan Git
├── .env                     <<< Environment variables (path)
├── README.md                <<< Ringkasan singkat
│
├── src/                     <<< Kode inti (JANGAN diubah kecuali perlu)
│   ├── config_loader.py     │   Baca/tulis config.yaml
│   ├── data_prep.py         │   Algorithm 1, Features, Selection, Sequences
│   ├── model_factory.py     │   PatchTST, GRU (dan model lain di masa depan)
│   ├── trainer.py           │   Training, Optuna, TSCV, MLflow
│   └── predictor.py         │   Evaluasi, CSI->Power, Target Testing
│
├── data/
│   ├── raw/                 <<< Taruh CSV data mentah di sini
│   ├── processed/           <<< Artefak .npy (otomatis dibuat oleh preprocessing)
│   └── target/              <<< Taruh CSV data Indonesia di sini
│
├── models/                  <<< Model tersimpan (.keras / .h5)
└── logs/
    ├── session/             <<< Data persistensi (training history, eval results)
    └── ...                  <<< Log pipeline lainnya
```

**Prinsip utama**: Anda hanya perlu berurusan dengan 2 hal:
1. `config.yaml` — untuk mengubah parameter
2. `app.py` (Web Dashboard) atau `main.py` (CLI/TUI) — untuk menjalankan pipeline

---

## 3. Memahami File Konfigurasi

File `config.yaml` adalah **satu-satunya tempat** untuk mengubah perilaku pipeline.
Berikut bagian-bagian pentingnya:

### 3.1 Data
```yaml
data:
  csv_path: "data/raw/dkasc_with_dc_power_10_5kw.csv"   # Path ke file CSV
  csv_separator: ";"                                       # Pemisah kolom
  target_col: "pv_output_kw"                               # Kolom target
```

### 3.2 Sistem PV
```yaml
pv_system:
  nameplate_capacity_kw: 10.5    # Kapasitas panel (kW)
  temp_coeff: -0.0045            # Koefisien temperatur
```

### 3.3 Splitting Data
```yaml
splitting:
  train_ratio: 0.9               # 90% training
  test_ratio: 0.1                # 10% testing
```

### 3.4 Model
```yaml
model:
  architecture: "patchtst"       # Pilihan: "patchtst" atau "gru"
  hyperparameters:
    lookback: 96                 # Jendela input (jam)
    patch_len: 24
    stride: 12
    d_model: 32
    n_heads: 8
    n_layers: 3
    dropout: 0.1
    learning_rate: 0.0002
    batch_size: 32
```

### 3.5 Fitur yang Aktif
```yaml
features:
  groups:
    weather: true     # Fitur cuaca (GHI, Temp, RH, Wind)
    lags: true        # Fitur lag (1h, 12h, 24h, 48h, 168h)
    physics: true     # Fitur fisika (clear sky, CSI)
    rolling: true     # Moving average & std
```
> Matikan grup fitur dengan mengubah `true` menjadi `false`.

### 3.6 Preprocessing (Data Cleaning)
```yaml
preprocessing:
  resample_1h: true            # Resample data ke interval 1 jam
  remove_outliers: true        # Aktifkan deteksi outlier
  ghi_high_pv_zero: true       # Hapus data PV=0 saat GHI tinggi (>200 W/m²)
  ghi_dark_pv_high: true       # Hapus data PV tinggi saat GHI gelap (<threshold)
```

> **Catatan**: Preprocessing menggunakan **Algorithm 1** yang mencakup:
> - Penanganan duplikat dan nilai negatif
> - Deteksi outlier berdasarkan batasan fisik (GHI >2000, Temp <-30, dll)
> - Deteksi inkonsistensi PV-GHI (PV=0 saat matahari terang)
> - Deteksi data "frozen" (sensor error - nilai stagnant >10 jam)

### 3.7 Toggle Tuning & TSCV
```yaml
tuning:
  enabled: false     # true = jalankan Optuna, false = pakai hyperparameters di atas
  n_trials: 50

tscv:
  enabled: false     # true = jalankan cross-validation
  n_splits: 5
```

---

## 4. Menjalankan Pipeline

Ada **3 cara** menjalankan pipeline:

| Mode | Terminal yang Didukung | Cara |
|------|----------------------|------|
| **Web Dashboard** | Semua (akses lewat browser) | `streamlit run app.py` |
| **TUI** (Menu Interaktif) | Anaconda Prompt, CMD, PowerShell | `python main.py` |
| **CLI** (Command Line) | Semua terminal termasuk VS Code | `python main.py <perintah>` |

### Penting: Pastikan pakai Python dari env `tf-gpu`
```bash
# Cara 1: Aktifkan conda dulu
conda activate tf-gpu
python main.py

# Cara 2: Panggil Python langsung (jika conda activate tidak jalan)
C:\Users\Lenovo\miniconda3\envs\tf-gpu\python.exe main.py
```

---

## 5. Mode Web Dashboard (Streamlit)

> **Cara yang DIREKOMENDASIKAN** — tampilan visual lengkap dengan monitoring real-time.

### 5.1 Cara Menjalankan
```bash
conda activate tf-gpu
cd "c:\Users\Lenovo\OneDrive\Pretrain GRU\Pre-train model PatchTST\Modular Pipeline v1"
streamlit run app.py
```
Atau jika `streamlit` tidak terdeteksi:
```bash
C:\Users\Lenovo\miniconda3\envs\tf-gpu\Scripts\streamlit.exe run app.py
```

Dashboard akan terbuka di browser: **http://localhost:8501**

### 5.2 Layout Dashboard

Dashboard terdiri dari **Sidebar** (kiri) dan **Area Utama** dengan beberapa tab:

#### Sidebar - Konfigurasi
Di sidebar kiri, Anda bisa mengatur:
- **Model Manager** — Pilih model mana yang aktif untuk evaluasi (lihat Bagian 5.3)
- **Arsitektur Model** — PatchTST atau GRU
- **Dataset** — Path CSV dan kapasitas PV
- **Horizon** — Jumlah jam prediksi ke depan
- **Data Split** — Rasio train/test
- **Preprocessing Config** — Toggle cleaning rules (resample, outlier detection)
- **Hyperparameters** — Lookback, d_model, n_heads, dll
- **Grup Fitur** — ON/OFF weather, lags, physics, rolling
- **Opsi Pipeline** — Toggle Optuna tuning dan TSCV
- **Simpan Konfigurasi** — Menyimpan perubahan ke `config.yaml`

#### Tab Utama (ML Pipeline Flow)

Dashboard didesain mengikuti alur kerja *Machine Learning* yang standar:

| Tab | Tujuan | Komponen Utama |
|-----|--------|----------------|
| **🧪 Feature Lab** | Perancangan Strategi | Preset Manager, Manual/Auto Feature Selection. |
| **📥 Preprocessing** | **Pusat Kendali Data** | Pilih Dataset CSV, Atur Feature Groups, Split Ratio, Scaling, Cleaning Rules. |
| **🔎 Data Insights** | Verifikasi Visual | Metamorfosis Data (Cleaning Stats), Correlation Matrix, Sequence Preview. |
| **🎯 Tuning** | Optimasi Otomatis | Optuna Hyperparameter Search. |
| **🧠 Training Center** | Eksekusi Training | **Pilih Versi Data**, Live Loss Curve, Model Registry. |
| **📈 Evaluation** | Analisis Performa | Metrik komprehensif (MAE, RMSE, R²), Visualisasi Prediksi. |
| **📦 Target Test** | Real-world Testing | Pemuatan data eksternal (indonesia) ke model terlatih. |
| **🚀 Runner** | Eksekusi Cepat | Tombol sekali klik untuk Prep, Train, dan Eval. |
| **📋 Logs** | Monitoring | Log sistem dan riwayat preprocessing terakhir. |

### 5.3 Model Manager (Fitur Baru)

Setiap kali Anda menjalankan training, model akan otomatis disimpan di folder `models/` dengan nama yang unik (contoh: `patchtst_20260212_1330.keras`).

**Model Manager** di sidebar memungkinkan Anda:
- **Melihat semua model** yang pernah dilatih
- **Memilih model aktif** untuk evaluasi dan testing
- Model terbaru otomatis terpilih setelah training selesai

Cara menggunakan:
1. Lihat dropdown **"Pilih Model untuk Evaluasi"** di sidebar
2. Pilih model yang diinginkan
3. Klik **"Run Evaluation"** di tab Runner
4. Hasil evaluasi akan muncul di tab Evaluation

> **Tips**: Ini sangat berguna untuk **membandingkan** performa antar model.
> Misalnya, latih PatchTST → evaluasi → ganti arsitektur ke GRU → latih → evaluasi → bandingkan hasilnya.

### 5.4 Live Training Monitor

Saat training berjalan, Anda akan melihat secara real-time:
- **Progress bar** dengan persentase dan estimasi waktu selesai (ETA)
- **Metric cards** — Epoch, Train Loss, Val Loss, ETA
- **Loss curve chart** — Grafik interaktif yang terupdate setiap epoch
- **Epoch log** — Detail setiap epoch dalam format teks

### 5.5 Data Insights Tab

Tab ini menampilkan hasil preprocessing secara visual:
- **Statistik** — Jumlah baris asli, setelah cleaning, yang dihapus, dan final sequences
- **Feature Analysis** — Daftar fitur yang dipilih dan kontribusinya
- **Correlation Heatmap** — Matriks korelasi antar fitur
- **Data Preview** — Sample data mentah

### 5.6 Persistensi Data (Refresh-Safe)

Data tracking Anda **TIDAK akan hilang** saat halaman di-refresh. Berikut mekanismenya:

| Data | Tersimpan di | Lokasi File |
|------|-------------|-------------|
| Training History (loss curves) | `logs/session/last_training_history.json` | + per-model: `history_{model_name}.json` |
| Evaluation Metrics (MAE, R², dll) | `logs/session/last_eval_results.json` | |
| Model Terpilih | `logs/session/selected_model.txt` | |
| Model File (.keras/.h5) | `models/` | |

**Yang tetap hilang** saat refresh (sifatnya sementara):
- Live progress bar (real-time, tidak bisa dipersist)
- Pipeline log (history aktivitas sementara)
- Prep metadata (detail insight preprocessing — jalankan ulang Step 1 untuk melihat kembali)

---

## 6. Mode TUI (Terminal Interaktif)

> **Syarat**: Jalankan dari Anaconda Prompt / CMD / PowerShell (bukan terminal VS Code bawaan)

### Langkah:
```bash
conda activate tf-gpu
cd "c:\Users\Lenovo\OneDrive\Pretrain GRU\Pre-train model PatchTST\Modular Pipeline v1"
python main.py
```

### Tampilan Menu:
```
┌─────────────────────────────────────────┐
│      PV Forecasting Pipeline            │
│      Universal Controller v1.0          │
│      Supports: PatchTST, GRU, and more  │
└─────────────────────────────────────────┘

┌────────────────┬───────────────────────────────┐
│ Parameter      │ Nilai                         │
├────────────────┼───────────────────────────────┤
│ Arsitektur     │ PATCHTST                      │
│ Dataset        │ dkasc_with_dc_power_10_5kw.csv│
│ Kapasitas      │ 10.5 kW                       │
│ Split          │ 90% / 10%                     │
│ Lookback       │ 96                            │
│ Horizon        │ 24 jam                        │
│ Optuna Tuning  │ OFF                           │
│ TSCV           │ OFF                           │
└────────────────┴───────────────────────────────┘

? Apa yang ingin Anda lakukan?
> 1. Preprocessing (Data > Artefak)
  2. Training (Latih Model)
  3. Hyperparameter Tuning (Optuna)
  4. TSCV (Cross-Validation)
  5. Evaluate (Metrik & Analisis)
  6. Target Testing (Data Indonesia)
  7. Full Pipeline (Semua Otomatis)
  8. Edit Konfigurasi
  9. Keluar
```

**Navigasi**: Panah atas/bawah untuk memilih, Enter untuk menjalankan.

---

## 7. Mode CLI (Command Line)

Cocok untuk terminal VS Code, automasi, atau jika TUI tidak berfungsi.

### Perintah yang Tersedia:
```bash
python main.py preprocess     # Preprocessing data (CSV -> artefak .npy)
python main.py train          # Training model
python main.py tune           # Hyperparameter tuning (Optuna)
python main.py tscv           # Time Series Cross-Validation
python main.py evaluate       # Evaluasi model terlatih
python main.py target         # Testing pada data Indonesia
python main.py full           # Semua otomatis (Preprocess -> Train -> Evaluate)
```

### Contoh: Jalankan Full Pipeline
```bash
C:\Users\Lenovo\miniconda3\envs\tf-gpu\python.exe main.py full
```

Output yang diharapkan:
```
[1/4] Preprocessing...
  Raw Split: Train=39463, Test=4385
  19 features selected
  Train: (39176, 96, 19), Test: (4266, 96, 19)
  Semua artefak tersimpan.

[2/4] Tuning dilewati (disabled)

[3/4] Training...
  Epoch 1/100 - loss: 0.0800 - val_loss: 0.0500
  ...
  Model disimpan: models/patchtst_20260212_1130.keras

[4/4] Evaluating...
  TRAIN - MAE=0.25, RMSE=0.45, R2=0.85
  TEST  - MAE=0.26, RMSE=0.46, R2=0.84

FULL PIPELINE SELESAI!
```

---

## 8. Mengedit Konfigurasi

### Cara 1: Edit file langsung
Buka `config.yaml` di text editor, ubah nilainya, simpan.

### Cara 2: Lewat Menu TUI
1. Jalankan `python main.py`
2. Pilih **"8. Edit Konfigurasi"**
3. Pilih bagian yang ingin diubah:

```
? Bagian mana yang ingin diubah?
> 1. Arsitektur Model         <- Ganti PatchTST / GRU
  2. Dataset & Kapasitas      <- Ganti file CSV atau kapasitas kW
  3. Split Ratio              <- Ubah rasio train/test
  4. Hyperparameters          <- Ubah lookback, d_model, dll
  5. Search Space (Tuning)    <- Ubah range pencarian Optuna
  6. Toggle: Tuning / TSCV    <- ON/OFF Optuna dan TSCV
  7. Feature Groups           <- ON/OFF grup fitur
  8. Simpan & Kembali         <- Simpan perubahan ke config.yaml
```

4. Ikuti prompt yang muncul
5. Pilih **"8. Simpan & Kembali"** untuk menyimpan

### Cara 3: Lewat Web Dashboard
1. Buka `streamlit run app.py`
2. Ubah parameter di **Sidebar** (kiri)
3. Klik **"Simpan Konfigurasi"** di bagian bawah sidebar

---

## 9. Mengganti Arsitektur Model

### Contoh: Ganti dari PatchTST ke GRU

**Cara 1 — Edit config.yaml:**
```yaml
model:
  architecture: "gru"    # ubah dari "patchtst" ke "gru"
```

**Cara 2 — Lewat TUI:**
1. `python main.py` → "8. Edit Konfigurasi" → "1. Arsitektur Model"
2. Pilih `gru`
3. "8. Simpan & Kembali"
4. Kembali ke menu utama, pilih "7. Full Pipeline"

**Cara 3 — Lewat Web Dashboard:**
1. Buka sidebar → ubah dropdown **"Arsitektur Model"** ke `gru`
2. Klik **"Simpan Konfigurasi"**
3. Buka tab Runner → klik **"Run Training"**

> **Catatan**: Setelah ganti arsitektur, Anda perlu menjalankan ulang  
> **Training** (atau Full Pipeline). Preprocessing tidak perlu diulang  
> jika data dan fitur tidak berubah.

---

## 10. Hyperparameter Tuning (Optuna)

### Cara Mengaktifkan:

**Di config.yaml:**
```yaml
tuning:
  enabled: true        # ubah dari false ke true
  n_trials: 50         # jumlah percobaan
```

**Atau lewat TUI:**
"8. Edit Konfigurasi" → "6. Toggle: Tuning / TSCV" → "Aktifkan Optuna Tuning? Yes"

**Atau lewat Web Dashboard:**
Sidebar → "Opsi Pipeline" → centang "Optuna Tuning"

### Cara Menjalankan:
```bash
# Khusus tuning saja:
python main.py tune

# Atau sebagai bagian dari Full Pipeline (jika enabled: true):
python main.py full
```

### Rentang Pencarian (Search Space):
Dapat diubah di config.yaml:
```yaml
tuning:
  search_space:
    patch_len: [12, 24, 4]        # [min, max, step]
    stride: [4, 12, 4]            # [min, max, step]
    d_model: [32, 64, 128, 256]   # categorical (pilihan)
    n_heads: [4, 8, 12]           # categorical
    n_layers: [2, 6]              # [min, max]
    dropout: [0.1, 0.3]           # [min, max]
    learning_rate: [0.00005, 0.0005]  # [min, max] (log scale)
    batch_size: [32]              # categorical
    lookback: [72, 336, 24]       # [min, max, step]
```

---

## 11. Time Series Cross-Validation (TSCV)

### Cara Mengaktifkan:
```yaml
tscv:
  enabled: true
  n_splits: 5
```

### Cara Menjalankan:
```bash
python main.py tscv
```

Output:
```
--- Fold 1/5 ---
  Train: 6500, Val: 6500
  MAE=0.25, RMSE=0.42, R2=0.84

--- Fold 2/5 ---
  ...

TSCV SUMMARY
  Avg MAE:  0.26
  Avg RMSE: 0.44
  Avg R2:   0.83
```

---

## 12. Target Domain Testing (Data Indonesia)

### Langkah:

1. **Letakkan file CSV** data Indonesia di folder `data/target/`
   ```
   data/target/pv_indonesia_hourly.csv
   ```

2. **Pastikan format CSV** sama dengan data training:
   - Kolom: timestamp, ghi_wm2, dhi_wm2, ambient_temp_c, dll.
   - Separator: `;` (atau sesuaikan di config.yaml)

3. **Pastikan sudah ada model** yang terlatih di folder `models/`

4. **Jalankan:**
   ```bash
   python main.py target
   ```

5. Sistem akan:
   - Menampilkan pilihan model (jika ada lebih dari satu)
   - Menampilkan pilihan file CSV target
   - Memproses data target menggunakan scaler dari data training
   - Menampilkan metrik prediksi

---

## 13. Fitur-Fitur Terbaru (v1)

### 13.1 Cyclical Time Encoding (Sin/Cos)
Alih-alih menggunakan angka jam (0-23) atau bulan (1-12) secara linear, sistem ini menggunakan transformasi trigonometri untuk menjaga kontinuitas temporal:
- **Hourly**: Menghubungkan jam 23:00 kembali ke jam 00:00 dengan mulus.
- **Monthly**: Menghubungkan Desember ke Januari untuk menangkap siklus musim tahunan.
- **Seasonal (DOY)**: Menggunakan *Day of Year* (1-365) untuk resolusi tinggi. SHAP analysis membuktikan fitur ini sangat krusial setelah suhu udara.

### 13.2 Preprocessing yang Diperkuat (Algorithm 1)
Preprocessing sekarang mencakup deteksi outlier yang lebih ketat:
- **Physical Extremes**: GHI > 2000, Temp < -30, RH di luar 0-100%, Wind Speed < 0.
- **Precipitation Clipping**: Membatasi nilai curah hujan ekstrem yang sering merusak gradien model.
- **PV Imputation**: Mengisi gap kecil pada data PV menggunakan korelasi berbasis CSI.

### 13.3 Data Meta-Insights
Tab Insights kini menampilkan **"Metamorfosis Data"**:
- Melihat bagaimana baris data berkurang/bertambah di setiap tahap cleaning.
- Meninjau korelasi fitur baru (seperti Lag atau CSI) sebelum model dilatih.

---

## 14. Manajemen Versi Data & Model

### 14.1 Pemilihan Versi Preprocessed
Setiap kali Anda klik **"Start Preprocessing"**, sistem membuat folder baru di `data/processed/` dengan nama `v_MMDD_HHMM_filename`. 

**Cara menggunakan versi tertentu untuk training:**
1. Masuk ke tab **🧠 Training Center**.
2. Cari expander **"📦 Pilih Versi Data Preprocessed"**.
3. Pilih versi yang diinginkan (defaultnya adalah "Latest").
4. Jika versi baru tidak muncul, klik tombol **"🔄 Refresh Daftar Versi"**.

### 14.2 Model Registry
Model disimpan di folder `models/` lengkap dengan `meta.json` dan `scaler.pkl`. Anda dapat memuat model lama kapan pun melalui Model Manager di sidebar.

---

## 15. Version Control (Git & GitHub)

Proyek ini terintegrasi dengan Git untuk melacak setiap perubahan kode Anda.

### 15.1 Workflow Harian (Push Perubahan)
Jika Anda mengubah file (misal menambahkan model baru atau mengubah `data_prep.py`):
1. Buka **Anaconda Prompt**.
2. `cd` ke folder proyek.
3. `git add .` (tambahkan semua perubahan).
4. `git commit -m "update: menambahkan fitur X"` (simpan progres dengan catatan).
5. `git push` (unggah ke GitHub).

### 15.2 Menarik Update dari GitHub (Pull)
Jika Anda mengerjakan proyek di komputer lain:
`git pull origin main`

### 15.3 Mengapa Menggunakan Git?
- **Pencadangan**: Kode Anda aman di GitHub jika laptop bermasalah.
- **Rollback**: Bisa kembali ke versi kode minggu lalu jika ada error besar.
- **Kolaborasi**: Memungkinkan pengerjaan proyek bersama tim secara rapi.

---

## 16. Troubleshooting

### Error: `No module named 'tensorflow'`
**Penyebab**: Python yang dipakai bukan dari env `tf-gpu`.  
**Solusi**: Gunakan path Python langsung:
```bash
C:\Users\Lenovo\miniconda3\envs\tf-gpu\python.exe main.py
```

### Error: `NoConsoleScreenBufferError`
**Penyebab**: Terminal tidak mendukung TUI interaktif (misal: VS Code terminal).  
**Solusi**: Gunakan mode CLI dengan menambahkan argumen:
```bash
python main.py full
```
Atau buka **Anaconda Prompt** untuk mode TUI.  
Atau gunakan **Web Dashboard** (`streamlit run app.py`) yang bekerja di semua terminal.

### Error: `UnicodeEncodeError`
**Penyebab**: Terminal Windows tidak mendukung karakter Unicode tertentu.  
**Solusi**: Sudah diperbaiki di versi terbaru. Jika masih muncul, jalankan:
```bash
chcp 65001
python main.py
```

### Error: `FileNotFoundError: Config file tidak ditemukan`
**Penyebab**: Menjalankan `python main.py` dari folder yang salah.  
**Solusi**: Pastikan terminal berada di folder `Modular Pipeline v1`:
```bash
cd "c:\Users\Lenovo\OneDrive\Pretrain GRU\Pre-train model PatchTST\Modular Pipeline v1"
```

### Error: `unable to synchronize` / `OOM` saat Evaluasi
**Penyebab**: GPU kehabisan memori karena masih menyimpan sisa training.  
**Solusi**: Sudah diperbaiki di versi terbaru — evaluasi otomatis membersihkan memori GPU.  
Jika masih terjadi:
1. Restart Streamlit (`Ctrl+C` lalu `streamlit run app.py`)
2. Atau kurangi batch size di config.yaml
3. Atau jalankan evaluasi terpisah (tidak langsung setelah training)

### Error: Shape Mismatch saat Evaluasi
**Penyebab**: Horizon/lookback di sidebar tidak cocok dengan model yang dimuat.  
**Solusi**: Sudah diperbaiki — evaluasi sekarang otomatis mendeteksi horizon dan lookback dari arsitektur model. Tidak perlu mengatur manual.

### Preprocessing lambat / data berkurang drastis
**Penyebab**: Banyak gap waktu di data → sequence yang valid sedikit.  
**Solusi**: Pertimbangkan mengubah split ratio di config.yaml:
```yaml
splitting:
  train_ratio: 0.8    # coba 80/20 atau 90/10
```
Atau nonaktifkan beberapa aturan cleaning yang terlalu agresif di sidebar.

### Port 8501 sudah dipakai (Streamlit)
**Penyebab**: Instance Streamlit sebelumnya masih berjalan.  
**Solusi**:
```bash
# Hentikan semua proses Python
taskkill /F /IM python.exe /T

# Jalankan ulang
streamlit run app.py
```

---

## Ringkasan Alur Kerja Harian

### Menggunakan Web Dashboard (Direkomendasikan):
```
1. Buka Anaconda Prompt
2. conda activate tf-gpu
3. cd ke folder "Modular Pipeline v1"
4. streamlit run app.py
5. Buka browser: http://localhost:8501

6. Di Dashboard:
   a. Atur konfigurasi di Sidebar (jika perlu)
   b. Tab Runner → "Run Preprocessing"
   c. Tab Runner → "Run Training" (monitor real-time)
   d. Tab Runner → "Run Evaluation"
   e. Lihat hasil di tab Training & Evaluation

7. Untuk eksperimen lain:
   - Ganti arsitektur/hyperparameters di Sidebar
   - Klik "Simpan Konfigurasi"
   - Ulangi langkah 6b-6d
   - Bandingkan model lewat Model Manager
```

### Menggunakan CLI (Cepat):
```
1. Buka Anaconda Prompt
2. conda activate tf-gpu
3. cd ke folder "Modular Pipeline v1"
4. python main.py full                # langsung jalan semua

5. Lihat hasil di:
   - Terminal (metrik langsung ditampilkan)
   - models/ (file model tersimpan)
   - logs/  (log detail)
```

**Untuk eksperimen cepat:**
```
Ubah config.yaml → python main.py full → Lihat metrik → Ulangi
```
