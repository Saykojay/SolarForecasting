# PV Forecasting Pipeline - Modular v1 🚀

Pipeline modular canggih untuk prediksi output PV menggunakan model Deep Learning (PatchTST, GRU). Versi ini dioptimalkan untuk riset akademis dengan fitur **Cyclical Encoding** yang presisi dan manajemen data versional.

## 🛠️ Quick Start (Recomended: Web Dashboard)

```bash
# 1. Aktifkan Environment
conda activate tf-gpu

# 2. Masuk ke Project
cd "Modular Pipeline v1"

# 3. Jalankan Dashboard
streamlit run app.py
```

## 📂 Struktur Utama
```
├── app.py                   # Web Dashboard (Pusat Kendali)
├── config.yaml              # Konfigurasi Global & Preset
├── src/                     # Source Code (Core Logic)
│   ├── data_prep.py         # Advanced Feature Engineering & Versioning
│   ├── model_factory.py     # PatchTST & GRU Architectures
│   └── ...
├── data/
│   ├── raw/                 # Letakkan file CSV mentah di sini
│   └── processed/           # Folder hasil preprocessing (Versional: v_MMDD_...)
└── models/                  # Model tersimpan (.keras) dan Metadata
```

## 🔄 Version Control (Git)

Proyek ini sudah terhubung ke GitHub: `https://github.com/Saykojay/SolarForecasting`.

**Cara menyimpan perubahan ke GitHub:**
1. Buka Anaconda Prompt di folder proyek.
2. Tambahkan perubahan: `git add .`
3. Commit (beri catatan): `git commit -m "Catatan perubahan Anda"`
4. Upload: `git push`

## 🧪 Fitur Unggulan v1
- **Tabbed Workflow**: Urutan logis ML Pipeline (Lab -> Prep -> Insights -> Train -> Eval).
- **Cyclical Time Encoding**: Transformasi Sin/Cos untuk Jam, Bulan, dan *Day of Year* (DOY).
- **Versioning System**: Menyimpan setiap hasil preprocessing dalam folder unik untuk menghindari konflik data.
- **Physics-Based Features**: Integrasi CSI (*Clear Sky Index*) untuk akurasi lebih tinggi.

---
*Untuk panduan detail penggunaan setiap tab, silakan baca [TUTORIAL.md](TUTORIAL.md).*
