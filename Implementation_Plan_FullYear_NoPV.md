# Implementation Plan: 1-Year Solar Forecasting (No PV History)
**Tujuan:** Membangun dan melatih model kecerdasan buatan (PatchTST/TimeTracker) menggunakan dataset penuh 1 tahun kalender untuk memprediksi PV Output pada PLTS baru (tanpa sejarah `pv_output`), murni mengandalkan data ekstrapolasi cuaca historis.

---

## FASE 1: Data Preprocessing & Feature Engineering
**Target:** Menciptakan tensor data bersih beresolusi 1-jam dari dataset 1 tahun tanpa *data leakage* (kebocoran) riwayat PV masa lalu.

1. Buka antarmuka Streamlit, navigasi ke tab **Data Preprocessing**.
2. **Kapasitas Dataset:** Pastikan *checkbox* "Batasi Jumlah Data (Limit Rows)" **TIDAK DICENTANG**. Kita menggunakan 100% dataset (kisaran ~8.760 baris per tahun).
3. **Konfigurasi Target (Y):** 
   - Centang **"Gunakan CSI Normalization"**.
   - Set *GHI Threshold* ke `50`. (Ini memastikan model menebak rasio ketebalan awan, bukan memprediksi kW langsung yang fluktuatif).
4. **Pemilihan Fitur Historis (X):**
   - Mode Seleksi: Pilih **Manual**.
   - **WAJIB DICENTANG:** Variabel cuaca (`ghi_wm2`, `ambient_temp_c`, `relative_humidity_pct`, `wind_speed_ms`).
   - **WAJIB DICENTANG:** Variabel siklus waktu (`time_hour`, `time_day`, `time_month`, `time_doy`).
   - **SANGAT DILARANG DICENTANG:** Segala fitur yang mengandung unsur `pv_output_kw` atau `csi_target` (termasuk versi *lag* atau *moving average*-nya).
5. Eksekusi **Mulai Preprocessing** dan tunggu hingga selesai.
6. Simpan konfigurasi ini pada *Preset Manager* (misal dengan nama: `FullYear_WeatherOnly`).

---

## FASE 2: Correlation & Feature Verification (Insights)
**Target:** Memverifikasi korelasi antar sensor cuaca secara statistik sebelum membuang waktu komputasi untuk *training*.

1. Navigasi ke tab **Data Insights**.
2. Amati **Pearson Correlation Matrix Heatmap**.
3. Pastikan fitur `ghi_wm2` memiliki korelasi pekat dengan target (> 0.6).
4. Catat fitur cuaca yang korelasinya sangat lemah (mendekati 0.0), misalnya mungkin `wind_direction`.
5. Jika ada fitur cuaca yang terbukti murni *noise* (benar-benar tidak berkorelasi di iklim Singapura), Anda bisa kembali ke Fase 1 untuk menghapus / *uncheck* fitur tersebut agar tidak membebani memori AI.

---

## FASE 3: Hyperparameter Tuning (Optuna)
**Target:** Menemukan *Learning Rate*, *Batch Size*, dan struktur blok arsitektur optimal untuk menelan data 1 tahun.

1. Navigasi ke tab **Tuning (Optuna)**.
2. Pada bagian *Data Source*, pastikan data yang terpilih adalah paket data *Preprocessing* hasil dari Fase 1 (`FullYear_WeatherOnly`).
3. Pilih arsitektur yang diinginkan (Misal: `PatchTST`).
4. **Batasi Search Space** agar komputasi tahunan tidak memakan waktu berhari-hari:
   - *Epochs:* Batasi ke 10-15 putaran per *trial*.
   - *n_layers:* Jangan diset melebihi 3.
   - *d_model:* Maksimal 128 atau 256.
5. Jalankan Tuning *(Run Tuning)* dan biarkan Optuna bekerja (estimasi waktu: beberapa jam hingga semalaman penuh, tergantung spesifikasi GPU).
6. Saat Tuning selesai, catat Parameter Terbaik (*Best Parameters*) yang ditampilkan sistem.

---

## FASE 4: Final Master Training
**Target:** Melatih model mahakarya final dengan parameter emas yang telah divalidasi Optuna.

1. Navigasi ke tab **Training Center**.
2. Pilih data hasil Fase 1.
3. Masukkan kombinasi *Best Parameters* (Learning Rate, Batch Size, dll) hasil temuan Optuna di Fase 3 ke dalam kotak pengaturan (atau biarkan otomatis terisi jika di dukung sistem).
4. Tekan **Mulai Training**.
5. Model akan berlatih selama seluruh *epoch* yang ditentukan (dengan fungsi *Early Stopping* di latar belakang untuk mencegah *Overfitting*).

---

## FASE 5: Evaluasi Kelayakan Lapangan (Inference Check)
**Target:** Menganalisa seberapa baik tebakan model terhadap masa depan tanpa melihat masa lalu.

1. Navigasi ke tab **Comparison / Evaluasi**.
2. Pilih model *Masterpiece* yang baru saja Anda latih.
3. Klik **Run Comparison Analysis**.
4. Di bagian **3. Visual Comparison**, fokus pada 2 grafik peringatan:
   - **MAE Score (Lower is Better):** Pastikan batangnya cukup memuaskan sesuai target error operasi pabrik listrik yang ditetapkan.
   - **Overfit Δ (Train R² - Test R²):** Pastikan tingginya mendekati angka `0`. Jika angka ini lebih dari `0.15` (misal 15% kesenjangan), ini menandakan model gagal beradaptasi ke data ujicoba (menghafal buta). Kembalilah ke Fase 3, tambahkan batas `Dropout`, lalu _Training_ ulang.
5. Model yang lulus ujian ini berstatus **PRODUCTION READY** dan siap untuk dieksekusi di pabrik / *device* manapun yang hanya berbekal termometer dan sensor matahari!

---

## APPENDIX: Data Volume & Horizon Ablation Study (SOP Uji Rentang Data)
**Target:** Mendesain eksperimen terstruktur untuk mencari tahu "Titik Jenuh" (Diminishing Returns) kuantitas data historis yang paling optimal untuk AI Pabrik Off-Grid, guna menghemat *Training Time* tanpa mengorbankan keamanan RMSE.

### Tabel Rencana Eksperimen Rentang Data Training

| Eksperimen ID | Kategori Uji Coba | Tanggal Mulai (Start) | Tanggal Akhir (End) | Durasi Data | Tujuan & Hipotesis yang Ingin Dibuktikan |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vol-1** | Baseline Minimal | 1 Okt 2024 | 31 Des 2024 | **3 Bulan** | *Underfitting Check*: Membuktikan AI akan gagal (error besar) jika hanya dilatih dengan cuaca 1 musim saja. |
| **Vol-2** | Setengah Siklus | 1 Jul 2024 | 31 Des 2024 | **6 Bulan** | Menilai apakah data setengah tahun cukup untuk menangani transisi cuaca kemarau ke hujan badai. |
| **Vol-3** | Siklus Penuh | 1 Jan 2024 | 31 Des 2024 | **1 Tahun** | **(Standar Emas)** Membuktikan perbaikan signifikan saat AI melihat seluruh 365 hari (kemarau, equinox, monsoon). |
| **Vol-4** | Siklus Ganda | 1 Jan 2023 | 31 Des 2024 | **2 Tahun** | Mencari "Titik Jenuh". Apakah menambah data 2x lipat menghasilkan akurasi (RMSE) yang sepadan dengan lama waktu *training*-nya? |
| **Vol-5** | Maksimum Historis | 1 Jan 2020 | 31 Des 2024 | **5 Tahun** | Menguji ketahanan AI terhadap iklim ekstrem (El Nino/La Nina) jangka panjang, namun rentan pada bias masa lalu. |
| **Era-1** | Uji Degradasi Umur | 1 Jan 2021 | 31 Des 2021 | **1 Tahun (Lama)** | Membuktikan bahwa cuaca bumi berubah (Concept Drift). Model ini diprediksi kalah akurat dibanding **Vol-3** saat diuji hari ini. |

### Cara Mengesekusi SOP Uji Rentang Data di Dashboard UI
1. Buka Tab **Data Preparation**.
2. Centang opsi **Limit Dataset (Subset)** (di kolom tengah) dan pilih metodenya: **"Rentang Tanggal (Date Range)"**.
3. Masukkan kombinasi *Start Date* dan *End Date* dari tabel di atas satu per satu (Misal: `01-10-2024` s/d `31-12-2024` untuk Vol-1).
4. Klik **Start Preprocessing**, lalu beri nama paket datanya (Misal: `Data_Vol1_3Bulan`).
5. Latih model satu per satu di tab **Training Center** menggunakan paket data masing-masing.
6. Arahkan ke tab **Comparison**, pilih keenam model (Vol-1 hingga Vol-5 + Era-1), lalu klik **Run Comparison Analysis**.
7. **Analisis Kurva U:** Perhatikan grafik balok **RMSE (Lower is Safer)**. Cari di *checkpoint* data mana baris batang RMSE tersebut paling landai sebelum akhirnya stagnan atau kembali meninggi. Itulah volume data historis paling sempurna untuk dibeli oleh pabrik pembangkit Anda!
