# Metodologi Riset Eksperimental: Optimasi Terstruktur Model AI Prediksi Solar (Photovoltaic)

Dokumen ini memuat prosedur standar operasional (SOP) dari alur eksperimen yang ketat, terstruktur, dan berjenjang (*Sequential Optimization*). Metode ini diciptakan untuk mencegah manipulasi data (data leakage), memastikan keaslian generalisasi model, dan menemukan keseimbangan tertinggi antara waktu pelatihan komputer (*Training Time Efficiency*) dengan keamanan operasional (*RMSE Safety*) dalam konteks Sistem Off-Grid.

---

## 🏗️ FASE 1: Data-Centric & Feature Ablation (Penghapusan Variabel)
**Tujuan Umum:** Menyajikan matriks *input* yang paling akurat secara kausalitas dan paling murni (tanpa bocoran) kepada otak AI, sebelum memperbaiki algoritma matematikanya.
**Kunci Metrik Pembanding:** Mempertahankan arsitektur bawaan/dasar pabrik (*Default LSTM/PatchTST Hyperparameters*) dan *Loss Function* standar yaitu `MSE` agar perbandingan antar data bersifat "Apple-to-Apple".

### Eksperimen 1.1: Pemblokiran *Data Leakage* (No PV History)
*   **Permasalahan:** Jika AI menghafal hasil produksi daya historis (`pv_output` masa lalu), ia akan terlalu cerdas menghafal angka ketimbang belajar rasio ketebalan awan dari cuaca. 
*   **Pelaksanaan:** Pada proses *Preprocessing*, seluruh kolom yang mengandung narasi Pembangkitan Daya (`pv_output`, `lag_pv`, `csi_target`) secara manual **DICABUT/DIABAIKAN**. 
*   **Input Fitur Fisika Murni (X):** `dhi_wm2`, `dni_wm2`, `ambient_temp_c` (Sensor Meteorologi Lokal).
*   **Input Fitur Rotasi Bumi (Siklikal Time):** `hour_sin`, `hour_cos` (Untuk pemahaman pasang-surut sudut datang matahari).
*   **Target Y (CSI Normalization):** Target diubah dari spektrum kW liar menjadi indeks stasioner (*Clear Sky Index*), agar AI fokus memprediksi "Berapa rasen (%/Oktas) ketebalan langit terhalang awan".

---

## 📊 FASE 2: Data Volume & Horizon Diminishing Returns (Siklus Pengalaman)
**Tujuan Umum:** Mengetahui *Titik Jenuh* (Kurva *Diminishing Returns*) antara seberapa banyak tumpukan data masa lalu dapat membuat rasio *Error* makin rendah seiring bengkaknya durasi latihan *(Train Time)* akibat kelemahan Konsep Bergeser (*Concept Drift*).

### Eksperimen 2.1: Ablasi Rentang Waktu Historis (Kurva-U)
Model Default (MSE) dilatih berkali-kali melintasi berbagai rentangan waktu. **Hipotesanya:** AI hanya membutuhkan data musiman yang cukup proporsional, terlalu kuno (5 tahun +) memicu bias, dan terlalu dikit (3 Bulan) memicu kegagalan prediksi *Outliers* ekstrem.

| ID Skenario | Kategori Rekaman Cuaca | Fokus Analisa Generalisasi Cuaca Terhadap Model |
| :--- | :--- | :--- |
| `Vol-1` | 3 Bulan Terakhir | Pembuktian bahwa *Error* terlihat 0.92 Murni karena AI menghafal pola stabil jangka pendek. Bahaya *Seasonality Bias* jika di *Deploy* musim lain. |
| `Vol-2` | 6 Bulan Berjalan | Fase kebingungan transisi iklim. RMSE diprediksi tertinggi/terburuk karena data mendua (kemarau & basah paruh tahun). |
| `Vol-4` | 2 Tahun (Siklus Standar) | Pencapaian Kematangan Perputaran Rotasi. Fase AI mulai menyadari pola iklim tahunan secara logis dan menurunkan nilai Kesalahan *Error*. |
| `Vol-5` | 5 Tahun Penuh (2020-2024)| Pencarian 'Titik Jenuh'. Bukti nRMSE (6.16%) hampir identik dengan 2 Tahun (6.16%), menandakan data 2020 sudah 'usang' / tidak berharga untuk belajar. |
| **`Vol-6`** | **5 Tahun Mutakhir (2021-2025)**| **Skenario Emas (Pemenang).** Setelah menghapus faktor keusangan 2020, *Sweet Spot* optimal dicapai pada R² 0.87 dan nRMSE teraman: **6.08%**. |

> **Pencapaian Fase 2:** Data berukuran 5 Tahun Terbaru tanpa bias masa lalu **(`Vol-6`) ditetapkan secara sah dan permanen** sebagai Fondasi Utama yang mengunci *Dataset* Sistem Pabrik.

---

## 🧠 FASE 3: Model-Centric & Optimizer Tuning (Ekstraksi Matematika - Anda Berada Di Sini)
**Tujuan Umum:** Membawa bongkahan Fondasi Emas Data (`Vol-6`) ke potensi kalkulasi *Deep Learning* terdalamnya dengan mengubah Hukum Gradien Error Model dan Pencarian Paramater Otomatis. Karena kita sudah tahu data 5 tahun ini **dipenuhi anomali cuaca ektsrem (badai & terik tak wajar)**, maka pendekatan hukuman Error AI wajib direvisi secara kalkulus.

### Eksperimen 3.1: Transformasi Loss Function (Konkordansi Outliers)
Algoritma dasar dihukum lewat metrik kuadrat `MSE`. Kini AI dipindahkan paksa eksekusinya ke parameter regresi *Robust-to-Outliers* yang toleran: `Huber Loss`.
*   **Cara Kerja Huber di Solar:** Jika *error* harian tipis (cuaca sedang cerah biasa), gradien akan diseret layaknya `MSE` (*Convex Mulus*). Namun, jika terjadi fluks prediksi meleset drastis gara-gara *Badai Tak Tertebak / Cumulonimbus Nyasar*, rumus langsung bertransformasi patah menjadi `MAE` (Linear).
*   **Hipotesis Eksekusi:** Wujud AI `Vol-6` yang memiliki nRMSE 6.08%, ketika dilatih ulang dengan parameter `loss_fn='huber'`, ditargetkan dapat menekan nilai agregat kesalahan kasarnya *(MAE base)* secara masif.

### Eksperimen 3.2: Hyperparameter Tuning via Optuna (TPE)
Mesin cerdas Bayesian TPE diluncurkan **secara spesifik di atas data `Vol-6` yang sudah memakai hukum `Huber Loss`**.
*   Karena struktur Data 5 Tahun sangat berat untuk RAM VRAM GPU, parameter kapasitas otak AI *(Model Capacity)* dibatasi secara rasional.
*   *Search Space Limits:* Jumlah Tumpukan *Layers* dikunci di maksimal `n_layers: [2, 3]`.
*   *Target Eksperimen:* Mengarahkan *Tuning* untuk hanya membongkar mencari kombinasi terindah dari **Learning Rate**, rasio pertahanan hafal buta **Dropout Rate**, dan besaran sirkulasi data **Batch Size**. 

---

## 🏆 FASE 4: Kesimpulan Kelayakan Manufaktur (Deploy Readiness)
Apabila model jebolan Fase Eksperimen 3.2 ini sanggup mematahkan rekor metrik bawaan dari Skenario Data-Centric (`nRMSE < 6.08%`), maka model tersebut 100% tervalidasi berlapis (Data-Proven & Model-Proven) berstatus **PRODUCTION-READY** dan diwajibkan untuk di-_Compile_ menjadi *Backbone Pipeline* Pembangkit Listrik Tenaga Surya (Off-Grid).
