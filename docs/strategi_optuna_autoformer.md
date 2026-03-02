# Strategi Search Space Optuna TPE untuk Model Autoformer (100 Trials)

Autoformer adalah model *Deep Learning* tingkat lanjut yang dirancang khusus untuk memecahkan kelemahan Transformer biasa pada deret waktu *(Time Series)* jarak jauh. Ia menggunakan mekanisme **Auto-Correlation** (menemukan pola berulang/musiman) dan **Decomposition** bawaan (memisahkan *trend* dan musiman di dalam arsitektur).

Berbeda dari PatchTST yang memotong data menjadi Patch, **Autoformer mencerna kurva waktu secara terus-menerus menggunakan dekomposisi pergerakan rata-rata (*Moving Average*)**.

Karena model ini memakan memori paralel seperti Transformer pada umumnya, **penguncian `batch_size : 128`** sangat disarankan bila Anda menjalankannya di GPU besar seperti RTX A5000 / RTX 3090 / 4090, untuk mempercepat 100 trials secara maksimal.

Berikut adalah susunan matematis ruang pencariannya.

---

## 🎯 Tabel Rekomendasi Search Space (Autoformer 100 Trials)

| Parameter (Hyperparameter) | Rentang Pilihan | Skala / Distribusi | Argumentasi Pakar & Alasan Matematis |
| :--- | :--- | :--- | :--- |
| **`lookback`** <br>*(Jendela Sejarah Masa Lalu)* | **`96` - `336`** <br>*(4 - 14 Hari)* | *Categorical (Step 24)* | Ini nyawa Autoformer! Karena ia mengandalkan analisis musim (*seasonality*), ia butuh sejarah minimal beberapa hari berulang (cth: siklus terbit-tenggelam matahari selama 7 hari atau 168 jam). Memberinya data pendek (< 96) mematikan fitur utama Auto-Correlation-nya. |
| **`moving_avg`** <br>*(Jendela Dekomposisi Tren)* | **`25` - `49`** | *Step 2 (Harus Ganjil)* | **Spesifik Autoformer!** Parameter ini bertugas meratakan kurva untuk menemukan garis *Trend* (Tren Utama). Standar mutlak dari paper aslinya adalah angka **25**. Nilainya **WAJIB GANJIL** agar titik tengah *sliding window* tetap simetris (contoh: 25, 27, 33, 49). |
| **`d_model`** <br>*(Dimensi Matriks internal)* | **`64`, `128`** | *Categorical (`2^n`)* | Ukuran representasi vektor. Angka **256** atau **512** terlalu rawan Overfit untuk dataset tabular energi sederhana. Cukup 64 atau 128. |
| **`n_layers`** <br>*(Kedalaman Encoder/Decoder)* | **`2` - `4`** | *Integer* | Tumpukan blok Auto-Correlation. Karena dekomposisinya sudah sangat pintar, menumpuk lebih dari 4 lapis berisiko degradasi. (Rekomendasi Optuna mencari di sekitar 2 atau 3). |
| **`n_heads`** <br>*(Kepala *Auto-Correlation*)* | **`8`, `16`** | *Categorical (`2^n`)* | Karena `d_model` dibatasi ke 64 atau 128, maka `n_heads` terbaik dipasang di 8 atau 16. Pastikan kembali **`d_model` selalu habis dibagi `n_heads`**. |
| **`ff_dim`** <br>*(Dimensi Feed Forward)* | **`256`, `512`** | *Categorical (`2^n`)* | Biasanya disetel 2× atau 4× dari `d_model`. |
| **`batch_size`** <br>*(Umpan Per Update)* | **`128`** <br>*(Dikunci/Statis)* | *Fixed* | GPU monster seperti A5000 sangat menyukai matriks besar. Menggunakan Batch Size 128 mencegah *bottleneck* transfer data CPU-ke-GPU. Mengunci di 128 juga mencegah risiko OOM (Out of Memory) jika Anda melakukan subsampling. |
| **`learning_rate`** <br>*(Kecepatan Belajar)* | **`0.00005` - `0.001`** | *Log-Uniform* | Layer Normalization pada Autoformer sensitif di awal pelatihan. Rate yang konservatif (notasi e-4) adalah standar. |
| **`dropout`** <br>*(Mencegah Hafalan)* | **`0.1` - `0.4`** | *Float* | Tingkat skeptisisme *(Dropout)* yang lumayan tinggi berguna agar model tidak murni menghafal letak lekukan matahari kemarin, tetapi mempelajari esensi polanya. |

---

## 🔬 Spesialisasi CLI Python untuk Autoformer

Jika Anda berencana memisahkan proses ini di eksekusi terminal (untuk GPU lain), Anda bisa menggunakan skrip CLI kita dengan tambahan sedikit injeksi fitur `moving_avg` untuk si Autoformer. 

Karena *Search Space* Autoformer tidak memakai `patch_len` maupun `stride` (itu milik adiknya, PatchTST), pastikan algoritma Optuna untuk `autoformer_hf` disetel dengan rapi menargetkan **Decomposition** dan membuang patching.

**Tips Performa Lintas-GPU:**
Jika GPU "Sewa Lain" yang Anda maksud juga berspesifikasi tinggi (A5000 / A6000), tuning Autoformer dengan **Trial 100 + Subsampling 30%** biasanya rampung dalam perhitungan ~90 menit karena batch_size 128-nya mempercepat iterasi per-epoch secara brutal.
