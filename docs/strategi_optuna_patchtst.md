# Strategi Search Space Optuna TPE untuk Model PatchTST (100 Trials)

PatchTST adalah salah satu model berbasis Transformer paling canggih saat ini untuk Time Series (*Channel Independent Patch Time Series Transformer*). Tidak seperti GRU yang memproses data langkah demi langkah, PatchTST memotong data *lookback* menjadi "patch" (potongan kecil) dan memprosesnya secara paralel. Oleh karena itu, *search space* untuk model ini membutuhkan konfigurasi yang sangat spesifik dan sinkron matematis (misalnya ukuran *patch* harus selaras dengan *stride* dan *lookback*).

Berdasarkan implementasi arsitektur **TensorFlow/Keras** khusus (`patchtst`) dan *best practice* eksekusi 100 trials menggunakan **TPE (Tree-structured Parzen Estimator)**, berikut adalah rancangan ruang pencariannya.

---

## 🎯 Tabel Rekomendasi Search Space (PatchTST dengan 100 Trials)

| Parameter (Hyperparameter) | Rentang Pilihan | Skala / Distribusi | Argumentasi Pakar & Alasan Matematis |
| :--- | :--- | :--- | :--- |
| **`patch_len`** <br>*(Panjang Potongan/Patch)* | **`8` - `32`** | *Step 8 (8, 16, 24, 32)* | Ini adalah jumlah jam (titik data) dalam satu token. Karena data energi surya memiliki siklus harian yang kuat (siang-malam), potongan 8, 16, atau maksimal 24 jam sangat direkomendasikan untuk menangkap pola lokal matahari terbit hingga terbenam. |
| **`stride`** <br>*(Langkah Pergeseran Patch)* | **`4` - `16`** | *Step 4 (4, 8, 12, 16)* | *Stride* harus selalu lebih kecil atau sama dengan `patch_len` agar antar-patch saling tumpang tindih (*overlapping*). Stride 8 pada patch_len 16 adalah standar emas (*50% overlap*). |
| **`d_model`** <br>*(Dimensi Konteks Vektor)* | **`32`, `64`, `128`** | *Categorical (`2^n`)* | PatchTST tidak rakus dimensi laten seperti Transformer terjemahan teks. Nilai 64 atau 128 sudah sangat masif untuk sekadar dataset *tabular* energi. Lebih dari itu berisiko memakan VRAM secara eksponensial tanpa *gain* akurasi. |
| **`n_layers`** <br>*(Kedalaman Encoder)* | **`1` - `4`** | *Integer* | Tumpukan balok Transformer. Sama seperti `d_model`, data cuaca jarang butuh abstraksi tingkat tinggi. Rentang 2 atau 3 layer sering kali menjadi yang terbaik. Lapis ke-4 opsional untuk menguji batas *underfitting*. |
| **`n_heads`** <br>*(Kepala *Attention*)* | **`4`, `8`, `16`** | *Categorical (`2^n`)* | Catatan Super Penting: **`d_model` HARUS habis dibagi oleh `n_heads`**. Sistem Anda di `trainer.py` sudah menangani auto-koreksi ini. Semakin banyak kepala, semakin model ini melihat berbagai "jenis pola" cuaca unik, tapi kalkulasinya memberatkan memori. |
| **`ff_dim`** <br>*(Forward Network internal)* | **`128`, `256`, `512`** | *Categorical (`2^n`)* | Biasanya disetel 2× atau 4× dari `d_model`. Bertugas mencerna dan meresapkan hasil saringan pola Attention Head. |
| **`lookback`** <br>*(Jendela History Cuaca)* | **`72` - `336`** <br>*(3 - 14 Hari)* | *Categorical (Step 24)* | Kelemahan utama GRU/LSTM (ingatan jangka pendek) adalah kekuatan utama PatchTST! Transformer **suka sekali data panjang**. Melihat sejarah sisa cuaca selama 7, 10, atau hingga 14 hari ke belakang sangat berguna. Pastikan input tidak ganjil. |
| **`batch_size`** <br>*(Umpan Per Update)* | **`128`** <br>*(Dikunci/Statis)* | *Fixed* | Berbeda dengan GRU yang memicu OOM (kebocoran memori sekuensial) jika angkanya tinggi, PatchTST adalah matriks raksasa yang dihitung memanjang secara horizontal di VRAM (Paralelisme Super). Untuk efisiensi Tensor Core GPU (RTX 3090/4090) pada operasi matriks besar, batch size sebaiknya dikunci keras di ukuran **128**. Ini dijamin memberikan kombinasi kecepatan kilat TPE dan stabilitas *gradient update* tanpa hambatan *bottleneck* iterasi loop keras seperti milik CPU. |
| **`learning_rate`** <br>*(Kecepatan Turun Gunung)* | **`0.00005` - `0.001`** | *Log-Uniform* | Transformer (dengan Layer Norm dan Attention-nya) gampang "meledak" bobotnya di awal. Dia sangat sensitif, butuh angka super persekian desimal (*Adam Optimizer*), dan umumnya bekerja ajaib di angka di sekitar ~0.0001 (notasi 1e-4). |
| **`dropout`** <br>*(Skeptisisme Neouron Acak)* | **`0.1` - `0.4`** | *Float* | Karena PatchTST sangat mudah mencerna pola komprehensif, ia amat berpotensi *"Overfitting"* jika dibiarkan rakus. Rentang dropout tinggi (sampai 0.4) sangat bermanfaat memperkuat antibodinya menghadapi data suhu/radiasi ekstrem yang tak terlihat. |

---

## 🔬 Teori Evolusi TPE Optuna untuk PatchTST (100 Trials)

PatchTST berlari jauh lebih lambat tiap iterasi pelatihannya di atas keras dibanding model dasar GRU. Menyiapkan **100 Trials** Optuna untuk model ini adalah balapan *Maraton (Endurance Test)* sesungguhnya untuk RunPod Anda.

1. **Trial 1 - 30 (Eksplorasi Jaring Lebar):** Optuna akan menebak-nebak *Search Space* ekstrem secara radikal. Ia akan memadukan *patch_len* besar (32) dengan *stride* raksasa (16), sementara layer ditumpuk setinggi-tingginya (`n_layers`=4) untuk mencari batasan kasar di mana *loss* mulai membengkak atau VRAM menjerit.
2. **Trial 31 - 70 (Fokus Geometri Area Attention Pusat):** Setelah memetakan dataran Loss Huber/MSE, Optuna akan menyadari bahwa kestabilan terbaik rata-rata ada pada potongan *patch* menengah (16 atau 24). Di titik ini, TPE akan berusaha gonta-ganti arsitektur *attention head* (4 atau 8 atau 16) yang paling kompatibel dengan proporsi *d_model* (64 atau 128) yang sanggup dicerna 14-hari *lookback* tanpa kepanasan.
3. **Trial 71 - 100 (Fine-Tuning Ketinggian Loss):** 30 percobaan terakhir adalah operasi bedah mikro. Arsitektur besar-nya biasanya sudah terkunci logikanya (Misal: Dipastikan Optuna memilih d_model 128, n_heads 8, patch 16). Di sini TPE murni hanya mengejar efisiensi pengali `learning_rate` yang paling kecil dari ~`0.00021` hingga ~`0.00029` untuk memeras 1-2 persen sisa keunggulan R-Squared.

**Pesan Penting Jika Anda Ingin Eksekusi PatchTST 100 Trials:**
Selalu gunakan **Taktik Data Subsampling (misalnya 20-30%)** yang bisa dichecklist di bagian konfigurasi streamLit! Percobaan Optuna mencari hyperparameter untuk PatchTST jika disuntikkan setahun penuh (100% dari 30.000 baris data) x 100 trials, bisa memakan waktu penyewaan GPU (runpod) hingga semalaman. Dengan 20-30% *subsampling* yang merepresentasikan iklim terakhir, ia bisa merampungkannya dalam 1-2 jam. Gunakan parameter terbaik dari tuning kilat subsampling ini baru ke pelatihan *Full Dataset* sesungguhnya.
