# Strategi Search Space Optuna TPE untuk Model LSTM

Dalam konteks *Time Series Forecasting* (Peramalan Runtun Waktu), parameter optimal untuk arsitektur *Long Short-Term Memory* (LSTM) sangat berbeda dengan model berbasis Transformer (seperti PatchTST). LSTM memiliki kerentanan tinggi terhadap masalah *vanishing gradient* (memudar/meluasnya gradien) dan *overfitting* (terlalu menghafal data) jika tidak dikonfigurasi dengan hati-hati. 

Berdasarkan konsensus komunitas peneliti *Machine Learning*, berikut adalah panduan mengatur *Search Space* (*Rentang Pencarian*) menggunakan algoritma optimasi **TPE (Tree-structured Parzen Estimator)** pada kerangka Optuna agar efisien, murah (khusus 20-50 Trials di GPU), dan mematikan.

---

## 🎯 Tabel Rekomendasi Search Space (LSTM)

| Parameter (Hyperparameter) | Rentang (Search Space) | Skala Distribusi | Argumentasi Pakar & Alasan Matematis |
| :--- | :--- | :--- | :--- |
| **`d_model`** <br>*(Hidden Size per Layer)* | **`16` - `128`** <br>*(Step/Kelipatan: 16)* | *Categorical / Int* | Tidak seperti Transformer yang diuntungkan oleh Dimensi raksasa (512+), **kapasitas sel memori LSTM di atas angka 128 untuk data tabular deret waktu (cuaca, dll) seringkali memicu degradasi performa**. Ia menjadi *overparameterized* (terlalu banyak pintu logika yang dihafal). Angka **`32`** dan **`64`** adalah "Sweet Spot" juara karena memaksa LSTM untuk mengekstrak *pola/tren* alih-alih menghafal mati titik data. |
| **`n_layers`** <br>*(Kedalaman/Tumpukan)* | **`1` - `3`** <br>*(Step: 1)* | *Integer* | *"Lebih dalam tidak selalu lebih pintar"*. Secara empiris, menumpuk LSTM lebih dari 3 lapis untuk data runtun waktu numerik sangat meresikokan hilangnya aliran gradien ke lapisan bawah. Model menjadi sangat lambat dikonvergensi. Sebuah model **1 Lapis yang Lebar (64 unit)** sangat sering mengungguli model 3 Lapis bersusun (32 unit). |
| **`dropout`** <br>*(Regularisasi)* | **`0.10` - `0.40`** <br>*(Step: 0.05)* | *Float (Linear)* | Arsitektur LSTM secara inheren adalah "mesin penghafal" *(Memorizer)*. Tanpa efek kejut perusakan ingatan seluler yang diatur komplotan *Dropout*, model akan gagal memprediksi pola masa depan yang belum terpetakan. Rentang **`0.20 - 0.30`** merupakan koridor ajaib standar industri untuk menjamin model tahan banting terhadap *noise*. |
| **`learning_rate`** <br>*(Kecepatan Belajar)* | **`0.0005` - `0.005`** | *Log-Uniform* | LSTM yang dipadukan dengan *optimizer Adam* amat sangat sensitif terhadap nilai eksponensial *Learning Rate*. Angka >`0.01` akan memperlihatkan kurva *Loss* yang bergetar pecah (*Oscillating/Diverging*). Sangat krusial meminta Optuna TPE untuk mencari di skala **Logaritmik** agar algoritmanya fokus mencari kedalaman (misal: beda antara `0.001` vs `0.0008`) daripada melebar ke angka statis. |
| **`lookback`** <br>*(Jendela Periode)* | **`24` - `168`** <br>*(Step: 24 (Harian))* | *Categorical / Int* | Inilah titik lemah utama (Kriptonit) LSTM: **Bottleneck memori jangka pendek!** Berbeda dengan pandangan tajam *PatchTST* ke masa lalu, LSTM mulai "Pikun/Lupa" kejadian di langkah *time-step* 1 jika disuruh menelan data sampai sepanjang 168 langkah. Disarankan menargetkan *Lookback* pada periode **`24` jam hingga `72` jam (1-3 hari)** saja agar relevansi sel "*Forget Gate*" tidak rusak. |
| **`batch_size`** <br>*(Umpan Ukuran Data)* | **`32` - `128`** <br>*(Pilihan: 32, 64, 128)* | *Categorical* | Sifat dasar RNN/LSTM yang memproses data berbalas-balasan (*Sequential*) sangat dirusak oleh *Batch Size* raksasa (>256). Ukuran *batch* besar mematikan derajat "*noise*" dari kalkulasi *Stochastic Gradient Descent* yang vital bagi LSTM untuk merangkak keluar dari lubang jalan buntu optimasi (*Local Minima*). Angka **`64`** adalah kecepatan hibrida terbaik. |

---

## 💡 Strategi Eksekusi: "Mode Sniper" (Budget 20 Trials)

Jika Anda memiliki batasan komputasi GPU / komersial dan hanya dapat menjalankan Optuna selama **20 kali percobaan (Trials)**, Anda wajib menyempitkan radar saringan (*Search Space*) sekuat mungkin untuk membantu TPE menemukan bongkahan emas.

Gunakan *konfigurasi ekstrem* ini di UI Streamlit Anda:

1. **Arsitektur:** `lstm`
2. **Lapisan (Layers):** Min `1` — Max `2` *(Cukup uji 1 dan 2 lapis, buang lapis 3)*.
3. **Unit Terdalam (d_model):** Min `32` — Max `64` — Step `32` *(Memaksa Optuna hanya memilih 32 atau 64)*.
4. **Jendela Masa Lalu (lookback):** Min `24` — Max `72` — Step `24` *(Hanya uji kekuatan ramal mundur 1, 2, atau 3 hari)*.

**Mengapa Menyempitkan Seperti Ini?**
Algoritma TPE *(Tree-structured Parzen Estimator)* berkerja dengan mengarahkan probabilitas uji coba ke area yang secara historis memberikan hasil *Loss/Error* terkecil. Dengan menyuguhkan "Pilihan Menu" yang sempit berisi angka-angka jagoan saja, Anda menghemat 15 *Trials* pertama yang biasanya dihabiskan Optuna untuk belajar angka-angka ekstrem yang pasti gagal (*seperti hidden size 512 atau lookback 336*). 

Dalam "Mode Sniper" ini, hampir 95% dari ke-20 model mutan hasil racikan mesin pencari Anda akan menghasilkan garis grafik valid yang kompetitif melawan model Baseline Linear Regression.
