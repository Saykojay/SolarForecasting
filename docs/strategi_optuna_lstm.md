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
| **`learning_rate`** <br>*(Kecepatan Belajar)* | **`0.001` - `0.01`** | *Log-Uniform* | Jika Anda memaksakan *Batch Size* besar (128-256), Anda **wajib** menyesuaikan *Learning Rate* menjadi lebih besar (Aturan *Linear Scaling Rule*). Karena pembaruan *gradient* lebih jarang terjadi (namun variansinya lebih stabil), menaikkan batas atas LR ke `0.01` akan mencegah model terjebak dalam *Underfitting* akibat langkah pembaruan bobot yang terlalu pelan. |
| **`lookback`** <br>*(Jendela Periode)* | **`24` - `168`** <br>*(Step: 24 (Harian))* | *Categorical / Int* | Inilah titik lemah utama (Kriptonit) LSTM: **Bottleneck memori jangka pendek!** Berbeda dengan pandangan tajam *PatchTST* ke masa lalu, LSTM mulai "Pikun/Lupa" kejadian di langkah *time-step* 1 jika disuruh menelan data sampai sepanjang 168 langkah. Disarankan menargetkan *Lookback* pada periode **`24` jam hingga `72` jam (1-3 hari)** saja agar relevansi sel "*Forget Gate*" tidak rusak. |
| **`batch_size`** <br>*(Umpan Ukuran Data)* | **`128` - `256`** <br>*(Pilihan: 128, 256)* | *Categorical* | Jika menggunakan GPU monster seperti A5000, rentang besar `128 - 256` akan membuat proses _Training_ 5-10x lebih cepat secara komputasi. Konsekuensinya model mungkin kehilangan sedikit efek '*noise*' regulasi *SGD*, namun bisa dikompensasi secara sempurna dengan *Learning Rate* yang lebih besar (lihat baris atas). |

---

## 💡 Strategi Eksekusi: "Mode Ekstensif" (Budget 100 Trials)

Dengan _budget_ **100 kali percobaan (Trials)** yang sangat leluasa dan didukung oleh komputasi GPU A5000, Anda bisa membiarkan mesin menjelajahi seluruh lautan probabilitas (Eksplorasi) sebelum ia menukik tajam (Eksploitasi). TPE akan punya cukup bensin untuk secara matematis membuktikan hipotesis parameter-parameter eksotik.

Gunakan *konfigurasi ekstensif* ini di UI Streamlit Anda:

1. **Arsitektur:** `lstm`
2. **Lapisan (Layers):** Min `1` — Max `3` *(Beri Optuna ruang untuk membuktikan bahwa 3 lapis bisa jadi jebakan, atau malah jackpot).*
3. **Unit Terdalam (d_model):** Min `16` — Max `128` — Step `16` *(Optuna akan menjelajahi ruang arsitektur langsing hingga gemuk).*
4. **Jendela Masa Lalu (lookback):** Min `24` — Max `168` — Step `24` *(Optuna akan menguji apakah menengok ke seminggu penuh ke belakang berguna bagi LSTM atau justru membuatnya "bingung").*

**Mengapa Melebarkan Rentang Seperti Ini?**
Dengan 100 *Trials*, Optuna TPE memiliki modal yang cukup untuk melakukan *Random Exploration* di 20-30 eksekusi pertama. Setelah ia memetakan area mana yang sukses (rendah eror) dan mana yang hancur berantakan (*vanishing gradient* di >128 _hidden units_ atau lapis ke-3), ia akan menggunakan sisa 70 trial fokus mengoptimasi angka yang menjanjikan.

Dalam "Mode Ekstensif" ini, hasil grafik paralel (*Parallel Coordinate Plot*) Optuna Anda akan terlihat sangat menakjubkan, menunjukkan evolusi kecerdasan buatan dari tebakan kasar (fase Eksplorasi) yang bergeser pelan-pelan ke area "Juara Mutlak" di 50 trial terakhir (fase Eksploitasi).
