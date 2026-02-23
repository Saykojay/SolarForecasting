# Strategi Search Space Optuna TPE untuk Model TimeTracker

Membangun *Search Space* untuk arsitektur **TimeTracker** sangat berbeda dengan LSTM baseline ataupun PatchTST murni. **TimeTracker** di dalam repositori ini dibangun menggunakan pondasi Transformer (Decoder-only) yang diperkuat dengan lapisan **Mixture of Experts (MoE)**. Konsep MoE memungkinkan model memiliki kapasitas parameter raksasa (banyak *sub-network expert*) namun komputasinya tetap efisien karena hanya sebagian *expert* yang aktif (berkat mekanisme *Routing Top-K*).

Berdasarkan literatur mutakhir seputar *Transformer-based MoE* untuk deret waktu, berikut adalah rekomendasi *Search Space* ekstrem ketika Anda memiliki *budget* eksekusi TPE sebanyak **100 Trials** di atas GPU setara NVIDIA RTX A5000.

---

## 🎯 Tabel Rekomendasi Search Space (TimeTracker)

| Parameter (Hyperparameter) | Rentang (100 Trials) | Skala Distribusi | Argumentasi Pakar & Alasan Matematis |
| :--- | :--- | :--- | :--- |
| **`patch_len` & `stride`** <br>*(Tokenisasi Waktu)* | Patch: **`8` - `24`** <br>Stride: **`4` - `12`** | *Categorical (Step 4)* | *TimeTracker* menggunakan *Patching* untuk memperpendek horizon komputasi Atensi. Angka `16` adalah standar emas untuk data cuaca per jam (setara jendela 16 jam). Karena *budget* trial besar (100), biarkan Optuna membuktikan apakah jendela yang lebih sempit (`8`) mampu merekam *noise* jangka pendek lebih baik. |
| **`d_model` & `n_heads`** <br>*(Kapasitas Atensi)* | d_model: **`32` - `128`** <br>n_heads: **`4` - `8`** | *Categorical* | Tidak perlu *d_model* raksasa (>256) seperti LLM tipe teks. Mengapa? Karena kapasitas penyimpanan bobot TimeTracker sudah di-_boosting_ secara eksponensial oleh lapisan **Mixture of Experts (MoE)**. D_model yang ramping (64) dipadukan dengan banyak *Expert* akan menghasilkan model cepat yang kebal *Overfitting*. |
| **`n_layers`** <br>*(Kedalaman Trafo)* | **`2` - `4`** | *Integer* | Berbeda dengan LSTM yang "bodoh" di kedalaman, arsitektur Transformer memiliki *Skip Connections/Residuals*. Namun untuk deret waktu tabular, lebih dari 4 lapis *MoE Blocks* cenderung memicu *Over-smoothing* (semua token atensi menjadi homogen dan kehilangan kekuatannya). |
| **MoE: `n_private_experts`** <br>*(Jumlah Pakar Pribadi)* | **`2` - `8`** <br>*(Pilihan: 2, 4, 8)* | *Categorical* | Ini adalah nyawa TimeTracker! Semakin banyak *expert*, model semakin pintar memecah masalah (*trend vs seasonality*). Secara *default*, `4` sangat stabil. Berikan kebebasan Optuna menguji `8` pakar karena VRAM 24GB A5000 sangat sanggup menelan pembengkakan parameternya. |
| **MoE: `top_k`** <br>*(Ahli yang Terpilih)* | **`1` atau `2`** | *Categorical* | Jika `n_private_experts` adalah 8, meminta *Router* untuk memilih `top_k=2` (2 ahli terbaik per token waktu) adalah standar penelitian MoE modern (seperti *Sparse Transformer* Google). Jika `top_k` terlalu besar pemerataan komputasi (*Sparseness*) MoE akan hilang fungsinya. |
| **`lookback`** <br>*(Jendela Sejarah)* | **`96` - `336`** <br>*(4 hari s/d 14 hari)* | *Categorical (Step 24)* | Transformer tidak pernah menderita amnesia (tidak ada *vanishing gradient* temporal). Beri perintah Optuna menjelajah memori panjang untuk mencari pola mingguan (168 jam) dan dwi-mingguan (336 jam). Semakin panjang *Lookback*, semakin cerdas lapisan *Attention*. |
| **`batch_size`** <br>*(Umpan Data)* | **`128` - `256`** | *Categorical* | Sama dengan hukum GPU kelas atas sebelumnya. Untuk model *Transformer+MoE*, ukurun *batch* yang besar membantu *Router Layer* di dalam MoE untuk mendistribusikan beban secara seimbang (*Load Balancing*) antar *Expert*. *Batch* yang terlalu kecil (32) akan membuat beberapa *Expert* "kelaparan" data (mati). |
| **`learning_rate`** <br>*(Kecepatan Belajar)* | **`0.001` - `0.004`** | *Log-Uniform* | Model Transformer dengan MoE memiliki topologi gradien yang berisik karena fungsi *Routing* yang tidak selalu mulus (*Non-smooth*). LR jangan dibuat ekstrem (terlalu tinggi). Rentang logaritmik konvensional `0.001` - `0.004` digabungkan dengan *Batch* raksasa akan menjaga *Router* belajar untuk tidak selalu menunjuk *Expert* yang sama berulang-ulang (*Collapse*). |

---

## 💡 Alur Pikir TPE (Tree-structured Parzen Estimator) pada MoE dengan 100 Trials

Dengan _budget_ **100 Trials**, strategi "Mode Ekstensif" ini menjadi ruang bereksperimen yang fantastis untuk algoritma Optuna TPE:

1. **Trial 1 - 25 (Fase Eksplorasi Liar):**
   TPE akan secara acak menjodohkan parameter mutan. Misalnya: Model dengan `8 Expert` disilang dengan *lookback* mematikan `336 jam`, namun dengan `d_model` sempit `32`. Hasil Loss mungkin akan sangat berfluktuasi (ada yang jeblok, ada yang bersinar).
2. **Trial 26 - 60 (Pembentukan Topologi Parzen):**
   TPE mulai memahami interaksi fitur. TPE akan secara matematis menyadari bahwa *"Oh, jika saya menggunakan top_k=2, ternyata batch_size=256 bekerja lebih baik ketimbang 128 untuk menyeimbangkan beban para expert."* Ia akan mulai membuang konfigurasi yang membuang-buang memori tanpa menambah akurasi.
3. **Trial 61 - 100 (Fase Eksploitasi & Fine-Tuning Celah Sempit):**
   Optuna memusatkan 40 tembakan terakhir pada rentang spesifik (misal: ia sudah pasti mengunci `patch_len=16` dan `lookback=168`, lalu murni hanya mencari perbedaan desimal pada *Learning Rate* dan *Dropout*).

**Eksekusi:**
Segera setelah uji PatchTST Anda selesai, beralihlah ke arsitektur `timetracker`, atur batas di expander Optuna sesuai tabel di atas, pastikan Taktik 3 (Data Subsampling 30-40%) menyala, dan lepaskan model MoE pertama Anda!
