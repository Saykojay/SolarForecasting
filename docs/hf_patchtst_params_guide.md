# Kategorisasi Parameter Hugging Face PatchTST

Berdasarkan dokumentasi resmi `transformers.PatchTSTConfig`, berikut adalah pengelompokkan parameter untuk model PatchTST bawaan *Hugging Face* dalam konteks pipeline `patchtst_hf` milik Anda. Pengelompokkan ini membedakan mana parameter yang dikendalikan oleh UI Streamlit secara live, mana yang di-hardcode demi stabilitas, dan mana yang tidak digunakan karena beda kasus penggunaannya.

---

### 🟢 1. ADJUSTABLE (Dapat Disetel di UI / Dioptimasi Optuna)
Parameter ini memiliki kontrol langsung di UI Streamlit Training/Tuning:

*   **`context_length`**: Dipetakan dari slider `lookback` (Window Size input Anda, misal: 72).
*   **`patch_length`**: Dipetakan dari input `patch_len` (Panjang setiap sub-series patch, misal: 16).
*   **`patch_stride`**: Dipetakan dari input `stride` (Jarak pergeseran antar titik awal sub-series).
*   **`num_hidden_layers`**: Dipetakan dari input `n_layers` (Jumlah blok utama Transformer Encoder).
*   **`d_model`**: Dipetakan dari pilihan `d_model` (Dimensi ukuran representasi *embedding* laten internal).
*   **`num_attention_heads`**: Dipetakan dari pilihan `n_heads` (Jumlah kepala mekanisme *Multi-Head Attention*).
*   **`ffn_dim`**: Dipetakan dari input `ff_dim` (Lebar dari bagian *Feed Forward Network* di setiap layer).
*   **🔥 Dropout Groups**: Slider *Dropout Rate* di UI Anda disuntikkan secara serentak ke 3 properti sekaligus demi kepraktisan regularisasi:
    *   **`attention_dropout`** 
    *   **`ff_dropout`** 
    *   **`head_dropout`** 

---

### 🟡 2. AUTO-LOCKED (Otomatis Menyesuaikan Konteks Data)
Parameter fundamental ini dikalkulasi otomatis oleh `src/data_prep.py` dan `config.yaml` sehingga Anda tidak perlu repot mengetiknya, agar model tidak bertabrakan dengan dimensi input:

*   **`num_input_channels`**: Otomatis mendeteksi jumlah `n_features` dari dataset Anda (misal: ada GHI, Module Temp, CSI, dst).
*   **`prediction_length`**: Otomatis mengikuti pengaturan global `forecast_horizon` yang Anda setel di tab Dashboard (misal: 24 jam).

---

### 🔴 3. HARDCODED (Sengaja Dikunci oleh Sistem)
Parameter ini saya tahan (*hardcoded*) di dalam skrip `src/model_hf.py` untuk menyelaraskan arsitekturnya persis dengan resep pemenang *"Original Paper Nie et al 2023"* — Channel Independence + Patching yang murni. Menjadikannya opsional hanya akan menambah kebingungan *search space* tanpa *impact* yang berarti bagi studi Power Forecasting ini:

*   **`activation_function` (Kunci: *"gelu"*)**: Standar industri modern yang tidak memiliki masalah "Dying ReLU" ("relu" didukung HF tapi kita tidak pakai).
*   **`norm_type` (Kunci: *"batchnorm"*)**: Ini sangat penting! Paper aslinya spesifik membuktikan bahwa *Batch Normalization* bekerja *sedikit* lebih baik untuk time series Channel-Independent ketimbang Layer Norm dalam PatchTST.
*   **`pre_norm` (Kunci: *True*)**: Pre-norm alias Normalisasi sebelum Attention Layer jauh lebih stabil saat kedalaman lapisannya membesar.
*   **`share_embedding` (Kunci: *True*)** & **`share_projection` (Kunci: *True*)**: Ini adalah kunci agar semua sinyal channel di-*embed* oleh 1 layer *Dense* linear yang sama sebelum masuk Transformer (tidak boros memori).
*   **`positional_encoding_type` (Kunci: *"sincos"*)**: Menggunakan rumus sinus-cosinus baku Transformer daripada inisialisasi *"random"* yang bisa melenceng di awal *training*.
*   **`pooling_type` (Kunci: *"mean"*)**: Cara model meringkas informasi patch beruntun (rata-rata) sebelum dipancarkan *(projected)* ke nilai titik waktu *forecast* akhir.

---

### ⚪ 4. TIDAK DIGUNAKAN (Diabaikan)
Parameter berlabel `(optional)` berikut ada di dalam dokumentasi Hugging Face **PatchTSTConfig**, namun sama sekali kita tidak gunakan (ter-bypass oleh nilai `False/None` bawaan):

#### 🚫 Mode *Probabilistic Distribution / Forecasting*
Jika di mode ini (`loss="nll"`), outputnya bukan angka eksak melainkan sebaran probabilitas (*mean* & *variance*). Karena di Pipeline Anda (`predictor.py`) sistem mengharapkan *Point Estimation* eksak dengan `Loss=Huber` atau `MSE`, kita bypass:
*   `distribution_output` (student_t, normal, dl)
*   `num_parallel_samples`

#### 🚫 Mode *Self-Supervised / Masked Pre-training*
Kekuatan rahasia HF PatchTST adalah Anda bisa melakukan pre-training terhadap dataset PVWatts tanpa target sama sekali (seperti MLM di BERT: menyembunyikan sebagian data agar ditebak), barulah di *fine-tune*. Karena saat ini kita melatih regresi langsung yang diawasi (Supervised), sistem bypass fitur ini:
*   `do_mask_input`
*   `mask_type`
*   `random_mask_ratio`
*   `num_forecast_mask_patches`
*   `channel_consistent_masking`
*   `unmasked_channel_indices`
*   `mask_value`

#### 🚫 Lainnya (Terkontrol di Luar Model)
*   `loss`: Alih-alih melimpahkan ke HF, Pipeline ini memotong jalur (*backward loss*) dan menggunakan fungsi loss kustom murni milik Anda via PyTorch Optimizer yang berfokus ke metrik regresi (`mse`, `huber`).
*   `scaling`: Tidak dipakai (Kita set `False`), karena Pipeline modular `data_prep.py` Anda sudah secara presisi menggunakan standardisasi Sklearn `MinMaxScaler/StandardScaler`.
*   `channel_attention`: Tidak dipakai, ini mengaktifkan Layer tambahan antar dimensi channel, merusak prinsip *murni* "Channel-Independence" dari PatchTST asli. 
*   `num_targets`: 1 (Ditetapkan oleh sistem).
*   `output_range`: Tidak dipakai (Sistem membatasi Output Model berdasarkan `nameplate_capacity` secara mekanikal (*hard bounds clipping*) di `predictor.py`).
*   `path_dropout`, `positional_dropout`, `norm_eps`, `init_std`, `bias`, `use_cls_token`: Diabaikan mengikuti *fallback* default PyTorch yang umum.
