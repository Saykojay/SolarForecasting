# Strategi Search Space Optuna TPE untuk Model Causal Transformer / Decoder-Only (100 Trials)

Causal Transformer (Decoder-Only) adalah arsitektur turunan GPT yang diterapkan pada *Time Series Forecasting*. Berbeda dari Autoformer yang punya dekomposisi bawaan atau PatchTST yang mengandalkan patching, **Causal Transformer mengandalkan sepenuhnya pada *Self-Attention* dengan Causal Mask** — artinya setiap timestep hanya boleh "melihat" timestep sebelumnya (bukan ke depan), persis seperti bahasa yang dibaca dari kiri ke kanan.

Arsitektur ini memiliki 3 komponen kunci:
1. **Input Projection** — proyeksi linear dari `n_features` ke `d_model`
2. **Positional Encoding** — injeksi informasi urutan waktu
3. **Stacked TransformerEncoder + Causal Mask** — self-attention bertopeng segitiga (seperti GPT)
4. **Prediction Head** — flatten `Lookback × d_model` → prediksi `Horizon × Features`

Karena **Prediction Head memuat** `lookback × d_model` neuron, kombinasi `lookback` dan `d_model` yang terlalu besar akan **meledakkan jumlah parameter model secara kuadratik**. Ini menjadi pertimbangan utama dalam menentukan search space.

---

## 🎯 Tabel Rekomendasi Search Space (Causal Transformer 100 Trials)

| Parameter (Hyperparameter) | Rentang Pilihan | Skala / Distribusi | Argumentasi Pakar & Alasan Matematis |
| :--- | :--- | :--- | :--- |
| **`lookback`** <br>*(Jendela Sejarah Masa Lalu)* | **`72` – `336`** <br>*(3 – 14 Hari)* | *Integer (Step 24)* | Kompromi kritis: lookback yang panjang → model melihat lebih banyak konteks musiman (baik), TAPI jumlah parameter Prediction Head naik **proporsional** (`lookback × d_model`). Rentang 72–336 seimbang antara informasi temporal dan efisiensi VRAM. Optuna akan menemukan sweet-spot-nya. |
| **`d_model`** <br>*(Dimensi Representasi Vektor)* | **`32`, `64`, `128`** | *Categorical (`2^n`)* | Karena head layer = `lookback × d_model → forecast`, `d_model` yang terlalu besar membuat model raksasa yang sangat mudah overfit pada data energi surya kecil. **Mulai dari 32** (berbeda dari Autoformer/PatchTST yang mulai 64), karena arsitektur decoder-only lebih efisien per-layer. Performa terbaik sering ditemukan di 64. |
| **`n_layers`** <br>*(Kedalaman Decoder Stack)* | **`2` – `6`** | *Integer* | Decoder-only **membutuhkan lebih banyak layer** dibanding encoder-decoder (seperti Autoformer) karena setiap informasi harus terakumulasi secara kausalistis dari bawah ke atas. Paper GPT dan literatur LLM menunjukkan decoder-only optimal di kedalaman lebih dalam, tapi untuk time series sederhana, 2–6 sudah cukup. |
| **`n_heads`** <br>*(Kepala Multi-Head Attention)* | **`4`, `8`** | *Categorical (`2^n`)* | Karena `d_model` bisa sekecil 32, maka kita batasi `n_heads` ke [4, 8] agar dimensi per-head (`d_model / n_heads`) tidak terlalu kecil (<8). Constraint: `d_model % n_heads == 0` **WAJIB** terpenuhi. Jika `d_model=32`, hanya `n_heads=4` yang valid. |
| **`ff_dim`** <br>*(Dimensi Feed-Forward Network)* | **`64`, `128`, `256`, `512`** | *Categorical (`2^n`)* | Standar Transformer: `ff_dim ≈ 2× s/d 4× d_model`. Namun karena d_model bisa 32, kita mulai dari 64. Perlu menjaga keseimbangan — ff_dim terlalu besar memperlambat training tanpa benefit signifikan di dataset energi yang kolomnya sedikit. |
| **`batch_size`** <br>*(Umpan Per Update)* | **`128`** <br>*(Dikunci/Statis)* | *Fixed* | **Dikunci di 128.** GPU RTX A5000 (24 GB VRAM) sangat optimal di batch 128 untuk arsitektur decoder-only ini. Batch kecil (16–32) memperlambat konvergensi; batch terlalu besar (256+) berisiko OOM jika lookback panjang. |
| **`learning_rate`** <br>*(Kecepatan Belajar)* | **`0.00005` – `0.005`** | *Log-Uniform* | Range lebih lebar dibanding Autoformer karena decoder-only cenderung **konvergen lebih lambat** (tidak punya shortcut dekomposisi). Log-uniform memastikan Optuna mengeksplorasi baik 1e-4 maupun 5e-4 secara merata. LR terlalu tinggi (>0.005) menyebabkan *loss explosion* pada Causal Mask attention. |
| **`dropout`** <br>*(Mencegah Hafalan)* | **`0.05` – `0.35`** | *Float* | Sedikit lebih rendah dari Autoformer/PatchTST karena decoder-only sudah memiliki regularisasi alami dari Causal Mask (token hanya melihat masa lalu, bukan keseluruhan sequence). Dropout terlalu tinggi (>0.4) mematikan aliran informasi di arsitektur yang hanya bisa melihat ke belakang. |

---

## ⚠️ Peringatan Kritis: Explosion of Parameters

Karena Prediction Head: `nn.Linear(lookback × d_model, forecast_horizon × n_features)`:

| Lookback | d_model | Parameter Head | Status |
| :--- | :--- | :--- | :--- |
| 72 | 32 | 2,304 × output | ✅ Ringan |
| 168 | 64 | 10,752 × output | ✅ Normal |
| 336 | 128 | 43,008 × output | ⚠️ Berat |
| 336 | 256 | 86,016 × output | ❌ OOM Risk! |

**Inilah alasan utama `d_model` dimulai dari 32 (bukan 64 seperti model lain)** — untuk memberikan ruang eksplorasi `lookback` yang panjang tanpa meledakkan VRAM.

---

## 🔬 Perbandingan dengan Arsitektur Lain

| Aspek | Causal Transformer | PatchTST | Autoformer |
| :--- | :--- | :--- | :--- |
| Mekanisme utama | Causal Mask (GPT-style) | Patching + Channel-Independent | Auto-Correlation + Decomposition |
| Parameter spesifik | Tidak ada (murni Transformer) | `patch_len`, `stride` | `moving_avg` |
| Sensitivitas lookback | **Sangat tinggi** (mempengaruhi # params) | Rendah (dipecah jadi patch) | Sedang |
| Layer optimal | 2–6 (perlu lebih dalam) | 2–4 | 2–4 |
| Kelebihan | Simpel, intuitif, efisien per-layer | State-of-art Long-term | Decomposisi otomatis |
| Kelemahan utama | Head membesar dgn lookback | Sensitif terhadap patch_len | Butuh data musiman >4 hari |

---

## 💡 Tips Performa untuk 100 Trials

1. **Gunakan Subsampling 20-30%** untuk mempercepat trial awal. Causal Transformer cenderung membutuhkan lebih banyak epoch (30-50) untuk konvergen dibanding PatchTST karena tidak ada shortcut patching.
2. **Pruner Median sangat efektif** — karena banyak kombinasi `lookback × d_model` yang langsung terlihat jelek di epoch awal, pruner bisa memangkas 60-80% trial dan menghemat waktu drastis.
3. **Jangan panik jika val_loss lebih tinggi dari PatchTST** — Causal Transformer secara empiris sering kalah dari PatchTST/Autoformer pada long-term forecasting murni, TAPI ia unggul di skenario dimana temporal ordering/kausalitas benar-benar penting (misal: efek cuaca hari ini terhadap output besok).
