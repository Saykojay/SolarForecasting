# 🔍 Audit: Implementasi PatchTST vs Paper Asli (ICLR 2023)

> Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
> Nie et al., Published at ICLR 2023
>
> File yang diaudit: `src/model_factory.py`

---

## Ringkasan Audit

| Komponen | Paper | Implementasi | Status |
|---|---|---|:---:|
| Patch Embedding | Conv1D projection | Conv1D projection | ✅ Sesuai |
| Positional Encoding | Learnable | Learnable | ✅ Sesuai |
| Transformer Encoder | MHA + FFN + LayerNorm | MHA + FFN + LayerNorm | ✅ Sesuai |
| Activation (FFN) | GELU | GELU | ✅ Sesuai |
| FFN Dimension | D → F(2D) → D | D → 4D → D | ⚠️ Berbeda |
| Norm Style | Pre-norm (paper default) | Post-norm (LayerNorm setelah residual) | ⚠️ Berbeda |
| Channel Independence | Key innovation | **TIDAK ADA** | ❌ Missing |
| Instance Normalization | RevIN (normalize input) | **TIDAK ADA** | ❌ Missing |
| Prediction Head | Flatten + Linear | Flatten + Dense(128,relu) + Dropout + Linear | ⚠️ Berbeda |
| n_layers default | 3 | 4 | ⚠️ Berbeda |
| n_heads default | 16 | 8 | ⚠️ Berbeda |
| d_model default | 128 | 128 | ✅ Sesuai |
| dropout default | 0.2 | 0.1 | ⚠️ Berbeda |
| Patch len / Stride | 16 / 8 | 16 / 8 | ✅ Sesuai |
| BatchNorm | Disebutkan superior untuk TS | Tidak digunakan | ⚠️ Tidak ada |
| Loss Function | MSE | MSE | ✅ Sesuai |

---

## Detail Perbedaan

### ❌ 1. Channel Independence (KRITIS — Inovasi utama paper)

**Paper (Section 3.1):**
> "Each input token to the Transformer is a **single univariate** time series...
> Channel-independence: each channel contains a single univariate time series that
> shares the same embedding and Transformer weights across all the channels."

Implementasi paper memproses setiap variabel **secara independen** melalui Transformer
yang sama, lalu menggabungkan hasilnya. Secara teknis:

```python
# Paper approach:
# Input: (Batch, Lookback, N_features)
# Reshape ke: (Batch * N_features, Lookback, 1)  ← CHANNEL INDEPENDENCE
# Proses melalui Transformer
# Reshape kembali: (Batch, N_features, Horizon)
# Flatten → Linear → Output
```

**Implementasi saat ini:**
```python
# Input: (Batch, Lookback, N_features)    ← SEMUA FITUR MASUK BERSAMAAN
# PatchEmbedding → Transformer → Flatten → Dense → Output
```

Semua fitur dimasukkan sekaligus ke Conv1D (channel-mixing), bukan diproses
independen. Ini **fundamental berbeda** dari paper.

**Dampak:**
- Channel independence mencegah overfitting pada korelasi antar variabel
- Meningkatkan generalisasi terutama pada dataset besar
- Ablation study paper menunjukkan channel independence + patching = best performance

---

### ❌ 2. Instance Normalization (RevIN)

**Paper (Section 3.1):**
> "It simply normalizes each time series instance x(i) with zero mean and unit
> standard deviation. We normalize each x(i) before patching and the mean and
> deviation are added back to the output prediction."

Ini **normalize per-instance** (bukan per-batch seperti BatchNorm):
```python
# Sebelum masuk model:
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)
x_norm = (x - mean) / (std + eps)

# Setelah prediksi:
output = output * std + mean  # denormalize
```

**Implementasi saat ini:** Tidak ada instance normalization. Hanya menggunakan
MinMaxScaler pada preprocessing (yang berbeda konsepnya).

**Dampak:**
- Table 11 paper menunjukkan instance norm memberikan improvement signifikan
  terutama pada dataset dengan distribution shift (ILI: 1.32 vs 3.56 MSE)

---

### ⚠️ 3. FFN Dimension (Minor)

**Paper (Appendix A.1.4):**
> "one projecting the hidden representation D = 128 to a new dimension **F = 256**"

Jadi FFN ratio = **2x** (D=128 → F=256 → D=128)

**Implementasi:**
```python
ff_dim=d_model * 4  # Line 110 → 128 * 4 = 512
```

FFN ratio = **4x** (D=128 → F=512 → D=128)

**Dampak:** FFN 4x lebih besar = lebih banyak parameter, bisa overfitting pada
dataset kecil. Paper menggunakan 2x yang lebih konservatif.

---

### ⚠️ 4. Normalization Order (Pre-norm vs Post-norm)

**Paper menggunakan standard Transformer encoder** yang pada implementasi referensi
(ICLR 2023 codebase) menggunakan **Pre-LayerNorm**:

```python
# Pre-norm (Paper):
x_norm = LayerNorm(x)
attn = MHA(x_norm, x_norm)
x = x + Dropout(attn)
x_norm = LayerNorm(x)
ffn = FFN(x_norm)
x = x + Dropout(ffn)
```

**Implementasi (Post-norm):**
```python
# Post-norm (model_factory.py lines 78-84):
attn_output = self.mha(x, x)
attn_output = self.dropout1(attn_output, training=training)
x = self.ln1(x + attn_output)      # ← Norm SETELAH residual
ffn_output = self.ffn(x)
ffn_output = self.dropout2(ffn_output, training=training)
return self.ln2(x + ffn_output)     # ← Norm SETELAH residual
```

**Dampak:** Post-norm umumnya lebih stabil untuk training tapi Pre-norm memberikan
gradient flow yang lebih baik untuk model yang dalam (banyak layer).

---

### ⚠️ 5. Prediction Head Terlalu Kompleks

**Paper:**
> Flatten → **Single Linear Layer** → Output

```python
# Paper: sederhana
x = Flatten(x)  # (batch, num_patches * d_model)
output = Linear(forecast_horizon)(x)
```

**Implementasi:**
```python
# model_factory.py lines 112-115: terlalu dalam
x = Flatten(x)
x = Dense(128, activation='relu')(x)    # ← Layer ekstra
x = Dropout(dropout)(x)                  # ← Dropout ekstra
outputs = Dense(forecast_horizon)(x)
```

**Dampak:** Layer ekstra menambah non-linearity yang tidak perlu. Paper sengaja
menggunakan linear head sederhana agar representation learning efektif.

---

### ⚠️ 6. Default Hyperparameters

| Parameter | Paper Default | Implementasi | Catatan |
|---|:---:|:---:|---|
| n_layers | **3** | 4 | Paper: "3 encoder layers" |
| n_heads | **16** | 8 | Paper: "head number H = 16" |
| dropout | **0.2** | 0.1 | Paper: "Dropout with probability 0.2" |
| d_model | 128 | 128 | ✅ Sama |
| patch_len | 16 | 16 | ✅ Sama |
| stride | 8 | 8 | ✅ Sama |
| FFN dim | 2 * D = 256 | 4 * D = 512 | Paper: F = 256 |

> Note: Paper menggunakan **reduced parameters** untuk dataset kecil:
> H=4, D=16, F=128 (ILI, ETTh1, ETTh2)

---

## Prioritas Perbaikan

### 1. 🔴 Channel Independence (HARUS diperbaiki)
Ini inovasi **utama** paper. Tanpa ini, model bukan "PatchTST" melainkan
"Patched Transformer" biasa. Channel independence memberikan:
- Mencegah overfitting korelasi antar variabel
- Parameter sharing yang efisien
- Generalisasi lebih baik

### 2. 🟠 Instance Normalization (Sangat disarankan)
Memberikan improvement signifikan terutama pada data dengan distribution shift.
PV data mengalami distribution shift musiman → sangat relevan.

### 3. 🟡 FFN Dimension (Disarankan)
Ubah dari 4x ke 2x untuk mengurangi overfitting.

### 4. 🟡 Prediction Head (Disarankan)
Sederhanakan ke single linear layer sesuai paper.

### 5. 🟢 Hyperparameter Defaults (Opsional)
Sesuaikan n_layers=3, n_heads=16, dropout=0.2.

---

## Contoh Perbaikan Code

### Channel Independence
```python
def build_patchtst(lookback, n_features, forecast_horizon, hp):
    patch_len = hp['patch_len']
    stride = hp['stride']
    d_model = hp['d_model']
    n_heads = hp['n_heads']
    n_layers = hp['n_layers']
    dropout = hp['dropout']

    inputs = Input(shape=(lookback, n_features), name='main_input')

    # === CHANNEL INDEPENDENCE ===
    # Proses setiap channel secara independen melalui Transformer yang sama
    # Reshape: (batch, lookback, n_features) → (batch*n_features, lookback, 1)

    # Transpose to (batch, n_features, lookback)
    x = tf.keras.layers.Permute((2, 1))(inputs)
    # Reshape to (batch * n_features, lookback, 1)
    x = tf.keras.layers.Reshape((n_features * lookback,))(x)
    x = tf.keras.layers.RepeatVector(1)(x)  # placeholder

    # Simpler approach: use TimeDistributed or manual loop
    # ... (implementation details)

    # === INSTANCE NORMALIZATION ===
    # Normalize per-instance before patching
    # mean, std saved for denormalization at output

    # Patch Embedding
    x = PatchEmbedding(patch_len, stride, d_model)(x)

    # Positional Encoding
    num_patches = (lookback - patch_len) // stride + 1
    x = PositionalEncoding(max_len=num_patches + 10, d_model=d_model)(x)

    # Transformer Encoder
    for i in range(n_layers):
        x = TransformerEncoderBlock(d_model, n_heads,
                                     ff_dim=d_model * 2,  # Paper: 2x not 4x
                                     dropout=dropout)(x)

    # Simple Linear Head (Paper: Flatten + Linear only)
    x = tf.keras.layers.Flatten()(x)
    outputs = Dense(forecast_horizon, activation='linear')(x)

    return tf.keras.Model(inputs, outputs, name='PatchTST')
```

### Instance Normalization Layer
```python
class InstanceNorm(tf.keras.layers.Layer):
    """RevIN: Reversible Instance Normalization"""
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = tf.reduce_mean(x, axis=1, keepdims=True)
            self.std = tf.math.reduce_std(x, axis=1, keepdims=True) + self.eps
            return (x - self.mean) / self.std
        else:  # denorm
            return x * self.std + self.mean
```

---

## Kesimpulan

Implementasi saat ini menangkap **ide dasar patching** dengan benar (Conv1D untuk
patch projection, positional encoding, Transformer encoder), tetapi **kehilangan
dua inovasi kunci** yang membuat PatchTST unggul di paper:

1. **Channel Independence** — memproses setiap variabel secara independen
2. **Instance Normalization** — normalize per-instance untuk mengatasi distribution shift

Tanpa dua komponen ini, model lebih mirip "Patched Vanilla Transformer" daripada
"PatchTST" yang sebenarnya. Menambahkan keduanya berpotensi memberikan improvement
yang signifikan pada performa forecasting.
