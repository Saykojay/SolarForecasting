# 🔬 Evaluasi Pipeline & Strategi Improvement

> Dokumen ini berisi analisis lengkap pipeline PV Forecasting dan strategi-strategi
> yang bisa diterapkan untuk meningkatkan performa model.
>
> **Tanggal evaluasi:** 19 Februari 2026

---

## Status Pipeline Saat Ini

| Aspek | Kondisi | Rating |
|---|---|:---:|
| **Data** | 51k rows, 14 variabel (NASA+OM), 5.9 tahun | ⭐⭐⭐⭐ |
| **Preprocessing** | Algorithm 1, outlier removal, physics-based | ⭐⭐⭐⭐ |
| **Feature Engineering** | Weather raw + time cyclical | ⭐⭐⭐ |
| **Model** | PatchTST, 128d, 8 heads, 4 layers | ⭐⭐⭐⭐ |
| **Training** | MSE loss, patience=15, lr schedule | ⭐⭐⭐ |
| **Best val_loss** | 0.0058 (3 fitur, lb=72) | — |

### Dataset: `singapore_pv_bestof_v2_2020_2025.csv`

| Sumber | Variabel | Korelasi PV |
|---|---|:---:|
| NASA POWER | `ghi_wm2` (Global Horizontal Irradiance) | +0.936 |
| NASA POWER | `dhi_wm2` (Diffuse Horizontal Irradiance) | +0.894 |
| NASA POWER | `ambient_temp_c` | +0.657 |
| NASA POWER | `relative_humidity_pct` | -0.677 |
| NASA POWER | `wind_speed_ms` | +0.335 |
| NASA POWER | `wind_direction_deg` | +0.061 |
| NASA POWER | `precipitation_mm` | -0.033 |
| Open-Meteo | `dni_wm2` (Direct Normal Irradiance) | +0.815 |
| Open-Meteo | `cloud_cover_pct` | -0.096 |
| Open-Meteo | `cloud_cover_low_pct` | +0.150 |
| Open-Meteo | `cloud_cover_mid_pct` | -0.063 |
| Open-Meteo | `cloud_cover_high_pct` | -0.135 |

### Model History (Best Configurations)

| Model | Features | val_loss | d_model | heads | layers | lookback | batch | dropout |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `_1414` | 19 | **0.0059** | 128 | 8 | 4 | 72 | 16 | 0.1 |
| `_1556` | 3 | **0.0058** | 128 | 8 | 4 | 72 | 16 | 0.1 |
| `_2150` | 3 | 0.0117 | 128 | 8 | 4 | 72 | 16 | 0.1 |
| `_1236` | 29 | 0.0346 | 128 | 8 | 4 | 168 | 32 | 0.1 |
| `_0807` | 9 | 0.0921 | 64 | 2 | 5 | 72 | 64 | 0.05 |

---

## Strategi Improvement

### 🥇 1. Feature Engineering — Lags & Rolling

**Expected improvement: +10-25% R²**

**Status saat ini:** OFF (`lags: false`, `rolling: false`)

Ini fitur terbuang paling besar. Mengaktifkan lags dan rolling features memberikan
model "shortcut" eksplisit untuk memahami pola temporal.

| Setting | Efek |
|---|---|
| `lags: true` | Menambah memori temporal: GHI 1h/12h/24h/48h lalu → model tahu tren |
| `rolling: true` | Stabilisasi noise: rata-rata 3 jam GHI → lebih smooth |

**Kenapa berdampak besar?**

PatchTST memang punya attention yang bisa "mundur" di lookback window, tapi
lag features memberikan shortcut eksplisit — model tidak harus belajar sendiri
bahwa "GHI 24 jam lalu penting."

**Risiko:** Menambah ~15-30 fitur → lebih banyak parameter → perlu lebih banyak data/epochs.

**Cara menerapkan:** Ubah di `config.yaml`:
```yaml
groups:
  lags: true
  rolling: true
```

---

### 🥈 2. Target: CSI vs Raw PV

**Expected improvement: +5-15% di productive hours**

**Status saat ini:** `use_csi: false` (prediksi raw kW langsung)

| Target | Pro | Kontra |
|---|---|---|
| **Raw PV (kW)** | Simple, interpretable | Range 0-16 kW, malam selalu 0 → model "buang" kapasitas belajar untuk malam |
| **CSI** | Normalized 0-1.2, hanya productive hours relevan | Perlu inverse transform, sedikit lebih kompleks |

**Kenapa CSI bisa lebih baik?**

CSI membuang variasi "mudah" (siklus siang-malam) dan fokus ke variasi "susah"
(cloudiness). Model menggunakan 100% kapasitasnya untuk memprediksi hal yang
benar-benar bervariasi.

**Cara menerapkan:**
```yaml
target:
  use_csi: true
features:
  groups:
    physics: true  # Diperlukan untuk menghitung CSI
```

---

### 🥉 3. Loss Function — Huber atau MAE

**Expected improvement: +3-8%**

**Status saat ini:** `loss: MSE`

| Loss | Sifat | Kapan Bagus |
|---|---|---|
| **MSE** | Hukum outlier sangat berat (kuadrat) | Data bersih, distribusi normal |
| **Huber** | MSE untuk error kecil, MAE untuk error besar | Data real-world (ada outlier) |
| **MAE** | Error linear, tidak sensitif outlier | Ingin error merata semua jam |

MSE membuat model fokus mengurangi error terbesar (misal peak hour) tapi
mengabaikan error kecil (misal senja). **Huber** memberikan keseimbangan lebih baik.

**Cara menerapkan:** Ubah di `src/model_factory.py` fungsi `compile_model`:
```python
model.compile(
    optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
    loss='huber'  # atau tf.keras.losses.Huber(delta=1.0)
)
```

---

### 4. Lookback Window Optimization

**Expected improvement: +2-5%**

**Status saat ini:** `lookback: 72` (3 hari)

Dari data model:
- `lb=72` → val_loss **0.0058** (terbaik)
- `lb=168` → val_loss 0.0136 (lebih buruk)

72 sudah optimal untuk dataset ini. Tapi bisa coba:
- **`lb=48`** (2 hari) — lebih cepat, mungkin cukup untuk cuaca tropis yang stabil
- **`lb=96`** (4 hari) — menangkap pola cuaca mingguan

**Cara menerapkan:**
```yaml
model:
  hyperparameters:
    lookback: 48  # atau 96
```

---

### 5. Scaler — StandardScaler vs MinMax

**Expected improvement: +2-5%**

**Status saat ini:** `scaler_type: minmax`

| Scaler | Range | Kapan Bagus |
|---|---|---|
| **MinMax** [0,1] | Bounded | Data tanpa outlier, distribusi uniform |
| **Standard** (z-score) | Unbounded | Data normal/skewed, ada outlier |
| **RobustScaler** | Quartile-based | Data dengan banyak outlier |

GHI punya distribusi **sangat skewed** (banyak 0 di malam, sedikit peaks).
**StandardScaler** atau **RobustScaler** sering lebih baik untuk data PV.

**Cara menerapkan:**
```yaml
features:
  scaler_type: standard    # atau robust
  target_scaler_type: standard
```

---

### 6. Hyperparameter Tuning (Optuna)

**Expected improvement: +3-10%**

**Observasi:** Semua model menggunakan konfigurasi yang sangat mirip
(`d_model=128, heads=8, layers=4, patch=16, stride=8`).

Hyperparameter yang belum dieksplorasi:

| Parameter | Saat Ini | Rekomendasi |
|---|:---:|---|
| **patch_len** | 16 | 4, 8, 12, 24 — patch pendek lebih sensitif cuaca cepat berubah |
| **stride** | 8 | 4 — overlap lebih banyak = informasi lebih detail |
| **d_model** | 128 | 64, 256 — 256 jika fitur banyak |
| **n_layers** | 4 | 2-3 — model lebih simple sering generalize lebih baik |
| **dropout** | 0.1 | 0.2-0.3 — mengurangi overfitting |
| **batch_size** | 16 | 32-64 — gradient lebih stabil |

**Cara menerapkan:**
```yaml
tuning:
  enabled: true
  n_trials: 30
  search_space:
    patch_len: [4, 24, 4]
    stride: [4, 16, 4]
    d_model: [64, 256]
    n_heads: [4, 8]
    n_layers: [2, 6]
    dropout: [0.1, 0.35]
    learning_rate: [0.00005, 0.001]
    batch_size: [16, 64]
    lookback: [48, 96, 24]
```

---

### 7. Data Augmentation

**Expected improvement: +2-5%**

Teknik khusus time series:
- **Jittering**: Tambah noise kecil ke input → model lebih robust
- **Window Slicing**: Gunakan sub-window acak dari lookback → lebih banyak training samples
- **Mixup temporal**: Campurkan 2 hari yang mirip → regularisasi

**Effort:** Requires code changes di training loop.

---

### 8. Ensemble / Multi-Model

**Expected improvement: +3-8%**

Bukan cuma 1 model, tapi gabungan:
- **PatchTST (best hp)** + **GRU** → average prediksi
- Atau: Train 3 PatchTST dengan seed berbeda → average → mengurangi variance

**Contoh implementasi:**
```python
pred1 = model_patchtst.predict(X_test)
pred2 = model_gru.predict(X_test)
final_pred = (pred1 + pred2) / 2  # simple average
# atau: weighted average berdasarkan val_loss masing-masing
```

---

### 9. Post-Processing

**Expected improvement: +1-3%**

Teknik sederhana yang sering diabaikan tapi efektif:

- **Clip negatif**: `pred = max(0, pred)` — PV tidak bisa negatif
- **Night zeroing**: Jika jam 20:00-05:00, paksa prediksi = 0
- **Capacity clipping**: `pred = min(pred, 15.93)` — tidak bisa melebihi kapasitas panel
- **Smoothing**: Moving average 3-step pada output

**Contoh implementasi:**
```python
pred = np.clip(pred, 0, capacity_kw)           # Physical bounds
pred[hour_mask_night] = 0                        # Night zeroing
pred = pd.Series(pred).rolling(3, center=True).mean().values  # Smooth
```

---

### 10. Validation Strategy — Time Series CV

**Expected improvement: Better generalization (tidak langsung improve R², tapi mengurangi overfitting)**

**Status saat ini:** `tscv: false`, simple train/test split.

Time Series Cross-Validation:
```
Fold 1: [Train: 2020-2021] [Val: 2022]
Fold 2: [Train: 2020-2022] [Val: 2023]
Fold 3: [Train: 2020-2023] [Val: 2024]
```

Memberikan estimasi performa yang lebih reliable dan mengurangi risiko
overfitting pada 1 split tertentu.

**Cara menerapkan:**
```yaml
tscv:
  enabled: true
  n_splits: 3
```

---

## Priority Matrix

| # | Strategi | Effort | Impact | Priority |
|:---:|---|:---:|:---:|:---:|
| 1 | **Lags + Rolling ON** | 🟢 Low (config) | 🔴 High | ⭐⭐⭐⭐⭐ |
| 2 | **CSI Target** | 🟢 Low (config) | 🟠 Medium-High | ⭐⭐⭐⭐ |
| 3 | **Huber Loss** | 🟡 Medium (code) | 🟠 Medium | ⭐⭐⭐⭐ |
| 4 | **StandardScaler** | 🟢 Low (config) | 🟡 Medium | ⭐⭐⭐ |
| 5 | **Optuna Tuning** | 🟢 Low (config) | 🟡 Medium | ⭐⭐⭐ |
| 6 | **Post-processing** | 🟡 Medium (code) | 🟡 Medium | ⭐⭐⭐ |
| 7 | **TSCV** | 🟢 Low (config) | 🟡 Medium | ⭐⭐⭐ |
| 8 | **Ensemble** | 🔴 High (infra) | 🟠 Medium-High | ⭐⭐ |
| 9 | **Data Augmentation** | 🔴 High (code) | 🟡 Medium | ⭐⭐ |
| 10 | **Lookback tuning** | 🟢 Low (config) | 🟢 Low | ⭐⭐ |

---

## Rekomendasi Langkah Pertama

1. Toggle `lags: true`, `rolling: true`, `physics: true` di config
2. Jalankan Preprocessing → Train
3. Bandingkan hasil dengan model sebelumnya menggunakan tab Model Comparison

Ini adalah **zero-code change** (hanya config) dengan expected improvement terbesar.

---

## Catatan Penting

### Val_loss vs Metrik Error

Val_loss yang rendah tidak selalu menggambarkan hasil metrik error karena:

1. **Loss dihitung pada data scaled** (0-1), **metrik pada data asli** (kW)
2. **Loss = rata-rata semua 24 timestep** — step awal mendominasi
3. **Validation split ≠ Test split** — pola cuaca berbeda
4. **MSE vs MAE/MAPE** — mengoptimasi hal yang sedikit berbeda

> **Rule of thumb:** Gunakan val_loss untuk membandingkan **epoch dalam 1 training run**.
> Gunakan R², MAE, RMSE pada data de-normalized untuk membandingkan **antar model/dataset**.

### Korelasi Negatif

Fitur dengan korelasi negatif **sama berharganya** dengan positif. Sistem threshold
korelasi di pipeline sudah menggunakan `.abs()` sehingga fitur negatif kuat
(seperti humidity -0.68) tetap terpilih.
