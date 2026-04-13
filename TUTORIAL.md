# Tutorial: Solar PV Forecasting Pipeline (v1)

This tutorial guides you through the complete lifecycle of solar forecasting, from raw weather data acquisition to generating industrial-grade production profiles.

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Step 0: Data Acquisition](#step-0-data-acquisition)
3. [Step 1: Typical Meteorological Year (TMY) Synthesis](#step-1-tmy-synthesis)
4. [Step 2: Training the Forecasting Model](#step-2-training)
5. [Step 3: Downstream Integration (HOMER Pro)](#step-3-downstream)
6. [Web Dashboard Guide](#web-dashboard-guide)

---

## 1. Prerequisites

### Environment Setup
```bash
conda create -n solar python=3.10
conda activate solar
pip install -r requirements.txt
```

---

## 2. Step 0: Data Acquisition

Before training, you need historical weather data. We provide scripts to fetch data from **NASA POWER** and **Open-Meteo**.

### Using NASA POWER (Long-term Historical)
NASA data is excellent for capturing 10-20 year trends (TMY basis).
```bash
python scripts/fetch_nasa_power_ntt.py
```
*   **Input**: Coordinates (lat/lon) defined in the script.
*   **Output**: `data/raw/nasa_power_ntt.csv`.

### Using Open-Meteo (High Resolution/Local)
Open-Meteo provides higher resolution (hourly) and recent data.
```bash
python scripts/build_clean_openmeteo.py
```

---

## 3. Step 1: Typical Meteorological Year (TMY) Synthesis

In solar research, we model representative years to avoid transient anomalies. 

### Generating a TMY Profile
```bash
python scripts/build_tmy_ntt.py
```
This script:
1.  Analyzes 10 years of NASA data.
2.  Selects the most "typical" months (P50/P90).
3.  Synthesizes a full 8,760-hour typical year.
4.  Outputs `data/raw/TMY_NTT_2027_Projected.csv`.

### Correcting Solar Noon (Physical Alignment)
If your sensors have a clock shift (very common), run the correction script:
```bash
python scripts/fix_tmy_shift.py
```

---

## 4. Step 2: Training the Forecasting Model

Once you have your TMY or historical CSV, launch the dashboard to train.

### Launching the Web Interface
```bash
streamlit run app.py
```

### Key Workflow in Dashboard:
1.  **Tab 1: Preprocessing**: Load your TMY CSV. Apply **Algorithm 1**. This cleans outliers where GHI > 0 but PV = 0.
2.  **Tab 3: Training**: Select **GRU** (Recommended). Set lookback to 24 (hours). 
3.  **Tab 5: Batch**: Run multiple experiments (PatchTST vs GRU) to find the best metrics for your specific site.

---

## 5. Step 3: Downstream Integration (HOMER Pro)

The end goal for most researchers is sizing the system. 

1.  **Train your Champion Model** (e.g., GRU on TMY 2027).
2.  **Run Inference**: Generate a full year forecast.
3.  **Export**: The pipeline saves a `normalized_power.csv`.
4.  **HOMER**: In HOMER Pro, import this as a `Custom Production Profile`. This is your Stage II input.

---

## 📝 Key Features Detail

### Algorithm 1: Physics-Based Cleaning
Located in `src/data_prep.py`. It doesn't just look at statistics; it looks at **Physics**:
-   Clamps values to $0$.
-   Ensures $GHI \le ClearSkyGHI$.
-   Removes nighttime noise (GHI < 20 W/m²).

---
*Questions? Reach out via the repository issues.*
