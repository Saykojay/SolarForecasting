"""
Typical Meteorological Year (TMY) Synthesis - NTT 2027
=======================================================
Memproses data historis 10 tahun NASA POWER menjadi profil TMY (8.760 jam)
berdasarkan metode Finkelstein-Schafer (FS) / Sandia yang dimodifikasi.

Input   : data/raw/ntt_historical_10y.csv
Output  : data/raw/ntt_tmy_2027.csv

Fitur output (sesuai kebutuhan model GRU skripsi):
    timestamp, dhi_wm2, dni_wm2, ambient_temp_c, hour_sin, hour_cos

Metode TMY (Finkelstein-Schafer / ISO 15927-4):
    Untuk setiap bulan (Jan-Des):
    1. Hitung long-term empirical CDF dari statistik harian (mean DHI,
       mean DNI, mean T2M) menggunakan SEMUA tahun yang tersedia.
    2. Untuk setiap tahun individual, hitung FS statistic:
         FS = (1/N) × Σ |F_longterm(x_i) - F_year(x_i)|
    3. Gabungkan FS dari ketiga variabel dengan bobot:
         DNI 40%, DHI 40%, T2M 20%
    4. Pilih tahun dengan WS (weighted FS) terkecil sebagai bulan "tipikal".
    5. Sambung ke-12 bulan terpilih -> profil 8.760 jam.
    6. Haluskan transisi antar-bulan (linear interpolasi 6 jam).
    7. Tambahkan hour_sin / hour_cos dan beri timestamp 2027.

Cara pakai:
    python scripts/build_tmy_ntt.py

Requirements: pandas, numpy, scipy
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Konfigurasi ─────────────────────────────────────────────────────────────
TMY_YEAR          = 2027        # Tahun yang akan di-assign ke profil TMY
TRANSITION_HOURS  = 6           # Panjang jendela smoothing antar-bulan (jam)

# Bobot FS per variabel (harus berjumlah 1.0)
WEIGHTS = {
    "dhi_mean": 0.40,   # DHI harian rata-rata
    "dni_mean": 0.40,   # DNI harian rata-rata
    "t2m_mean": 0.20,   # T2M harian rata-rata
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","Mei","Jun",
               "Jul","Agu","Sep","Okt","Nov","Des"]

# ── Helper: Empirical CDF ────────────────────────────────────────────────────

def empirical_cdf(values: np.ndarray):
    """
    Kembalikan fungsi F(x) = fraksi nilai ≤ x dari array 'values'.
    Menggunakan interpolasi linear antara titik-titik terurut.
    """
    sorted_v = np.sort(values[~np.isnan(values)])
    n        = len(sorted_v)
    probs    = np.arange(1, n + 1) / n      # Weibull plotting position

    def F(x):
        # np.searchsorted memberi posisi; interpolasi untuk nilai tepat
        return np.interp(x, sorted_v, probs, left=0.0, right=1.0)

    return F


def fs_statistic(year_values: np.ndarray, lt_cdf_func) -> float:
    """
    Hitung FS statistic antara distribusi satu tahun dengan
    long-term CDF.

    FS = (1/N) × Σ |F_lt(x_i) - F_year(x_i)|
    """
    v_sorted = np.sort(year_values[~np.isnan(year_values)])
    n        = len(v_sorted)
    if n == 0:
        return np.inf
    f_year = np.arange(1, n + 1) / n          # CDF tahun individual
    f_lt   = lt_cdf_func(v_sorted)             # CDF long-term pada titik yang sama
    return float(np.mean(np.abs(f_lt - f_year)))


# ── Langkah 1: Load Data ─────────────────────────────────────────────────────

def load_historical(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", parse_dates=["timestamp"],
                     dayfirst=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Pastikan tidak ada nilai negatif
    df["dhi_wm2"] = df["dhi_wm2"].clip(lower=0)
    df["dni_wm2"] = df["dni_wm2"].clip(lower=0)

    # Tambah kolom bantu
    df["year"]       = df["timestamp"].dt.year
    df["month"]      = df["timestamp"].dt.month
    df["date"]       = df["timestamp"].dt.date
    df["hour"]       = df["timestamp"].dt.hour

    return df


# ── Langkah 2: TMY Month Selection ───────────────────────────────────────────

def select_typical_year_for_month(df_month_all: pd.DataFrame) -> int:
    """
    Dari semua data untuk satu bulan tertentu (semua tahun),
    pilih tahun yang paling 'tipikal' menggunakan metode FS.

    Kembalikan integer tahun yang dipilih.
    """
    # Hitung statistik harian
    daily = (
        df_month_all
        .groupby(["year", "date"])
        .agg(
            dhi_mean=("dhi_wm2", "mean"),
            dni_mean=("dni_wm2", "mean"),
            t2m_mean=("ambient_temp_c", "mean"),
        )
        .reset_index()
    )
    daily["year"] = pd.to_datetime(daily["date"].astype(str)).dt.year

    years = sorted(daily["year"].unique())

    if len(years) < 2:
        return years[0]   # hanya 1 tahun, langsung pilih

    # Long-term CDF untuk setiap variabel
    lt_cdfs = {
        var: empirical_cdf(daily[var].values)
        for var in WEIGHTS
    }

    # Hitung weighted FS untuk setiap tahun
    ws_scores = {}
    for yr in years:
        dy = daily[daily["year"] == yr]
        total_ws = 0.0
        for var, weight in WEIGHTS.items():
            fs = fs_statistic(dy[var].values, lt_cdfs[var])
            total_ws += weight * fs
        ws_scores[yr] = total_ws

    best_year = min(ws_scores, key=ws_scores.get)
    return best_year, ws_scores


# ── Langkah 3: Rakit Profil TMY ──────────────────────────────────────────────

def assemble_tmy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pilih 1 tahun tipikal per bulan, lalu sambung menjadi
    satu DataFrame 8.760 baris (non-leap year 2027).
    """
    segments   = []
    selection_log = []

    for m in range(1, 13):
        df_m = df[df["month"] == m].copy()

        best_year, ws_scores = select_typical_year_for_month(df_m)

        # Ambil data jam dari tahun terpilih untuk bulan ini
        segment = df_m[df_m["year"] == best_year][
            ["hour", "dhi_wm2", "dni_wm2", "ambient_temp_c", "timestamp"]
        ].copy()

        # Pastikan terurut
        segment.sort_values("timestamp", inplace=True)

        segments.append(segment)
        selection_log.append({
            "month"    : MONTH_NAMES[m - 1],
            "best_year": best_year,
            "ws_score" : round(ws_scores[best_year], 5),
            "n_hours"  : len(segment),
        })
        print(f"  Bulan {MONTH_NAMES[m-1]:3s}: tahun {best_year} dipilih "
              f"(WS={ws_scores[best_year]:.4f}, {len(segment)} jam)")

    return segments, selection_log


# ── Langkah 4: Assign Timestamp 2027 ─────────────────────────────────────────

def build_2027_index() -> pd.DatetimeIndex:
    """Buat index hourly untuk tahun 2027 (bukan leap year -> 8.760 jam)."""
    idx = pd.date_range(
        start=f"{TMY_YEAR}-01-01 00:00",
        end  =f"{TMY_YEAR}-12-31 23:00",
        freq ="h",
    )
    assert len(idx) == 8760, f"Expected 8760 hours, got {len(idx)}"
    return idx


def reassign_timestamps(segments: list, idx_2027: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Sambung semua segmen bulan, potong/pad ke tepat 8.760 jam,
    lalu assign timestamp 2027.

    Pendekatan: mapping berdasarkan urutan jam (month x day-of-month x hour)
    sehingga mismatch jumlah baris akibat shift timezone tidak menimbulkan NaN.
    """
    target = 8760

    # Bangun lookup: untuk setiap (month, hour_in_month) -> nilai meteorologi
    # dari segmen yang dipilih, lalu petakan ke idx_2027
    # Cara robust: ambil data per bulan dari segmen,
    # buat profil "template" berdasarkan (day_of_month, hour),
    # lalu isi 2027 jam per jam
    combined_raw = pd.concat(segments, ignore_index=True)
    combined_raw = combined_raw[["timestamp", "dhi_wm2", "dni_wm2",
                                  "ambient_temp_c"]].copy()
    combined_raw["month"]        = combined_raw["timestamp"].dt.month
    combined_raw["day_of_month"] = combined_raw["timestamp"].dt.day
    combined_raw["hour"]         = combined_raw["timestamp"].dt.hour

    result_rows = []
    for ts in idx_2027:
        m   = ts.month
        dom = ts.day
        h   = ts.hour
        # Cari baris dengan bulan, hari, jam yang sama di segmen
        mask = (
            (combined_raw["month"]        == m) &
            (combined_raw["day_of_month"] == dom) &
            (combined_raw["hour"]         == h)
        )
        match = combined_raw[mask]
        if not match.empty:
            row = match.iloc[0]
            result_rows.append({
                "dhi_wm2"       : row["dhi_wm2"],
                "dni_wm2"       : row["dni_wm2"],
                "ambient_temp_c": row["ambient_temp_c"],
            })
        else:
            # Fallback: ambil rata-rata jam tersebut pada bulan yang sama
            mask_fallback = (
                (combined_raw["month"] == m) &
                (combined_raw["hour"]  == h)
            )
            fb = combined_raw[mask_fallback]
            if not fb.empty:
                result_rows.append({
                    "dhi_wm2"       : fb["dhi_wm2"].mean(),
                    "dni_wm2"       : fb["dni_wm2"].mean(),
                    "ambient_temp_c": fb["ambient_temp_c"].mean(),
                })
            else:
                # Sangat jarang: isi dengan 0 / rata-rata suhu global
                result_rows.append({
                    "dhi_wm2"       : 0.0,
                    "dni_wm2"       : 0.0,
                    "ambient_temp_c": combined_raw["ambient_temp_c"].mean(),
                })

    combined = pd.DataFrame(result_rows)
    assert len(combined) == target, f"Baris hasil: {len(combined)} (expected {target})"

    combined.index      = idx_2027
    combined.index.name = "timestamp"
    combined.reset_index(inplace=True)

    return combined


# ── Langkah 5: Smoothing Transisi Antar-Bulan ────────────────────────────────

def smooth_transitions(df: pd.DataFrame, window: int = TRANSITION_HOURS) -> pd.DataFrame:
    """
    Di setiap pergantian bulan, interpolasi linear selama 'window' jam
    untuk menghindari loncatan tajam pada profil.
    """
    if window <= 0:
        return df

    df = df.reset_index(drop=True)   # pastikan index integer 0..8759

    # Temukan posisi baris (integer) pertama setiap bulan
    months      = df["timestamp"].dt.month
    month_start_positions = [
        int(months[months == m].index[0]) for m in range(1, 13)
    ]

    for col in ["dhi_wm2", "dni_wm2", "ambient_temp_c"]:
        for start_pos in month_start_positions[1:]:   # skip bulan Jan
            lo = max(0, start_pos - window)
            hi = min(len(df) - 1, start_pos + window)
            # Interpolasi linear antara nilai di lo dan hi
            val_lo = df.at[lo, col]
            val_hi = df.at[hi, col]
            xs = np.arange(lo, hi + 1)
            df.loc[lo:hi, col] = np.interp(xs, [lo, hi], [val_lo, val_hi])

    return df


# ── Langkah 6: Tambah Fitur Cyclical ─────────────────────────────────────────

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan hour_sin dan hour_cos berdasarkan jam lokal.
    hour_sin = sin(2π × hour / 24)
    hour_cos = cos(2π × hour / 24)
    """
    hour         = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).round(8)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).round(8)
    return df


# ── Langkah 7: Validasi Akhir ─────────────────────────────────────────────────

def validate_tmy(df: pd.DataFrame) -> None:
    """Pastikan output memenuhi syarat untuk input model GRU."""
    assert len(df) == 8760, f"Baris tidak tepat 8.760: {len(df)}"
    assert df["timestamp"].dt.year.nunique() == 1, "Tahun tidak seragam"
    assert df["timestamp"].dt.year.iloc[0]   == TMY_YEAR, f"Bukan tahun {TMY_YEAR}"
    assert df["dhi_wm2"].isna().sum()        == 0, "Masih ada NaN di dhi_wm2"
    assert df["dni_wm2"].isna().sum()        == 0, "Masih ada NaN di dni_wm2"
    assert df["ambient_temp_c"].isna().sum() == 0, "Masih ada NaN di ambient_temp_c"
    assert df["hour_sin"].isna().sum()       == 0, "Masih ada NaN di hour_sin"
    assert df["hour_cos"].isna().sum()       == 0, "Masih ada NaN di hour_cos"
    assert (df["dhi_wm2"] >= 0).all(),  "Nilai DHI negatif ditemukan"
    assert (df["dni_wm2"] >= 0).all(),  "Nilai DNI negatif ditemukan"
    print("  [OK] Validasi lulus: 8.760 baris, tidak ada NaN, nilai valid.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent
    in_path    = script_dir.parent / "data" / "raw" / "ntt_historical_10y.csv"
    out_path   = script_dir.parent / "data" / "raw" / "ntt_tmy_2027.csv"

    print("=" * 60)
    print(f"TMY Synthesis - NTT -> {TMY_YEAR}")
    print(f"  Input  : {in_path}")
    print(f"  Output : {out_path}")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────────────
    if not in_path.exists():
        raise FileNotFoundError(
            f"File historis tidak ditemukan: {in_path}\n"
            "Jalankan terlebih dahulu: python scripts/fetch_nasa_power_ntt.py"
        )
    print("\n[1/6] Memuat data historis ...")
    df = load_historical(in_path)
    years_available = sorted(df["year"].unique())
    print(f"  Baris loaded   : {len(df):,}")
    print(f"  Tahun tersedia : {years_available}")

    # ── 2. Pilih bulan tipikal ────────────────────────────────────────────
    print("\n[2/6] Memilih bulan tipikal (metode Finkelstein-Schafer) ...")
    segments, log = assemble_tmy(df)

    print("\n  Rekapitulasi pemilihan:")
    print(f"  {'Bulan':>5} | {'Tahun Dipilih':>13} | {'WS Score':>9} | {'Jam':>5}")
    print("  " + "-" * 42)
    for entry in log:
        print(f"  {entry['month']:>5} | {entry['best_year']:>13} | {entry['ws_score']:>9.5f} | {entry['n_hours']:>5}")

    # ── 3. Rakit & Assign timestamp 2027 ─────────────────────────────────
    print(f"\n[3/6] Merakit profil {TMY_YEAR} (8.760 jam) ...")
    idx_2027 = build_2027_index()
    tmy      = reassign_timestamps(segments, idx_2027)
    print(f"  Jumlah baris: {len(tmy)}")

    # ── 4. Smoothing transisi ─────────────────────────────────────────────
    print(f"\n[4/6] Menghaluskan transisi antar-bulan ({TRANSITION_HOURS} jam) ...")
    tmy = smooth_transitions(tmy, window=TRANSITION_HOURS)

    # ── 5. Fitur cyclical ─────────────────────────────────────────────────
    print("\n[5/6] Menambahkan hour_sin dan hour_cos ...")
    tmy = add_cyclical_features(tmy)

    # ── 6. Validasi & Simpan ──────────────────────────────────────────────
    print("\n[6/6] Validasi dan menyimpan ...")
    validate_tmy(tmy)

    # Urutkan kolom sesuai kebutuhan pipeline
    col_order = ["timestamp", "dhi_wm2", "dni_wm2", "ambient_temp_c",
                 "hour_sin", "hour_cos"]
    tmy = tmy[col_order]

    tmy.to_csv(out_path, index=False, sep=";",
               date_format="%d/%m/%Y %H:%M")

    # ── Ringkasan ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SELESAI - Ringkasan TMY")
    print("=" * 60)
    print(f"  Total baris    : {len(tmy):,}")
    print(f"  Periode        : {tmy['timestamp'].iloc[0]}  s/d  {tmy['timestamp'].iloc[-1]}")
    print(f"\n  Statistik:\n{tmy.describe().round(3)}")

    # Cek distribusi: berapa jam PV aktif (DNI > 0 atau DHI > 10)
    n_siang = (tmy["dhi_wm2"] > 10).sum()
    print(f"\n  Jam siang (DHI > 10 W/m²) : {n_siang} / 8.760  ({n_siang/8760*100:.1f}%)")

    print(f"\n  Disimpan ke: {out_path}")
    print("\nData TMY 2027 siap digunakan sebagai input prediksi model GRU.")
    print("Selanjutnya: muat ntt_tmy_2027.csv ke pipeline Streamlit")
    print("  -> tab 'Target Testing' atau 'Prediction / Eval'")
    print("  -> pilih model Final_GRU_FineTuned")


if __name__ == "__main__":
    main()
