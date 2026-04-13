"""
NASA POWER Data Acquisition – NTT Site
=======================================
Menarik data meteorologi historis 10 tahun (2014–2024) dari NASA POWER API
pada koordinat lokasi studi NTT untuk keperluan skripsi.

Lokasi  : -10.297625°S, 123.59123°E  (NTT, Indonesia)
Timezone: UTC+8 (WITA – Waktu Indonesia Tengah)
Fitur   : DHI (W/m²), DNI (W/m²), T2M (°C)
Output  : data/raw/ntt_historical_10y.csv

Cara pakai:
    python scripts/fetch_nasa_power_ntt.py

Requirements: requests, pandas, numpy
"""

import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Konfigurasi ─────────────────────────────────────────────────────────────
LAT = -10.297625
LON = 123.59123
START_YEAR = 2014
END_YEAR   = 2024
UTC_OFFSET = 8          # WITA = UTC+8

# Parameter NASA POWER (RE community, hourly)
#   ALLSKY_SFC_SW_DIFF → DHI  (W/m²)  ← "DIFF" bukan "DIF" untuk hourly endpoint
#   ALLSKY_SFC_SW_DNI  → DNI  (W/m²)
#   T2M                → Suhu udara 2 m (°C)
NASA_PARAMS  = "ALLSKY_SFC_SW_DIFF,ALLSKY_SFC_SW_DNI,T2M"
FILL_VALUE   = -999.0           # nilai sentinel NASA POWER untuk data kosong
BASE_URL     = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# ── Helper functions ─────────────────────────────────────────────────────────

def fetch_year(year: int, max_retries: int = 3) -> dict:
    """Unduh satu tahun data hourly dari NASA POWER API (JSON)."""
    url = (
        f"{BASE_URL}"
        f"?parameters={NASA_PARAMS}"
        f"&community=RE"
        f"&longitude={LON}"
        f"&latitude={LAT}"
        f"&start={year}0101"
        f"&end={year}1231"
        f"&format=JSON"
    )
    headers = {"Accept": "application/json"}

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [{year}] Fetching ... (attempt {attempt}/{max_retries})")
            r = requests.get(url, headers=headers, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            print(f"  [{year}] Timeout. Menunggu {10 * attempt}s ...")
            time.sleep(10 * attempt)
        except requests.exceptions.HTTPError as e:
            print(f"  [{year}] HTTP error: {e}")
            time.sleep(5 * attempt)
        except Exception as e:
            print(f"  [{year}] Error: {e}")
            time.sleep(5 * attempt)

    raise RuntimeError(f"Gagal mengambil data untuk tahun {year} setelah {max_retries} percobaan.")


def parse_response(data: dict) -> pd.DataFrame:
    """
    Parse JSON response NASA POWER menjadi DataFrame.

    Format key timestamp dari API: "YYYYMMDD HHMM" (UTC)
    Kolom output: timestamp_utc, dhi_wm2, dni_wm2, ambient_temp_c
    """
    try:
        params = data["properties"]["parameter"]
    except KeyError:
        # Beberapa versi API menggunakan struktur alternatif
        params = data["features"][0]["properties"]["parameter"]

    raw_dhi = params["ALLSKY_SFC_SW_DIFF"]
    raw_dni = params["ALLSKY_SFC_SW_DNI"]
    raw_t2m = params["T2M"]

    # Semua key harus identik; gunakan key dari DHI sebagai acuan
    # Format key API: "YYYYMMDDHH"  (contoh: "2020010100" = 2020-01-01 00:00 UTC)
    keys = list(raw_dhi.keys())

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(keys, format="%Y%m%d%H"),
        "dhi_wm2"       : [raw_dhi[k] for k in keys],
        "dni_wm2"       : [raw_dni[k] for k in keys],
        "ambient_temp_c": [raw_t2m[k] for k in keys],
    })

    # Ganti fill value dengan NaN
    df.replace(FILL_VALUE, np.nan, inplace=True)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan data: konversi ke waktu lokal, klip nilai negatif,
    interpolasi gap kecil, dan reset index.
    """
    # NASA Power Hourly data for RE community appears to be in LST or 
    # already synchronized with the local sun for specific point queries.
    # No manual shift needed to avoid double-offsetting.
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Klip radiasi negatif ke 0 (kadang muncul dari fill value yang diinterpolasi)
    df["dhi_wm2"] = df["dhi_wm2"].clip(lower=0)
    df["dni_wm2"] = df["dni_wm2"].clip(lower=0)

    # Interpolasi gap kecil (≤ 3 jam berturut-turut)
    for col in ["dhi_wm2", "dni_wm2", "ambient_temp_c"]:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            print(f"  Interpolasi {n_missing} NaN di kolom '{col}' ...")
            df[col] = df[col].interpolate(method="linear", limit=3)
        remaining = df[col].isna().sum()
        if remaining > 0:
            print(f"  PERINGATAN: Masih ada {remaining} NaN di '{col}' setelah interpolasi "
                  f"(gap > 3 jam). Akan diisi dengan forward-fill.")
            df[col] = df[col].ffill().bfill()

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Tentukan path output relatif terhadap lokasi script ini
    script_dir = Path(__file__).resolve().parent
    out_path   = script_dir.parent / "data" / "raw" / "ntt_historical_10y.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NASA POWER Data Acquisition – NTT Site")
    print(f"  Koordinat : {LAT}°S, {LON}°E")
    print(f"  Periode   : {START_YEAR}–{END_YEAR}  ({END_YEAR - START_YEAR + 1} tahun)")
    print(f"  Output    : {out_path}")
    print("=" * 60)

    all_dfs = []
    for year in range(START_YEAR, END_YEAR + 1):
        try:
            raw = fetch_year(year)
            df  = parse_response(raw)
            all_dfs.append(df)
            print(f"  [{year}] OK  – {len(df):,} baris diterima")
        except RuntimeError as e:
            print(f"  [{year}] SKIP karena error: {e}", file=sys.stderr)

        # Jeda sopan antar request (~2 detik) agar tidak membebani server
        time.sleep(2)

    if not all_dfs:
        print("Tidak ada data yang berhasil diambil. Periksa koneksi internet.", file=sys.stderr)
        sys.exit(1)

    print("\nMenggabungkan dan membersihkan data ...")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = clean_data(combined)

    # Urutkan kolom output
    combined = combined[["timestamp", "dhi_wm2", "dni_wm2", "ambient_temp_c"]]

    # Simpan
    combined.to_csv(out_path, index=False, sep=";",
                    date_format="%d/%m/%Y %H:%M")

    # ── Ringkasan ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SELESAI – Ringkasan Data")
    print("=" * 60)
    print(f"  Total baris    : {len(combined):,}")
    print(f"  Periode lokal  : {combined['timestamp'].min()} s/d {combined['timestamp'].max()}")
    print(f"  NaN tersisa    : {combined.isna().sum().sum()}")
    print(f"\n  Statistik:\n{combined.describe().round(2)}")
    print(f"\n  Disimpan ke: {out_path}")


if __name__ == "__main__":
    main()
