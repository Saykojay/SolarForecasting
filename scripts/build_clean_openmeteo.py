"""
Best-of dataset v2:
- GHI, DHI, temp, humidity, wind, precip: FROM NASA (better aligned, r=0.936)
- Cloud cover (4 levels), DNI: FROM Open-Meteo (exclusive features, no shift needed)
- DNI shifted by -2h since it's an OM-exclusive feature with same offset
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import requests, time as tm

lat, lon = 1.3620, 103.8626

# 1. Download fresh Open-Meteo
print("Downloading fresh Open-Meteo data...")
variables = 'shortwave_radiation,direct_radiation,diffuse_radiation,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high'
all_dfs = []
for start, end in [('2020-01-01','2021-12-31'),('2022-01-01','2023-12-31'),('2024-01-01','2025-11-29')]:
    r = requests.get('https://archive-api.open-meteo.com/v1/archive',
        params={'latitude':lat,'longitude':lon,'start_date':start,'end_date':end,
                'hourly':variables,'timezone':'Asia/Singapore'}, timeout=60)
    chunk = pd.DataFrame(r.json()['hourly'])
    all_dfs.append(chunk)
    print(f"  {start}-{end}: {len(chunk)} rows")
    tm.sleep(1)

df_om = pd.concat(all_dfs, ignore_index=True)
df_om['time'] = pd.to_datetime(df_om['time'])
df_om = df_om.sort_values('time').drop_duplicates(subset='time').reset_index(drop=True)

# Apply -2h shift to OM data
df_om['time'] = df_om['time'] - pd.Timedelta(hours=2)

# 2. Load NASA+PV data (source of truth for GHI, temp, etc.)
df_nasa = pd.read_csv('data/raw/singapore_pv_weather_2020_2025.csv', sep=';')
df_nasa['timestamp'] = pd.to_datetime(df_nasa['timestamp'], dayfirst=True)

# 3. Prepare OM exclusive features only (rename to merge)
df_om_excl = df_om[['time', 'direct_radiation', 'cloud_cover', 
                     'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high']].rename(columns={
    'time': 'timestamp',
    'direct_radiation': 'dni_wm2',
    'cloud_cover': 'cloud_cover_pct',
    'cloud_cover_low': 'cloud_cover_low_pct',
    'cloud_cover_mid': 'cloud_cover_mid_pct',
    'cloud_cover_high': 'cloud_cover_high_pct',
})

# 4. Merge: NASA base + OM exclusives
df_final = pd.merge(df_nasa, df_om_excl, on='timestamp', how='inner')

# 5. Verify
print(f"\nShape: {df_final.shape}")
print(f"Columns: {df_final.columns.tolist()}")
print(f"Missing PV: {df_final['pv_output_kw'].isna().sum()}")

# Verify Jan 5 2020
print("\n=== Jan 5, 2020 ===")
mask = df_final['timestamp'].dt.date == pd.Timestamp('2020-01-05').date()
cols = ['timestamp','pv_output_kw','ghi_wm2','dhi_wm2','dni_wm2','cloud_cover_pct']
sample = df_final[mask][cols].copy()
sample['timestamp'] = sample['timestamp'].dt.strftime('%H:%M')
print(sample.to_string(index=False))

# Correlations
print("\nCorrelations:")
pv_valid = df_final.dropna(subset=['pv_output_kw'])
pv = pv_valid['pv_output_kw']
for col in ['ghi_wm2','dhi_wm2','dni_wm2','ambient_temp_c','relative_humidity_pct',
            'cloud_cover_pct','cloud_cover_low_pct','cloud_cover_mid_pct','cloud_cover_high_pct','wind_speed_ms']:
    c = pv_valid[col].corr(pv)
    print(f"  {col:28s}: {c:+.4f}")

# Save
out = 'data/raw/singapore_pv_bestof_v2_2020_2025.csv'
df_final.to_csv(out, sep=';', index=False, date_format='%d/%m/%Y %H:%M')
print(f"\nSaved: {out} ({len(df_final)} rows, {len(df_final.columns)} cols)")
