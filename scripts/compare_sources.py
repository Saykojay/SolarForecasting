import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
Head-to-head analysis: NASA POWER vs Open-Meteo
For each matching feature, pick the source with highest PV correlation.
Then create a 'best-of' merged dataset.
"""
import pandas as pd
import numpy as np

# Load merged dataset
df = pd.read_csv('data/raw/singapore_pv_weather_openmeteo_2020_2025.csv', sep=';')
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
df = df.dropna(subset=['pv_output_kw'])

pv = df['pv_output_kw']

# ============================================================
# HEAD-TO-HEAD: Same features, different sources
# ============================================================
matchups = [
    ('GHI (W/m²)',       'ghi_wm2',              'om_ghi_wm2'),
    ('DHI (W/m²)',       'dhi_wm2',              'om_dhi_wm2'),
    ('Temperature (°C)', 'ambient_temp_c',       'om_temp_c'),
    ('Humidity (%)',      'relative_humidity_pct', 'om_humidity_pct'),
    ('Wind Speed (m/s)', 'wind_speed_ms',         'om_wind_ms'),
    ('Wind Dir (°)',     'wind_direction_deg',    'om_wind_dir_deg'),
    ('Precipitation',    'precipitation_mm',      'om_precip_mm'),
]

print("=" * 80)
print("HEAD-TO-HEAD: NASA POWER vs Open-Meteo")
print("=" * 80)
print(f"{'Feature':<22} | {'NASA (corr)':>12} | {'Open-Meteo (corr)':>18} | {'Winner':>12} | {'Diff':>8}")
print("-" * 80)

winners = {}  # feature_name -> winning_col
for name, nasa_col, om_col in matchups:
    corr_nasa = abs(pv.corr(df[nasa_col]))
    corr_om = abs(pv.corr(df[om_col]))
    
    if corr_nasa >= corr_om:
        winner = "NASA"
        win_col = nasa_col
    else:
        winner = "Open-Meteo"
        win_col = om_col
    
    diff = abs(corr_nasa - corr_om)
    winners[name] = {'col': win_col, 'corr': max(corr_nasa, corr_om), 'source': winner}
    
    flag = "✅" if winner == "NASA" else "🌐"
    print(f"{name:<22} | {corr_nasa:12.4f} | {corr_om:18.4f} | {flag} {winner:<8} | {diff:8.4f}")

# ============================================================
# OPEN-METEO EXCLUSIVE FEATURES (no NASA equivalent)
# ============================================================
print("\n" + "=" * 80)
print("OPEN-METEO EXCLUSIVE FEATURES (tidak ada di NASA POWER)")
print("=" * 80)

exclusive_cols = [
    ('DNI (W/m²)',          'om_dni_wm2'),
    ('Cloud Cover Total',   'cloud_cover_pct'),
    ('Cloud Cover Low',     'cloud_cover_low_pct'),
    ('Cloud Cover Mid',     'cloud_cover_mid_pct'),
    ('Cloud Cover High',    'cloud_cover_high_pct'),
]

for name, col in exclusive_cols:
    corr = pv.corr(df[col])
    print(f"  {name:<25}: corr = {corr:+.4f}")

# ============================================================
# PRODUCTIVE HOURS ANALYSIS (GHI > 50 only)
# ============================================================
print("\n" + "=" * 80)
print("PRODUCTIVE HOURS ONLY (GHI > 50)")
print("=" * 80)
mask_prod = df['ghi_wm2'] > 50
df_prod = df[mask_prod]
pv_prod = df_prod['pv_output_kw']

print(f"{'Feature':<22} | {'NASA (corr)':>12} | {'Open-Meteo (corr)':>18} | {'Winner':>12}")
print("-" * 80)

best_of_cols = {}  # For building the best-of dataset
for name, nasa_col, om_col in matchups:
    corr_nasa = abs(pv_prod.corr(df_prod[nasa_col]))
    corr_om = abs(pv_prod.corr(df_prod[om_col]))
    
    if corr_nasa >= corr_om:
        winner = "NASA"
        best_of_cols[name] = nasa_col
    else:
        winner = "Open-Meteo"
        best_of_cols[name] = om_col
    
    flag = "✅" if winner == "NASA" else "🌐"
    print(f"{name:<22} | {corr_nasa:12.4f} | {corr_om:18.4f} | {flag} {winner:<8}")

# Cloud cover in productive hours
print("\nCloud features (productive hours):")
for name, col in exclusive_cols:
    corr = pv_prod.corr(df_prod[col])
    print(f"  {name:<25}: corr = {corr:+.4f}")

# ============================================================
# BUILD BEST-OF DATASET
# ============================================================
print("\n" + "=" * 80)
print("BUILDING BEST-OF DATASET")
print("=" * 80)

# Standard column mapping for the pipeline (use winners)
col_remap = {}
for name, nasa_col, om_col in matchups:
    # The pipeline expects these standard column names
    standard_name = nasa_col  # Use NASA column names as standard
    winning_col = best_of_cols[name]
    col_remap[standard_name] = winning_col
    if winning_col != standard_name:
        print(f"  {standard_name:25s} <- {winning_col} (Open-Meteo wins)")
    else:
        print(f"  {standard_name:25s} <- {winning_col} (NASA wins)")

# Build dataset
df_best = pd.DataFrame()
df_best['timestamp'] = df['timestamp']
df_best['pv_output_kw'] = df['pv_output_kw']

# Add winning features with standard names
for std_name, winning_col in col_remap.items():
    df_best[std_name] = df[winning_col].values

# Add Open-Meteo exclusive features
df_best['om_dni_wm2'] = df['om_dni_wm2'].values
df_best['cloud_cover_pct'] = df['cloud_cover_pct'].values
df_best['cloud_cover_low_pct'] = df['cloud_cover_low_pct'].values
df_best['cloud_cover_mid_pct'] = df['cloud_cover_mid_pct'].values
df_best['cloud_cover_high_pct'] = df['cloud_cover_high_pct'].values

# Save
out = 'data/raw/singapore_pv_bestof_2020_2025.csv'
df_best.to_csv(out, sep=';', index=False, date_format='%d/%m/%Y %H:%M')

print(f"\nSaved: {out}")
print(f"Columns ({len(df_best.columns)}): {df_best.columns.tolist()}")
print(f"Rows: {len(df_best)}")

# Final correlation summary
print("\n" + "=" * 80)
print("FINAL BEST-OF DATASET - CORRELATION RANKING")
print("=" * 80)
corrs = df_best.drop(columns=['timestamp']).corrwith(df_best['pv_output_kw']).sort_values(ascending=False)
for col, c in corrs.items():
    bar = '█' * int(abs(c) * 30)
    sign = '+' if c >= 0 else '-'
    print(f"  {col:25s}: {sign}{abs(c):.4f} {bar}")
