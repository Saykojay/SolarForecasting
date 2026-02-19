"""Merge PV data with Open-Meteo weather data."""
import pandas as pd
import numpy as np

# Load both datasets
df_pv = pd.read_csv('data/raw/singapore_pv_weather_2020_2025.csv', sep=';')
df_pv['timestamp'] = pd.to_datetime(df_pv['timestamp'], format='%d/%m/%Y %H:%M')

df_om = pd.read_csv('data/raw/open_meteo_singapore_2020_2025.csv')
df_om['timestamp'] = pd.to_datetime(df_om['timestamp'])

print(f"PV data: {len(df_pv)} rows, {df_pv.timestamp.min()} to {df_pv.timestamp.max()}")
print(f"Open-Meteo: {len(df_om)} rows, {df_om.timestamp.min()} to {df_om.timestamp.max()}")

# Merge on timestamp
df_merged = pd.merge(df_pv, df_om, on='timestamp', how='inner')
print(f"Merged: {len(df_merged)} rows")
print(f"Columns: {df_merged.columns.tolist()}")

# Check GHI comparison between NASA POWER and Open-Meteo
ghi_mask = (df_merged['ghi_wm2'] > 50) & (df_merged['om_ghi_wm2'] > 50)
corr = df_merged.loc[ghi_mask, 'ghi_wm2'].corr(df_merged.loc[ghi_mask, 'om_ghi_wm2'])
nasa_mean = df_merged.loc[ghi_mask, 'ghi_wm2'].mean()
om_mean = df_merged.loc[ghi_mask, 'om_ghi_wm2'].mean()
print(f"\nGHI comparison (NASA vs Open-Meteo, productive hours):")
print(f"  Correlation: {corr:.4f}")
print(f"  NASA mean: {nasa_mean:.1f}")
print(f"  OM mean: {om_mean:.1f}")

# Check cloud cover correlation with PV
pv_mask = df_merged['pv_output_kw'].notna()
print(f"\nNew feature correlations with PV output:")
new_cols = ['cloud_cover_pct', 'cloud_cover_low_pct', 'cloud_cover_mid_pct', 
            'cloud_cover_high_pct', 'om_ghi_wm2', 'om_dni_wm2', 'om_dhi_wm2',
            'om_temp_c', 'om_humidity_pct']
for col in new_cols:
    c = df_merged.loc[pv_mask, col].corr(df_merged.loc[pv_mask, 'pv_output_kw'])
    print(f"  {col:25s}: {c:.4f}")

# Save merged dataset
out_path = 'data/raw/singapore_pv_weather_openmeteo_2020_2025.csv'
df_merged.to_csv(out_path, sep=';', index=False, date_format='%d/%m/%Y %H:%M')
print(f"\nSaved merged dataset: {out_path}")
print(f"Total columns: {len(df_merged.columns)}")
