import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/tangerang_pv_weather_2020_2024.csv', sep=';')
# Rename columns just in case
df.columns = [c.lower() for c in df.columns]

# Convert to numeric
cols_to_fix = ['pv_output_kw', 'ghi_wm2', 'ambient_temp_c']
for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("--- Data Analysis for tangerang_pv_weather_2020_2024.csv ---")
print(f"Total rows: {len(df)}")
print(f"PV Output Max: {df['pv_output_kw'].max():.2f} kW")
print(f"PV Output Mean (Daylight): {df[df['ghi_wm2'] > 50]['pv_output_kw'].mean():.2f} kW")

# Check values > 4.76
over_nominal = df[df['pv_output_kw'] > 4.76]
print(f"Rows exceeding nominal capacity (4.76kW): {len(over_nominal)}")

# Distribution
print("\nQuantiles (PV Output):")
print(df['pv_output_kw'].describe())

# Check correlation at high GHI
high_ghi = df[df['ghi_wm2'] > 800]
print(f"\nStats at high GHI (>800 W/m2):")
print(f"  Count: {len(high_ghi)}")
print(f"  Max PV: {high_ghi['pv_output_kw'].max():.2f} kW")
print(f"  Mean PV: {high_ghi['pv_output_kw'].mean():.2f} kW")
print(f"  Max GHI: {high_ghi['ghi_wm2'].max():.2f} W/m2")

# Correlation
corr = df[['pv_output_kw', 'ghi_wm2']].corr().iloc[0,1]
print(f"\nCorrelation (PV vs GHI): {corr:.4f}")
