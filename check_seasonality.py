import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/tangerang_pv_weather_2020_2024.csv', sep=';')
df.columns = [c.lower() for c in df.columns]
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
df.set_index('timestamp', inplace=True)

# Fix types
for c in ['pv_output_kw', 'ghi_wm2']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df['month'] = df.index.month
df['ratio'] = df['pv_output_kw'] / df['ghi_wm2']
# Filter for high ghi only to get clean ratio
mask = (df['ghi_wm2'] > 600) & (df['pv_output_kw'] > 0.5)
monthly_ratio = df[mask].groupby('month')['ratio'].mean()

print("Monthly PV/GHI Ratio (at high irradiance):")
print(monthly_ratio)
