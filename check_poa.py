import pandas as pd
import numpy as np
import pvlib
from datetime import datetime

# Tangerang approx
lat, lon = -6.17, 106.63
tz = 'Asia/Jakarta'

df = pd.read_csv('data/raw/tangerang_pv_weather_2020_2024.csv', sep=';')
df.columns = [c.lower() for c in df.columns]
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
df.set_index('timestamp', inplace=True)
df = df.sort_index()

# Fix types
for c in ['pv_output_kw', 'ghi_wm2', 'dhi_wm2', 'ambient_temp_c']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Drop NaNs for correlation
df_clean = df.dropna(subset=['pv_output_kw', 'ghi_wm2', 'dhi_wm2'])

# Get solar position
solpos = pvlib.solarposition.get_solarposition(df_clean.index, lat, lon)

# Calculate POA
# Surface Tilt: 15, Surface Azimuth: 0 (North in Southern Hemisphere logic?)
# Wait, in pvlib, Azimuth 0 = North, 180 = South.
# Since Tangerang is at -6 (South), a North orientation means Azimuth = 0.
tilt = 15
azimuth = 0

# Calculate extraterrestrial radiation
dni_extra = pvlib.irradiance.get_extra_radiation(df_clean.index)

# Estimate DNI from GHI and DHI
# GHI = DNI * cos(zenith) + DHI  => DNI = (GHI - DHI) / cos(zenith)
dni = (df_clean['ghi_wm2'] - df_clean['dhi_wm2']) / np.cos(np.radians(solpos['zenith']))
dni = dni.clip(0, 1400) # Sanity

poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azimuth,
    solar_zenith=solpos['zenith'],
    solar_azimuth=solpos['azimuth'],
    dni=dni,
    ghi=df_clean['ghi_wm2'],
    dhi=df_clean['dhi_wm2'],
    dni_extra=dni_extra
)

df_clean['poa_calc'] = poa['poa_global']

# Compare correlations
corr_ghi = df_clean['pv_output_kw'].corr(df_clean['ghi_wm2'])
corr_poa = df_clean['pv_output_kw'].corr(df_clean['poa_calc'])

print(f"Correlation with GHI: {corr_ghi:.4f}")
print(f"Correlation with Calculated POA (15N): {corr_poa:.4f}")

# Check monthly maxes to see if tilt matches seasonality
df_clean['month'] = df_clean.index.month
monthly_max = df_clean.groupby('month')[['pv_output_kw', 'ghi_wm2', 'poa_calc']].max()
print("\nMonthly Maxima:")
print(monthly_max)
