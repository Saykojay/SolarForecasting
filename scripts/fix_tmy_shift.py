import pandas as pd
import numpy as np
import os

def fix_shift():
    raw_path = r'c:\Users\Lenovo\OneDrive\Pretrain GRU\Pre-train model PatchTST\Modular Pipeline v1\data\raw\ntt_tmy_2027.csv'
    out_path = r'c:\Users\Lenovo\OneDrive\Pretrain GRU\Pre-train model PatchTST\Modular Pipeline v1\data\raw\ntt_tmy_2027_fixed.csv'
    
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    print(f"Reading {raw_path}...")
    df = pd.read_csv(raw_path, sep=';')
    
    # 1. Automatically detect the peak hour in the original data
    df['hour'] = pd.to_datetime(df['timestamp'], dayfirst=True).dt.hour
    df['ghi_proxy'] = df['dhi_wm2'] + df['dni_wm2']
    hourly_avg = df.groupby('hour')['ghi_proxy'].mean()
    peak_hour = hourly_avg.idxmax()
    
    print(f"Detected original peak hour: {peak_hour}:00")
    
    # 2. Calculate shift amount to move peak to 12:00
    # Shift = peak_hour - 12
    shift_amount = int(peak_hour - 12)
    print(f"Applying circular shift of {-shift_amount} steps to align peak to 12:00...")
    
    # 3. Columns to shift (all except timestamp and cyclical features)
    cols_to_shift = ['dhi_wm2', 'dni_wm2', 'ambient_temp_c']
    
    # Apply shift
    for col in cols_to_shift:
        if col in df.columns:
            df[col] = np.roll(df[col].values, -shift_amount)
            
    # 4. Verify peak alignment
    df['ghi_proxy'] = df['dhi_wm2'] + df['dni_wm2']
    new_hourly_avg = df.groupby('hour')['ghi_proxy'].mean()
    new_peak_hour = new_hourly_avg.idxmax()
    print(f"New Solar Peak Hour (average): {new_peak_hour}:00")
    
    if new_peak_hour == 12 or new_peak_hour == 13:
        print("Success! Data is now synchronized with local solar time.")
    else:
        print(f"Warning: Peak hour is {new_peak_hour}:00, expected ~12:00.")
        
    # Drop temp columns
    df.drop(columns=['ghi_proxy', 'hour'], inplace=True)
    
    print(f"Saving to {out_path}...")
    df.to_csv(out_path, index=False, sep=';')
    print("Done.")

if __name__ == "__main__":
    fix_shift()
