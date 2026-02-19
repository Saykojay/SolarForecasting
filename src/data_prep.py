"""
data_prep.py - Preprocessing, Feature Engineering, dan Sequence Creation
Berisi semua logika dari notebook: Algorithm 1, create_features,
calculate_clear_sky_pv, select_features_hybrid, create_sequences_with_indices.
"""
import numpy as np
import pandas as pd
import logging
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


# ============================================================
# ALGORITHM 1: DATA PRE-PROCESSING
# ============================================================
def preprocess_algorithm1(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Algorithm 1: Data Pre-processing (Rigorous cleaning based on Physics & Statistics)."""
    cols = cfg['data']
    pcfg = cfg.get('preprocessing', {})
    df = df.copy()
    
    print("=" * 60)
    print("ALGORITHM 1: DATA PRE-PROCESSING")
    print("=" * 60)

    # 1. Rename & Set Index
    time_col = cols['time_col']
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], format=cols.get('time_format', None))
        df = df.set_index(time_col)
        df = df.sort_index()

    # 2. Resample (Optional but recommended for consistency)
    if pcfg.get('resample_1h', True):
        if hasattr(df.index, 'freq') and df.index.freq is None:
            inferred_freq = pd.infer_freq(df.index[:100])
            if inferred_freq and 'h' not in str(inferred_freq).lower():
                print(f"  Resampling dari {inferred_freq} ke 1h...")
                df = df.resample('1h').mean()

    # 3. Handle Duplicates
    df = df[~df.index.duplicated(keep='first')]

    # 4. Outlier Detection (Notebook Logic)
    if pcfg.get('remove_outliers', True):
        ghi_col = cols['ghi_col']
        target_col = cols['target_col']
        temp_col = cols['temp_col']
        
        # Rule 1: Physical Parameters Extreme
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        if ghi_col in df.columns: outlier_mask |= (df[ghi_col] > 2000)
        if temp_col in df.columns: outlier_mask |= (df[temp_col] < -30)
        if cols['rh_col'] in df.columns: 
            outlier_mask |= (df[cols['rh_col']] < 0) | (df[cols['rh_col']] > 100)
        if cols['wind_speed_col'] in df.columns:
            outlier_mask |= (df[cols['wind_speed_col']] < 0)
        
        # Rule 2: PV-GHI Inconsistency (PV 0 when GHI is high)
        if pcfg.get('ghi_high_pv_zero', True) and target_col in df.columns and ghi_col in df.columns:
            outlier_mask |= ((df[target_col] <= 0) & (df[ghi_col] > 200))
            
        # Rule 3: PV High when GHI is Dark (Sensor Error)
        if pcfg.get('ghi_dark_pv_high', True) and target_col in df.columns and ghi_col in df.columns:
            cap = cfg['pv_system']['nameplate_capacity_kw']
            thresh = cfg['pv_system'].get('csi_ghi_threshold', 20)
            outlier_mask |= ((df[target_col] > 0.1 * cap) & (df[ghi_col] < thresh))
            
        if target_col in df.columns:
            is_nonzero = df[target_col] > 0
            consecutive_same = (df[target_col] == df[target_col].shift(1)) & is_nonzero
            consecutive_count = consecutive_same.astype(int).groupby((consecutive_same != consecutive_same.shift()).cumsum()).cumsum()
            outlier_mask |= (consecutive_count >= 10)
        
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            print(f"  Detected {n_outliers} outlier rows based on physics/consistency rules")
            # Set to NaN instead of dropping immediately to maintain index continuity
            df.loc[outlier_mask, :] = np.nan

    # 5. Advanced Cleaning (Optional/Customizable)
    # 5a. GHI vs DHI Correction
    if pcfg.get('fix_ghi_dhi', True) and 'ghi_wm2' in df.columns and 'dhi_wm2' in df.columns:
        inconsistent = df['ghi_wm2'] < df['dhi_wm2']
        if inconsistent.any():
            print(f"  [Cleaning] Fixed {inconsistent.sum()} rows where GHI < DHI.")
            df.loc[inconsistent, 'ghi_wm2'] = df.loc[inconsistent, 'dhi_wm2']
            
    # 5b. Precipitation Clipping
    precip_limit = pcfg.get('clip_precipitation')
    if precip_limit and 'precipitation_mm' in df.columns:
        excess = df['precipitation_mm'] > precip_limit
        if excess.any():
            print(f"  [Cleaning] Clipping {excess.sum()} precipitation values above {precip_limit} mm/h.")
            df.loc[excess, 'precipitation_mm'] = precip_limit

    # 5c. Missing PV Imputation (Simple CSI-based)
    if pcfg.get('impute_missing_pv', False) and cols['target_col'] in df.columns and cols['ghi_col'] in df.columns:
        target_col = cols['target_col']
        ghi_col = cols['ghi_col']
        
        # We only impute during day time (GHI > threshold)
        thresh = cfg['target'].get('csi_ghi_threshold', 50)
        missing_mask = df[target_col].isnull() & (df[ghi_col] > thresh)
        
        if missing_mask.any():
            print(f"  [Impute] Estimating {missing_mask.sum()} missing PV values based on GHI...")
            # Calculate a simple average efficiency (PV/GHI) from non-null hours
            valid_mask = df[target_col].notnull() & (df[ghi_col] > thresh)
            if valid_mask.any():
                avg_efficiency = (df.loc[valid_mask, target_col] / df.loc[valid_mask, ghi_col]).median()
                df.loc[missing_mask, target_col] = df.loc[missing_mask, ghi_col] * avg_efficiency
            else:
                # Fallback to linear interpolate if no valid reference
                df[target_col] = df[target_col].interpolate(method='linear')

    # 6. Negative Clamping
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].clip(lower=0)

    # 6. Reindex & Gap Management
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1h')
    df = df.reindex(full_range)
    df.index.name = time_col

    print(f"  Final shape: {df.shape}")
    print("=" * 60)
    return df


# ============================================================
# CLEAR SKY PV & CSI CALCULATION
# ============================================================
def calculate_clear_sky_pv(ghi, poa, temp_ambient, wind_speed=None,
                           nameplate_capacity=10.5, temp_coeff=-0.0045,
                           ref_temp=25, system_efficiency=0.86):
    """Menghitung output PV clear-sky teoritis berbasis model fisik sederhana."""
    # Cell temperature (model Ross)
    noct = 45
    t_cell = temp_ambient + (noct - 20) / 800 * poa

    # Temperature derating
    temp_factor = 1 + temp_coeff * (t_cell - ref_temp)

    # Clear sky power (kW)
    pv_clear_sky = nameplate_capacity * (poa / 1000) * temp_factor * system_efficiency
    pv_clear_sky = np.clip(pv_clear_sky, 0, nameplate_capacity * 1.1)
    return pv_clear_sky


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def create_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Membuat fitur turunan dari data mentah."""
    df = df.copy()
    cols = cfg['data']
    pv_cfg = cfg['pv_system']
    target_cfg = cfg['target']

    time_col = cols['time_col']
    ghi_col = cols['ghi_col']
    dhi_col = cols['dhi_col']
    temp_col = cols['temp_col']
    rh_col = cols['rh_col']
    wind_col = cols['wind_speed_col']
    poa_col = cols['poa_col']
    target_col = cols['target_col']

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(time_col)

    # Simpan timestamp
    df['timestamp_col'] = df.index

    # 1. CLEAR SKY PV & CSI TARGET (only if use_csi=true or physics=true)
    fg = cfg['features']['groups']
    need_csi = target_cfg.get('use_csi', False) or fg.get('physics', False)
    
    if need_csi:
        print("Calculating Clear Sky PV & CSI...")
        
        # POA Fallback: Use GHI if POA is not present
        actual_poa = df[poa_col].values if poa_col in df.columns else df[ghi_col].values
        if poa_col not in df.columns:
            print(f"  Warning: {poa_col} missing. Using {ghi_col} as proxy.")
            
        df['pv_clear_sky'] = calculate_clear_sky_pv(
            ghi=df[ghi_col].values,
            poa=actual_poa,
            temp_ambient=df[temp_col].values,
            wind_speed=df[wind_col].values if wind_col in df.columns else None,
            nameplate_capacity=pv_cfg['nameplate_capacity_kw'],
            temp_coeff=pv_cfg['temp_coeff'],
            ref_temp=pv_cfg['ref_temp'],
            system_efficiency=pv_cfg['system_efficiency'],
        )

        ghi_threshold = target_cfg['csi_ghi_threshold']
        csi_max = target_cfg['csi_max']

        productive_mask = df[ghi_col] > ghi_threshold
        df['csi_target'] = 0.0
        safe_cs = df.loc[productive_mask, 'pv_clear_sky'].replace(0, np.nan)
        df.loc[productive_mask, 'csi_target'] = (
            df.loc[productive_mask, target_col] / safe_cs
        ).clip(0, csi_max)
        df['csi_target'] = df['csi_target'].fillna(0)

        productive_csi = df.loc[productive_mask, 'csi_target']
        print(f"  CSI stats (productive hours):")
        print(f"    Mean: {productive_csi.mean():.4f}")
        print(f"    Std:  {productive_csi.std():.4f}")

        # Normalized PV clear sky
        df['pv_cs_normalized'] = df['pv_clear_sky'] / pv_cfg['nameplate_capacity_kw']
    else:
        print("Skipping CSI/Clear-Sky (use_csi=false, physics=false)")
        df['pv_clear_sky'] = 0.0
        df['csi_target'] = 0.0
        df['pv_cs_normalized'] = 0.0


    # 2. TIME FEATURES (Cyclical & Linear)
    fg = cfg['features']['groups']
    if fg.get('time_hour', True) if 'time_hour' in fg else fg.get('time', True):
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
    if fg.get('time_month', True) if 'time_month' in fg else fg.get('time', True):
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
    if fg.get('time_doy', True) if 'time_doy' in fg else fg.get('time', True):
        # Using 365.25 to account for leap years roughly
        df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        
    if fg.get('time_year', False):
        df['year_linear'] = df.index.year

    # 3. LAG & ROLLING FEATURES (Adaptive to all available weather/system columns)
    # CRITICAL: Always include the target series itself as a feature for history!
    act_target = 'csi_target' if target_cfg['use_csi'] else target_col
    weather_cols = [ghi_col, dhi_col, temp_col, rh_col, wind_col, act_target]
    # Add any other numeric columns from data config that might exist
    if 'wind_dir_col' in cols and cols['wind_dir_col'] in df.columns: weather_cols.append(cols['wind_dir_col'])
    # Detect precipitation, cloud, or Open-Meteo supplementary columns
    for c in df.columns:
        if any(x in c.lower() for x in ['precip', 'rain', 'press', 'cloud', 'om_ghi', 'om_dni', 'om_dhi', 'om_temp', 'om_humid', 'om_wind']):
            if c not in weather_cols: weather_cols.append(c)

    # Filter to existing columns only
    existing_weather = [c for c in weather_cols if c in df.columns]
    
    # Apply Lags
    if cfg['features']['groups'].get('lags', True):
        for col in existing_weather:
            prefix = col.split('_')[0] if '_' in col else col
            for lag_h in [1, 12, 24]: # Standard lags
                df[f"{prefix}_lag_{lag_h}h"] = df[col].shift(lag_h)
            if col in [ghi_col, 'csi_target']: # Long lags only for criticals
                df[f"{prefix}_lag_48h"] = df[col].shift(48)
                df[f"{prefix}_lag_168h"] = df[col].shift(168)

    # Apply Rolling
    if cfg['features']['groups'].get('rolling', True):
        for col in [ghi_col, temp_col, 'csi_target']: # Rolling only on primary signals to avoid explosion
            if col in df.columns:
                prefix = col.split('_')[0] if '_' in col else col
                df[f'{prefix}_ma_3h'] = df[col].rolling(3).mean()
                df[f'{prefix}_std_3h'] = df[col].rolling(3).std()

    # 5. DC POWER FEATURE
    dc_col = 'pv_output_dc_kw'
    if dc_col in df.columns:
        pass  # sudah ada
    elif target_col in df.columns:
        df[dc_col] = df[target_col]

    print(f"Features created. Total columns: {len(df.columns)}")
    print("=" * 60)
    return df


# ============================================================
# HYBRID FEATURE SELECTION
# ============================================================
def select_features_hybrid(df: pd.DataFrame, target_col: str, cfg: dict):
    """Hybrid Feature Selection: Statistical + Domain Knowledge."""
    # 0. Manual Override for Experiments
    if cfg['features'].get('selection_mode', 'auto') == 'manual':
        manual_list = cfg['features'].get('manual_features', [])
        # Filter to only those that actually exist in df
        selected = [c for c in manual_list if c in df.columns]
        print(f"🛠️ Manual Selection Mode: {len(selected)} features selected.")
        return selected, df[selected].corr()

    time_col = cfg['data']['time_col']
    corr_threshold = cfg['features']['corr_threshold']
    multicol_threshold = cfg['features']['multicol_threshold']

    # Exclude auxiliary/intermediate computation columns (not real features)
    # NOTE: pv_output_kw and csi_target at PAST timesteps in the lookback window 
    # are legitimate autoregressive features, NOT data leakage.
    # Data leakage would be if we used FUTURE values, which sequence creation prevents.
    exclude_cols = [time_col, 'pv_clear_sky', 'pv_cs_normalized', 
                    'pv_output_dc_kw', 'timestamp_col']
    candidate_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                      if c not in exclude_cols]

    
    # Filter by Group
    grps = cfg['features'].get('groups', {})
    filtered_candidates = []
    for c in candidate_cols:
        is_lag = 'lag_' in c
        is_ma = '_ma_' in c or '_std_' in c
        
        # Identify specific time features
        is_hour = 'hour_' in c
        is_month = 'month_' in c
        is_doy = 'day_of_year_' in c
        is_year = 'year_linear' in c
        
        if is_lag:
            if grps.get('lags', True): filtered_candidates.append(c)
        elif is_ma:
            if grps.get('rolling', True): filtered_candidates.append(c)
        elif is_hour:
            if grps.get('time_hour', True) or grps.get('time', True): filtered_candidates.append(c)
        elif is_month:
            if grps.get('time_month', True) or grps.get('time', True): filtered_candidates.append(c)
        elif is_doy:
            if grps.get('time_doy', True) or grps.get('time', True): filtered_candidates.append(c)
        elif is_year:
            if grps.get('time_year', False) or grps.get('time', True): filtered_candidates.append(c)
        else:
            # Likely raw weather or other base features
            if grps.get('weather', True): filtered_candidates.append(c)
            
    candidate_cols = filtered_candidates

    # 1. Correlation ranking
    corr = df[candidate_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    print(f"Top 15 Features by Correlation")
    print(corr.head(15).to_string())

    selected = corr[corr >= corr_threshold].index.tolist()
    print(f"\nStatistically selected: {len(selected)} features")

    # 2. Remove multicollinearity
    corr_matrix = df[selected].corr().abs()
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > multicol_threshold:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                # Drop the one with lower correlation to target
                if corr.get(col_i, 0) < corr.get(col_j, 0):
                    to_drop.add(col_i)
                else:
                    to_drop.add(col_j)
    selected = [c for c in selected if c not in to_drop]
    print(f"After removing multicollinearity: {len(selected)} features")

    # 3. Physics-based features (always include)
    physics_cols = [cfg['data']['temp_col'], cfg['data']['rh_col'], cfg['data']['wind_speed_col']]
    print("\n=== Adding Physics-Based Features ===")
    if cfg['features']['groups'].get('physics', True):
        for pc in physics_cols:
            if pc in df.columns:
                if pc in selected:
                    print(f"  ✓ {pc} already selected")
                else:
                    selected.append(pc)
                    print(f"  + {pc} added (physics)")

    print(f"\n=== FINAL: {len(selected)} features selected ===")
    
    # Return the full correlation matrix for the final selected features
    final_corr_matrix = df[selected].corr()
    return selected, final_corr_matrix


# ============================================================
# SEQUENCE CREATION
# ============================================================
def create_sequences_with_indices(X, y, timestamps, lookback, horizon=1):
    """Membuat sequences dengan pengecekan kontinuitas waktu."""
    Xs, ys, indices = [], [], []
    ts = pd.to_datetime(timestamps).values.astype('datetime64[s]').astype(np.int64)
    expected_diff = (lookback + horizon - 1) * 3600
    for i in range(lookback, len(X) - horizon + 1):
        if (ts[i + horizon - 1] - ts[i - lookback]) == expected_diff:
            Xs.append(X[i - lookback:i])
            ys.append(y[i:i + horizon] if horizon > 1 else y[i])
            indices.append(i)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(indices)


# ============================================================
# FULL PREPROCESSING PIPELINE
# ============================================================
def run_preprocessing(cfg: dict, version_name: str = None):
    """Menjalankan seluruh pipeline preprocessing dan menyimpan hasilnya."""
    from src.config_loader import get_root_cols, ensure_dirs
    ensure_dirs(cfg)

    cols = cfg['data']
    split_cfg = cfg['splitting']
    target_cfg = cfg['target']
    forecast_horizon = cfg['forecasting']['horizon']
    lookback = cfg['model']['hyperparameters']['lookback']

    # 1. Load data
    print(f"\nLoading data from {cols['csv_path']}...")
    df = pd.read_csv(cols['csv_path'], sep=cols['csv_separator'])
    
    # NEW: Data Trimming (Mangkas Data)
    pcfg = cfg.get('preprocessing', {})
    trim_rows = pcfg.get('trim_rows')
    if trim_rows and isinstance(trim_rows, int) and trim_rows > 0:
        print(f"✂️ Trimming data to first {trim_rows} rows...")
        df = df.head(trim_rows)
    
    # Robust date parsing
    try:
        fmt = cols.get('time_format')
        if fmt and fmt.lower() not in ['auto', 'none', 'infer']:
            df[cols['time_col']] = pd.to_datetime(df[cols['time_col']], format=fmt)
        else:
            df[cols['time_col']] = pd.to_datetime(df[cols['time_col']], dayfirst=False)
    except Exception as e:
        print(f"  Warning: Format '{fmt}' failed ({e}). Trying mixed format auto-detection...")
        df[cols['time_col']] = pd.to_datetime(df[cols['time_col']], format='mixed')
    
    print(f"Original shape: {df.shape}")

    # 2. Preprocessing Algorithm 1
    df_clean = preprocess_algorithm1(df, cfg)

    # 3. Split
    print("=" * 60)
    print("ALGORITHM: SPLIT -> DROPNA -> CREATE FEATURES -> SELECT -> SCALE -> SEQUENCE")
    train_size = int(split_cfg['train_ratio'] * len(df_clean))
    df_train_raw = df_clean.iloc[:train_size].copy()
    df_test_raw = df_clean.iloc[train_size:].copy()
    print(f"Raw Split: Train={len(df_train_raw)}, Test={len(df_test_raw)}")

    # 4. Drop missing
    root_cols_all = get_root_cols(cfg)
    # Filter only columns that exist in the dataframe
    root_cols = [c for c in root_cols_all if c in df_train_raw.columns]
    
    print(f"\nDropping rows with missing values in existing ROOT_COLS: {root_cols}")
    n_tb = len(df_train_raw)
    n_vb = len(df_test_raw)
    df_train_clean = df_train_raw.dropna(subset=root_cols).copy()
    df_test_clean = df_test_raw.dropna(subset=root_cols).copy()
    print(f"  Train: {n_tb} -> {len(df_train_clean)} ({n_tb - len(df_train_clean)} rows dropped)")
    print(f"  Test:  {n_vb} -> {len(df_test_clean)} ({n_vb - len(df_test_clean)} rows dropped)")

    # 5. Create features
    print("\nCreating derived features...")
    df_train_feats = create_features(df_train_clean, cfg)
    df_train_feats = df_train_feats.dropna()

    bridge_len = min(168, len(df_train_clean))
    df_test_with_bridge = pd.concat([df_train_clean.tail(bridge_len), df_test_clean])
    df_test_feats_full = create_features(df_test_with_bridge, cfg)
    df_test_feats = df_test_feats_full.iloc[bridge_len:].copy().dropna()
    print(f"  After feature creation: Train={len(df_train_feats)}, Test={len(df_test_feats)}")

    # 6. Feature selection
    act_target = 'csi_target' if target_cfg['use_csi'] else cols['target_col']
    print(f"\nSelecting features based on: {act_target}")
    selected_features, corr_matrix = select_features_hybrid(df_train_feats, act_target, cfg)

    # 7. Scaling (Separate scalers for X features and y target)
    scaler_type = cfg['features'].get('scaler_type', 'minmax').lower()
    target_scaler_type = cfg['features'].get('target_scaler_type', scaler_type).lower()
    
    # X scaler
    if scaler_type == 'standard':
        print("Using StandardScaler for X features (Mean=0, Std=1)...")
        X_scaler = StandardScaler()
    else:
        print("Using MinMaxScaler for X features (Range=[0, 1])...")
        X_scaler = MinMaxScaler()
    
    # Y scaler (can differ from X scaler for better target distribution handling)
    if target_scaler_type == 'standard':
        print("Using StandardScaler for Target (Mean=0, Std=1)...")
        y_scaler = StandardScaler()
    else:
        print("Using MinMaxScaler for Target (Range=[0, 1])...")
        y_scaler = MinMaxScaler()

    X_train_scaled = X_scaler.fit_transform(df_train_feats[selected_features])
    y_train_scaled = y_scaler.fit_transform(df_train_feats[[act_target]]).flatten()

    X_test_scaled = X_scaler.transform(df_test_feats[selected_features])
    y_test_scaled = y_scaler.transform(df_test_feats[[act_target]]).flatten()


    # 8. Create sequences
    print("\nCreating sequences...")
    X_train_seq, y_train_seq, train_indices = create_sequences_with_indices(
        X_train_scaled, y_train_scaled, df_train_feats.index, lookback, forecast_horizon)
    X_test_seq, y_test_seq, test_indices = create_sequences_with_indices(
        X_test_scaled, y_test_scaled, df_test_feats.index, lookback, forecast_horizon)

    print(f"\nFinal Dataset Shapes:")
    # 9. Save artifacts
    # FIXED: Recursively find 'processed' root to avoid nesting sub-versions inside versions
    raw_root = os.path.abspath(cfg['paths']['processed_dir'])
    root_out_dir = raw_root
    
    while True:
        bname = os.path.basename(root_out_dir).lower()
        if bname == 'processed':
            break
        parent = os.path.dirname(root_out_dir)
        if not parent or parent == root_out_dir:
            break
        root_out_dir = parent

    # Create shortened versioned directory to avoid path-too-long issues
    timestamp_str = datetime.now().strftime('%m%d_%H%M') # Shorter timestamp
    input_filename = os.path.basename(cols['csv_path']).split('.')[0]
    # Limit filename part to 15 chars
    short_name = input_filename[:15]
    
    if version_name:
        dir_name = version_name
    else:
        dir_name = f"v_{timestamp_str}_{short_name}"
    
    out_dir = os.path.join(root_out_dir, dir_name)
    os.makedirs(out_dir, exist_ok=True)
    
    def save_to(target_path):
        os.makedirs(target_path, exist_ok=True)
        np.save(os.path.join(target_path, 'X_train.npy'), X_train_seq)
        np.save(os.path.join(target_path, 'y_train.npy'), y_train_seq)
        np.save(os.path.join(target_path, 'X_test.npy'), X_test_seq)
        np.save(os.path.join(target_path, 'y_test.npy'), y_test_seq)
        np.save(os.path.join(target_path, 'train_indices.npy'), train_indices)
        np.save(os.path.join(target_path, 'test_indices.npy'), test_indices)
        joblib.dump(X_scaler, os.path.join(target_path, 'X_scaler.pkl'))
        joblib.dump(y_scaler, os.path.join(target_path, 'y_scaler.pkl'))
        df_train_feats.to_pickle(os.path.join(target_path, 'df_train_feats.pkl'))
        df_test_feats.to_pickle(os.path.join(target_path, 'df_test_feats.pkl'))

    # Save to both latest (root) and versioned
    save_to(root_out_dir)
    save_to(out_dir)

    # Save summary to both
    summary_data = {
        'stats': {
            'original_rows': len(df),
            'after_algorithm1': len(df_clean),
            'train_size_raw': train_size,
            'test_size_raw': len(df_clean) - train_size,
            'train_final': len(X_train_seq),
            'test_final': len(X_test_seq),
            'dropped_missing': (n_tb - len(df_train_clean)) + (n_vb - len(df_test_clean))
        },
        'selected_features': selected_features,
        'all_features': df_train_feats.columns.tolist(),
        'n_features': int(X_train_seq.shape[2]),
        'target_col': act_target,
        'use_csi': target_cfg['use_csi'],
        'lookback': lookback,
        'horizon': forecast_horizon,
        'csv_source': cols['csv_path'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'corr_matrix': corr_matrix.to_dict() 
    }
    
    import json
    for p in [root_out_dir, out_dir]:
        with open(os.path.join(p, 'prep_summary.json'), 'w') as f:
            json.dump(summary_data, f, indent=2)

    print(f"\n✅ Preprocessing selesai!")
    print(f"   - Hasil terbaru: {root_out_dir}")
    print(f"   - Versi diarsipkan: {out_dir}")

    # Return full metadata package for Dashboard
    return {
        'X_train': X_train_seq, 'y_train': y_train_seq,
        'X_test': X_test_seq, 'y_test': y_test_seq,
        'train_indices': train_indices, 'test_indices': test_indices,
        'X_scaler': X_scaler, 'y_scaler': y_scaler,
        'df_train': df_train_feats, 'df_test': df_test_feats,
        'selected_features': selected_features,
        'all_features': df_train_feats.columns.tolist(),
        'n_features': X_train_seq.shape[2],
        'corr_matrix': corr_matrix,
        'stats': {
            'original_rows': len(df),
            'after_algorithm1': len(df_clean),
            'train_size_raw': train_size,
            'test_size_raw': len(df_clean) - train_size,
            'train_final': len(X_train_seq),
            'test_final': len(X_test_seq),
            'dropped_missing': (n_tb - len(df_train_clean)) + (n_vb - len(df_test_clean))
        }
    }


# Needed for run_preprocessing when called as module
import os
