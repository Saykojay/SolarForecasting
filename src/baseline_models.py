import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

logger = logging.getLogger(__name__)

def evaluate_ml_baseline(model_type, X_train, y_train, X_test, y_test, y_scaler=None):
    """
    Melatih dan mengevaluasi Model ML Tradisional (Linear Regression, Ridge, Random Forest).
    Diasumsikan X adalah Tensor (Batch, Lookback, Features). Kita ratakan (flatten) untuk ML 2D.
    """
    print(f"Melatih {model_type} baseline...")
    
    # Flatten the Time Series Window
    X_tr_flat = X_train.reshape(X_train.shape[0], -1)
    X_te_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Flatten y for multi-output regression
    y_tr_flat = y_train.reshape(y_train.shape[0], -1)
    
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Ridge Regression':
        model = Ridge(alpha=1.0)
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Tipe model {model_type} tidak dikenal.")
        
    start_time = datetime.now()
    model.fit(X_tr_flat, y_tr_flat)
    train_time = (datetime.now() - start_time).total_seconds()
    
    # Predict
    y_pred_scaled = model.predict(X_te_flat)
    
    # Reshape back to (Batch, Horizon)
    if len(y_test.shape) == 2:
        y_pred_scaled = y_pred_scaled.reshape(y_test.shape)
        
    # Inverse Transform
    if y_scaler is not None:
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_actual = y_scaler.inverse_transform(y_test)
    else:
        y_pred = y_pred_scaled
        y_actual = y_test
        
    # Hard Clipping
    y_pred = np.maximum(0, y_pred)
    
    # Hitung metrik sederhana
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    
    return {
        'model': model_type,
        'train_time': train_time,
        'metrics': {'R²': r2, 'MAE': mae, 'RMSE': rmse},
        'y_pred': y_pred,
        'y_actual': y_actual
    }

def evaluate_physics_baseline(model_type, df_test, capacity_kw=None):
    """
    Mengevaluasi Model Fisika menggunakan PVLib (PVWatts, Single Diode, dll).
    Mengasumsikan df_test sudah mengandung kolom 'GHI', 'DNI', 'DHI', 'temp_air', 'wind_speed'.
    """
    import pvlib
    from pvlib.pvsystem import PVSystem
    from pvlib.location import Location
    from pvlib.modelchain import ModelChain
    from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

    print(f"Menjalankan simulasi fisika: {model_type}")
    
    # Default koordinat (asumsi umum jika tidak ada di config, sebaiknya diupdate sesuai dataset)
    lat, lon = -7.25, 112.75 # Default Surabaya
    location = Location(latitude=lat, longitude=lat, tz='Asia/Jakarta')
    
    # Ekstraksi Cuaca
    expected_cols = ['GHI', 'DNI', 'DHI', 'temp_air', 'wind_speed']
    weather = pd.DataFrame(index=df_test.index)
    col_map = {}
    for c in expected_cols:
        matches = [col for col in df_test.columns if col.lower() == c.lower()]
        if matches:
            col_map[c] = matches[0]
        else:
            # Cari substring jika tidak exact match
            matches = [col for col in df_test.columns if c.lower() in col.lower()]
            if matches: col_map[c] = matches[0]
    
    try:
        weather['ghi'] = df_test[col_map.get('GHI', 'ghi')]
        weather['dni'] = df_test[col_map.get('DNI', 'dni')]
        weather['dhi'] = df_test[col_map.get('DHI', 'dhi')]
        weather['temp_air'] = df_test[col_map.get('temp_air', 'temp_air')]
        weather['wind_speed'] = df_test[col_map.get('wind_speed', 'wind_speed')]
    except KeyError as e:
        raise ValueError(f"Kolom cuaca tidak lengkap untuk {model_type}. Butuh GHI,DNI,DHI,temp_air,wind_speed. Error: {e}")

    # Build PV System
    if capacity_kw is None:
        capacity_kw = 1000 # Default 1 MW
        
    if model_type == 'PVWatts':
        # PVWatts is the simplest physical model
        system = PVSystem(surface_tilt=15, surface_azimuth=180, 
                          module_parameters={'pdc0': capacity_kw*1000, 'gamma_pdc': -0.004},
                          inverter_parameters={'pdc0': capacity_kw*1000, 'eta_inv_nom': 0.96},
                          temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass'])
        mc = ModelChain(system, location, aoi_model='physical', spectral_model='no_loss')
        
    elif model_type in ['Single Diode (SAPM)', 'NOCT Model']:
        # More complex physical models often need exact module specs. We fallback to PVWatts internally
        # with different temperature models for demonstration if true specs aren't available.
        system = PVSystem(surface_tilt=15, surface_azimuth=180, 
                          module_parameters={'pdc0': capacity_kw*1000, 'gamma_pdc': -0.004},
                          inverter_parameters={'pdc0': capacity_kw*1000, 'eta_inv_nom': 0.96},
                          temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm']['roof_mount_glass_polymer'])
        mc = ModelChain(system, location, aoi_model='physical', spectral_model='no_loss')
    else:
        raise ValueError(f"Fisika model {model_type} tidak dikenali.")

    # Jalankan simulasi
    mc.run_model(weather)
    
    # Ambil output AC power (W) lalu convert ke kW
    pred_power = mc.results.ac / 1000.0
    pred_power = pred_power.fillna(0).clip(lower=0).values
    
    # Ambil target aktual jika ada
    target_col = [col for col in df_test.columns if 'target' in col.lower() or 'power' in col.lower() or 'pv' in col.lower()]
    if target_col:
        y_actual = df_test[target_col[0]].values
        
        # Hard clip pred (cannot be more than 120% of capacity)
        pred_power = np.minimum(pred_power, capacity_kw * 1.2)
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        # Selaraskan panjang jika perlu
        min_len = min(len(y_actual), len(pred_power))
        mae = mean_absolute_error(y_actual[:min_len], pred_power[:min_len])
        rmse = np.sqrt(mean_squared_error(y_actual[:min_len], pred_power[:min_len]))
        r2 = r2_score(y_actual[:min_len], pred_power[:min_len])
    else:
        y_actual = None
        mae, rmse, r2 = 0, 0, 0

    return {
        'model': model_type,
        'train_time': 0, # Physics models don't "train"
        'metrics': {'R²': r2, 'MAE': mae, 'RMSE': rmse},
        'y_pred': pred_power,
        'y_actual': y_actual
    }
