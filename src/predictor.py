"""
predictor.py - Inferensi, Evaluasi Metrik, dan Target Domain Testing.
Berisi: safe_predict, calculate_full_metrics, CSI->Power transform, target testing.
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ============================================================
# SAFE PREDICT (GPU Memory-safe)
# ============================================================
def safe_predict(model, X, batch_size=32):
    """
    Prediksi yang lebih stabil untuk menghindari OOM atau iterator crash.
    Menghindari penggunaan generator yang sering menyebabkan OUT_OF_RANGE di Keras 3.
    """
    try:
        # Gunakan predict standar
        return model.predict(X, batch_size=batch_size, verbose=0)
    except Exception as e:
        print(f"[WARN] Standard predict failed: {e}. Using manual batching fallback...")
        preds = []
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            # Call model directly as a function (faster for small batches)
            p = model(batch, training=False)
            preds.append(p.numpy() if hasattr(p, 'numpy') else p)
        return np.concatenate(preds, axis=0)


# ============================================================
# METRICS CALCULATION
# ============================================================
def calculate_full_metrics(actual, predicted, ghi_all=None, set_name="",
                           capacity_kw=10.5):
    """Menghitung metrik lengkap (MAE, RMSE, R², MAPE, NormMAE, NormRMSE)."""
    act = actual.flatten()
    pred = predicted.flatten()

    if ghi_all is not None:
        ghi_flat = ghi_all.flatten()
        mask = ghi_flat > 50  # productive hours
        act = act[mask]
        pred = pred[mask]
        label = f"{set_name} - PRODUCTIVE HOURS (GHI > 50)"
    else:
        label = f"{set_name} - ALL HOURS"

    mae = mean_absolute_error(act, pred)
    rmse = np.sqrt(mean_squared_error(act, pred))
    r2 = r2_score(act, pred)

    # MAPE: use threshold at 10% of capacity to exclude near-zero values
    # that cause MAPE to explode (e.g. 0.05 kW actual -> 200% error)
    mape_threshold = capacity_kw * 0.10  # 10% of nameplate capacity
    significant = act > mape_threshold
    if significant.sum() > 0:
        raw_pct = np.abs((act[significant] - pred[significant]) / act[significant])
        # Cap individual errors at 100% to prevent outlier explosion
        capped_pct = np.clip(raw_pct, 0, 1.0)
        mape = np.mean(capped_pct) * 100
    else:
        mape = 0.0

    # Weighted MAPE (WMAPE): immune to near-zero division
    # WMAPE = sum(|error|) / sum(actual) — naturally robust
    total_actual = np.sum(np.abs(act))
    wmape = (np.sum(np.abs(act - pred)) / total_actual * 100) if total_actual > 0 else 0.0

    norm_mae = mae / capacity_kw if capacity_kw > 0 else 0
    norm_rmse = rmse / capacity_kw if capacity_kw > 0 else 0

    print(f"\n>>> {label} <<<")
    print(f"  Absolute: MAE={mae:.4f} kW, RMSE={rmse:.4f} kW, R2={r2:.4f}")
    print(f"  MAPE={mape:.2f}% (capped, >{mape_threshold:.1f}kW only), WMAPE={wmape:.2f}%")
    print(f"  Normalized: NormMAE={norm_mae:.4f} ({norm_mae*100:.2f}%), NormRMSE={norm_rmse:.4f}")

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'wmape': wmape,
            'norm_mae': norm_mae, 'norm_rmse': norm_rmse}


# ============================================================
# FULL EVALUATION PIPELINE (CSI -> Power)
# ============================================================
def evaluate_model(model: tf.keras.Model, cfg: dict, data: dict = None, scaler_dir: str = None):
    """
    Melakukan evaluasi lengkap pada set Train dan Test.
    scaler_dir: Jika diberikan, ambil scaler dan metadata dari folder model bundle.
    """
    from src.data_prep import create_sequences_with_indices
    # 0. GPU Context (safe for multiple evaluations)
    # Ref: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    # (Checking if growth is already set is better but simpler for now)
    # We use a localized import and config here to avoid global side effects 
    # if it would destroy the model object passed as parameter)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    root_proc = cfg['paths']['processed_dir']
    # If scaler_dir is provided, it means we are using a bundled model
    proc = scaler_dir if scaler_dir else root_proc
    pv_cfg = cfg['pv_system']
    
    # Detect horizon from model instead of relying solely on config
    horizon = model.output_shape[1]
    lookback = model.input_shape[1]
    
    print(f"Evaluation Context: lookback={lookback}, horizon={horizon}")
    print(f"Using Artifacts from: {proc}")

    # Load data if needed
    if data is None:
        data = {}
        
    # --- AUTO-RESOLVE DATA SOURCE ---
    # If the active data folder has a mismatch, we should try to load from the model's original source
    data_proc_src = root_proc
    if scaler_dir and os.path.exists(os.path.join(scaler_dir, 'meta.json')):
        with open(os.path.join(scaler_dir, 'meta.json'), 'r') as f:
            m_meta = json.load(f)
            orig_ds = m_meta.get('data_source', '').replace('\\', '/')
            if orig_ds and os.path.exists(orig_ds):
                # Check if this original source is different from active root
                if os.path.abspath(orig_ds) != os.path.abspath(root_proc):
                    print(f"🔄 Model expects specific data. Switching data source to: {orig_ds}")
                    data_proc_src = orig_ds
        
    try:
        X_train = data.get('X_train', np.load(os.path.join(data_proc_src, 'X_train.npy')))
        X_test = data.get('X_test', np.load(os.path.join(data_proc_src, 'X_test.npy')))
        # Handle Feature Count mismatch (Common when switching Feature Engineering config)
        expected_n_features = model.input_shape[2]
        actual_n_features = X_train.shape[2]
        
        if expected_n_features != actual_n_features:
            print(f"⚠️ Feature count mismatch: Model expects {expected_n_features}, Data has {actual_n_features}.")
            
            # Try to align if we have the feature list in the model bundle
            model_feat_path = os.path.join(proc, 'selected_features.json')
            
            # If not in bundle, try to find it in the original data_source from meta
            if not os.path.exists(model_feat_path):
                meta_path = os.path.join(proc, 'meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        m_data = json.load(f)
                        orig_data_path = m_data.get('data_source', '')
                        
                        if orig_data_path:
                            # 1. Try path in meta directly (normalize slashes)
                            path_direct = orig_data_path.replace('\\', '/')
                            candidate = os.path.join(path_direct, 'selected_features.json')
                            
                            if os.path.exists(candidate):
                                model_feat_path = candidate
                            else:
                                # 2. Smart Recovery: Search for the leaf folder name in data/processed
                                # This fixes broken paths caused by nesting or system moves
                                leaf_name = os.path.basename(orig_data_path.rstrip('\\/'))
                                if leaf_name:
                                    print(f"   Searching for original data folder '{leaf_name}' recursively...")
                                    # root_proc is likely the base 'data/processed' if we are here
                                    # Let's find the absolute root of data/processed
                                    search_root = root_proc
                                    while os.path.basename(search_root).startswith(('v_', 'version_')):
                                        search_root = os.path.dirname(search_root)
                                    
                                    found_path = None
                                    for root, dirs, files in os.walk(search_root):
                                        if os.path.basename(root) == leaf_name:
                                            if 'selected_features.json' in files:
                                                found_path = os.path.join(root, 'selected_features.json')
                                                break
                                    
                                    if found_path:
                                        print(f"   Recovery successful! Found feature list at: {found_path}")
                                        model_feat_path = found_path

            data_feat_path = os.path.join(data_proc_src, 'selected_features.json')
            
            if os.path.exists(model_feat_path) and os.path.exists(data_feat_path):
                with open(model_feat_path, 'r') as f: model_feat_list = json.load(f)
                with open(data_feat_path, 'r') as f: data_feat_list = json.load(f)
                
                # If all model features exist in the current data, we can slice
                if all(f in data_feat_list for f in model_feat_list):
                    print(f"   Aligning data: Extracting the {expected_n_features} features used during training...")
                    indices = [data_feat_list.index(f) for f in model_feat_list]
                    X_train = X_train[:, :, indices]
                    X_test = X_test[:, :, indices]
                else:
                    missing = [f for f in model_feat_list if f not in data_feat_list]
                    raise ValueError(f"Inkompatibilitas Fitur: Model ini dilatih menggunakan fitur {missing} "
                                     f"yang tidak ada dalam data preprocessed saat ini. "
                                     f"Silakan pilih versi data yang tepat di Training Center.")
            else:
                raise ValueError(f"Mismatch Fitur: Model mengharapkan {expected_n_features} fitur, "
                                 f"tapi data memiliki {actual_n_features}. "
                                 f"Pastikan Versi Data dan Model yang dipilih sinkron.")

        # Handle lookback mismatch in evaluation
        actual_data_lookback = X_train.shape[1]
        if lookback != actual_data_lookback:
            print(f"⚠️ Evaluation Lookback mismatch: Model wants {lookback}, Data has {actual_data_lookback}.")
            if lookback < actual_data_lookback:
                print(f"   Truncating evaluation sequences to last {lookback} steps...")
                X_train = X_train[:, -lookback:, :]
                X_test = X_test[:, -lookback:, :]
            else:
                raise ValueError(f"Data lookback ({actual_data_lookback}) lebih kecil dari "
                                 f"ekspektasi model ({lookback}). Jalankan ulang Preprocessing Step 1.")

        # Scaler and Dataframes: If scaler_dir exists, take pkl from there, otherwise from root_proc
        y_scaler_path = os.path.join(proc, 'y_scaler.pkl')
        if not os.path.exists(y_scaler_path): # Fallback to root if not in bundle
            y_scaler_path = os.path.join(root_proc, 'y_scaler.pkl')
            
        y_scaler = data.get('y_scaler', joblib.load(y_scaler_path))
        
        # Dataframes are usually too big for bundles, so we stick to data_proc_src for these
        df_train = data.get('df_train', pd.read_pickle(os.path.join(data_proc_src, 'df_train_feats.pkl')))
        df_test = data.get('df_test', pd.read_pickle(os.path.join(data_proc_src, 'df_test_feats.pkl')))
    except Exception as e:
        if isinstance(e, ValueError): raise e
        raise ValueError(f"Gagal memuat data preprocessing: {e}. Pastikan sudah menjalankan Step 1.")

    # Predict
    print("Generating predictions...")
    y_train_pred_scaled = safe_predict(model, X_train)
    y_test_pred_scaled = safe_predict(model, X_test)

    # Actual values
    target_col = cfg['data']['target_col']
    
    # Check if we should use CSI to Power conversion
    # We look at the prep_summary.json in the bundle (proc) first
    use_csi_conversion = True
    summary_path = os.path.join(proc, 'prep_summary.json')
    if not os.path.exists(summary_path): 
        summary_path = os.path.join(data_proc_src, 'prep_summary.json')
        
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            train_target = summary.get('target_col', 'csi_target')
            if train_target != 'csi_target':
                use_csi_conversion = False
                print("💡 Model detected as Absolute Power model (No CSI). Skipping conversion.")

    # Inverse transform
    y_train_pred_raw = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).reshape(y_train_pred_scaled.shape)
    y_test_pred_raw = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).reshape(y_test_pred_scaled.shape)

    steps = np.arange(horizon)
    if use_csi_conversion:
        print("Converting CSI to Power...")
        # Recreate indices for this evaluation based on ACTUAL model shapes
        _, _, train_idx_local = create_sequences_with_indices(
            np.zeros((len(df_train), 1)), np.zeros(len(df_train)),
            df_train.index, lookback, horizon)
        _, _, test_idx_local = create_sequences_with_indices(
            np.zeros((len(df_test), 1)), np.zeros(len(df_test)),
            df_test.index, lookback, horizon)

        # Align
        n_train = min(len(train_idx_local), len(y_train_pred_raw))
        n_test = min(len(test_idx_local), len(y_test_pred_raw))
        train_idx_local = train_idx_local[:n_train]
        test_idx_local = test_idx_local[:n_test]
        y_train_pred_raw = y_train_pred_raw[:n_train]
        y_test_pred_raw = y_test_pred_raw[:n_test]

        # Clear sky values at prediction points
        pv_cs_train = df_train['pv_clear_sky'].values[train_idx_local[:, np.newaxis] + steps]
        pv_cs_test = df_test['pv_clear_sky'].values[test_idx_local[:, np.newaxis] + steps]

        # CSI -> Power
        if len(y_train_pred_raw.shape) == 3 and y_train_pred_raw.shape[2] == 1:
            pv_train_pred = np.clip(y_train_pred_raw[:, :, 0] * pv_cs_train, 0, pv_cfg['nameplate_capacity_kw'])
            pv_test_pred = np.clip(y_test_pred_raw[:, :, 0] * pv_cs_test, 0, pv_cfg['nameplate_capacity_kw'])
        else:
            pv_train_pred = np.clip(y_train_pred_raw * pv_cs_train, 0, pv_cfg['nameplate_capacity_kw'])
            pv_test_pred = np.clip(y_test_pred_raw * pv_cs_test, 0, pv_cfg['nameplate_capacity_kw'])
    else:
        # Already power
        if len(y_train_pred_raw.shape) == 3 and y_train_pred_raw.shape[2] == 1:
            pv_train_pred = np.clip(y_train_pred_raw[:, :, 0], 0, pv_cfg['nameplate_capacity_kw'] * 1.2)
            pv_test_pred = np.clip(y_test_pred_raw[:, :, 0], 0, pv_cfg['nameplate_capacity_kw'] * 1.2)
        else:
            pv_train_pred = np.clip(y_train_pred_raw, 0, pv_cfg['nameplate_capacity_kw'] * 1.2)
            pv_test_pred = np.clip(y_test_pred_raw, 0, pv_cfg['nameplate_capacity_kw'] * 1.2)
        
        # Need match indices for actuals too
        _, _, train_idx_local = create_sequences_with_indices(
            np.zeros((len(df_train), 1)), np.zeros(len(df_train)),
            df_train.index, lookback, horizon)
        _, _, test_idx_local = create_sequences_with_indices(
            np.zeros((len(df_test), 1)), np.zeros(len(df_test)),
            df_test.index, lookback, horizon)
        
        n_train = min(len(train_idx_local), len(pv_train_pred))
        n_test = min(len(test_idx_local), len(pv_test_pred))
        train_idx_local = train_idx_local[:n_train]
        test_idx_local = test_idx_local[:n_test]
        pv_train_pred = pv_train_pred[:n_train]
        pv_test_pred = pv_test_pred[:n_test]

    # Actual values
    pv_train_actual = df_train[target_col].values[train_idx_local[:, np.newaxis] + steps]
    pv_test_actual = df_test[target_col].values[test_idx_local[:, np.newaxis] + steps]

    # GHI for filtering productive hours
    ghi_col = cfg['data']['ghi_col']
    ghi_train = df_train[ghi_col].values[train_idx_local[:, np.newaxis] + steps]
    ghi_test = df_test[ghi_col].values[test_idx_local[:, np.newaxis] + steps]

    # ============================================================
    # HOUR-BASED WRAPPER: Zero out predictions during nighttime
    # Strategy: Train on full 24h data for temporal continuity,
    # but force predictions to 0 when GHI < threshold (no sun = no power).
    # This eliminates nighttime "hallucination" noise from metrics.
    # ============================================================
    wrapper_threshold = cfg.get('preprocessing', {}).get('productive_hours_threshold', 50)
    night_mask_train = ghi_train < wrapper_threshold
    night_mask_test = ghi_test < wrapper_threshold
    
    n_zeroed_train = night_mask_train.sum()
    n_zeroed_test = night_mask_test.sum()
    print(f"\n🌙 Hour-Based Wrapper (GHI < {wrapper_threshold} W/m²):")
    print(f"  Train: {n_zeroed_train:,} predictions forced to 0 "
          f"({n_zeroed_train/night_mask_train.size*100:.1f}%)")
    print(f"  Test:  {n_zeroed_test:,} predictions forced to 0 "
          f"({n_zeroed_test/night_mask_test.size*100:.1f}%)")
    
    pv_train_pred[night_mask_train] = 0.0
    pv_test_pred[night_mask_test] = 0.0

    # ============================================================
    # METRICS: ALL HOURS (Primary) + Productive Hours (Secondary)
    # ============================================================
    print("\nFINAL PERFORMANCE ANALYSIS")
    
    # PRIMARY: ALL hours (model gets credit for nighttime zeros via wrapper)
    m_train = calculate_full_metrics(pv_train_actual, pv_train_pred,
                                      None, "TRAIN", pv_cfg['nameplate_capacity_kw'])
    m_test = calculate_full_metrics(pv_test_actual, pv_test_pred,
                                     None, "TEST", pv_cfg['nameplate_capacity_kw'])
    
    # SECONDARY: Productive hours only (diagnostic)
    m_train_prod = calculate_full_metrics(pv_train_actual, pv_train_pred,
                                           ghi_train, "TRAIN", pv_cfg['nameplate_capacity_kw'])
    m_test_prod = calculate_full_metrics(pv_test_actual, pv_test_pred,
                                          ghi_test, "TEST", pv_cfg['nameplate_capacity_kw'])
    
    # ============================================================
    # PER-STEP R² DIAGNOSTICS
    # Shows how prediction quality degrades with forecast horizon
    # ============================================================
    per_step_r2 = {}
    for step_idx in range(horizon):
        step_actual = pv_test_actual[:, step_idx]
        step_pred = pv_test_pred[:, step_idx]
        if len(step_actual) > 10 and np.std(step_actual) > 0:
            per_step_r2[step_idx + 1] = float(r2_score(step_actual, step_pred))
    
    print(f"\n📊 Per-Step R² (Test):")
    for s in [1, 6, 12, 24]:
        if s in per_step_r2:
            print(f"  Step t+{s:2d}: R²={per_step_r2[s]:.4f}")
    
    # ============================================================
    # PER-HOUR-OF-DAY ERROR DIAGNOSTICS
    # Shows which clock hours are hardest to predict
    # ============================================================
    time_col = cfg['data']['time_col']
    hourly_errors = {}
    try:
        # Get timestamps for test predictions  
        test_timestamps = pd.to_datetime(df_test[time_col].values)
        for step_idx in range(horizon):
            # Get the actual clock hour for each prediction at this step
            pred_indices = test_idx_local + step_idx
            valid = pred_indices < len(test_timestamps)
            if valid.sum() == 0:
                continue
            hours = test_timestamps[pred_indices[valid]].hour
            actual_vals = pv_test_actual[valid, step_idx]
            pred_vals = pv_test_pred[valid, step_idx]
            errors = np.abs(actual_vals - pred_vals)
            
            for h, err, act, prd in zip(hours, errors, actual_vals, pred_vals):
                if h not in hourly_errors:
                    hourly_errors[h] = {'mae_sum': 0, 'count': 0, 
                                        'actual_sum': 0, 'pred_sum': 0,
                                        'sq_err_sum': 0, 'actual_list': [], 'pred_list': []}
                hourly_errors[h]['mae_sum'] += err
                hourly_errors[h]['sq_err_sum'] += err**2
                hourly_errors[h]['count'] += 1
                hourly_errors[h]['actual_sum'] += act
                hourly_errors[h]['pred_sum'] += prd
                hourly_errors[h]['actual_list'].append(act)
                hourly_errors[h]['pred_list'].append(prd)
        
        # Compute final per-hour metrics
        hourly_metrics = {}
        for h in sorted(hourly_errors.keys()):
            d = hourly_errors[h]
            n = d['count']
            h_mae = d['mae_sum'] / n
            h_rmse = np.sqrt(d['sq_err_sum'] / n)
            act_arr = np.array(d['actual_list'])
            prd_arr = np.array(d['pred_list'])
            h_r2 = float(r2_score(act_arr, prd_arr)) if np.std(act_arr) > 0 else 0.0
            hourly_metrics[int(h)] = {'mae': round(h_mae, 4), 'rmse': round(h_rmse, 4), 
                                       'r2': round(h_r2, 4), 'count': n}
        
        print(f"\n🕐 Per-Hour-of-Day Error (Test):")
        print(f"  {'Hour':>4} | {'MAE':>8} | {'RMSE':>8} | {'R²':>8} | {'N':>6}")
        print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
        for h in range(24):
            if h in hourly_metrics:
                m = hourly_metrics[h]
                print(f"  {h:4d} | {m['mae']:8.4f} | {m['rmse']:8.4f} | {m['r2']:8.4f} | {m['count']:6d}")
    except Exception as e:
        hourly_metrics = {}
        per_step_r2 = {}
        print(f"⚠️ Could not compute hourly diagnostics: {e}")

    return {
        'metrics_train': m_train, 'metrics_test': m_test,
        'metrics_train_productive': m_train_prod, 'metrics_test_productive': m_test_prod,
        'per_step_r2': per_step_r2,
        'hourly_metrics': hourly_metrics,
        'pv_train_actual': pv_train_actual, 'pv_train_pred': pv_train_pred,
        'pv_test_actual': pv_test_actual, 'pv_test_pred': pv_test_pred,
        'ghi_train': ghi_train, 'ghi_test': ghi_test,
        'df_train': df_train, 'df_test': df_test,
        'train_indices': train_idx_local, 'test_indices': test_idx_local,
    }




# ============================================================
# TARGET DOMAIN TESTING (Preprocessed Data)
# ============================================================
def test_on_preprocessed_target(model_path: str, target_dir: str, cfg: dict):
    """
    Menguji model yang sudah dilatih pada folder preprocessed data target.
    
    CRITICAL: Untuk transfer learning yang benar:
    - INPUT harus di-scale ulang menggunakan SOURCE X_scaler (dari model bundle)
    - OUTPUT harus di-inverse-transform menggunakan SOURCE y_scaler (dari model bundle)
    - Fitur harus diurutkan sesuai urutan yang dipakai saat training
    - Konversi CSI → kW menggunakan clear_sky TARGET (adaptif kapasitas)
    """
    from src.model_factory import get_custom_objects
    from src.data_prep import create_sequences_with_indices

    print(f"\n{'='*60}")
    print(f"TARGET DOMAIN TESTING (PREPROCESSED)")
    print(f"Model: {model_path}")
    print(f"Target Data Dir: {target_dir}")
    print(f"{'='*60}")

    model_root = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)

    # 1. Load model
    actual_model_path = model_path
    is_hf = os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'config.json'))
    
    if is_hf:
        from src.model_hf import load_hf_wrapper
        model = load_hf_wrapper(model_path)
    else:
        if os.path.isdir(model_path):
            for ext in ['.h5', '.keras']:
                candidate = os.path.join(model_path, f'model{ext}')
                if os.path.exists(candidate):
                    actual_model_path = candidate
                    break
        
        # WORKAROUND: .keras as HDF5
        import zipfile
        if actual_model_path.endswith('.keras') and not zipfile.is_zipfile(actual_model_path):
            print(f"[WARN] {os.path.basename(actual_model_path)} terdeteksi sebagai HDF5 file.")
            h5_path = actual_model_path.replace('.keras', '.h5')
            if not os.path.exists(h5_path):
                import shutil
                shutil.copy(actual_model_path, h5_path)
            actual_model_path = h5_path
            
        model_loaded = False
        try:
            model = tf.keras.models.load_model(actual_model_path, custom_objects=get_custom_objects(), safe_mode=False)
            model_loaded = True
        except Exception as load_err:
            if "bad marshal data" in str(load_err).lower() or "unknown type code" in str(load_err).lower():
                print(f"[RECOVER] Terjadi error marshal (Python version mismatch). Mencoba membangun ulang model...")
                from src.model_factory import build_model
                
                # Coba baca meta.json untuk dapet info architecture dll
                m_meta = {}
                meta_path = os.path.join(model_root, 'meta.json')
                if os.path.exists(meta_path):
                    import json
                    with open(meta_path, 'r') as f: m_meta = json.load(f)
                
                arch = m_meta.get('architecture', cfg['model']['architecture'])
                lb = m_meta.get('lookback', cfg['model']['hyperparameters']['lookback'])
                nf = m_meta.get('n_features', 0)
                hz = m_meta.get('horizon', cfg['forecasting']['horizon'])
                hp = m_meta.get('hyperparameters', cfg['model']['hyperparameters'])
                
                # Build fresh model
                model = build_model(arch, lb, nf, hz, hp)
                
                # Load weights only
                try:
                    model.load_weights(actual_model_path)
                    print(f"[OK] Model berhasil dibangun ulang dan bobot dimuat (BYPASS MARSHAL).")
                    model_loaded = True
                except Exception as w_err:
                    print(f"[ERR] Gagal memuat bobot: {w_err}")
                    raise load_err # Lempar error asli jika gagal
            else:
                raise load_err # Lempar error asli jika bukan masalah marshal
    
    from src.model_factory import fix_lambda_tf_refs
    if model_loaded:
        fix_lambda_tf_refs(model)
        print(f"Model loaded successfully.")
    else:
        raise ValueError("Model failed to load.")

    expected_n_features = model.input_shape[2] if hasattr(model, 'input_shape') else None
    lookback = model.input_shape[1] if hasattr(model, 'input_shape') else cfg['model']['hyperparameters']['lookback']
    horizon = model.output_shape[1] if hasattr(model, 'output_shape') else cfg['forecasting']['horizon']

    # 2. Load TARGET data (raw features, UNSCALED)
    import joblib
    try:
        df_test = pd.read_pickle(os.path.join(target_dir, 'df_test_feats.pkl'))
    except Exception as e:
        raise ValueError(f"Gagal memuat df_test_feats.pkl dari {target_dir}. Detail: {e}")

    # 3. Load SOURCE scalers (from MODEL bundle) - CRITICAL for correct transfer
    source_x_scaler_path = os.path.join(model_root, 'X_scaler.pkl')
    source_y_scaler_path = os.path.join(model_root, 'y_scaler.pkl')
    
    if not os.path.exists(source_x_scaler_path) or not os.path.exists(source_y_scaler_path):
        print("[WARN] Source scalers not found in model bundle. Falling back to target scalers.")
        print("       For accurate cross-domain results, ensure X_scaler.pkl & y_scaler.pkl are in the model folder.")
        source_x_scaler = joblib.load(os.path.join(target_dir, 'X_scaler.pkl'))
        source_y_scaler = joblib.load(os.path.join(target_dir, 'y_scaler.pkl'))
        using_source_scaler = False
    else:
        source_x_scaler = joblib.load(source_x_scaler_path)
        source_y_scaler = joblib.load(source_y_scaler_path)
        using_source_scaler = True
        print(f"[OK] Using SOURCE scalers from model bundle for correct cross-domain evaluation.")

    # 4. Determine feature ordering from SOURCE (model training)
    # Look for prep_summary.json or selected_features.json in model bundle
    source_features = None
    
    # Check model meta for data_source path
    meta_path = os.path.join(model_root, 'meta.json')
    if os.path.exists(meta_path):
        import json
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        data_source = meta.get('data_source', '')
        # Fallback: if absolute path doesn't exist (e.g., trained on different machine),
        # try relative path using the folder name
        if data_source and not os.path.exists(data_source):
            relative_ds = os.path.join('data', 'processed', os.path.basename(data_source))
            if os.path.exists(relative_ds):
                data_source = relative_ds
        # Try to load source features from the training data directory
        for candidate_path in [data_source, model_root]:
            for fname in ['prep_summary.json', 'selected_features.json']:
                fpath = os.path.join(candidate_path, fname)
                if os.path.exists(fpath):
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                    if fname == 'prep_summary.json':
                        source_features = data.get('selected_features')
                    else:
                        source_features = data
                    if source_features:
                        print(f"[OK] Source feature order loaded from: {fpath}")
                        break
            if source_features:
                break

    # Load target features order
    target_summary_path = os.path.join(target_dir, 'prep_summary.json')
    target_features = None
    use_csi = True
    if os.path.exists(target_summary_path):
        import json
        with open(target_summary_path, 'r') as f:
            summ = json.load(f)
        target_features = summ.get('selected_features')
        if summ.get('target_col') != 'csi_target':
            use_csi = False

    print(f"\n--- Feature Alignment ---")
    print(f"  Source features: {source_features}")
    print(f"  Target features: {target_features}")
    print(f"  Target mode: {'CSI' if use_csi else 'Direct Power'}")

    # 5. Determine target column for ground truth
    act_target = 'csi_target' if use_csi else cfg['data']['target_col']

    # 6. Re-scale target data using SOURCE X_scaler with correct feature ordering
    if source_features and target_features:
        # Check if all source features exist in target data
        missing = [f for f in source_features if f not in df_test.columns]
        if missing:
            raise ValueError(f"Target data missing features required by model: {missing}")
        
        # Extract features in SOURCE order
        X_raw = df_test[source_features].values
        print(f"  Re-ordering target features to match source: {source_features}")
    else:
        # Fallback: use whatever features are available
        if target_features:
            X_raw = df_test[target_features].values
        else:
            # Last resort: load pre-scaled X_test
            X_test = np.load(os.path.join(target_dir, 'X_test.npy'))
            print("[WARN] Could not determine feature order. Using pre-scaled X_test directly.")
            # In this fallback, we still need y data
            y_test = np.load(os.path.join(target_dir, 'y_test.npy'))
            source_features = target_features
            # Skip re-scaling, go directly to prediction
            X_raw = None
    
    if X_raw is not None:
        # Scale with SOURCE X_scaler
        X_scaled = source_x_scaler.transform(X_raw)
        
        # Get target values
        y_raw = df_test[act_target].values
        y_scaled = source_y_scaler.transform(y_raw.reshape(-1, 1)).flatten()
        
        # Create sequences
        print(f"\nCreating sequences (lookback={lookback}, horizon={horizon})...")
        X_test, y_test, test_indices = create_sequences_with_indices(
            X_scaled, y_scaled, df_test.index, lookback, horizon
        )
        print(f"  Sequences created: X_test={X_test.shape}, y_test={y_test.shape}")
    else:
        # Fallback path
        y_test = np.load(os.path.join(target_dir, 'y_test.npy'))
        _, _, test_indices = create_sequences_with_indices(
            np.zeros((len(df_test), 1)), np.zeros(len(df_test)),
            df_test.index, lookback, horizon
        )

    if X_test.shape[2] != expected_n_features:
        raise ValueError(f"Feature count mismatch after alignment: got {X_test.shape[2]}, model expects {expected_n_features}")

    # 7. Predict using model
    print("Running inference...")
    y_pred_scaled = safe_predict(model, X_test)
    
    # Inverse transform with SOURCE y_scaler (model's output space!)
    y_pred_raw = source_y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)

    # 8. Convert to physical units and calculate metrics
    pv_cfg = cfg['pv_system']
    capacity_kw = pv_cfg['nameplate_capacity_kw']

    steps = np.arange(horizon)
    n_test = min(len(test_indices), len(y_pred_raw))
    test_indices = test_indices[:n_test]
    y_pred_raw = y_pred_raw[:n_test]

    target_col = cfg['data']['target_col']
    if target_col not in df_test.columns:
        # Try common alternatives
        for alt in ['pv_output_kw', 'power_kw', 'pv_kw']:
            if alt in df_test.columns:
                target_col = alt
                break

    if use_csi:
        # CSI mode: CSI * clear_sky_target → power in target capacity
        # DYNAMIC RECALCUALTION: Re-calculate Clear Sky using CURRENT capacity 
        # to ensure scaling is correct even if preprocessed with wrong capacity.
        from src.data_prep import calculate_clear_sky_pv
        
        # Get weather columns needed for CS
        ghi_col = cfg['data']['ghi_col']
        poa_col = cfg['data'].get('poa_col', ghi_col)
        temp_col = cfg['data']['temp_col']
        wind_col = cfg['data'].get('wind_speed_col')
        
        # Recalculate full CS array for df_test
        current_pv_cs = calculate_clear_sky_pv(
            ghi=df_test[ghi_col].values,
            poa=df_test[poa_col].values if poa_col in df_test.columns else df_test[ghi_col].values,
            temp_ambient=df_test[temp_col].values,
            wind_speed=df_test[wind_col].values if wind_col and wind_col in df_test.columns else None,
            nameplate_capacity=capacity_kw,
            temp_coeff=pv_cfg['temp_coeff'],
            ref_temp=pv_cfg['ref_temp'],
            system_efficiency=pv_cfg['system_efficiency']
        )
        
        pv_cs_test = current_pv_cs[test_indices[:, np.newaxis] + steps]
        
        if len(y_pred_raw.shape) == 3 and y_pred_raw.shape[2] == 1:
            pv_test_pred = np.clip(y_pred_raw[:, :, 0] * pv_cs_test, 0, capacity_kw)
        else:
            pv_test_pred = np.clip(y_pred_raw * pv_cs_test, 0, capacity_kw)
        
        # Log difference to debug
        original_cs_max = df_test['pv_clear_sky'].max()
        print(f"[DEBUG] Recap CS: Original Max={original_cs_max:.2f}kW, Dynamic Max={current_pv_cs.max():.2f}kW")
    else:
        if len(y_pred_raw.shape) == 3 and y_pred_raw.shape[2] == 1:
            pv_test_pred = np.clip(y_pred_raw[:, :, 0], 0, capacity_kw * 1.2)
        else:
            pv_test_pred = np.clip(y_pred_raw, 0, capacity_kw * 1.2)

    pv_test_actual = df_test[target_col].values[test_indices[:, np.newaxis] + steps]
    
    # Apply productive hours mask
    ghi_col = cfg['data']['ghi_col']
    if ghi_col in df_test.columns:
        ghi_test = df_test[ghi_col].values[test_indices[:, np.newaxis] + steps]
    else:
        ghi_test = np.ones_like(pv_test_actual) * 1000

    wrapper_thresh = cfg.get('preprocessing', {}).get('productive_hours_threshold', 50)
    pv_test_pred[ghi_test < wrapper_thresh] = 0.0

    # 9. Calculate metrics
    print("Menghitung metrik akhir...")
    
    # Debug stats
    print(f"\n--- Debug Transfer Stats ---")
    print(f"  y_pred_raw (CSI) mean: {y_pred_raw.mean():.4f}, std: {y_pred_raw.std():.4f}")
    print(f"  pv_test_pred (kW) mean: {pv_test_pred.mean():.4f}, max: {pv_test_pred.max():.4f}")
    print(f"  pv_test_actual (kW) mean: {pv_test_actual.mean():.4f}, max: {pv_test_actual.max():.4f}")
    print(f"  Using source scaler: {using_source_scaler}")
    print(f"  Capacity: {capacity_kw} kW")
    
    m_test = calculate_full_metrics(pv_test_actual, pv_test_pred, None, f"TARGET ({os.path.basename(target_dir)})", capacity_kw)
    
    # NEW: Create a detailed dataframe for export and visualization
    # We take the 0-th step of the prediction for simplicity in time-series plotting
    # or flatten everything if needed. Usually, 1-step or first step is standard for a quick view.
    timestamps_test = df_test.index[test_indices + 1] # Approximately the first step of horizon
    
    df_results = pd.DataFrame({
        'Timestamp': timestamps_test,
        'Actual_kW': pv_test_actual[:, 0],
        'Predicted_kW': pv_test_pred[:, 0],
        'Error_kW': pv_test_actual[:, 0] - pv_test_pred[:, 0]
    })
    
    print(f"\n[OK] Target domain testing selesai!")
    return {
        'metrics': m_test,
        'df_results': df_results
    }
