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
    """Predict menggunakan generator agar aman untuk GPU memory."""
    def input_generator():
        for i in range(len(X)):
            yield X[i]

    dataset = tf.data.Dataset.from_generator(
        input_generator,
        output_signature=tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return model.predict(dataset, verbose=1)


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

    nonzero = act > 0.01
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((act[nonzero] - pred[nonzero]) / act[nonzero])) * 100
    else:
        mape = 0.0

    norm_mae = mae / capacity_kw if capacity_kw > 0 else 0
    norm_rmse = rmse / capacity_kw if capacity_kw > 0 else 0

    print(f"\n>>> {label} <<<")
    print(f"  Absolute: MAE={mae:.4f} kW, RMSE={rmse:.4f} kW, R2={r2:.4f}, MAPE={mape:.2f}%")
    print(f"  Normalized: NormMAE={norm_mae:.4f} ({norm_mae*100:.2f}%), NormRMSE={norm_rmse:.4f}")

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
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

    # Load data if needed (Data always comes from the ACTIVE processed dir)
    if data is None:
        data = {}
        
    try:
        X_train = data.get('X_train', np.load(os.path.join(root_proc, 'X_train.npy')))
        X_test = data.get('X_test', np.load(os.path.join(root_proc, 'X_test.npy')))
        
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
        
        # Dataframes are usually too big for bundles, so we stick to root_proc for these
        df_train = data.get('df_train', pd.read_pickle(os.path.join(root_proc, 'df_train_feats.pkl')))
        df_test = data.get('df_test', pd.read_pickle(os.path.join(root_proc, 'df_test_feats.pkl')))
    except Exception as e:
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
        summary_path = os.path.join(root_proc, 'prep_summary.json')
        
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
        pv_train_pred = np.clip(y_train_pred_raw * pv_cs_train, 0, pv_cfg['nameplate_capacity_kw'])
        pv_test_pred = np.clip(y_test_pred_raw * pv_cs_test, 0, pv_cfg['nameplate_capacity_kw'])
    else:
        # Already power
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

    # Calculate metrics
    print("\nFINAL PERFORMANCE ANALYSIS")
    m_train = calculate_full_metrics(pv_train_actual, pv_train_pred,
                                      ghi_train, "TRAIN", pv_cfg['nameplate_capacity_kw'])
    m_test = calculate_full_metrics(pv_test_actual, pv_test_pred,
                                     ghi_test, "TEST", pv_cfg['nameplate_capacity_kw'])

    return {
        'metrics_train': m_train, 'metrics_test': m_test,
        'pv_train_actual': pv_train_actual, 'pv_train_pred': pv_train_pred,
        'pv_test_actual': pv_test_actual, 'pv_test_pred': pv_test_pred,
        'ghi_train': ghi_train, 'ghi_test': ghi_test,
        'df_train': df_train, 'df_test': df_test,
        'train_indices': train_idx_local, 'test_indices': test_idx_local,
    }


# ============================================================
# TARGET DOMAIN TESTING
# ============================================================
def test_on_target(model_path: str, target_csv: str, cfg: dict):
    """
    Menguji model yang sudah dilatih pada data target (Indonesia).
    Hanya membutuhkan: model .keras/.h5, scalers, dan CSV data target.
    """
    from src.model_factory import get_custom_objects
    from src.data_prep import preprocess_algorithm1, create_features, create_sequences_with_indices

    proc = cfg['paths']['processed_dir']
    pv_cfg = cfg['pv_system']
    horizon = cfg['forecasting']['horizon']
    lookback = cfg['model']['hyperparameters']['lookback']

    print(f"\n{'='*60}")
    print(f"TARGET DOMAIN TESTING")
    print(f"Model: {model_path}")
    print(f"Data:  {target_csv}")
    print(f"{'='*60}")

    # 1. Load model
    model = tf.keras.models.load_model(model_path, custom_objects=get_custom_objects())
    print(f"Model loaded: {model.name}")

    # 2. Load scalers
    X_scaler = joblib.load(os.path.join(proc, 'X_scaler.pkl'))
    y_scaler = joblib.load(os.path.join(proc, 'y_scaler.pkl'))

    # 3. Load & preprocess target data
    df_target = pd.read_csv(target_csv, sep=cfg['data']['csv_separator'])
    df_target[cfg['data']['time_col']] = pd.to_datetime(
        df_target[cfg['data']['time_col']], format=cfg['data'].get('time_format'))
    df_target_clean = preprocess_algorithm1(df_target, cfg)

    # 4. Create features
    df_target_feats = create_features(df_target_clean, cfg).dropna()

    # 5. Load feature list
    with open(os.path.join(proc, 'selected_features.json'), 'r') as f:
        selected_features = json.load(f)

    # 6. Scale
    X_target_scaled = X_scaler.transform(df_target_feats[selected_features])
    act_target = 'csi_target' if cfg['target']['use_csi'] else cfg['data']['target_col']
    y_target_scaled = y_scaler.transform(df_target_feats[[act_target]]).flatten()

    # 7. Create sequences
    X_seq, y_seq, indices = create_sequences_with_indices(
        X_target_scaled, y_target_scaled, df_target_feats.index, lookback, horizon)
    print(f"Target sequences: {X_seq.shape}")

    # 8. Predict & evaluate
    y_pred_scaled = safe_predict(model, X_seq)
    y_pred_csi = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)

    steps = np.arange(horizon)
    pv_cs = df_target_feats['pv_clear_sky'].values[indices[:, np.newaxis] + steps]
    pv_pred = np.clip(y_pred_csi * pv_cs, 0, pv_cfg['nameplate_capacity_kw'])
    pv_actual = df_target_feats[cfg['data']['target_col']].values[indices[:, np.newaxis] + steps]
    ghi = df_target_feats[cfg['data']['ghi_col']].values[indices[:, np.newaxis] + steps]

    metrics = calculate_full_metrics(pv_actual, pv_pred, ghi, "TARGET",
                                      pv_cfg['nameplate_capacity_kw'])

    print(f"\n✅ Target domain testing selesai!")
    return metrics
