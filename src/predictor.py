"""
predictor.py - Inferensi, Evaluasi Metrik, dan Target Domain Testing.
Berisi: safe_predict, calculate_full_metrics, CSI->Power transform, target testing.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import logging
import sys
import copy

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ============================================================
# NUMPY CROSS-VERSION PICKLE COMPATIBILITY
# ============================================================
# NumPy 2.0+ renamed numpy.core -> numpy._core internally.
# Pickles saved with NumPy 2.x contain 'numpy._core.xxx' references
# that fail on NumPy 1.x (and vice versa). This custom unpickler
# transparently remaps module paths at load time.
# ============================================================
# Determine NumPy version ONCE at import time (before any unpickling side effects)
_NUMPY_MAJOR_VERSION = int(np.__version__.split('.')[0])

class _NumpyCompatUnpickler(pickle.Unpickler):
    """Custom unpickler that handles numpy.core <-> numpy._core renames."""
    def find_class(self, module, name):
        if _NUMPY_MAJOR_VERSION < 2:
            # Running NumPy 1.x: remap numpy._core.xxx -> numpy.core.xxx
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core', 1)
        else:
            # Running NumPy 2.x: remap numpy.core.xxx -> numpy._core.xxx
            if module.startswith('numpy.core'):
                module = module.replace('numpy.core', 'numpy._core', 1)
        return super().find_class(module, name)


def safe_read_pickle(path):
    """Read a pickle file with cross-version NumPy compatibility.
    
    Uses a custom unpickler to handle numpy.core <-> numpy._core
    module renames between NumPy 1.x and 2.x.
    """
    try:
        # First try standard pandas read_pickle (fastest)
        return pd.read_pickle(path)
    except (ModuleNotFoundError, ImportError) as e:
        if 'numpy' in str(e):
            logger.info(f"Standard pickle failed ({e}), using NumPy-compat unpickler for {path}")
            with open(path, 'rb') as f:
                return _NumpyCompatUnpickler(f).load()
        raise


# ============================================================
# SAFE PREDICT (GPU Memory-safe)
# ============================================================
def safe_predict(model, X, batch_size=32):
    """
    Prediksi yang lebih stabil untuk menghindari OOM atau iterator crash.
    Menerapkan strategi dynamic batch reduction jika terjadi ResourceExhaustedError.
    """
    import gc
    
    # Auto-detect Autoformer or extremely large input/lookback
    model_name = getattr(model, 'name', 'unknown').lower()
    is_autoformer = 'autoformer' in model_name or (hasattr(model, 'model') and 'autoformer' in getattr(model.model, 'name', '').lower())
    
    # Pre-emptive reduction for memory-heavy architectures
    if is_autoformer and batch_size > 8:
        print(f"   [Auto-Scale] Autoformer detected. Reducing batch size {batch_size} -> 8 for VRAM safety.")
        batch_size = 8

    # Attempt recursive batch reduction strategy
    current_batch = batch_size
    while current_batch >= 1:
        try:
            # 1. Try standard Keras predict
            if hasattr(model, 'predict'):
                # Handle HFModelWrapper which might not like batch_size in its custom predict
                if model_name == "hf_model_wrapper" or hasattr(model, '_is_hf'):
                    return model.predict(X, verbose=0)
                else:
                    return model.predict(X, batch_size=current_batch, verbose=0)
            
            # 2. Try calling model as function (e.g. TF Graph or Layer)
            return model(X, training=False).numpy()
            
        except (tf.errors.ResourceExhaustedError, Exception) as e:
            err_msg = str(e).lower()
            is_oom = "oom" in err_msg or "resource exhausted" in err_msg or "allocation" in err_msg
            
            if is_oom and current_batch > 1:
                new_batch = max(1, current_batch // 4)
                print(f"⚠️ [VRAM OOM] Gagal pada batch {current_batch}. Mencoba kembali dengan batch {new_batch}...")
                current_batch = new_batch
                # Force cleanup
                tf.keras.backend.clear_session()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            
            # If it's batch=1 and still OOM, try CPU fallback
            if is_oom and current_batch == 1:
                print(f"🚨 [VRAM CRITICAL] Batch 1 pun OOM. Mencoba paksa ke CPU...")
                with tf.device('/CPU:0'):
                    # We might need to handle the model transfer or just basic np predict
                    try:
                        # For Keras, calling with tf.device should work
                        p = model(X, training=False)
                        return p.numpy() if hasattr(p, 'numpy') else p
                    except:
                        pass
            
            # If not OOM or CPU fallback failed, raise the original error or handle manual fallback
            print(f"[ERROR] Prediksi gagal total: {e}")
            break

    # Manual line-by-line fallback as last resort (Very slow but guaranteed safety)
    print("   [Fallback] Menggunakan manual batching (batch=1)...")
    preds = []
    for i in range(len(X)):
        xb = X[i:i+1]
        try:
            p = model(xb, training=False)
            preds.append(p.numpy() if hasattr(p, 'numpy') else p)
        except:
            p = model.predict(xb, verbose=0)
            preds.append(p)
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
    pv_cfg = copy.deepcopy(cfg['pv_system'])
    found_orig_pv = False
    
    # Detect horizon from model instead of relying solely on config
    horizon = model.output_shape[1]
    lookback = model.input_shape[1]
    
    print(f"Evaluation Context: lookback={lookback}, horizon={horizon}")
    print(f"Using Artifacts from: {proc}")

    # Load data if needed
    if data is None:
        data = {}
        
    # --- IDENTIFY TEST DATA AND MODEL ORIGIN ---
    data_proc_src = root_proc # TEST DATA (Where we evaluate)
    model_origin_src = proc   # MODEL METADATA (Where we get scalers/features)
    
    if scaler_dir and os.path.exists(os.path.join(scaler_dir, 'meta.json')):
        try:
            with open(os.path.join(scaler_dir, 'meta.json'), 'r') as f:
                m_meta = json.load(f)
                orig_ds = m_meta.get('data_source', '').replace('\\', '/')
                if orig_ds:
                    if os.path.exists(orig_ds): model_origin_src = orig_ds
                    else:
                        leaf = os.path.basename(orig_ds.rstrip('\\/'))
                        if leaf.lower() in ['processed', 'data']:
                            parts = orig_ds.rstrip('\\/').split('/')
                            if len(parts) >= 2: leaf = parts[-2]
                        search_root = os.path.abspath(root_proc)
                        for _ in range(3): search_root = os.path.dirname(search_root)
                        for root, _, files in os.walk(search_root):
                            if os.path.basename(root) == leaf and ('X_train.npy' in files or 'df_train_feats.pkl' in files):
                                model_origin_src = root.replace('\\', '/')
                                break
                if 'pv_system' in m_meta: pv_cfg.update(m_meta['pv_system'])
        except: pass

    try:
        expected_n_features = model.input_shape[2] if hasattr(model, 'input_shape') else None
        lookback = model.input_shape[1] if hasattr(model, 'input_shape') else cfg['model']['hyperparameters']['lookback']
        horizon = model.output_shape[1] if hasattr(model, 'output_shape') else cfg['forecasting']['horizon']

        X_train, X_test = None, None
        train_indices, test_indices = None, None
        data_reconstructed = False
        
        # Helper to get feature lists
        def _get_ft_list(folder):
            fp = os.path.join(folder, 'selected_features.json').replace('\\', '/')
            if os.path.exists(fp):
                try:
                    with open(fp, 'r') as _f: return json.load(_f)
                except: pass
            pp = os.path.join(folder, 'prep_summary.json').replace('\\', '/')
            if os.path.exists(pp):
                try:
                    with open(pp, 'r') as _f: return json.load(_f).get('selected_features')
                except: pass
            return None

        model_feat_list = _get_ft_list(proc)
        if not model_feat_list: model_feat_list = _get_ft_list(model_origin_src)
        data_feat_list = _get_ft_list(data_proc_src)

        # Do we need reconstruction?
        need_reconstruction = False
        
        npy_path = os.path.join(data_proc_src, 'X_train.npy').replace('\\', '/')
        idx_tr_p = os.path.join(data_proc_src, 'train_indices.npy').replace('\\', '/')
        idx_ts_p = os.path.join(data_proc_src, 'test_indices.npy').replace('\\', '/')
        
        if not (os.path.exists(npy_path) and os.path.exists(idx_tr_p) and os.path.exists(idx_ts_p)):
            need_reconstruction = True
        else:
            # Check shape compatibility
            tmp_X = np.load(npy_path, mmap_mode='r')
            if tmp_X.shape[1] != lookback:
                need_reconstruction = True
            
            # Check feature matching
            if model_feat_list and data_feat_list and (model_feat_list != data_feat_list):
                # If feature counts match but order/names differ, NPY is incorrect. Force reconstruction.
                need_reconstruction = True

        df_train = data.get('df_train')
        if df_train is None:
            tr_p = os.path.join(data_proc_src, 'df_train_feats.pkl').replace('\\', '/')
            df_train = pd.read_pickle(tr_p) if os.path.exists(tr_p) else None
            
        df_test = data.get('df_test')
        if df_test is None:
            ts_p = os.path.join(data_proc_src, 'df_test_feats.pkl').replace('\\', '/')
            df_test = pd.read_pickle(ts_p) if os.path.exists(ts_p) else None

        if not need_reconstruction and data.get('X_train') is None:
            # Use pre-saved NPY
            X_train = np.load(npy_path)
            X_test = np.load(os.path.join(data_proc_src, 'X_test.npy').replace('\\', '/'))
            train_indices = np.load(idx_tr_p)
            test_indices = np.load(idx_ts_p)
            print(f"✅ Synced NPY loaded: {data_proc_src}")
        elif data.get('X_train') is None:
            # Reconstruct from PKL
            print(f"🚀 Syncing: Reconstructing sequences from .pkl in {data_proc_src}...")
            sc_path = os.path.join(proc, 'X_scaler.pkl').replace('\\', '/')
            if not os.path.exists(sc_path): sc_path = os.path.join(model_origin_src, 'X_scaler.pkl').replace('\\', '/')
            
            if df_train is not None and df_test is not None and os.path.exists(sc_path):
                x_scaler = joblib.load(sc_path)
                
                ft_list = model_feat_list if model_feat_list else data_feat_list
                y_c = 'csi_target' if 'csi_target' in df_train.columns else cfg['data']['target_col']
                
                if ft_list: 
                    valid_ft = [f for f in ft_list if f in df_train.columns]
                    if len(valid_ft) != len(ft_list):
                        missing = set(ft_list) - set(valid_ft)
                        raise ValueError(f"Target data ({os.path.basename(data_proc_src)}) lacks features required by model: {missing}")
                    X_tr_df, X_ts_df = df_train[valid_ft], df_test[valid_ft]
                else: 
                    X_tr_df = df_train.select_dtypes(include=[np.number]).drop(columns=[y_c] if y_c in df_train.columns else [])
                    X_ts_df = df_test.select_dtypes(include=[np.number]).drop(columns=[y_c] if y_c in df_test.columns else [])
                
                X_train_scaled, X_test_scaled = x_scaler.transform(X_tr_df), x_scaler.transform(X_ts_df)
                
                from src.data_prep import create_sequences_with_indices
                X_train, _, train_indices = create_sequences_with_indices(X_train_scaled, np.zeros(len(df_train)), df_train.index, lookback, horizon)
                X_test, _, test_indices = create_sequences_with_indices(X_test_scaled, np.zeros(len(df_test)), df_test.index, lookback, horizon)
                data_reconstructed = True
            else: 
                raise ValueError(f"Missing essential files in {data_proc_src} or {model_origin_src}")

        if data.get('X_train') is not None:
            X_train = data['X_train']
            X_test = data['X_test']
            train_indices = data.get('train_indices')
            test_indices = data.get('test_indices')

        actual_n_features = X_train.shape[2]
        if expected_n_features and expected_n_features != actual_n_features:
            raise ValueError(f"Feature count mismatch: Model expects {expected_n_features}, Data has {actual_n_features}.")

        # Label/Inverse Scaler MUST come from origin
        y_sc_p = os.path.join(proc, 'y_scaler.pkl').replace('\\', '/')
        if not os.path.exists(y_sc_p): y_sc_p = os.path.join(model_origin_src, 'y_scaler.pkl').replace('\\', '/')
        y_scaler = data.get('y_scaler', joblib.load(y_sc_p))
    except Exception as e:
        if isinstance(e, ValueError): raise e
        raise ValueError(f"Gagal memuat data preprocessing: {e}. Pastikan sudah menjalankan Step 1.")

    # --- INFERENCE ---
    import time
    
    start_time = time.time()
    y_ts_s = safe_predict(model, X_test)
    end_time = time.time()
    inf_time_ms = ((end_time - start_time) / max(1, len(X_test))) * 1000
    
    y_tr_s = safe_predict(model, X_train)
    
    y_tr_r = y_scaler.inverse_transform(y_tr_s.reshape(-1, 1)).reshape(y_tr_s.shape)
    y_ts_r = y_scaler.inverse_transform(y_ts_s.reshape(-1, 1)).reshape(y_ts_s.shape)

    # --- CONVERSION ---
    steps = np.arange(horizon)
    use_csi = True
    sum_p = os.path.join(proc, 'prep_summary.json')
    if not os.path.exists(sum_p): sum_p = os.path.join(data_proc_src, 'prep_summary.json')
    if os.path.exists(sum_p):
        with open(sum_p, 'r') as f:
            if json.load(f).get('target_col') != 'csi_target': use_csi = False

    if use_csi:
        from src.data_prep import calculate_clear_sky_pv
        def get_cs(df):
            if 'pv_clear_sky' in df.columns:
                return df['pv_clear_sky'].values
                
            poa_c = cfg['data'].get('poa_col', cfg['data']['ghi_col'])
            if poa_c not in df.columns: poa_c = cfg['data']['ghi_col']
            wind_c = cfg['data'].get('wind_speed_col')
            if wind_c not in df.columns: wind_c = None
            
            return calculate_clear_sky_pv(df[cfg['data']['ghi_col']].values, df[poa_c].values,
                                          df[cfg['data']['temp_col']].values, df[wind_c].values if wind_c else None,
                                          pv_cfg['nameplate_capacity_kw'], pv_cfg.get('temp_coeff', -0.004), pv_cfg.get('ref_temp', 25.0), pv_cfg.get('system_efficiency', 0.85))
        
        cs_arr_tr = get_cs(df_train)
        cs_arr_ts = get_cs(df_test)
        cs_tr = cs_arr_tr[train_indices[:, np.newaxis] + steps]
        cs_ts = cs_arr_ts[test_indices[:, np.newaxis] + steps]
        pv_tr_p = np.clip(y_tr_r[:,:,0] if len(y_tr_r.shape)==3 else y_tr_r, 0, 1.2) * cs_tr
        pv_ts_p = np.clip(y_ts_r[:,:,0] if len(y_ts_r.shape)==3 else y_ts_r, 0, 1.2) * cs_ts
        
        # Robustly guess original capacity to avoid nMAE / nRMSE exploding
        # Clear Sky maximum is practically identical to effective Nameplate Capacity
        eval_capacity_kw = max(np.max(cs_arr_tr), np.max(cs_arr_ts))
    else:
        eval_capacity_kw = pv_cfg['nameplate_capacity_kw']
        pv_tr_p = np.clip(y_tr_r, 0, eval_capacity_kw * 1.2)
        pv_ts_p = np.clip(y_ts_r, 0, eval_capacity_kw * 1.2)

    t_col = cfg['data']['target_col']
    # Fallback to whatever is available in the dataframe if the config one doesn't exist but the other does
    if t_col not in df_train.columns and 'pv_output_kw' in df_train.columns: t_col = 'pv_output_kw'
    
    pv_tr_a = df_train[t_col].values[train_indices[:, np.newaxis] + steps]
    pv_ts_a = df_test[t_col].values[test_indices[:, np.newaxis] + steps]
    ghi_ts = df_test[cfg['data']['ghi_col']].values[test_indices[:, np.newaxis] + steps]
    pv_ts_p[ghi_ts < cfg.get('preprocessing', {}).get('productive_hours_threshold', 50)] = 0

    m_train = calculate_full_metrics(pv_tr_a, pv_tr_p, None, "TRAIN", eval_capacity_kw)
    m_test = calculate_full_metrics(pv_ts_a, pv_ts_p, None, "TEST", eval_capacity_kw)

    return {
        'metrics_train': m_train, 'metrics_test': m_test,
        'pv_train_actual': pv_tr_a, 'pv_train_pred': pv_tr_p,
        'pv_test_actual': pv_ts_a, 'pv_test_pred': pv_ts_p,
        'ghi_test': ghi_ts,
        'df_train': df_train, 'df_test': df_test,
        'train_indices': train_indices, 'test_indices': test_indices,
        'data_path': f"{os.path.basename(data_proc_src)}{' (Synced)' if data_reconstructed else ''}",
        'history': getattr(model, 'history', None),
        'inference_time_ms': inf_time_ms
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
            for candidate in [
                os.path.join(model_path, f'model.keras'),
                os.path.join(model_path, f'model.h5'),
                os.path.join(model_path, f'pytorch_model.bin'),
                model_path  # The directory itself might be a SavedModel (contains saved_model.pb)
            ]:
                if os.path.exists(candidate):
                    if candidate == model_path and not os.path.exists(os.path.join(model_path, 'saved_model.pb')):
                        continue # If directory and no saved_model.pb, try next
                    actual_model_path = candidate
                    
                    # If it's a pytorch bin but config.json is missing, load_hf_wrapper won't trigger above.
                    # We should force HF load here to be safe.
                    if candidate.endswith('.bin') and not is_hf:
                        is_hf = True
                        from src.model_hf import load_hf_wrapper
                        model = load_hf_wrapper(model_path)
                    
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
            
        model_loaded = is_hf # Already loaded above if HF
        
        if not is_hf:
            try:
                try:
                    model = tf.keras.models.load_model(actual_model_path, custom_objects=get_custom_objects(), safe_mode=False)
                except TypeError:
                    model = tf.keras.models.load_model(actual_model_path, custom_objects=get_custom_objects())
                model_loaded = True
            except Exception as load_err:
                err_str = str(load_err).lower()
                
                # CASE 1: Keras 3 ZIP format or OneDrive access issue
                if "signature not found" in err_str or "unable to synchronously open" in err_str or "invalid argument" in err_str:
                    import zipfile as zf
                    if actual_model_path.endswith('.keras') and zf.is_zipfile(actual_model_path):
                        print(f"[RECOVER] Keras 3 ZIP format terdeteksi. Mengekstrak dan membangun ulang...")
                        import json as _json
                        from src.model_factory import build_model, compile_model, manual_load_k3_weights
                        
                        extract_dir = os.path.join(model_root, '_extracted_k3')
                        os.makedirs(extract_dir, exist_ok=True)
                        with zf.ZipFile(actual_model_path, 'r') as z:
                            z.extractall(extract_dir)
                        
                        weights_h5 = os.path.join(extract_dir, 'model.weights.h5')
                        
                        # Read meta.json
                        m_meta = {}
                        meta_path = os.path.join(model_root, 'meta.json')
                        if os.path.exists(meta_path):
                            try:
                                with open(meta_path, 'r', encoding='utf-8') as f: m_meta = _json.load(f)
                            except: pass
                        
                        arch = m_meta.get('architecture', cfg['model']['architecture'])
                        lb = m_meta.get('lookback', cfg['model']['hyperparameters']['lookback'])
                        nf = m_meta.get('n_features', 0)
                        hz = m_meta.get('horizon', cfg['forecasting']['horizon'])
                        hp = m_meta.get('hyperparameters', cfg['model']['hyperparameters'])
                        
                        if nf == 0:
                            prep_path = os.path.join(model_root, 'prep_summary.json')
                            if os.path.exists(prep_path):
                                try:
                                    with open(prep_path, 'r') as f:
                                        p_meta = _json.load(f)
                                        if n_f := p_meta.get('n_features'): nf = n_f
                                        if l_b := p_meta.get('lookback'): lb = l_b
                                        if h_z := p_meta.get('horizon'): hz = h_z
                                except: pass
                        
                        if nf == 0:
                            raise ValueError(f"Rebuild gagal: n_features=0 di meta.json. Lokasi: {model_root}")
                            
                        model = build_model(arch, lb, nf, hz, hp)
                        compile_model(model, hp.get('learning_rate', 0.001))
                        
                        if os.path.exists(weights_h5):
                            manual_load_k3_weights(model, weights_h5)
                            print(f"[OK] Model rebuild berhasil (Keras 3 ZIP).")
                            model_loaded = True
                    else:
                        # Truly unreadable
                        raise ValueError(
                            f"File model '{os.path.basename(actual_model_path)}' tidak dapat dibaca. "
                            f"Coba: Klik Kanan folder '{os.path.basename(model_root)}' -> 'Always keep on this device'."
                        )
                
                # CASE 2: Python version mismatch (marshal error)
                elif "bad marshal data" in err_str or "unknown type code" in err_str:
                    print(f"[RECOVER] Terjadi error marshal. Mencoba membangun ulang model...")
                    from src.model_factory import build_model, manual_load_k3_weights
                    
                    # ... reuse m_meta logic ...
                    m_meta = {}
                    meta_path = os.path.join(model_root, 'meta.json')
                    if os.path.exists(meta_path):
                        import json as _json
                        with open(meta_path, 'r', encoding='utf-8') as f: m_meta = _json.load(f)
                    
                    arch = m_meta.get('architecture', cfg['model']['architecture'])
                    lb = m_meta.get('lookback', cfg['model']['hyperparameters']['lookback'])
                    nf = m_meta.get('n_features', 0)
                    hz = m_meta.get('horizon', cfg['forecasting']['horizon'])
                    hp = m_meta.get('hyperparameters', cfg['model']['hyperparameters'])
                    
                    if nf == 0:
                        prep_path = os.path.join(model_root, 'prep_summary.json')
                        if os.path.exists(prep_path):
                            try:
                                import json as _json
                                with open(prep_path, 'r') as f:
                                    p_m = _json.load(f)
                                    if n_f := p_m.get('n_features'): nf = n_f
                            except: pass
                    
                    model = build_model(arch, lb, nf, hz, hp)
                    manual_load_k3_weights(model, actual_model_path)
                    model_loaded = True
                else:
                    raise load_err
                
        if not is_hf:
            from src.model_factory import fix_lambda_tf_refs
            if model_loaded:
                fix_lambda_tf_refs(model)
            else:
                raise ValueError("Gagal memuat model.")

    # Read meta.json for model shape info (used as fallback, especially for HF models)
    import json as _json_meta
    meta_info = {}
    meta_json_path = os.path.join(model_root, 'meta.json')
    if os.path.exists(meta_json_path):
        try:
            with open(meta_json_path, 'r', encoding='utf-8') as f:
                meta_info = _json_meta.load(f)
        except: pass
    
    if is_hf:
        # HF models don't have .input_shape / .output_shape
        expected_n_features = meta_info.get('n_features', None)
        # lookback may be at top-level or inside hyperparameters
        lookback = meta_info.get('lookback', 
                     meta_info.get('hyperparameters', {}).get('lookback', 
                       cfg['model']['hyperparameters']['lookback']))
        # horizon may be 'forecast_horizon' or 'horizon'
        horizon = meta_info.get('forecast_horizon', 
                    meta_info.get('horizon', 
                      cfg['forecasting']['horizon']))
    else:
        expected_n_features = model.input_shape[2] if hasattr(model, 'input_shape') else None
        lookback = model.input_shape[1] if hasattr(model, 'input_shape') else cfg['model']['hyperparameters']['lookback']
        horizon = model.output_shape[1] if hasattr(model, 'output_shape') else cfg['forecasting']['horizon']

    # 2. Load TARGET data (raw features, UNSCALED)
    import joblib
    try:
        df_test = safe_read_pickle(os.path.join(target_dir, 'df_test_feats.pkl'))
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
    import time
    start_time = time.time()
    y_pred_scaled = safe_predict(model, X_test)
    inference_time = time.time() - start_time
    
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
    print(f"  Inference Time: {inference_time:.4f}s")
    
    m_test = calculate_full_metrics(pv_test_actual, pv_test_pred, None, f"TARGET ({os.path.basename(target_dir)})", capacity_kw)
    
    # NEW: Prepare data for export and multi-step visualization
    timestamps_test = df_test.index[test_indices]
    
    print(f"\n[OK] Target domain testing selesai!")
    return {
        'metrics': m_test,
        'inference_time': inference_time,
        'timestamps': timestamps_test,
        'actual_full': pv_test_actual,
        'pred_full': pv_test_pred,
        'horizon': horizon
    }
