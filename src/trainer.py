"""
trainer.py - Logic untuk Training, Optuna Tuning, dan TSCV.
"""
import os
import gc
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import logging
import joblib

logger = logging.getLogger(__name__)
# Ensure basic logging is configured if not already
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _get_callbacks(cfg: dict):
    """Membuat callback standar (EarlyStopping + ReduceLROnPlateau)."""
    t = cfg['training']
    return [
        EarlyStopping(
            monitor='val_loss', patience=t['patience'],
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=t['lr_factor'],
            patience=t['lr_patience'], min_lr=t['min_lr'], verbose=1
        ),
    ]


# ============================================================
# STANDARD TRAINING
# ============================================================
def train_model(cfg: dict, data: dict = None, extra_callbacks: list = None, custom_model_id: str = None, loss_fn: str = 'mse'):
    """Melatih model dengan hyperparameters dari config."""
    from src.model_factory import build_model, compile_model

    hp = cfg['model']['hyperparameters']
    arch = cfg['model']['architecture']
    horizon = cfg['forecasting']['horizon']
    lookback = hp['lookback']

    # Load data jika belum di-pass
    if data is None:
        proc = cfg['paths']['processed_dir']
        print(f"📂 Loading data from {proc}...")
        import sys
        sys.stdout.flush()
        
        data = {
            'X_train': np.load(os.path.join(proc, 'X_train.npy')),
            'y_train': np.load(os.path.join(proc, 'y_train.npy')),
            'X_test': np.load(os.path.join(proc, 'X_test.npy')),
            'y_test': np.load(os.path.join(proc, 'y_test.npy')),
            'n_features': None,  # akan dideteksi dari shape
        }
    
    print(f"📊 Data shapes: X={data['X_train'].shape}, y={data['y_train'].shape}")
    print(f"Using hyperparameters: {hp}")
    print(f"Architecture: {arch}")
    sys.stdout.flush()

    # Handle lookback mismatch (Lookback in config vs Lookback in preprocessed data)
    actual_data_lookback = data['X_train'].shape[1]
    if lookback != actual_data_lookback:
        print(f"⚠️ Lookback mismatch: Config wants {lookback}, Data has {actual_data_lookback}.")
        if lookback < actual_data_lookback:
            print(f"   Truncating data sequences to last {lookback} steps...")
            data['X_train'] = data['X_train'][:, -lookback:, :]
            data['X_test'] = data['X_test'][:, -lookback:, :]
        else:
            print(f"   ⚠️ Data lookback ({actual_data_lookback}) is smaller than config lookback ({lookback}).")
            print(f"   Falling back to data lookback: {actual_data_lookback}")
            lookback = actual_data_lookback
            hp['lookback'] = lookback

    n_features = data.get('n_features') or data['X_train'].shape[2]
    model = build_model(arch, lookback, n_features, horizon, hp)
    compile_model(model, hp['learning_rate'], loss_fn=loss_fn)
    model.summary()

    cbs = _get_callbacks(cfg)
    if extra_callbacks:
        cbs.extend(extra_callbacks)

    import time
    start_time = time.time()
    
    # Create model folder for bundling
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_id = custom_model_id if custom_model_id else f"{arch}_{timestamp}"
    
    root_model_dir = cfg['paths']['models_dir']
    model_folder = os.path.join(root_model_dir, model_id)
    os.makedirs(model_folder, exist_ok=True)

    print(f"\n🚀 Start Training Model: {model_id}")
    print(f"   (Please wait for the first epoch log, it may take a moment...)")
    import sys
    sys.stdout.flush()

    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_test'], data['y_test']),
        epochs=cfg['training']['epochs'],
        batch_size=hp['batch_size'],
        callbacks=cbs,
        verbose=2 # One line per epoch is more stable for CMD logging
    )
    
    end_time = time.time()
    training_duration = end_time - start_time

    # Save model
    model_path = os.path.join(model_folder, "model.keras")
    try:
        model.save(model_path)
    except Exception as e:
        print(f"⚠️ Gagal menyimpan dalam format .keras ({e}), mencoba .h5...")
        model_path = os.path.join(model_folder, "model.h5")
        model.save(model_path)

    # Copy Scalers from processed_dir to bundle (CRITICAL for Transfer Learning)
    import shutil
    proc_dir = cfg['paths']['processed_dir']
    for s_name in ['X_scaler.pkl', 'y_scaler.pkl', 'selected_features.json']:
        s_src = os.path.join(proc_dir, s_name)
        if os.path.exists(s_src):
            shutil.copy(s_src, os.path.join(model_folder, s_name))
            print(f"📦 Artifact '{s_name}' dipaketkan bersama model.")

    print(f"\n✅ Model Bundle disimpan di: {model_folder}")
    print(f"⏱️ Time Elapsed: {training_duration:.2f} seconds")

    # Save metadata
    meta = {
        'model_id': model_id,
        'architecture': arch,
        'hyperparameters': hp,
        'model_file': os.path.basename(model_path),
        'train_loss': float(min(history.history['loss'])),
        'val_loss': float(min(history.history['val_loss'])),
        'epochs_trained': len(history.history['loss']),
        'training_time_seconds': round(training_duration, 2),
        'timestamp': timestamp,
        'forecast_horizon': horizon,
        'n_features': n_features,
        'data_source': proc_dir
    }

    # Try to record git hash
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                            stderr=subprocess.DEVNULL).decode().strip()
        meta['git_commit'] = git_hash
    except Exception:
        meta['git_commit'] = 'N/A'

    meta_path = os.path.join(model_folder, "meta.json")
    with open(meta_path, 'w') as f:
        import json
        json.dump(meta, f, indent=2)

    return model, history, meta

# ============================================================
# OPTUNA HYPERPARAMETER TUNING
# ============================================================
def run_optuna_tuning(cfg: dict, data: dict = None, extra_callbacks: list = None, force_cpu: bool = False, loss_fn: str = 'mse'):
    """Menjalankan Optuna untuk mencari hyperparameter terbaik."""
    import optuna
    import mlflow
    from src.model_factory import build_model, compile_model
    from src.data_prep import create_sequences_with_indices

    # --- FORCE CPU MODE ---
    if force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # Disable all GPUs so TF only sees CPU
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass
        logger.info("🖥️ Force CPU mode enabled for tuning.")
        print("🖥️ Tuning berjalan dalam mode CPU (GPU dinonaktifkan).")
    
    # Clear any previous session
    tf.keras.backend.clear_session()
    gc.collect()

    horizon = cfg['forecasting']['horizon']
    space = cfg['tuning']['search_space']

    # Load preprocessed scaled data
    if data is None:
        proc = cfg['paths']['processed_dir']
        data = {
            'X_train': np.load(os.path.join(proc, 'X_train.npy')),
            'y_train': np.load(os.path.join(proc, 'y_train.npy')),
        }
    n_features = data['X_train'].shape[2]

    # FORCE DISABLE GIT TRACKING FOR MLFLOW TO PREVENT MEMORY ERRORS
    os.environ["MLFLOW_TRACKING_GIT_DISABLE"] = "true"
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"

    # MLflow setup
    mlflow_uri = os.path.abspath(cfg['mlflow']['tracking_uri']).replace('\\', '/')
    mlflow.set_tracking_uri(f"file:///{mlflow_uri}")
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])

    def objective(trial):
        # Aggressive cleanup
        tf.keras.backend.clear_session()
        for _ in range(3):
            gc.collect()

        arch = cfg['model']['architecture']

        # Suggest hyperparameters
        hp = {}
        hp['patch_len'] = trial.suggest_int('patch_len', space['patch_len'][0], space['patch_len'][1], step=space['patch_len'][2])
        hp['stride'] = trial.suggest_int('stride', space['stride'][0], space['stride'][1], step=space['stride'][2])
        hp['d_model'] = trial.suggest_categorical('d_model', space['d_model'])
        hp['n_heads'] = trial.suggest_categorical('n_heads', space['n_heads'])
        hp['n_layers'] = trial.suggest_int('n_layers', space['n_layers'][0], space['n_layers'][1])
        hp['dropout'] = trial.suggest_float('dropout', space['dropout'][0], space['dropout'][1])
        hp['learning_rate'] = trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True)
        hp['batch_size'] = trial.suggest_categorical('batch_size', space['batch_size'])
        hp['lookback'] = trial.suggest_int('lookback', space['lookback'][0], space['lookback'][1], step=space['lookback'][2])

        # --- CONSTRAINT 1: patch_len must fit within lookback ---
        if hp['patch_len'] > hp['lookback']:
            hp['patch_len'] = hp['lookback'] // 2

        # --- CONSTRAINT 2: stride must be <= patch_len ---
        if hp['stride'] > hp['patch_len']:
            hp['stride'] = hp['patch_len']

        # --- CONSTRAINT 3: d_model must be divisible by n_heads ---
        if hp['d_model'] % hp['n_heads'] != 0:
            # Find the nearest valid n_heads (largest divisor <= current n_heads)
            valid_heads = [h for h in space['n_heads'] if hp['d_model'] % h == 0]
            if valid_heads:
                hp['n_heads'] = max(valid_heads)
            else:
                # Fallback: use n_heads=1 or 2 (always divides)
                hp['n_heads'] = max(h for h in [1, 2, 4] if hp['d_model'] % h == 0)
            logger.info(f"Auto-corrected n_heads to {hp['n_heads']} (d_model={hp['d_model']})")

        X_full = data['X_train']
        y_full = data['y_train']

        # If lookback changed, we need to re-sequence... but X_train is already sequenced.
        # For simplicity, we do an inner train/val split on the existing sequences.
        n_train = int(0.8 * len(X_full))
        X_tr, y_tr = X_full[:n_train], y_full[:n_train]
        X_va, y_va = X_full[n_train:], y_full[n_train:]

        model = build_model(arch, hp['lookback'], n_features, horizon, hp)

        # If lookback doesn't match, truncate/pad input
        actual_lookback = X_tr.shape[1]
        if hp['lookback'] != actual_lookback:
            if hp['lookback'] < actual_lookback:
                X_tr = X_tr[:, -hp['lookback']:, :]
                X_va = X_va[:, -hp['lookback']:, :]
            else:
                # Can't extend, use original lookback
                hp['lookback'] = actual_lookback
                model = build_model(arch, actual_lookback, n_features, horizon, hp)

        compile_model(model, hp['learning_rate'], loss_fn=loss_fn)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_tr, y_tr, validation_data=(X_va, y_va),
            epochs=50, batch_size=hp['batch_size'],
            callbacks=[early_stop], verbose=0
        )

        val_loss = min(history.history['val_loss'])

        try:
            with mlflow.start_run(nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric('val_loss', val_loss)
        except Exception as e:
            print(f"⚠️ MLflow logging failed (ignoring): {e}")

        return val_loss


    print("=" * 70)
    print("RUNNING HYPERPARAMETER TUNING (Optuna)")
    print("=" * 70)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=cfg['tuning']['n_trials'], 
                   show_progress_bar=True, callbacks=extra_callbacks)

    best = study.best_params
    print(f"\n✅ Best params: {best}")
    print(f"   Best val_loss: {study.best_value:.6f}")

    # Update config with best params
    cfg['model']['hyperparameters'].update(best)

    return best, study


# ============================================================
# TIME SERIES CROSS-VALIDATION (TSCV)
# ============================================================
def run_tscv(cfg: dict, data: dict = None):
    """Menjalankan Time Series Cross-Validation."""
    from src.model_factory import build_model, compile_model
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    hp = cfg['model']['hyperparameters']
    arch = cfg['model']['architecture']
    horizon = cfg['forecasting']['horizon']
    n_splits = cfg['tscv']['n_splits']

    if data is None:
        proc = cfg['paths']['processed_dir']
        data = {
            'X_train': np.load(os.path.join(proc, 'X_train.npy')),
            'y_train': np.load(os.path.join(proc, 'y_train.npy')),
        }

    X_full, y_full = data['X_train'], data['y_train']
    n_features = X_full.shape[2]
    
    # Handle lookback mismatch in TSCV
    lookback = hp['lookback']
    actual_data_lookback = X_full.shape[1]
    if lookback != actual_data_lookback:
        print(f"⚠️ TSCV Lookback mismatch: Config wants {lookback}, Data has {actual_data_lookback}.")
        if lookback < actual_data_lookback:
            print(f"   Truncating TSCV sequences to last {lookback} steps...")
            X_full = X_full[:, -lookback:, :]
        else:
            print(f"   ⚠️ Data lookback ({actual_data_lookback}) is smaller than config lookback ({lookback}).")
            hp['lookback'] = actual_data_lookback
            lookback = actual_data_lookback
    
    # update X_full in data for actual use if modified
    data['X_train'] = X_full

    print("=" * 70)
    print(f"RUNNING TIME SERIES CROSS-VALIDATION ({n_splits} Folds)")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_tr, y_tr = X_full[train_idx], y_full[train_idx]
        X_va, y_va = X_full[val_idx], y_full[val_idx]
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

        tf.keras.backend.clear_session()
        gc.collect()

        model = build_model(arch, hp['lookback'], n_features, horizon, hp)
        compile_model(model, hp['learning_rate'])

        model.fit(
            X_tr, y_tr, validation_data=(X_va, y_va),
            epochs=cfg['training']['epochs'], batch_size=hp['batch_size'],
            callbacks=_get_callbacks(cfg), verbose=0
        )

        y_pred = model.predict(X_va, verbose=0)
        mae = mean_absolute_error(y_va, y_pred)
        rmse = np.sqrt(mean_squared_error(y_va, y_pred))
        r2 = r2_score(y_va.flatten(), y_pred.flatten())
        metrics_per_fold.append({'fold': fold + 1, 'mae': mae, 'rmse': rmse, 'r2': r2})
        print(f"  MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    print("\n" + "=" * 70)
    print("TSCV SUMMARY")
    avg_mae = np.mean([m['mae'] for m in metrics_per_fold])
    avg_rmse = np.mean([m['rmse'] for m in metrics_per_fold])
    avg_r2 = np.mean([m['r2'] for m in metrics_per_fold])
    print(f"  Avg MAE:  {avg_mae:.4f}")
    print(f"  Avg RMSE: {avg_rmse:.4f}")
    print(f"  Avg R²:   {avg_r2:.4f}")
    print("=" * 70)

    return metrics_per_fold
