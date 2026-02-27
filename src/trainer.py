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
    
    if arch in ['patchtst_hf', 'autoformer_hf', 'causal_transformer_hf']:
        from src.model_hf import build_patchtst_hf, build_autoformer_hf, build_causal_transformer_hf
        if arch == 'patchtst_hf':
            model = build_patchtst_hf(lookback, n_features, horizon, hp)
        elif arch == 'autoformer_hf':
            model = build_autoformer_hf(lookback, n_features, horizon, hp)
        elif arch == 'causal_transformer_hf':
            model = build_causal_transformer_hf(lookback, n_features, horizon, hp)
    else:
        model = build_model(arch, lookback, n_features, horizon, hp)
        compile_model(model, hp['learning_rate'], loss_fn=loss_fn)
        model.summary()

    # Define callbacks later when evaluating model paths
    
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

    use_batch_scheduling = cfg['training'].get('use_batch_scheduling', False)
    total_epochs = cfg['training']['epochs']
    init_batch_size = hp['batch_size']
    
    # Initialize Callbacks before using them
    cbs = _get_callbacks(cfg)
    if extra_callbacks:
        for cb in extra_callbacks:
            if hasattr(cb, 'on_train_begin'):
                cb.on_train_begin()
    
    import time
    start_time = time.time()
    
    if extra_callbacks:
        cbs.extend(extra_callbacks)

    if arch in ['patchtst_hf', 'autoformer_hf', 'causal_transformer_hf']:
        from src.model_hf import train_eval_pytorch_model
        # Use simple epochs since Custom PyTorch loop handles patience
        hp['epochs'] = total_epochs
        hp['batch_size'] = init_batch_size
        hp['loss'] = loss_fn
        # Send only UI callbacks so Streamlit UI updates per epoch without crashing via Keras EarlyStopping
        history, model = train_eval_pytorch_model(
            model, data['X_train'], data['y_train'], data['X_test'], data['y_test'], hp, callbacks=extra_callbacks
        )
    elif use_batch_scheduling:
        print(f"\n📈 [Taktik 3] Batch Size Scheduling diaktifkan. Start Batch: {init_batch_size}")
        import copy
        from keras.callbacks import History
        
        full_history = History()
        full_history.history = {}
        
        p1_epochs = int(total_epochs * 0.4)
        p2_epochs = int(total_epochs * 0.3)
        p3_epochs = total_epochs - p1_epochs - p2_epochs
        
        max_bs = cfg['training'].get('max_batch_size', 512)
        phases = [
            {'epochs': p1_epochs, 'batch_size': min(init_batch_size, max_bs)},
            {'epochs': p2_epochs, 'batch_size': min(init_batch_size * 2, max_bs)},
            {'epochs': p3_epochs, 'batch_size': min(init_batch_size * 4, max_bs)}
        ]
        
        current_epoch = 0
        early_stopped = False
        
        for i, phase in enumerate(phases):
            p_epochs = phase['epochs']
            if p_epochs <= 0 or early_stopped:
                break
                
            p_bs = phase['batch_size']
            print(f"   [Phase {i+1}] Training for {p_epochs} epochs with Batch Size: {p_bs}")
            sys.stdout.flush()
            
            h = model.fit(
                data['X_train'], data['y_train'],
                validation_data=(data['X_test'], data['y_test']),
                initial_epoch=current_epoch,
                epochs=current_epoch + p_epochs,
                batch_size=p_bs,
                callbacks=cbs,
                verbose=2
            )
            
            # Merge history
            for k, v in h.history.items():
                if k not in full_history.history:
                    full_history.history[k] = []
                full_history.history[k].extend(v)
                
            current_epoch += p_epochs
            
            for cb in cbs:
                if hasattr(cb, 'stopped_epoch') and getattr(cb, 'stopped_epoch', 0) > 0:
                    early_stopped = True
                    print(f"   [Early Stop] Triggered in Phase {i+1}")
                    break
                    
        history = full_history
        history.model = model
        history.params = {'epochs': total_epochs}
    else:
        history = model.fit(
            data['X_train'], data['y_train'],
            validation_data=(data['X_test'], data['y_test']),
            epochs=total_epochs,
            batch_size=init_batch_size,
            callbacks=cbs,
            verbose=2 # One line per epoch is more stable for CMD logging
        )
    
    end_time = time.time()
    training_duration = end_time - start_time

    if arch in ['patchtst_hf', 'autoformer_hf', 'causal_transformer_hf']:
        import torch
        model_path = os.path.join(model_folder, "model_hf")
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(model_path)
        else:
            # For CustomAutoformerForPrediction
            torch.save(model.state_dict(), os.path.join(model_folder, "pytorch_model.bin"))
    else:
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
def run_optuna_tuning(cfg: dict, data: dict = None, extra_callbacks: list = None, force_cpu: bool = False, loss_fn: str = 'mse', verbose: bool = True):
    """Menjalankan Optuna untuk mencari hyperparameter terbaik."""
    import optuna
    try:
        from optuna_integration import TFKerasPruningCallback
    except ImportError:
        from optuna.integration import TFKerasPruningCallback
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
        
        # General Hyperparameters (Supported across most models)
        if 'd_model' in space:
            if isinstance(space['d_model'], list) and len(space['d_model']) == 2 and isinstance(space['d_model'][0], int):
                hp['d_model'] = trial.suggest_categorical('d_model', [2**i for i in range(int(np.log2(space['d_model'][0])), int(np.log2(space['d_model'][1]))+1)])
            else:
                hp['d_model'] = trial.suggest_categorical('d_model', space['d_model'])
                
        if 'n_layers' in space:
            hp['n_layers'] = trial.suggest_int('n_layers', space['n_layers'][0], space['n_layers'][1])
            
        if 'dropout' in space:
            hp['dropout'] = trial.suggest_float('dropout', space['dropout'][0], space['dropout'][1])
            
        if 'learning_rate' in space:
            hp['learning_rate'] = trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True)
            
        if 'batch_size' in space:
            if isinstance(space['batch_size'], list) and len(space['batch_size']) == 2 and isinstance(space['batch_size'][0], int):
                hp['batch_size'] = trial.suggest_categorical('batch_size', [2**i for i in range(int(np.log2(space['batch_size'][0])), int(np.log2(space['batch_size'][1]))+1)])
            else:
                hp['batch_size'] = trial.suggest_categorical('batch_size', space['batch_size'])
                
        if 'lookback' in space:
            hp['lookback'] = trial.suggest_int('lookback', space['lookback'][0], space['lookback'][1], step=space['lookback'][2])


        # Architecture-Specific Hyperparameters
        if arch in ['patchtst', 'patchtst_hf', 'autoformer_hf', 'autoformer', 'causal_transformer_hf', 'timetracker']:
            if 'n_heads' in space:
                if isinstance(space['n_heads'], list) and len(space['n_heads']) == 2 and isinstance(space['n_heads'][0], int):
                    hp['n_heads'] = trial.suggest_categorical('n_heads', [2**i for i in range(int(np.log2(space['n_heads'][0])), int(np.log2(space['n_heads'][1]))+1)])
                else:
                    hp['n_heads'] = trial.suggest_categorical('n_heads', space['n_heads'])
            else:
                hp['n_heads'] = 8
                
            if 'ff_dim' in space:
                if isinstance(space['ff_dim'], list) and len(space['ff_dim']) == 2 and isinstance(space['ff_dim'][0], int):
                    hp['ff_dim'] = trial.suggest_categorical('ff_dim', [2**i for i in range(int(np.log2(space['ff_dim'][0])), int(np.log2(space['ff_dim'][1]))+1)])
                else:
                    hp['ff_dim'] = trial.suggest_categorical('ff_dim', space['ff_dim'])
                    
            if 'd_model' in hp and hp['d_model'] % hp['n_heads'] != 0:
                valid_heads = [h for h in [1, 2, 4, 8, 16, 32] if hp['d_model'] % h == 0]
                if valid_heads:
                    hp['n_heads'] = max(valid_heads)
                else:
                    hp['n_heads'] = max(h for h in [1, 2, 4] if hp['d_model'] % h == 0)
                logger.info(f"Auto-corrected n_heads to {hp['n_heads']} (d_model={hp['d_model']})")

        if arch in ['patchtst', 'patchtst_hf', 'timetracker']:
            if 'patch_len' in space:
                hp['patch_len'] = trial.suggest_int('patch_len', space['patch_len'][0], space['patch_len'][1], step=space['patch_len'][2])
            else:
                hp['patch_len'] = 16
                
            if 'stride' in space:
                hp['stride'] = trial.suggest_int('stride', space['stride'][0], space['stride'][1], step=space['stride'][2])
            else:
                hp['stride'] = 8
                
            if hp['patch_len'] > hp['lookback']:
                hp['patch_len'] = max(2, hp['lookback'] // 2)
            if hp['stride'] > hp['patch_len']:
                hp['stride'] = hp['patch_len']

        if arch in ['autoformer_hf', 'autoformer']:
            if 'moving_avg' in space:
                ma_min, ma_max = space['moving_avg'][0], space['moving_avg'][1]
                if ma_min % 2 == 0: ma_min += 1
                if ma_max % 2 == 0: ma_max -= 1
                if ma_min <= ma_max:
                    hp['moving_avg'] = trial.suggest_int('moving_avg', ma_min, ma_max, step=2)
                else:
                    hp['moving_avg'] = 25

        if arch == 'timetracker':
            if 'n_shared_experts' in space:
                hp['n_shared_experts'] = trial.suggest_int('n_shared_experts', space['n_shared_experts'][0], space['n_shared_experts'][1])
            if 'n_private_experts' in space:
                hp['n_private_experts'] = trial.suggest_int('n_private_experts', space['n_private_experts'][0], space['n_private_experts'][1])
            if 'top_k' in space:
                hp['top_k'] = trial.suggest_int('top_k', space['top_k'][0], space['top_k'][1])

        if arch in ['gru', 'lstm', 'rnn']:
            if 'use_bidirectional' in space:
                hp['use_bidirectional'] = trial.suggest_categorical('use_bidirectional', space['use_bidirectional'])
            elif 'use_bidirectional' in cfg['model']['hyperparameters']:
                hp['use_bidirectional'] = cfg['model']['hyperparameters']['use_bidirectional']
            if 'use_revin' in cfg['model']['hyperparameters']:
                hp['use_revin'] = cfg['model']['hyperparameters']['use_revin']

        X_full = data['X_train']
        y_full = data['y_train']

        # --- TACTIC 1: DATA SUBSAMPLING ---
        if cfg['tuning'].get('use_subsampling', False):
            ratio = cfg['tuning'].get('subsample_ratio', 0.20)
            subset_len = max(100, int(len(X_full) * ratio))
            # Mengambil data terbaru (dari ujung belakang dataset) mempertahankan integritas urutan waktu
            X_full = X_full[-subset_len:]
            y_full = y_full[-subset_len:]

        # If lookback changed, we need to re-sequence... but X_train is already sequenced.
        # For simplicity, we do an inner train/val split on the existing sequences.
        n_train = int(0.8 * len(X_full))
        X_tr, y_tr = X_full[:n_train], y_full[:n_train]
        X_va, y_va = X_full[n_train:], y_full[n_train:]

        # If lookback doesn't match, truncate/pad input
        actual_lookback = X_tr.shape[1]
        
        # Build model and train based on Architecture Type
        if arch in ['patchtst_hf', 'autoformer_hf', 'causal_transformer_hf']:
            # PyTorch specific model building and training
            if hp['lookback'] != actual_lookback:
                if hp['lookback'] < actual_lookback:
                    X_tr = X_tr[:, -hp['lookback']:, :]
                    X_va = X_va[:, -hp['lookback']:, :]
                else:
                    hp['lookback'] = actual_lookback
            
            # Using custom HF builder and PyTorch training loop
            from src.model_hf import build_patchtst_hf, build_autoformer_hf, build_causal_transformer_hf, train_eval_pytorch_model
            
            if arch == 'patchtst_hf':
                model = build_patchtst_hf(hp['lookback'], n_features, horizon, hp)
            elif arch == 'autoformer_hf':
                model = build_autoformer_hf(hp['lookback'], n_features, horizon, hp)
            else:
                model = build_causal_transformer_hf(hp['lookback'], n_features, horizon, hp)
            # Make sure we pass batch_size, epochs etc. to PyTorch trainer
            if 'batch_size' not in hp:
                hp['batch_size'] = cfg['training'].get('batch_size', 32)
            hp['epochs'] = 100 # Standard tuning limit
            hp['loss'] = loss_fn
            
            history, model = train_eval_pytorch_model(model, X_tr, y_tr, X_va, y_va, hp, trial=trial, verbose=verbose)
            val_loss = min(history.history['val_loss'])
        else:
            if hp['lookback'] != actual_lookback:
                if hp['lookback'] < actual_lookback:
                    X_tr = X_tr[:, -hp['lookback']:, :]
                    X_va = X_va[:, -hp['lookback']:, :]
                else:
                    hp['lookback'] = actual_lookback
                    
            model = build_model(arch, hp['lookback'], n_features, horizon, hp)
            compile_model(model, hp['learning_rate'], loss_fn=loss_fn)
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            pruning_cb = TFKerasPruningCallback(trial, 'val_loss')

            import sys
            class SingleLineTrainingCallback(tf.keras.callbacks.Callback):
                def __init__(self, t_num):
                    super().__init__()
                    self.t_num = t_num
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    v_loss = logs.get('val_loss', 0.0)
                    t_loss = logs.get('loss', 0.0)
                    sys.stdout.write(f"\r  └➜ [Trial {self.t_num}] Epoch {epoch+1:03d}/100 | loss: {t_loss:.4f} | val_loss: {v_loss:.4f}     ")
                    sys.stdout.flush()
                def on_train_end(self, logs=None):
                    sys.stdout.write("\n")
                    sys.stdout.flush()
            
            sl_cb = SingleLineTrainingCallback(trial.number)
            
            history = model.fit(
                X_tr, y_tr, validation_data=(X_va, y_va),
                epochs=100, batch_size=hp['batch_size'],
                callbacks=[early_stop, pruning_cb, sl_cb], verbose=0
            )
            val_loss = min(history.history['val_loss'])

        try:
            with mlflow.start_run(nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric('val_loss', val_loss)
        except Exception as e:
            print(f"⚠️ MLflow logging failed (ignoring): {e}")

        # --- Aggressive Final Memory Cleanup ---
        del model
        del history
        tf.keras.backend.clear_session()
        for _ in range(3):
            gc.collect()

        return val_loss


    print("=" * 70)
    print("RUNNING HYPERPARAMETER TUNING (Optuna with Pruning)")
    print("=" * 70)

    # Implement SuccessiveHalvingPruner
    # min_resource: epoch minimum sebelum pruning dipertimbangkan (10 epoch)
    # reduction_factor: factor pembagian trial tiap iterasi
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=10, reduction_factor=4)
    study = optuna.create_study(
        study_name=f'tuning_{cfg["model"]["architecture"]}', 
        storage='sqlite:///optuna_history.db', 
        load_if_exists=True,
        direction='minimize', 
        pruner=pruner
    )
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


def fine_tune_model(cfg, source_model_path, data=None, ft_config=None):
    """
    Fine-tune an existing model on new data with optional granular configuration.
    ft_config keys:
        - epochs: int
        - learning_rate: float
        - freeze_backbone: bool
        - trainable_last_n: int (default: 2)
    """
    from src.model_factory import get_custom_objects, fix_lambda_tf_refs, compile_model
    import sys

    if ft_config is None:
        ft_config = {}

    # 1. Load Pre-trained Model
    print(f"\n🚀 FINE-TUNING MODE")
    print(f"   Source Model: {source_model_path}")
    
    # Absolute path check — prefer .h5 first (more compatible with this project)
    model_file = source_model_path
    if os.path.isdir(source_model_path):
        for ext in ['.h5', '.keras']:
            candidate = os.path.join(source_model_path, f'model{ext}')
            if os.path.exists(candidate):
                model_file = candidate
                break
    
    print(f"   Loading model from: {model_file}")
    
    # Try loading with fallback
    try:
        try:
            model = tf.keras.models.load_model(model_file, custom_objects=get_custom_objects(), safe_mode=False)
        except TypeError:
            model = tf.keras.models.load_model(model_file, custom_objects=get_custom_objects())
    except Exception as e1:
        err_str = str(e1).lower()
        # Keras 3 ZIP format on TF 2.x
        if ("signature not found" in err_str or "unable to synchronously open" in err_str) and model_file.endswith('.keras'):
            import zipfile
            if zipfile.is_zipfile(model_file):
                print(f"   [RECOVER] Keras 3 ZIP format detected. Extracting and rebuilding...")
                import json as _json
                model_root = os.path.dirname(model_file)
                extract_dir = os.path.join(model_root, '_extracted_k3')
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(model_file, 'r') as zf:
                    zf.extractall(extract_dir)
                weights_h5 = os.path.join(extract_dir, 'model.weights.h5')
                
                m_meta = {}
                meta_path = os.path.join(model_root, 'meta.json')
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f: m_meta = _json.load(f)
                    except: pass
                from src.model_factory import build_model, compile_model
                arch = m_meta.get('architecture', cfg['model']['architecture'])
                hp = m_meta.get('hyperparameters', cfg['model']['hyperparameters'])
                lb = m_meta.get('lookback', cfg['model']['hyperparameters']['lookback'])
                nf = m_meta.get('n_features', 0)
                hz = m_meta.get('horizon', cfg['forecasting']['horizon'])
                
                if nf == 0:
                    prep_p = os.path.join(model_root, 'prep_summary.json')
                    if os.path.exists(prep_p):
                        try:
                            with open(prep_p, 'r') as f:
                                p_m = _json.load(f)
                                if n_f := p_m.get('n_features'): nf = n_f
                                if l_b := p_m.get('lookback'): lb = l_b
                                if h_z := p_m.get('horizon'): hz = h_z
                        except: pass
                        
                model = build_model(arch, lb, nf, hz, hp)
                model.load_weights(weights_h5)
                print(f"   [OK] Model '{arch}' rebuilt from Keras 3 ZIP and weights loaded.")
            else:
                raise e1
        else:
            print(f"   [WARN] Failed to load {model_file}: {e1}")
            alt_file = model_file.replace('.keras', '.h5') if model_file.endswith('.keras') else model_file.replace('.h5', '.keras')
            if os.path.exists(alt_file):
                print(f"   Trying alternative: {alt_file}")
                try:
                    model = tf.keras.models.load_model(alt_file, custom_objects=get_custom_objects(), safe_mode=False)
                except TypeError:
                    model = tf.keras.models.load_model(alt_file, custom_objects=get_custom_objects())
            else:
                try:
                    model = tf.keras.models.load_model(model_file, compile=False, safe_mode=False)
                except TypeError:
                    model = tf.keras.models.load_model(model_file, compile=False)
    
    fix_lambda_tf_refs(model)

    # NEW: Layer Freezing
    if ft_config.get('freeze_backbone', False):
        last_n = ft_config.get('trainable_last_n', 2)
        print(f"   Freezing backbone, keeping last {last_n} layers trainable...")
        for layer in model.layers[:-last_n]:
            layer.trainable = False
        for layer in model.layers[-last_n:]:
            layer.trainable = True
    
    # 2. Prepare Data
    if data is None:
        proc = cfg['paths']['processed_dir']
        data = {
            'X_train': np.load(os.path.join(proc, 'X_train.npy')),
            'y_train': np.load(os.path.join(proc, 'y_train.npy')),
            'X_test': np.load(os.path.join(proc, 'X_test.npy')),
            'y_test': np.load(os.path.join(proc, 'y_test.npy')),
        }

    # Align lookback
    model_lookback = model.input_shape[1]
    data_lookback = data['X_train'].shape[1]
    
    if data_lookback > model_lookback:
        print(f"   Truncating data lookback: {data_lookback} -> {model_lookback}")
        data['X_train'] = data['X_train'][:, -model_lookback:, :]
        data['X_test'] = data['X_test'][:, -model_lookback:, :]
    elif data_lookback < model_lookback:
        raise ValueError(f"Data Target memiliki lookback ({data_lookback}) lebih kecil dari Model ({model_lookback}). Silakan preprocess data target dengan lookback {model_lookback}.")
    
    print(f"   Train Shape: {data['X_train'].shape}, Test/Val Shape: {data['X_test'].shape}")
    
    # 3. Re-compile with Lower Learning Rate
    orig_lr = cfg['model']['hyperparameters'].get('learning_rate', 0.001)
    ft_lr = ft_config.get('learning_rate', orig_lr * 0.1)
    print(f"   Fine-tuning LR: {ft_lr:.6f} (Original config was {orig_lr:.6f})")
    
    compile_model(model, ft_lr, loss_fn=cfg['training'].get('loss_fn', 'huber'))
    
    # 4. Train
    callbacks = _get_callbacks(cfg)
    
    # Training
    t_cfg = cfg['training']
    epochs = ft_config.get('epochs', max(5, t_cfg.get('epochs', 20) // 2))
    
    print(f"   Fine-tuning for {epochs} epochs...")
    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_test'], data['y_test']),
        epochs=epochs,
        batch_size=cfg['model']['hyperparameters'].get('batch_size', 32),
        callbacks=callbacks,
        shuffle=True, # Critical for stability
        verbose=1
    )
    
    # 5. Save as New Version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Custom Name Logic
    custom_name = ft_config.get('custom_name', '').strip()
    if custom_name:
        # Sanitize name
        import re
        custom_name = re.sub(r'[^\w\-_\.]', '_', custom_name)
        model_id = f"ft_{custom_name}"
    else:
        model_id = f"ft_{timestamp}_{cfg['model']['architecture']}"
        
    save_dir = os.path.join(cfg['paths']['models_dir'], model_id)
    
    # Handle existing directory to avoid overwriting unless intended (or just add suffix)
    if os.path.exists(save_dir) and custom_name:
        model_id = f"{model_id}_{timestamp}"
        save_dir = os.path.join(cfg['paths']['models_dir'], model_id)
        
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, "model.keras")
    model.save(model_path)
    
    # Save meta
    meta = {
        'model_id': model_id,
        'base_model': source_model_path,
        'architecture': cfg['model']['architecture'],
        'fine_tuned': True,
        'timestamp': timestamp,
        'history': history.history
    }
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
        
    # Copy scalers from current processed dir
    import shutil
    for f in ['X_scaler.pkl', 'y_scaler.pkl', 'prep_summary.json']:
        src_f = os.path.join(cfg['paths']['processed_dir'], f)
        if os.path.exists(src_f):
            shutil.copy(src_f, os.path.join(save_dir, f))
            
    print(f"\n✅ Fine-tuning Berhasil! Model disimpan di: {save_dir}")
    return model, history, model_id
