import os
import sys
import yaml
import time
import argparse
from datetime import datetime

# Prevent TF from preallocating all GPU memory
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from src.trainer import run_optuna_tuning
from app import save_config_to_file

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_cli_tuning(arch_name, n_trials=50, use_subsample=False, subsample_ratio=0.2):
    print(f"==================================================")
    print(f"🚀 memulai CLI Optuna Tuning untuk Model: {arch_name.upper()}")
    print(f"==================================================")
    
    # 1. Load active config
    cfg = load_config()
    proc_dir = cfg['paths']['processed_dir']
    
    if not os.path.exists(os.path.join(proc_dir, 'X_train.npy')):
        print(f"❌ ERROR: Dataset X_train.npy tidak ditemukan di {proc_dir}")
        print("Pastikan Anda sudah menjalankan Preprocessing di Streamlit!")
        sys.exit(1)
        
    # 2. Modify Config safely for this run without breaking Streamlit's config
    cfg['model']['architecture'] = arch_name
    cfg['tuning']['n_trials'] = n_trials
    cfg['tuning']['use_subsampling'] = use_subsample
    cfg['tuning']['subsample_ratio'] = subsample_ratio
    
    # 3. Define Architecture-Specific Search Spaces
    if arch_name in ['gru', 'lstm', 'rnn']:
        cfg['tuning']['search_space'] = {
            'd_model': [32, 256],
            'n_layers': [1, 3],
            'dropout': [0.1, 0.4],
            'learning_rate': [0.0001, 0.005],
            'batch_size': [32, 32], # Statis seperti anjuran
            'lookback': [24, 168, 24],
            'use_bidirectional': [True] # <-- Fixed to True only
        }
    elif arch_name in ['patchtst', 'patchtst_hf']:
        cfg['tuning']['search_space'] = {
            'patch_len': [8, 32, 8],
            'stride': [4, 16, 4],
            'd_model': [64, 128],
            'n_layers': [2, 4],
            'n_heads': [4, 8, 16],
            'ff_dim': [128, 512],
            'dropout': [0.1, 0.4],
            'learning_rate': [5e-5, 1e-3],
            'batch_size': [128, 128], # Dikunci demi stabilitas paralel
            'lookback': [168, 336, 24]
        }
    elif arch_name in ['autoformer_hf', 'autoformer']:
        cfg['tuning']['search_space'] = {
            'moving_avg': [25, 49], # Window trend dekonsruksi (harus ganjil/odd array)
            'd_model': [64, 128],
            'n_layers': [2, 4],
            'n_heads': [8, 16],
            'ff_dim': [256, 512],
            'dropout': [0.1, 0.4],
            'learning_rate': [5e-5, 1e-3],
            'batch_size': [128, 128], # Dikunci demi konsistensi Transformer VRAM
            'lookback': [96, 336, 24] # Minimal 4 hari (96 jam) agar pola musim terlihat
        }
    elif arch_name == 'causal_transformer_hf':
        cfg['tuning']['search_space'] = {
            'patch_len': [8, 32, 8],
            'stride': [4, 16, 4],
            'd_model': [64, 128],
            'n_layers': [2, 4],
            'n_heads': [4, 8, 16],
            'ff_dim': [128, 512],
            'dropout': [0.1, 0.4],
            'learning_rate': [5e-5, 1e-3],
            'batch_size': [128, 128], # DIKUNCI MATI di 128 sesuai request user
            'lookback': [168, 336, 24]
        }
    else:
        print(f"⚠️ Search space default digunakan untuk {arch_name}.")
        
    # Save the modified space temporarily to ensure trainer.py reads it correctly
    save_config_to_file(cfg)
    
    print(f"\n⚙️ Konfigurasi Tuning:")
    print(f"   Architecture : {arch_name}")
    print(f"   Trials       : {n_trials}")
    print(f"   Subsampling  : {use_subsample} ({subsample_ratio * 100}%)")
    print(f"   Search Space : Menyesuaikan {arch_name} secara otomatis")
    print("\nMenunggu GPU bersiap...")
    time.sleep(3)
    
    # Run the tuning Engine (Quiet mode for CLI so it doesn't spam console too heavily)
    best_params, study = run_optuna_tuning(cfg=cfg, force_cpu=False, verbose=False)
    
    # Save Result to separate CSV to avoid overwriting Streamlit's file
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    csv_file = f"tuning_{arch_name}_{n_trials}trial_CLI_{timestamp}.csv"
    
    df_trials = study.trials_dataframe()
    df_trials.to_csv(csv_file, index=False)
    print(f"\n✅ Tuning Selesai! Riwayat Detail (Excel) disimpan di: {csv_file}")
    print(f"   Akses SQLite Database: optuna_history.db (Study: tuning_{arch_name})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jalankan Optuna Tuning secara Paksa via CLI (Paralel dengan UI).")
    parser.add_argument("arch", type=str, choices=['gru', 'lstm', 'rnn', 'patchtst', 'patchtst_hf', 'timetracker', 'autoformer_hf', 'autoformer', 'causal_transformer_hf'], help="Arsitektur model yang ingin di-tuning (contoh: rnn)")
    parser.add_argument("--trials", type=int, default=50, help="Jumlah percobaan Optuna")
    parser.add_argument("--subsample", action='store_true', help="Aktifkan fitur subsample data agar cepat")
    parser.add_argument("--ratio", type=float, default=0.2, help="Rasio subsample data (0.05 s/d 1.0)")
    
    args = parser.parse_args()
    
    run_cli_tuning(args.arch, n_trials=args.trials, use_subsample=args.subsample, subsample_ratio=args.ratio)
