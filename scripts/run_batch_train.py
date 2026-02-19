"""
run_batch_train.py - Skrip otomatis untuk melatih beberapa model sekaligus.
"""
import os
import yaml
import time
import json
import logging
import numpy as np
from src.trainer import train_model

# Setup logging sederhana ke console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Load Master Config (sebagai base template)
    with open('config.yaml', 'r') as f:
        master_cfg = yaml.safe_load(f)

    # 2. Definisikan Antrean Eksperimen (Sesuai request/rekomendasi)
    experiments = [
        {
            "name": "PatchTST_Standard_ICLR",
            "architecture": "patchtst",
            "hp": {
                "lookback": 168, "patch_len": 16, "stride": 8, "d_model": 64, 
                "n_layers": 3, "n_heads": 16, "ff_dim": 128, "dropout": 0.2, 
                "learning_rate": 0.0001, "batch_size": 32
            }
        },
        {
            "name": "GRU_Stacked_Bi",
            "architecture": "gru",
            "hp": {
                "lookback": 72, "d_model": 64, "n_layers": 2, "dropout": 0.2, 
                "use_bidirectional": True, "learning_rate": 0.0005, "batch_size": 32
            }
        },
        {
            "name": "LSTM_Stacked_Bi",
            "architecture": "lstm",
            "hp": {
                "lookback": 72, "d_model": 64, "n_layers": 2, "dropout": 0.2, 
                "use_bidirectional": True, "learning_rate": 0.0005, "batch_size": 32
            }
        },
        {
            "name": "RNN_Simple_Bi",
            "architecture": "rnn",
            "hp": {
                "lookback": 48, "d_model": 32, "n_layers": 2, "dropout": 0.1, 
                "use_bidirectional": True, "learning_rate": 0.001, "batch_size": 32
            }
        }
    ]

    batch_results = []
    total_start = time.time()

    logger.info(f"==== MEMULAI BATCH TRAINING ({len(experiments)} MODEL) ====")

    for i, exp in enumerate(experiments):
        logger.info(f"\n[Model {i+1}/{len(experiments)}] Menjalankan: {exp['name']} ({exp['architecture']})")
        
        # Clone master config dan update dengan spek eksperimen
        cfg = master_cfg.copy()
        cfg['model']['architecture'] = exp['architecture']
        cfg['model']['hyperparameters'] = exp['hp']
        
        try:
            # Jalankan training
            # Karena ini script otomatis, kita tidak perlu live progress Streamlit
            model, history, meta = train_model(cfg)
            
            res = {
                "name": exp['name'],
                "arch": exp['architecture'],
                "val_loss": meta['val_loss'],
                "time_sec": meta.get('training_time_seconds', 0),
                "epochs": meta['epochs_trained'],
                "model_id": meta['model_id']
            }
            batch_results.append(res)
            logger.info(f"✅ BERHASIL: {res['model_id']} | Loss: {res['val_loss']:.6f} | Waktu: {res['time_sec']}s")
            
        except Exception as e:
            logger.error(f"❌ GAGAL pada model {exp['name']}: {e}")

    total_duration = time.time() - total_start
    
    # 3. Simpan Ringkasan Hasil
    summary_path = os.path.join(master_cfg['paths']['models_dir'], f"batch_summary_{int(total_start)}.json")
    with open(summary_path, 'w') as f:
        json.dump(batch_results, f, indent=2)

    logger.info(f"\n==== BATCH SELESAI ====")
    logger.info(f"Total Waktu: {total_duration/60:.2f} menit")
    logger.info(f"Ringkasan disimpan di: {summary_path}")

if __name__ == "__main__":
    main()
