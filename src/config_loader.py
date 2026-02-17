"""
config_loader.py - Membaca & Menulis config.yaml
Modul ini menyediakan fungsi untuk memuat konfigurasi dari YAML,
serta menyimpan perubahan kembali ke file.
"""
import os
import yaml
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')


def load_config(path: str = None) -> dict:
    """Memuat konfigurasi dari file YAML."""
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file tidak ditemukan: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config dimuat dari {path}")
    return cfg


def save_config(cfg: dict, path: str = None):
    """Menyimpan konfigurasi ke file YAML."""
    path = path or DEFAULT_CONFIG_PATH
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info(f"Config disimpan ke {path}")


def get_column_mapping(cfg: dict) -> dict:
    """Mengembalikan mapping nama kolom dari config."""
    d = cfg['data']
    return {
        'time': d['time_col'],
        'target': d['target_col'],
        'ghi': d['ghi_col'],
        'dhi': d['dhi_col'],
        'temp': d['temp_col'],
        'rh': d['rh_col'],
        'wind_speed': d['wind_speed_col'],
        'wind_dir': d['wind_dir_col'],
        'poa': d['poa_col'],
    }


def get_root_cols(cfg: dict) -> list:
    """Mengembalikan daftar kolom utama (ROOT_COLS) untuk pengecekan missing values."""
    d = cfg['data']
    return [d['ghi_col'], d['dhi_col'], d['temp_col'], d['rh_col'],
            d['wind_speed_col'], d['poa_col'], d['target_col']]


def ensure_dirs(cfg: dict):
    """Membuat semua direktori yang dibutuhkan jika belum ada."""
    for key in ['processed_dir', 'models_dir', 'logs_dir', 'target_data_dir']:
        dirpath = cfg['paths'][key]
        os.makedirs(dirpath, exist_ok=True)
    # Also ensure data/raw exists
    os.makedirs('data/raw', exist_ok=True)
