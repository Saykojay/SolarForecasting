import os
import yaml
import logging

logger = logging.getLogger(__name__)

# Base directory adalah folder tempat config.yaml berada (root folder project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')


def load_config(path: str = None) -> dict:
    """Memuat konfigurasi dari file YAML dan menormalisasi path."""
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file tidak ditemukan: {path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # --- NORMALISASI PATH AGAR ADAPTIF DI BERBAGAI DEVICE ---
    if 'paths' in cfg:
        for key, val in cfg['paths'].items():
            if isinstance(val, str):
                # Jika path mengandung '\' dari Windows tapi di OS lain, normalize
                val = val.replace('\\', '/')
                
                # Jika path absolut tapi dari laptop Lenovo (user lama), paksa jadi relatif
                if "Users/Lenovo" in val or "Users/user" in val:
                    # Ambil bagian setelah 'Modular Pipeline v1' atau ambil daunnya saja
                    if "Modular Pipeline v1" in val:
                        val = val.split("Modular Pipeline v1/")[-1]
                    else:
                        # Fallback: ambil 2 komponen terakhir (misal data/processed)
                        parts = val.split('/')
                        if len(parts) >= 2:
                            val = os.path.join(parts[-2], parts[-1])
                
                # Jika path relatif, ubah jadi absolut berdasarkan BASE_DIR saat runtime
                if not os.path.isabs(val):
                    cfg['paths'][key] = os.path.normpath(os.path.join(BASE_DIR, val))
                else:
                    cfg['paths'][key] = os.path.normpath(val)
                    
    logger.info(f"Config dimuat dan dinormalisasi dari {path}")
    return cfg


def save_config(cfg: dict, path: str = None):
    """Menyimpan konfigurasi ke file YAML (tetap simpan path sebagai relatif agar portable)."""
    path = path or DEFAULT_CONFIG_PATH
    
    # Buat copy agar tidak merubah cfg di memory yang sudah absolut
    cfg_to_save = yaml.safe_load(yaml.dump(cfg)) 
    
    if 'paths' in cfg_to_save:
        for key, val in cfg_to_save['paths'].items():
            # Kembalikan ke path relatif sebelum disimpan ke file
            if os.path.isabs(val) and val.startswith(BASE_DIR):
                cfg_to_save['paths'][key] = os.path.relpath(val, BASE_DIR).replace('\\', '/')

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg_to_save, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info(f"Config disimpan ke {path} (portable format)")


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
        dirpath = cfg['paths'].get(key)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
    
    # Ensure data/raw exists relative to root
    raw_path = os.path.join(BASE_DIR, 'data', 'raw')
    os.makedirs(raw_path, exist_ok=True)
