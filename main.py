"""
main.py - Universal Interactive Controller (TUI)
Satu pintu untuk menjalankan seluruh pipeline:
  Preprocessing, Training, Tuning, TSCV, Evaluation, Target Testing.
"""
import os
import sys
import logging
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pipeline_{datetime.now():%Y%m%d_%H%M}.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('main')

# ============================================================
# RICH CONSOLE & QUESTIONARY
# ============================================================
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    import questionary
    # TUI hanya aktif jika terminal interaktif (bukan piped/redirected)
    HAS_TUI = sys.stdin.isatty()
    if not HAS_TUI:
        print("[INFO] Terminal non-interaktif terdeteksi. Menggunakan mode CLI.")
except ImportError:
    HAS_TUI = False
    print("[!] Library 'rich' dan 'questionary' tidak ditemukan.")
    print("    Install dengan: pip install rich questionary")
    print("    Pipeline akan berjalan dalam mode CLI sederhana.\n")

console = Console(force_terminal=True) if HAS_TUI else None


def show_banner():
    if HAS_TUI:
        console.print(Panel.fit(
            "[bold cyan]PV Forecasting Pipeline[/bold cyan]\n"
            "[dim]Universal Controller v1.0[/dim]\n"
            "[dim]Supports: PatchTST, GRU, and more[/dim]",
            border_style="bright_blue"
        ))
    else:
        print("=" * 50)
        print("  PV Forecasting Pipeline - Controller v1.0")
        print("=" * 50)


def show_current_config(cfg):
    """Tampilkan ringkasan konfigurasi saat ini."""
    if HAS_TUI:
        table = Table(title="Konfigurasi Aktif", show_lines=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Nilai", style="green")
        table.add_row("Arsitektur", cfg['model']['architecture'].upper())
        table.add_row("Dataset", os.path.basename(cfg['data']['csv_path']))
        table.add_row("Kapasitas", f"{cfg['pv_system']['nameplate_capacity_kw']} kW")
        table.add_row("Split", f"{cfg['splitting']['train_ratio']*100:.0f}% / {cfg['splitting']['test_ratio']*100:.0f}%")
        table.add_row("Lookback", str(cfg['model']['hyperparameters']['lookback']))
        table.add_row("Horizon", f"{cfg['forecasting']['horizon']} jam")
        table.add_row("Optuna Tuning", "[green]ON[/green]" if cfg['tuning']['enabled'] else "[red]OFF[/red]")
        table.add_row("TSCV", "[green]ON[/green]" if cfg['tscv']['enabled'] else "[red]OFF[/red]")
        console.print(table)
    else:
        print(f"\nKonfigurasi: {cfg['model']['architecture']} | "
              f"{os.path.basename(cfg['data']['csv_path'])} | "
              f"Lookback={cfg['model']['hyperparameters']['lookback']} | "
              f"Horizon={cfg['forecasting']['horizon']}")


# ============================================================
# MENU ACTIONS
# ============================================================
def action_preprocess(cfg):
    """Menjalankan preprocessing pipeline."""
    from src.data_prep import run_preprocessing
    return run_preprocessing(cfg)


def action_train(cfg, data=None):
    """Melatih model."""
    from src.trainer import train_model
    return train_model(cfg, data)


def action_tune(cfg, data=None):
    """Menjalankan Optuna tuning."""
    from src.trainer import run_optuna_tuning
    return run_optuna_tuning(cfg, data)


def action_tscv(cfg, data=None):
    """Menjalankan TSCV."""
    from src.trainer import run_tscv
    return run_tscv(cfg, data)


def action_evaluate(cfg, model=None, data=None):
    """Evaluasi model terlatih."""
    from src.predictor import evaluate_model
    if model is None:
        model = _load_latest_model(cfg)
    return evaluate_model(model, cfg, data)


def action_target_test(cfg):
    """Testing di data target domain."""
    from src.predictor import test_on_target

    # Cek 1: Ada model yang sudah dilatih?
    model_dir = cfg['paths']['models_dir']
    os.makedirs(model_dir, exist_ok=True)
    model_files = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5'))]
    if not model_files:
        print("\n" + "=" * 60)
        print("[!] BELUM ADA MODEL TERLATIH")
        print("=" * 60)
        print("Anda harus melatih model terlebih dahulu sebelum target testing.")
        print("\nLangkah:")
        print("  1. Jalankan 'python main.py full' atau 'python main.py train'")
        print("  2. Setelah model tersimpan, jalankan target testing lagi.")
        print("=" * 60)
        return None

    # Cek 2: Ada file CSV target?
    target_dir = cfg['paths']['target_data_dir']
    os.makedirs(target_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
    if not csv_files:
        abs_target = os.path.abspath(target_dir)
        print("\n" + "=" * 60)
        print("[!] BELUM ADA DATA TARGET")
        print("=" * 60)
        print("Target Testing membutuhkan file CSV data PV dari lokasi target.")
        print(f"\nLetakkan file CSV di folder berikut:")
        print(f"  {abs_target}")
        print("\nFormat CSV harus memiliki kolom yang sama dengan data training:")
        print(f"  timestamp, {cfg['data']['ghi_col']}, {cfg['data']['dhi_col']}, ")
        print(f"  {cfg['data']['temp_col']}, {cfg['data']['rh_col']}, ")
        print(f"  {cfg['data']['wind_speed_col']}, {cfg['data']['poa_col']}, ")
        print(f"  {cfg['data']['target_col']}")
        print(f"\nSeparator: '{cfg['data']['csv_separator']}'")
        print("=" * 60)
        return None

    # Semua siap — lanjut
    model_path = _pick_model(cfg)
    target_csv = _pick_target_data(cfg)
    return test_on_target(model_path, target_csv, cfg)


def action_full_pipeline(cfg):
    """Full Pipeline: Preprocess -> [Tune] -> Train -> [TSCV] -> Evaluate."""
    print("\n[FULL PIPELINE MODE]")
    print("=" * 60)

    # Step 1: Preprocess
    print("\n[1/4] Preprocessing...")
    data = action_preprocess(cfg)

    # Step 2: Tune (opsional)
    if cfg['tuning']['enabled']:
        print("\n[2/4] Hyperparameter Tuning...")
        best, _ = action_tune(cfg, data)
        cfg['model']['hyperparameters'].update(best)
    else:
        print("\n[2/4] Tuning dilewati (disabled)")

    # Step 3: Train
    print("\n[3/4] Training...")
    model, history, meta = action_train(cfg, data)

    # Step 4: TSCV or Evaluate
    if cfg['tscv']['enabled']:
        print("\n[3.5/4] TSCV...")
        action_tscv(cfg, data)

    print("\n[4/4] Evaluating...")
    results = action_evaluate(cfg, model, data)

    print("\n" + "=" * 60)
    print("FULL PIPELINE SELESAI!")
    print("=" * 60)
    return results


# ============================================================
# CONFIG EDITOR
# ============================================================
def action_edit_config(cfg):
    """Menu interaktif untuk mengedit konfigurasi."""
    from src.config_loader import save_config

    if not HAS_TUI:
        print("Config editor membutuhkan library 'questionary'. Install: pip install questionary")
        return cfg

    while True:
        section = questionary.select(
            "Bagian mana yang ingin diubah?",
            choices=[
                "1. Arsitektur Model",
                "2. Dataset & Kapasitas",
                "3. Split Ratio",
                "4. Hyperparameters",
                "5. Search Space (Tuning)",
                "6. Toggle: Tuning / TSCV",
                "7. Feature Groups",
                "8. Simpan & Kembali",
            ]
        ).ask()

        if section is None or "Simpan" in section:
            save_config(cfg)
            print("Config tersimpan!")
            break

        elif "Arsitektur" in section:
            arch = questionary.select(
                "Pilih arsitektur model:",
                choices=["patchtst", "gru"]
            ).ask()
            if arch:
                cfg['model']['architecture'] = arch

        elif "Dataset" in section:
            csv = questionary.text("Path CSV:", default=cfg['data']['csv_path']).ask()
            if csv:
                cfg['data']['csv_path'] = csv
            cap = questionary.text("Kapasitas (kW):", default=str(cfg['pv_system']['nameplate_capacity_kw'])).ask()
            if cap:
                cfg['pv_system']['nameplate_capacity_kw'] = float(cap)

        elif "Split" in section:
            train = questionary.text("Train Ratio (0-1):", default=str(cfg['splitting']['train_ratio'])).ask()
            if train:
                cfg['splitting']['train_ratio'] = float(train)
                cfg['splitting']['test_ratio'] = round(1 - float(train), 2)

        elif "Hyperparameters" in section:
            hp = cfg['model']['hyperparameters']
            for key in ['lookback', 'patch_len', 'stride', 'd_model', 'n_heads',
                         'n_layers', 'batch_size']:
                val = questionary.text(f"  {key}:", default=str(hp[key])).ask()
                if val:
                    hp[key] = int(val)
            for key in ['dropout', 'learning_rate']:
                val = questionary.text(f"  {key}:", default=str(hp[key])).ask()
                if val:
                    hp[key] = float(val)

        elif "Search Space" in section:
            space = cfg['tuning']['search_space']
            print("\nFormat: [min, max, step] atau [val1, val2, ...] untuk categorical")
            for key, val in space.items():
                new_val = questionary.text(f"  {key} {val}:", default=str(val)).ask()
                if new_val:
                    try:
                        space[key] = eval(new_val)
                    except Exception:
                        print(f"  Format salah, skip.")

        elif "Toggle" in section:
            cfg['tuning']['enabled'] = questionary.confirm(
                "Aktifkan Optuna Tuning?", default=cfg['tuning']['enabled']).ask()
            cfg['tscv']['enabled'] = questionary.confirm(
                "Aktifkan TSCV?", default=cfg['tscv']['enabled']).ask()
            n = questionary.text("Jumlah Trials Optuna:", default=str(cfg['tuning']['n_trials'])).ask()
            if n:
                cfg['tuning']['n_trials'] = int(n)

        elif "Feature" in section:
            groups = cfg['features']['groups']
            for g in groups:
                groups[g] = questionary.confirm(f"  Aktifkan grup '{g}'?", default=groups[g]).ask()

    return cfg


# ============================================================
# HELPERS
# ============================================================
def _load_latest_model(cfg):
    """Memuat model terbaru dari folder models/."""
    import tensorflow as tf
    from src.model_factory import get_custom_objects
    model_dir = cfg['paths']['models_dir']
    files = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5'))]
    if not files:
        raise FileNotFoundError(f"Tidak ada model di '{model_dir}/'")
    latest = sorted(files)[-1]
    path = os.path.join(model_dir, latest)
    print(f"Loading model: {path}")
    model = tf.keras.models.load_model(path, custom_objects=get_custom_objects(), safe_mode=False)
    from src.model_factory import fix_lambda_tf_refs
    fix_lambda_tf_refs(model)
    return model


def _pick_model(cfg):
    """Pilih file model dari folder models/."""
    model_dir = cfg['paths']['models_dir']
    files = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5'))]
    if not files:
        raise FileNotFoundError(f"Tidak ada model di '{model_dir}/'")
    if HAS_TUI:
        choice = questionary.select("Pilih model:", choices=files).ask()
        return os.path.join(model_dir, choice)
    return os.path.join(model_dir, sorted(files)[-1])


def _pick_target_data(cfg):
    """Pilih file CSV target dari folder data/target/."""
    target_dir = cfg['paths']['target_data_dir']
    os.makedirs(target_dir, exist_ok=True)
    files = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"Tidak ada CSV di '{target_dir}/'. Letakkan file CSV Indonesia di sana.")
    if HAS_TUI:
        choice = questionary.select("Pilih data target:", choices=files).ask()
        return os.path.join(target_dir, choice)
    return os.path.join(target_dir, files[0])


# ============================================================
# MAIN MENU
# ============================================================
def main():
    from src.config_loader import load_config, ensure_dirs

    # Setup GPU
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Memory Growth enabled for {len(gpus)} devices")

    # Load config
    cfg = load_config()
    ensure_dirs(cfg)

    show_banner()

    if not HAS_TUI:
        # Fallback CLI mode
        print("\nMode CLI (tanpa TUI). Gunakan argumen:")
        print("  python main.py preprocess")
        print("  python main.py train")
        print("  python main.py tune")
        print("  python main.py tscv")
        print("  python main.py evaluate")
        print("  python main.py target")
        print("  python main.py full")

        if len(sys.argv) > 1:
            cmd = sys.argv[1].lower()
            if cmd == 'preprocess': action_preprocess(cfg)
            elif cmd == 'train': action_train(cfg)
            elif cmd == 'tune': action_tune(cfg)
            elif cmd == 'tscv': action_tscv(cfg)
            elif cmd == 'evaluate': action_evaluate(cfg)
            elif cmd == 'target': action_target_test(cfg)
            elif cmd == 'full': action_full_pipeline(cfg)
            else: print(f"Perintah tidak dikenal: {cmd}")
        return

    # Interactive TUI loop
    while True:
        show_current_config(cfg)
        print()

        choice = questionary.select(
            "Apa yang ingin Anda lakukan?",
            choices=[
                "1. Preprocessing (Data > Artefak)",
                "2. Training (Latih Model)",
                "3. Hyperparameter Tuning (Optuna)",
                "4. TSCV (Cross-Validation)",
                "5. Evaluate (Metrik & Analisis)",
                "6. Target Testing (Data Indonesia)",
                "7. Full Pipeline (Semua Otomatis)",
                "8. Edit Konfigurasi",
                "9. Keluar",
            ]
        ).ask()

        if choice is None or "Keluar" in choice:
            print("\nSampai jumpa!")
            break

        try:
            if "Preprocessing" in choice:
                action_preprocess(cfg)
            elif "Training" in choice:
                action_train(cfg)
            elif "Tuning" in choice:
                action_tune(cfg)
            elif "TSCV" in choice:
                action_tscv(cfg)
            elif "Evaluate" in choice:
                action_evaluate(cfg)
            elif "Target" in choice:
                action_target_test(cfg)
            elif "Full Pipeline" in choice:
                action_full_pipeline(cfg)
            elif "Konfigurasi" in choice:
                cfg = action_edit_config(cfg)
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"[ERROR] {e}")

        print("\n" + "-" * 60)


if __name__ == '__main__':
    main()
