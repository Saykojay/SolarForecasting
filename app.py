"""
app.py - Streamlit Web Dashboard untuk PV Forecasting Pipeline
Jalankan: streamlit run app.py
"""
import os
import sys
import json
import time
import copy
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from src.lang import gt
from src.model_factory import get_custom_objects, compile_model
from src.predictor import evaluate_model
import gc

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

# ============================================================
# GPU & MEMORY CONFIGURATION
# ============================================================
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' # Fix for fragmentation/OOM
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce log noise
# Disable MLflow Git tracking to prevent "fatal: Memory allocation failure"
os.environ['MLFLOW_TRACKING_GIT_DISABLE'] = 'true'
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

# ============================================================
# GPU CONFIGURATION
# ============================================================
import tensorflow as tf

@st.cache_resource
def get_gpu_info():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            info = f"✅ Running in GPU Mode. Available GPUs: {len(gpus)}"
        else:
            info = "💡 Running in CPU Mode. No GPU detected or GPU disabled."
        
        time_str = datetime.now().strftime("%H:%M:%S")
        print(f"[{time_str}] PID:{os.getpid()} {info}")
        return info
    except Exception as e:
        return f"Error initializing GPU: {e}"

gpus_info = get_gpu_info()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PV Forecasting Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Google Fonts: Manrope */
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');
    
    /* Apply font globally */
    html, body, [class*="css"], .stApp,
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    .stTextInput label, .stSelectbox label, .stSlider label,
    .stNumberInput label, .stCheckbox label,
    button, input, select, textarea,
    [data-testid="stSidebar"],
    [data-testid="stWidgetLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    .stTabs [data-baseweb="tab"],
    .stCaption, .stCode {
        font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Typography hierarchy */
    .stMarkdown h1 { font-weight: 800 !important; letter-spacing: -0.02em; }
    .stMarkdown h2 { font-weight: 700 !important; letter-spacing: -0.01em; }
    .stMarkdown h3 { font-weight: 700 !important; }
    .stMarkdown h4 { font-weight: 600 !important; }
    .stMarkdown p, .stMarkdown li { font-weight: 400; line-height: 1.6; }
    
    /* Sidebar labels */
    [data-testid="stWidgetLabel"] p {
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    /* Tab labels */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    /* Button text */
    .stButton button {
        font-family: 'Manrope', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Dark pro theme */
    .stApp { background-color: #0e1117; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16192b 100%);
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
    }
    .metric-value {
        font-family: 'Manrope', sans-serif !important;
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-family: 'Manrope', sans-serif !important;
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 4px;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'Manrope', sans-serif !important;
    }
    .status-ready { background: #065f46; color: #6ee7b7; }
    .status-missing { background: #7f1d1d; color: #fca5a5; }
    
    /* Fix for selectbox scrolling and visibility */
    div[data-baseweb="popover"] {
        z-index: 10000 !important;
    }
    div[data-baseweb="menu"] {
        max-height: 350px !important;
        overflow-y: auto !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1729 0%, #131b2e 100%);
    }
    
    .pipeline-step {
        font-family: 'Manrope', sans-serif !important;
        background: #1a1f2e;
        border-left: 3px solid #6366f1;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        font-weight: 400;
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #2d3348;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# STATE & CONFIG
# ============================================================
@st.cache_data
def load_config_cached():
    from src.config_loader import load_config
    return load_config()

def save_config_to_file(cfg):
    from src.config_loader import save_config
    save_config(cfg)


# ============================================================
# PERSISTENCE HELPERS (survive page refreshes)
# ============================================================
# Gunakan path absolut relatif terhadap root folder project agar lebih robust
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(APP_ROOT, 'logs', 'session')
PRESETS_DIR = os.path.join(APP_ROOT, 'configs', 'presets')

def list_feature_presets():
    """List available feature engineering presets."""
    if not os.path.exists(PRESETS_DIR):
        os.makedirs(PRESETS_DIR, exist_ok=True)
    return sorted([f.replace('.json', '') for f in os.listdir(PRESETS_DIR) if f.endswith('.json')])

def save_feature_preset(name, cfg_part):
    """Save a subset of config as a preset JSON."""
    os.makedirs(PRESETS_DIR, exist_ok=True)
    # Filter only relevant keys to save to preset
    data = {
        'features': cfg_part.get('features', {}),
        'target': cfg_part.get('target', {}),
        'timestamp': datetime.now().isoformat()
    }
    path = os.path.join(PRESETS_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_feature_preset(name):
    """Load a subset of config from preset JSON."""
    path = os.path.join(PRESETS_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def _persist_path(name):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    return os.path.join(PERSIST_DIR, name)

@st.cache_data(ttl=60) # Cache for 60s to avoid heavy disk I/O on every rerun
def label_format_with_time(name, base_path):
    """Format string for selectbox to show name (timestamp)."""
    if name in ["Latest (Default)", "Aktif (Dinamis)", "-- Silakan Pilih --", "Current (Dynamic)", "Current / Last Run"]:
         return name
    full_path = os.path.join(base_path, name)
    if os.path.exists(full_path):
        mtime = os.path.getmtime(full_path)
        dt = datetime.fromtimestamp(mtime).strftime('%d/%m %H:%M')
        return f"{name} ({dt})"
    return name

def save_training_history(history_dict, model_name=None):
    """Save training history to disk so it survives refresh."""
    data = {k: [float(v) for v in vals] for k, vals in history_dict.items()}
    path = _persist_path('last_training_history.json')
    with open(path, 'w') as f:
        json.dump(data, f)
    # Also save per-model
    if model_name:
        per_model = _persist_path(f'history_{model_name}.json')
        with open(per_model, 'w') as f:
            json.dump(data, f)

def load_training_history():
    """Load training history from disk."""
    path = _persist_path('last_training_history.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def save_eval_results_to_disk(results):
    """Save evaluation metrics and large arrays to disk."""
    # 1. Save JSON metrics
    serializable = {
        'metrics_train': results['metrics_train'],
        'metrics_test': results['metrics_test'],
    }
    path = _persist_path('last_eval_results.json')
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    # 2. Save large arrays as .npy (to survive refresh)
    array_keys = ['pv_train_actual', 'pv_train_pred', 'pv_test_actual', 'pv_test_pred', 'ghi_train', 'ghi_test', 'train_indices', 'test_indices']
    for k in array_keys:
        if k in results:
            np.save(_persist_path(f'eval_{k}.npy'), results[k])

def load_eval_results_from_disk():
    """Load evaluation metrics and arrays from disk."""
    json_path = _persist_path('last_eval_results.json')
    if not os.path.exists(json_path):
        return None
        
    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        # Load arrays if they exist
        array_keys = ['pv_train_actual', 'pv_train_pred', 'pv_test_actual', 'pv_test_pred', 'ghi_train', 'ghi_test', 'train_indices', 'test_indices']
        for k in array_keys:
            npy_path = _persist_path(f'eval_{k}.npy')
            if os.path.exists(npy_path):
                results[k] = np.load(npy_path)
        
        return results
    except Exception:
        return None

def save_selected_model(name):
    """Persist which model is currently selected."""
    path = _persist_path('selected_model.txt')
    with open(path, 'w') as f:
        f.write(name)

def load_selected_model():
    """Load previously selected model name."""
    path = _persist_path('selected_model.txt')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read().strip()
    return None

def save_tuning_results(results):
    """Save Optuna tuning results to disk (last + timestamped history)."""
    # 1. Save "last" for immediate reload
    path = _persist_path('last_tuning_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 2. Save to history with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hist_dir = _persist_path('tuning_history')
    os.makedirs(hist_dir, exist_ok=True)
    hist_path = os.path.join(hist_dir, f"tune_{timestamp}.json")
    with open(hist_path, 'w') as f:
        json.dump(results, f, indent=2)

def list_tuning_history():
    """List available tuning history files."""
    hist_dir = _persist_path('tuning_history')
    if not os.path.exists(hist_dir):
        return []
    files = sorted([f for f in os.listdir(hist_dir) if f.endswith('.json')], reverse=True)
    return files

def load_specific_tuning_result(filename):
    """Load a specific tuning result file."""
    if filename == 'last_tuning_results.json':
        return load_tuning_results()
    
    hist_path = os.path.join(_persist_path('tuning_history'), filename)
    if os.path.exists(hist_path):
        with open(hist_path, 'r') as f:
            return json.load(f)
    return None

def load_tuning_results():
    """Load Optuna tuning results from disk."""
    path = _persist_path('last_tuning_results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def load_prep_metadata_from_disk(p_dir):
    """Load the summary metadata if it exists."""
    path = os.path.join(p_dir, 'prep_summary.json')
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                summary = json.load(f)
            # Reconstruct DataFrame if present
            if 'corr_matrix' in summary:
                summary['corr_matrix'] = pd.DataFrame(summary['corr_matrix'])
            return summary
        except Exception:
            return None
    return None

if 'cfg' not in st.session_state:
    st.session_state.cfg = load_config_cached()
if 'pipeline_log' not in st.session_state:
    st.session_state.pipeline_log = []
if 'training_history' not in st.session_state:
    st.session_state.training_history = load_training_history()
if 'eval_results' not in st.session_state:
    st.session_state.eval_results = load_eval_results_from_disk()
if 'tuning_results' not in st.session_state:
    st.session_state.tuning_results = load_tuning_results()
if 'prep_metadata' not in st.session_state:
    # We need proc_dir which is defined below, move it up or use direct path
    p_dir = st.session_state.cfg['paths']['processed_dir']
    st.session_state.prep_metadata = load_prep_metadata_from_disk(p_dir)
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = load_selected_model()
if 'lang' not in st.session_state:
    st.session_state.lang = 'ID' # Default to Indonesian

cfg = st.session_state.cfg
# IMPROVED: Robustly find the 'processed' root directory
raw_root = cfg['paths']['processed_dir']
root_out_dir = os.path.abspath(raw_root)
while True:
    bname = os.path.basename(root_out_dir).lower()
    if bname == 'processed': break
    parent = os.path.dirname(root_out_dir)
    if not parent or parent == root_out_dir: break
    root_out_dir = parent
proc_dir = root_out_dir

model_dir = cfg['paths']['models_dir']
target_dir = cfg['paths']['target_data_dir']


# ============================================================
# SIDEBAR - CONFIGURATION
# ============================================================
with st.sidebar:
    # --- LANGUAGE SELECTOR ---
    st.markdown(f"### {gt('lang_select', st.session_state.lang)}")
    sel_lang = st.radio("Select Language", ["ID", "EN"], 
                        index=0 if st.session_state.lang == 'ID' else 1, 
                        horizontal=True, label_visibility="collapsed")
    if sel_lang != st.session_state.lang:
        st.session_state.lang = sel_lang
        st.rerun()

    st.markdown(f"## {gt('sidebar_settings', st.session_state.lang)}")
    
    # GPU status indicator & Debug
    try:
        # Force clear if somehow inherited
        if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
             del os.environ['CUDA_VISIBLE_DEVICES']
             
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            st.success(f"GPU Active: `{gpus[0].name.split(':')[-1]}`")
            with st.expander("ℹ️ GPU Details"):
                st.write(f"TensorFlow Version: `{tf.__version__}`")
                st.write(f"Devices found: `{gpus}`")
        else:
            st.warning("⚠️ Running on CPU (No GPU detected)")
            with st.expander("🔍 Mengapa GPU tidak terdeteksi?"):
                st.write("1. **Pastikan server di-restart**: Jika Anda baru saja mengganti kode, Streamlit harus di-stop (Ctrl+C di terminal) dan dijalankan lagi melalui `launch_dashboard.bat`.")
                st.write("2. **Cek Environment**: Pastikan Anda menggunakan environment `tf-gpu`.")
                st.write(f"3. **Python Path**: `{sys.executable}`")
                st.write(f"4. **TF Version**: `{tf.__version__}`")
                st.code(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    except Exception as e:
        st.error(f"Error checking GPU: {e}")
        
    st.markdown("---")
    
    # Device Selector
    st.markdown("##### 🖥️ Device Acceleration")
    device_options = ["GPU", "CPU"]
    
    # Re-check actual GPU availability for default value
    actual_gpus = tf.config.list_physical_devices('GPU')
    default_device = "GPU" if actual_gpus else "CPU"
    
    selected_device = st.radio(
        "Gunakan Hardware:",
        device_options,
        index=device_options.index(st.session_state.get('execution_device', default_device)),
        horizontal=True,
        help="GPU = Cepat (RTX 3050), CPU = Lambat tapi stabil"
    )
    st.session_state.execution_device = selected_device
    
    if selected_device == "GPU" and actual_gpus:
        st.info("🚀 Training dioptimalkan untuk GPU.")
    elif selected_device == "GPU" and not actual_gpus:
        st.warning("⚠️ GPU tidak terdeteksi, akan fallback ke CPU.")
    else:
        st.warning("🐌 Training dipaksa ke CPU (Safe Mode).")
    
    # Model Selector
    st.markdown("##### Model Manager")
    if os.path.exists(model_dir):
        all_items = os.listdir(model_dir)
        # Find folders that look like bundles (start with arch names)
        bundled_models = [f for f in all_items if os.path.isdir(os.path.join(model_dir, f)) and 
                          (f.startswith('patchtst') or f.startswith('gru'))]
        # Traditional legacy files
        legacy_files = [f for f in all_items if f.endswith(('.keras', '.h5'))]
        
        model_options = sorted(bundled_models + legacy_files, reverse=True)
        
        if model_options:
            curr_idx = 0
            if st.session_state.get('selected_model') in model_options:
                curr_idx = model_options.index(st.session_state.selected_model)
            
            selected = st.selectbox("Pilih Model untuk Evaluasi", model_options, index=curr_idx,
                                   format_func=lambda x: label_format_with_time(x, model_dir))
            st.session_state.selected_model = selected
            st.caption(f"Aktif: `{selected}`")
        else:
            st.warning("Belum ada model tersimpan.")
    else:
        st.info("Folder models belum ada.")
    
    # Stop processes button
    st.markdown("---")
    if st.button("🛑 Stop Semua Proses", width='stretch', type="secondary"):
        import subprocess
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe', '/FI', 'MEMUSAGE gt 500000'], 
                      capture_output=True, text=True)
        st.session_state.is_running = False
        st.warning("Proses dihentikan.")
    
    if st.button("💾 Simpan Master Config", width='stretch'):
        save_config_to_file(cfg)
        st.success("Master Config tersimpan!")
        st.cache_data.clear()

    st.markdown("---")
    st.markdown("##### 🎯 Tuning Controller")
    cfg['tuning']['enabled'] = st.toggle("Enable Optuna Tuning", cfg['tuning'].get('enabled', False))
    if cfg['tuning']['enabled']:
        cfg['tuning']['n_trials'] = st.number_input("Number of Trials", 5, 200, cfg['tuning'].get('n_trials', 50), 5)
        cfg['tscv']['enabled'] = st.toggle("Use TSCV in Tuning", cfg['tscv'].get('enabled', True))


# ============================================================
# MAIN AREA - HEADER
# ============================================================
# Status Cards & Definitions
new_arch = cfg['model'].get('architecture', 'patchtst')
has_data = os.path.exists(os.path.join(proc_dir, 'X_train.npy'))
has_model = any(f.endswith(('.keras', '.h5')) for f in os.listdir(model_dir)) if os.path.exists(model_dir) else False
has_target = any(f.endswith('.csv') for f in os.listdir(target_dir)) if os.path.exists(target_dir) else False

st.markdown(f"""
<div style="text-align:center; margin-bottom: 2rem;">
    <h1 style="font-family: 'Manrope', sans-serif;
               background: linear-gradient(135deg, #818cf8, #6366f1, #4f46e5); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em;
               margin-bottom: 0.5rem;">
        {gt('page_title', st.session_state.lang)}
    </h1>
    <p style="font-family: 'Manrope', sans-serif; color: #64748b; 
              font-size: 1.1rem; font-weight: 400;">
        Universal Dashboard &mdash; PatchTST, GRU, {gt('active_arch', st.session_state.lang)}: {new_arch.upper()}
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    status = "ready" if has_data else "missing"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{"✅" if has_data else "⏳"}</div>
        <div class="metric-label">{gt('data_preprocessed', st.session_state.lang)}</div>
        <span class="status-badge status-{status}">{gt('status_ready', st.session_state.lang) if has_data else gt('status_missing', st.session_state.lang)}</span>
    </div>""", unsafe_allow_html=True)
with col2:
    status = "ready" if has_model else "missing"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{"✅" if has_model else "⏳"}</div>
        <div class="metric-label">{gt('model_trained', st.session_state.lang)}</div>
        <span class="status-badge status-{status}">{gt('status_ready', st.session_state.lang) if has_model else gt('status_missing', st.session_state.lang)}</span>
    </div>""", unsafe_allow_html=True)
with col3:
    status = "ready" if has_target else "missing"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{"📂" if has_target else "⏳"}</div>
        <div class="metric-label">{gt('target_data', st.session_state.lang)}</div>
        <span class="status-badge status-{status}">{gt('status_present', st.session_state.lang) if has_target else gt('status_missing', st.session_state.lang)}</span>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{new_arch.upper()}</div>
        <div class="metric-label">{gt('active_arch', st.session_state.lang)}</div>
        <span class="status-badge status-ready">{gt('status_ready', st.session_state.lang)}</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab_data, tab_prep_features, tab_train, tab_batch, tab_tuning, tab_eval, tab_compare, tab_transfer, tab_troubleshoot = st.tabs([
    gt('data_insights', st.session_state.lang),
    gt('data_prep', st.session_state.lang),
    gt('training_center', st.session_state.lang),
    gt('batch_experiments', st.session_state.lang),
    gt('optuna_tuning', st.session_state.lang),
    gt('prediction_eval', st.session_state.lang),
    gt('leaderboard', st.session_state.lang),
    gt('target_testing', st.session_state.lang),
    gt('troubleshooting', st.session_state.lang)
])

# --- TAB: DATA PREP & FEATURES ---
with tab_prep_features:
    st.markdown("### 🛠️ Data Preparation & Feature Engineering")
    st.markdown("Ubah data mentah (CSV) menjadi tensor siap train dengan fitur yang optimal.")
    
    # --- SUB-SECTION 1: FEATURE LAB ---
    with st.expander("🧪 Feature Engineering Lab (Preset Manager)", expanded=True):
        st.markdown("Eksperimen dengan kombinasi fitur dan simpan sebagai preset.")
        p1, p2 = st.columns([2, 1])
        with p1:
            presets = list_feature_presets()
            selected_preset = st.selectbox("Pilih Preset untuk Dimuat", ["-- Silakan Pilih --"] + presets, key="prep_f_preset")
            if selected_preset != "-- Silakan Pilih --":
                if st.button("📥 Load Preset", key="btn_load_f"):
                    loaded = load_feature_preset(selected_preset)
                    if loaded:
                        cfg['features'].update(loaded.get('features', {}))
                        cfg['target'].update(loaded.get('target', {}))
                        st.success(f"Preset '{selected_preset}' berhasil dimuat!")
                        st.session_state.cfg = cfg
                        st.rerun()
        with p2:
            new_preset_name = st.text_input("Simpan Preset Baru", placeholder="v1_lags_only", key="prep_new_p")
            if st.button("💾 Save as Preset", key="btn_save_f"):
                if new_preset_name.strip():
                    save_feature_preset(new_preset_name.strip(), cfg)
                    st.success(f"Preset '{new_preset_name}' disimpan!")
                    st.rerun()

    st.markdown("---")

    # --- SUB-SECTION 2: PREPROCESSING ---
    st.markdown("#### 📥 Pipeline Preprocessing")
    st.markdown("### 📥 Data Preprocessing")
    st.markdown("Ubah data mentah (CSV) menjadi tensor siap train.")
    
    # --- LAST LOG PERSISTENCE ---
    if st.session_state.get('last_prep_log'):
        with st.expander("📜 Log Preprocessing Terakhir", expanded=False):
            st.code(st.session_state.last_prep_log, language="text")
            if st.button("🗑️ Clear Log", key="clear_log_prep"):
                st.session_state.last_prep_log = None
                st.rerun()

    # --- DATASET SELECTION (Moved from Data Insight) ---
    with st.expander("📊 Pilih Dataset Sumber", expanded=not has_data):
        raw_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
        if os.path.exists(raw_dir):
            raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
            current_csv_name = os.path.basename(cfg['data']['csv_path'])
            selected_file_p = st.selectbox("Pilih File CSV untuk Diproses:", raw_files, 
                                         index=raw_files.index(current_csv_name) if current_csv_name in raw_files else 0,
                                         key="dataset_select_prep")
            cfg['data']['csv_path'] = f"data/raw/{selected_file_p}"
            
            if st.button("👁️ Preview Raw Data", key="btn_preview_prep"):
                df_prev = pd.read_csv(cfg['data']['csv_path'], sep=cfg['data']['csv_separator'], nrows=5)
                st.dataframe(df_prev)
        else:
            st.warning("Folder data/raw tidak ditemukan.")

    # --- CONFIGURATION SECTION ---
    with st.expander("⚙️ Konfigurasi Preprocessing & Features", expanded=not has_data):
        c1, c2 = st.columns(2)
        with c1:
            # --- CLEAR SEPARATION: TARGET SELECTION ---
            st.markdown("##### 🎯 1. Variabel Target (Yang Ingin Diprediksi)")
            st.info("Pilih apa yang menjadi Final Output (Y) hasil ramalan AI kita di hari esok.")
            use_csi = st.checkbox("Prediksi Rasio Cuaca Bersih (CSI Normalization)", 
                                   value=cfg['target'].get('use_csi', True),
                                   help="True: AI memprediksi CSI (Rasio radiasi tembus). False: AI memprediksi Daya (kW) secara langsung.",
                                   key="p_csi")
            cfg['target']['use_csi'] = use_csi
            if use_csi:
                cfg['target']['csi_ghi_threshold'] = st.number_input(
                    "GHI Threshold (W/m²)", 
                    value=cfg['target'].get('csi_ghi_threshold', 50),
                    min_value=0, max_value=500, key="p_csi_th"
                )

            st.markdown("---")
            # --- CLEAR SEPARATION: INPUT FEATURES ---
            st.markdown("##### 📦 2. Fitur Input Historis (Data Masa Lalu)")
            st.caption("Pilih metrik sejarah (X) apa saja yang digunakan AI untuk *meramal* target di atas.")
            g = cfg['features'].get('groups', {})
            # time features
            with st.expander("🕒 Cyclical Time Features (Sin/Cos)", expanded=False):
                g['time_hour'] = st.checkbox("Hourly (Hour Sin/Cos)", value=g.get('time_hour', True), 
                                            help="Menangkap pola harian (pagi-siang-malam).", key="p_time_h")
                g['time_day'] = st.checkbox("Daily (Day of Month Sin/Cos)", value=g.get('time_day', True), 
                                             help="Menangkap pola tanggal dalam suatu bulan.", key="p_time_day")
                g['time_month'] = st.checkbox("Monthly (Month Sin/Cos)", value=g.get('time_month', True), 
                                             help="Menangkap pola musiman bulanan.", key="p_time_m")
                g['time_doy'] = st.checkbox("Seasonal (DOY Sin/Cos)", value=g.get('time_doy', True), 
                                           help="Day of Year: Resolusi tinggi untuk dinamika musiman.", key="p_time_d")
                g['time_year'] = st.checkbox("Yearly (Linear)", value=g.get('time_year', False), 
                                            help="Menangkap tren jangka panjang/tahunan.", key="p_time_y")
            g['weather'] = st.checkbox("Weather (Suhu, GHI, dll)", value=g.get('weather', True), key="p_weather")
            g['lags'] = st.checkbox("Time Lags (Riwayat Waktu Mundur)", value=g.get('lags', True), key="p_lags")
            g['rolling'] = st.checkbox("Moving Average", value=g.get('rolling', True), key="p_roll")
            g['physics'] = st.checkbox("Physics-based (Memasukkan target sebagai input riwayat)", value=g.get('physics', True), key="p_phys")
            cfg['features']['groups'] = g
            
            st.markdown("---")
            st.markdown("##### 🎛️ Feature Selection Mode")
            sel_mode = st.radio(
                "Mode Seleksi Fitur",
                ["auto", "manual"],
                index=0 if cfg['features'].get('selection_mode', 'auto') == 'auto' else 1,
                horizontal=True,
                help="**Auto**: Seleksi otomatis berbasis korelasi. **Manual**: Pilih fitur sendiri.",
                key="p_sel_mode"
            )
            cfg['features']['selection_mode'] = sel_mode
            
            if sel_mode == 'manual':
                st.caption("⚠️ Pilih fitur input secara manual. Fitur auxiliar (pv_clear_sky, pv_output_dc_kw) otomatis diblokir.")
                try:
                    # Try to load ACTUAL feature names from last preprocessing run
                    feats_pkl_path = os.path.join(proc_dir, 'df_train_feats.pkl')
                    blocked_cols = {'pv_clear_sky', 'pv_cs_normalized', 'pv_output_dc_kw', 
                                   cfg['data']['time_col'], 'timestamp_col'}
                    
                    if os.path.exists(feats_pkl_path):
                        df_feats_sample = pd.read_pickle(feats_pkl_path)
                        all_available = sorted([c for c in df_feats_sample.columns 
                                              if c not in blocked_cols])
                        st.info(f"📋 Memuat {len(all_available)} fitur dari preprocessing terakhir")
                    else:
                        # Fallback: read CSV + estimate derived names
                        csv_path = cfg['data']['csv_path']
                        df_cols_preview = pd.read_csv(csv_path, sep=cfg['data']['csv_separator'], nrows=1)
                        raw_cols = [c for c in df_cols_preview.columns if c not in blocked_cols]
                        derived = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                                  'day_of_year_sin', 'day_of_year_cos', 'year_linear',
                                  'csi_target', 'pv_output_kw']
                        for wc in raw_cols:
                            prefix = wc.split('_')[0]
                            for lag in ['1h', '12h', '24h', '48h', '168h']:
                                derived.append(f"{prefix}_lag_{lag}")
                            derived.extend([f"{prefix}_ma_3h", f"{prefix}_std_3h"])
                        all_available = sorted(set(raw_cols + derived))
                        st.warning("⚠️ Belum ada data preprocessing. Nama fitur adalah estimasi. "
                                  "Jalankan Preprocessing (mode Auto) dulu untuk mendapatkan daftar fitur yang akurat.")
                    
                    # Group features by category for easier browsing
                    cat_raw = [c for c in all_available if 'lag_' not in c and '_ma_' not in c 
                              and '_std_' not in c and '_sin' not in c and '_cos' not in c 
                              and 'year_' not in c and 'csi' not in c and 'pv_clear' not in c]
                    cat_lag = [c for c in all_available if 'lag_' in c]
                    cat_roll = [c for c in all_available if '_ma_' in c or '_std_' in c]
                    cat_time = [c for c in all_available if '_sin' in c or '_cos' in c or 'year_' in c]
                    cat_physics = [c for c in all_available if 'csi' in c.lower()]
                    
                    st.caption(f"**Tersedia:** {len(cat_raw)} Raw, {len(cat_lag)} Lag, "
                              f"{len(cat_roll)} Rolling, {len(cat_time)} Time, {len(cat_physics)} Physics")
                    
                    current_manual = cfg['features'].get('manual_features', [])
                    valid_defaults = [f for f in current_manual if f in all_available]
                    
                    selected_manual = st.multiselect(
                        "Pilih Fitur Input Historis (X):",
                        options=all_available,
                        default=valid_defaults,
                        help="PERHATIAN: Ini BUKAN target. Jika Anda mencentang 'csi_target' di sini, model akan melihat sejarah 'csi_target' 72 jam ke belakang untuk meramal masa depan.",
                        key="p_manual_feats"
                    )
                    cfg['features']['manual_features'] = selected_manual
                    
                    if selected_manual:
                        st.success(f"✅ {len(selected_manual)} fitur dipilih.")
                    else:
                        st.warning("⚠️ Belum ada fitur yang dipilih! Pilih minimal 1 fitur.")
                except Exception as e:
                    st.error(f"Gagal membaca fitur: {e}")

            else:
                st.caption("Fitur akan dipilih otomatis berdasarkan korelasi dengan target.")
                corr_th = st.slider("Correlation Threshold", 0.01, 0.5, 
                                   cfg['features'].get('corr_threshold', 0.1), 0.01,
                                   help="Fitur dengan korelasi < threshold akan dibuang.",
                                   key="p_corr_th")
                cfg['features']['corr_threshold'] = corr_th
                
                multicol_th = st.slider("Multicollinearity Threshold", 0.7, 1.0, 
                                       cfg['features'].get('multicol_threshold', 0.95), 0.01,
                                       help="Fitur dengan korelasi antar-fitur > threshold akan dibuang (mencegah redundancy).",
                                       key="p_multicol_th")
                cfg['features']['multicol_threshold'] = multicol_th

            st.markdown("##### 📐 Data Split & Scaling")
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                cap_p = st.number_input("Kapasitas (kW)", value=cfg['pv_system']['nameplate_capacity_kw'],
                                       min_value=0.1, step=0.5, key="p_cap")
                cfg['pv_system']['nameplate_capacity_kw'] = cap_p
                
                train_ratio_p = st.slider("Train Ratio (Prep)", 0.5, 0.95, cfg['splitting']['train_ratio'], 0.05, key="p_split")
                cfg['splitting']['train_ratio'] = train_ratio_p
                cfg['splitting']['test_ratio'] = round(1 - train_ratio_p, 2)
            
            with c_s2:
                horizon_p = st.number_input("Horizon (jam)", value=cfg['forecasting']['horizon'],
                                           min_value=1, max_value=168, step=1, key="p_hor")
                cfg['forecasting']['horizon'] = horizon_p
                
                scaler_options = ["minmax", "standard"]
                current_scaler = cfg['features'].get('scaler_type', 'minmax').lower()
                selected_scaler = st.selectbox("Metode Scaling (Prep)", scaler_options,
                                             index=scaler_options.index(current_scaler) if current_scaler in scaler_options else 0,
                                             key="p_scale")
                cfg['features']['scaler_type'] = selected_scaler

            st.markdown("---")
            # Target transform moved to the top of c1

        with c2:
            st.markdown("##### 🧹 Cleaning (Algorithm 1)")
            pcfg = cfg.get('preprocessing', {})
            pcfg['resample_1h'] = st.checkbox("Resample Hourly", value=pcfg.get('resample_1h', True), key="p_res")
            pcfg['remove_outliers'] = st.checkbox("Remove Outliers", value=pcfg.get('remove_outliers', True), key="p_out")
            
            if pcfg['remove_outliers']:
                st.caption("Outlier Rules:")
                pcfg['ghi_high_pv_zero'] = st.checkbox("PV 0 saat GHI Terang", value=pcfg.get('ghi_high_pv_zero', True), key="p_ghi_pv")
                pcfg['ghi_dark_pv_high'] = st.checkbox("PV Tinggi saat GHI Gelap", value=pcfg.get('ghi_dark_pv_high', True), key="p_dark_pv")
            
            st.markdown("---")
            st.markdown("##### ✨ Advanced Cleaning")
            pcfg['fix_ghi_dhi'] = st.checkbox("Fix Physical Consistency (GHI < DHI)", value=pcfg.get('fix_ghi_dhi', True), key="p_fix")
            
            do_clip = st.checkbox("Clip Precipitation Outliers", value=bool(pcfg.get('clip_precipitation')), key="p_clip_bool")
            if do_clip:
                current_clip = pcfg.get('clip_precipitation')
                # Guard against bools since isinstance(False, int) is True
                init_val = current_clip if isinstance(current_clip, (int, float)) and not isinstance(current_clip, bool) else 100
                pcfg['clip_precipitation'] = st.number_input("Max Rain (mm/h)", 
                                                           value=float(max(1, init_val)),
                                                           min_value=1.0, key="p_clip_val")
            else:
                pcfg['clip_precipitation'] = False
                
            pcfg['impute_missing_pv'] = st.checkbox("Impute Missing PV (CSI-based)", value=pcfg.get('impute_missing_pv', False), key="p_imp")
            
            st.markdown("---")
            st.markdown("##### ✂️ Limit Dataset (Subset)")
            subset_mode = pcfg.get('subset_mode', 'semua_data' if not pcfg.get('trim_rows') else 'baris')
            
            smode_list = ["Semua Data", "Batasi Jumlah Baris", "Rentang Tanggal (Date Range)"]
            smode_idx = 0
            if subset_mode == 'baris': smode_idx = 1
            elif subset_mode == 'tanggal': smode_idx = 2
            
            sel_smode = st.radio("Metode Pemotongan Data:", smode_list, index=smode_idx, key="p_smode_radio")
            
            if sel_smode == "Semua Data":
                pcfg['subset_mode'] = 'semua_data'
                pcfg['trim_rows'] = False
            elif sel_smode == "Batasi Jumlah Baris":
                pcfg['subset_mode'] = 'baris'
                pcfg['trim_rows'] = st.number_input("Ambil X Baris Pertama:", 
                                                   min_value=100, 
                                                   value=int(pcfg.get('trim_rows', 5000)) if pcfg.get('trim_rows') else 5000,
                                                   step=1000,
                                                   key="p_trim_val")
            elif sel_smode == "Rentang Tanggal (Date Range)":
                pcfg['subset_mode'] = 'tanggal'
                pcfg['trim_rows'] = False
                
                # Default dates fallback
                default_start = datetime(2021, 1, 1).date()
                default_end = datetime(2021, 12, 31).date()
                
                st.caption("Pilih rentang tanggal dataset yang ingin dipakai.")
                c_d1, c_d2 = st.columns(2)
                
                with c_d1:
                    curr_start = pcfg.get('start_date', default_start)
                    if isinstance(curr_start, str):
                        try: curr_start = datetime.strptime(curr_start, "%Y-%m-%d").date()
                        except: curr_start = default_start
                    d_start = st.date_input("Mulai:", value=curr_start, key="p_dstart")
                    pcfg['start_date'] = str(d_start)
                    
                with c_d2:
                    curr_end = pcfg.get('end_date', default_end)
                    if isinstance(curr_end, str):
                        try: curr_end = datetime.strptime(curr_end, "%Y-%m-%d").date()
                        except: curr_end = default_end
                    d_end = st.date_input("Sampai:", value=curr_end, key="p_dend")
                    pcfg['end_date'] = str(d_end)

            cfg['preprocessing'] = pcfg

        st.markdown("---")
        with st.expander("🛠️ Dataset Mapping (Columns)"):
            cfg['data']['csv_separator'] = st.text_input("CSV Separator (Prep)", value=cfg['data']['csv_separator'])
            cfg['data']['time_format'] = st.text_input("Time Format (Prep)", value=cfg['data']['time_format'])
            col_map = cfg['data']
            col_map['time_col'] = st.text_input("Kolom Waktu (Prep)", value=col_map.get('time_col', 'timestamp'))
            col_map['target_col'] = st.text_input("Kolom Target (PV) (Prep)", value=col_map.get('target_col', 'pv_output_kw'))
            col_map['ghi_col'] = st.text_input("Kolom GHI (Prep)", value=col_map.get('ghi_col', 'ghi_wm2'))
            cfg['data'] = col_map
            
        if st.button("💾 Simpan Konfigurasi Preprocessing", key="save_prep_cfg"):
            save_config_to_file(cfg)
            st.success("Konfigurasi preprocessing disimpan!")

    st.markdown("---")
    st.info("💡 **Alur**: Data Mentah → Cleaning → Feature Engineering → Scaling → Sequencing")
    
    col_prep_l, col_prep_r = st.columns([2, 1])
    with col_prep_l:
        st.markdown('<div class="pipeline-step">', unsafe_allow_html=True)
        st.markdown("**Run Preprocessing Pipeline**")
        st.caption("Proses ini akan menghasilkan file .npy dan scaler di folder data/processed.")
        v_name_prep = st.text_input("Nama Versi (Opsional)", placeholder="misal: v1_weather_only", key="v_name_prep")
        run_preprocess = st.button("▶️ Start Preprocessing", type="primary", width="stretch", key="btn_prep_main")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_prep_r:
        st.markdown("**Status Dataset**")
        if st.session_state.get('prep_metadata'):
            st.success("✅ Data Siap")
            # Show condensed stats
            stats = st.session_state.prep_metadata.get('stats', {})
            st.metric("Final Train Rows", stats.get('train_final', 0))
            st.metric("Final Test Rows", stats.get('test_final', 0))
        else:
            st.warning("⚠️ Data Belum Diproses")

    if run_preprocess:
        with st.spinner("Preprocessing sedang berjalan..."):
            import io, contextlib
            try:
                from src.data_prep import run_preprocessing
                stdout_capture = io.StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    # Pass the custom version name if provided
                    v_name = v_name_prep.strip() if v_name_prep.strip() else None
                    metadata = run_preprocessing(cfg, version_name=v_name)
                st.session_state.prep_metadata = metadata
                st.session_state.last_prep_log = stdout_capture.getvalue()
                st.session_state.pipeline_log.append(
                    f"[{datetime.now():%H:%M:%S}] Preprocessing selesai. "
                    f"Train: {metadata['X_train'].shape}"
                )
                st.success(f"Preprocessing selesai!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# --- TAB: EVALUATION RUNNER ---
with tab_eval:
    st.markdown("### Pipeline Runner")
    
    # Show last prep log if available
    if st.session_state.get('last_prep_log'):
        with st.expander("📜 Log Preprocessing Terakhir", expanded=False):
            st.code(st.session_state.last_prep_log, language="text")
            if st.button("🗑️ Clear Log"):
                st.session_state.last_prep_log = None
                st.rerun()
    st.markdown("Pilih tahap pipeline yang ingin dijalankan:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="pipeline-step">', unsafe_allow_html=True)
        st.markdown("**Run Evaluation**")
        st.caption("Visualisasikan performa model pada data validasi.")
        run_eval = st.button("▶️ Run Evaluation", width='stretch', key="btn_eval_runner",
                              disabled=not has_model)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="pipeline-step">', unsafe_allow_html=True)
        st.markdown("**Full Pipeline**")
        st.caption("Prep + Train + Eval dalam satu klik.")
        run_full = st.button("🚀 Full Pipeline", width='stretch',
                              type="primary", key="btn_full_runner")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === RUNNER EXECUTION ===
    output_container_runner = st.container()

    if run_eval:
        with output_container_runner:
            with st.spinner("Evaluasi sedang berjalan..."):
                try:
                    import tensorflow as tf
                    import gc
                    
                    # Clear GPU memory FIRST
                    tf.keras.backend.clear_session()
                    gc.collect()
                    
                    # Import AFTER clear_session so custom objects re-register
                    from src.model_factory import get_custom_objects
                    from src.predictor import evaluate_model
                    
                    # Load selected model
                    model_id = st.session_state.selected_model
                    model_path = os.path.join(model_dir, model_id)
                    model_root = model_dir # default parent
                    
                    # Handle bundled folder
                    if os.path.isdir(model_path):
                        model_root = model_path
                        # Try model.keras then model.h5
                        possible_files = [os.path.join(model_path, 'model.keras'), 
                                         os.path.join(model_path, 'model.h5')]
                        found = False
                        for pf in possible_files:
                            if os.path.exists(pf):
                                model_path = pf
                                found = True
                                break
                        if not found:
                            st.error(f"Bundle model tidak valid (model.keras/h5 tidak ditemukan): {model_id}")
                            st.stop()
                    
                    if not os.path.exists(model_path):
                        st.error(f"File model tidak ditemukan: {model_path}")
                        st.stop()
                        
                    st.info(f"Loading model: {model_id}")
                    custom_objs = get_custom_objects()
                    with tf.keras.utils.custom_object_scope(custom_objs):
                        # Use compile=False to avoid HDF5 object not found errors related to optimizer state
                        model = tf.keras.models.load_model(model_path, compile=False)
                    
                    # Re-compile manually with standard Adam
                    from src.model_factory import compile_model
                    compile_model(model, cfg['model']['hyperparameters']['learning_rate'])
                    
                    # Use model root for scalers
                    scaler_dir = model_root if os.path.isdir(model_root) else None
                    
                    import io, contextlib
                    stdout_capture = io.StringIO()
                    # Use data from session if available to avoid reloading from disk
                    data = st.session_state.get('prep_metadata', None)
                    
                    with contextlib.redirect_stdout(stdout_capture):
                        results = evaluate_model(model, cfg, data=data, scaler_dir=scaler_dir)
                    # Add model_id to results metadata
                    results['model_id'] = model_id
                    
                    st.session_state.eval_results = results
                    save_eval_results_to_disk(results)
                    st.session_state.pipeline_log.append(
                        f"[{datetime.now():%H:%M:%S}] Evaluasi selesai. "
                        f"Model: {model_id} | Test R²: {results['metrics_test']['r2']:.4f}"
                    )
                    st.success("Evaluasi selesai!")
                    with st.expander("📜 Output Detail"):
                        st.code(stdout_capture.getvalue(), language="text")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc(), language="text")

    if run_full:
        with output_container_runner:
            progress = st.progress(0, text="Memulai Full Pipeline...")
            try:
                import io, contextlib
                
                # Step 1
                progress.progress(10, text="[1/4] Preprocessing...")
                stdout_capture = io.StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    from src.data_prep import run_preprocessing
                    data = run_preprocessing(cfg)
                
                # Step 2
                if cfg['tuning']['enabled']:
                    progress.progress(30, text="[2/4] Optuna Tuning...")
                    with contextlib.redirect_stdout(stdout_capture):
                        from src.trainer import run_optuna_tuning
                        best, _ = run_optuna_tuning(cfg, data)
                        cfg['model']['hyperparameters'].update(best)
                else:
                    progress.progress(30, text="[2/4] Tuning dilewati")
                
                # Step 3: Training with live monitoring
                progress.progress(50, text="[3/4] Training...")
                st.markdown("#### Live Training Progress")
                full_progress = st.progress(0, text="Starting training...")
                full_chart = st.empty()
                full_lr = st.empty()
                
                import tensorflow as tf
                
                class FullPipelineLiveCallback(tf.keras.callbacks.Callback):
                    def __init__(self):
                        super().__init__()
                        self.epoch_data = []
                        self.start_time = None
                    
                    def on_train_begin(self, logs=None):
                        self.start_time = time.time()
                    
                    def on_epoch_end(self, epoch, logs=None):
                        elapsed = time.time() - self.start_time
                        total = self.params['epochs']
                        eta = (elapsed / (epoch+1)) * (total - epoch - 1)
                        eta_str = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
                        loss = logs.get('loss', 0)
                        val_loss = logs.get('val_loss', 0)
                        
                        self.epoch_data.append({'epoch': epoch+1, 'loss': loss, 'val_loss': val_loss})
                        
                        full_progress.progress(
                            (epoch+1)/total,
                            text=f"Epoch {epoch+1}/{total} | Loss: {loss:.6f} | Val: {val_loss:.6f} | ETA: {eta_str}"
                        )
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=[d['epoch'] for d in self.epoch_data],
                            y=[d['loss'] for d in self.epoch_data],
                            mode='lines', name='Train', line=dict(color='#818cf8', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=[d['epoch'] for d in self.epoch_data],
                            y=[d['val_loss'] for d in self.epoch_data],
                            mode='lines', name='Val', line=dict(color='#f472b6', width=2)
                        ))
                        fig.update_layout(
                            template="plotly_dark", height=300,
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=20, b=30),
                        )
                        full_chart.plotly_chart(fig, width='stretch')
                        
                        best_val = min(d['val_loss'] for d in self.epoch_data)
                        full_lr.caption(f"Best Val Loss: {best_val:.6f} | Elapsed: {elapsed:.0f}s")
                
                full_cb = FullPipelineLiveCallback()
                from src.trainer import train_model
                model, history, meta = train_model(cfg, data, extra_callbacks=[full_cb])
                st.session_state.training_history = history.history
                model_id = meta['model_id']
                st.session_state.selected_model = model_id
                save_training_history(history.history, model_id)
                save_selected_model(model_id)
                
                # Step 4
                progress.progress(80, text="[4/4] Evaluating...")
                with contextlib.redirect_stdout(stdout_capture):
                    from src.predictor import evaluate_model
                    results = evaluate_model(model, cfg, data)
                st.session_state.eval_results = results
                save_eval_results_to_disk(results)
                
                progress.progress(100, text="Pipeline selesai!")
                st.session_state.pipeline_log.append(
                    f"[{datetime.now():%H:%M:%S}] Full Pipeline selesai. "
                    f"R²={results['metrics_test']['r2']:.4f}, MAE={results['metrics_test']['mae']:.4f}"
                )
                st.success("Full Pipeline selesai!")
                st.session_state.training_history = load_training_history()
                st.session_state.eval_results = load_eval_results_from_disk()
                with st.expander("📜 Full Output"):
                    st.code(stdout_capture.getvalue(), language="text")
            except Exception as e:
                st.error(f"Error: {e}")
            st.rerun()

    # Move run_tune and run_tscv to their respective tabs

    # run_tune moved to Tab Tuning
    # run_tscv logic added below in Tab Training


# --- TAB: DATA INSIGHTS ---
with tab_data:
    st.markdown(f"### 🔎 {gt('data_insights', st.session_state.lang)}")
    st.caption("Detail metamorfosis data dari CSV mentah menjadi tensor yang siap dilatih oleh model.")
    
    # --- VERSION SELECTOR FOR INSIGHTS ---
    v_col1, v_col2 = st.columns([2, 1])
    with v_col1:
        # proc_dir is the root 'processed' folder found in the init section
        if os.path.exists(proc_dir):
            # RECURSIVE SEARCH: Find all subdirectories that contain prep_summary.json (up to 2 levels)
            all_versions_paths = []
            for d1 in os.listdir(proc_dir):
                p1 = os.path.join(proc_dir, d1)
                if os.path.isdir(p1):
                    if os.path.exists(os.path.join(p1, 'prep_summary.json')):
                        all_versions_paths.append(p1)
                    # Check one level deeper (for nested versions)
                    for d2 in os.listdir(p1):
                        p2 = os.path.join(p1, d2)
                        if os.path.isdir(p2) and os.path.exists(os.path.join(p2, 'prep_summary.json')):
                            all_versions_paths.append(p2)
            
            # Convert back to relative paths for display and storage
            all_versions = sorted([os.path.relpath(p, proc_dir).replace('\\', '/') for p in all_versions_paths], reverse=True)
            all_versions = list(dict.fromkeys(all_versions)) # Remove duplicates
            
            # Identify current active version name for the index
            current_active_path = os.path.abspath(st.session_state.cfg['paths']['processed_dir'])
            current_rel_v = os.path.relpath(current_active_path, proc_dir).replace('\\', '/')
            
            options = ["Latest (Default)"] + all_versions
            default_idx = 0
            if current_rel_v in all_versions and os.path.abspath(current_active_path) != os.path.abspath(proc_dir):
                default_idx = all_versions.index(current_rel_v) + 1
                
            sel_v = st.selectbox("📁 Load Archived Data Version:", options, index=default_idx, key="sel_v_insight_global")
            
            if sel_v == "Latest (Default)":
                target_path = proc_dir
            else:
                target_path = os.path.join(proc_dir, sel_v)
                
            # If user switches version, update the global active dir and reload metadata
            if os.path.abspath(target_path) != os.path.abspath(cfg['paths']['processed_dir']):
                cfg['paths']['processed_dir'] = target_path
                st.session_state.cfg = cfg
                st.session_state.prep_metadata = load_prep_metadata_from_disk(target_path)
                st.rerun()
        else:
            st.warning("Folder 'data/processed' belum ditemukan.")

    with v_col2:
        st.write("") # padding
        st.write("") 
        if st.button("🔄 Refresh Versions", width="stretch"):
            st.rerun()

    st.markdown("---")
    m = st.session_state.get('prep_metadata')
    if m:
        stats = m['stats']
        sel_f = m['selected_features']
        all_f = m['all_features']
        
        # --- PHASE 1: DATA CLEANING (ALGORITHM 1) ---
        st.markdown("#### 🛡️ Phase 1: Cleaning & Integrity (Algorithm 1)")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Data", f"{stats['original_rows']:,}", help="Total baris awal dari file CSV.")
        c2.metric("Post-Cleaning", f"{stats['after_algorithm1']:,}", 
                  delta=f"{stats['after_algorithm1'] - stats['original_rows']:,}",
                  help="Baris tersisa setelah Filter Fisika & Outlier.")
        c3.metric("NaN Dropped", f"{stats['dropped_missing']:,}", delta_color="inverse",
                  help="Baris yang dibuang karena memiliki nilai kosong (NaN).")
        c4.metric("Valid Sequences", f"{stats['train_final']:,}",
                  help="Total sequence (X, y) yang berhasil dibuat setelah pengecekan kontinuitas waktu.")

        # Visual Flow of Data Reduction
        flow_data = pd.DataFrame({
            'Stage': ['Original', 'After Cleaning', 'After NaN Drop', 'Final Sequences'],
            'Rows': [stats['original_rows'], stats['after_algorithm1'], 
                     stats['after_algorithm1'] - stats['dropped_missing'], stats['train_final']]
        })
        fig_flow = px.area(flow_data, x='Stage', y='Rows', title="Data Pipeline Flow (Volume Retention)")
        fig_flow.update_layout(template="plotly_dark", height=300, 
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_flow, width='stretch')

        st.markdown("---")
        
        # --- PHASE 2: FEATURE ENGINEERING & SELECTION ---
        st.markdown("#### 🎯 Phase 2: Feature Engineering & Selection")
        
        col_f1, col_f2 = st.columns([1, 1.5])
        with col_f1:
            raw_f = [f for f in all_f if not any(x in f for x in ['_lag_', '_ma_', '_std_', '_sin', '_cos'])]
            eng_f = [f for f in all_f if f not in raw_f]
            
            st.markdown(f"**Total Features Explored:** `{len(all_f)}`")
            
            # Pie Chart of Feature Types
            feat_types = pd.DataFrame({
                'Type': ['Raw Input', 'Engineered (Lag/Roll/Cyc)'],
                'Count': [len(raw_f), len(eng_f)]
            })
            fig_types = px.pie(feat_types, values='Count', names='Type', 
                               color_discrete_sequence=['#818cf8', '#f472b6'],
                               hole=0.4)
            fig_types.update_layout(template="plotly_dark", height=250, showlegend=False,
                                    margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_types, width='stretch')
            
            st.markdown(f"**Features Selected by Algorithm:** `{len(sel_f)}`")
            with st.expander("Lihat Daftar Fitur Final"):
                for i, f in enumerate(sel_f):
                    st.markdown(f"{i+1}. `{f}`")
        
        with col_f2:
            st.markdown("**Correlation Matrix (Selected Features)**")
            corr = m['corr_matrix']
            if corr is not None:
                # Filter heatmap to focus on target correlation if too many features
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    colorscale='RdBu',
                    zmin=-1, zmax=1
                ))
                fig_corr.update_layout(
                    template="plotly_dark", height=450,
                    margin=dict(t=30, b=30, l=30, r=30),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_corr, width='stretch')
            else:
                st.info("Matriks korelasi tidak tersedia.")

        st.markdown("---")
        
        # --- PHASE 2b: ROLLING CORRELATION ANALYSIS ---
        st.markdown("#### ⏳ Rolling Correlation Analysis")
        st.caption("Lihat bagaimana nilai korelasi antara fitur target dan fitur lainnya berubah seiring baris data (waktu).")
        
        target_candidates = ['csi_target', 'pv_cs_normalized', 'pv_output_kw', cfg['data']['target_col']]
        target_col_opts = [c for c in all_f if c in target_candidates]
        if not target_col_opts:
            target_col_opts = [all_f[-1]]
            
        c_r1, c_r2, c_r3 = st.columns([1, 2, 1])
        with c_r1:
            roll_target = st.selectbox("Fitur Target (Y):", all_f, index=all_f.index(target_col_opts[0]), key="roll_target")
        with c_r2:
            default_feats = [f for f in sel_f if f != roll_target][:3]
            if not default_feats:
                default_feats = [f for f in all_f if f != roll_target][:3]
            roll_features = st.multiselect("Bandingkan Korelasi dengan (X):", [f for f in all_f if f != roll_target], default=default_feats, key="roll_features")
        with c_r3:
            roll_window = st.number_input("Rolling Window Size", min_value=24, max_value=5000, value=720, step=24, help="Jumlah baris data (misal 720 jam = 30 hari) untuk 1 nilai korelasi.", key="roll_window")
            
        if roll_features:
            train_feats_path = os.path.join(cfg['paths']['processed_dir'], 'df_train_feats.pkl')
            if os.path.exists(train_feats_path):
                if st.button("📊 Generate Rolling Correlation Chart", type="primary"):
                    with st.spinner("Menghitung rolling correlation sepanjang waktu..."):
                        import numpy as np
                        df_roll = pd.read_pickle(train_feats_path)
                        
                        fig_rc = go.Figure()
                        x_axis = np.arange(len(df_roll))
                        for f in roll_features:
                            # Hitung korelasi
                            rolling_corr = df_roll[roll_target].rolling(window=roll_window).corr(df_roll[f])
                            fig_rc.add_trace(go.Scatter(
                                x=x_axis, 
                                y=rolling_corr, 
                                mode='lines', 
                                name=f"{f} vs {roll_target}",
                                line=dict(width=2)
                            ))
                            
                        # Add zero line
                        fig_rc.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                            
                        fig_rc.update_layout(
                            template="plotly_dark",
                            title=f"Perubahan Dinamis Korelasi (Window={roll_window} baris)",
                            xaxis_title="Indeks Baris Data (Waktu)",
                            yaxis_title="Nilai Korelasi (Pearson)",
                            yaxis=dict(range=[-1.05, 1.05]),
                            height=400,
                            margin=dict(t=50, b=20, l=20, r=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_rc, width="stretch")
            else:
                st.warning("File `df_train_feats.pkl` tidak ditemukan di folder processed. Jalankan ulang preprocessing.")
        else:
            st.info("Pilih minimal 1 fitur untuk dibandingkan.")
            
        st.markdown("---")
        
        # --- PHASE 3: DATA SPLITTING & SEQUENCING ---
        st.markdown("#### ✂️ Phase 3: Dataset Splitting & Scaling")
        
        c1, c2 = st.columns(2)
        with c1:
            split_data = pd.DataFrame({
                'Set': ['Training', 'Validation'],
                'Sequences': [stats['train_final'], stats['test_final']]
            })
            fig_split = px.bar(split_data, x='Set', y='Sequences', color='Set',
                               color_discrete_map={'Training': '#818cf8', 'Validation': '#6366f1'})
            fig_split.update_layout(template="plotly_dark", height=300, showlegend=False,
                                    title="Train/Val Distribution")
            st.plotly_chart(fig_split, width='stretch')
            
        with c2:
            st.markdown("**Input Template (Sequence Sample)**")
            
            # Check if X_train is in memory
            if 'X_train' in m and m['X_train'] is not None:
                seq_sample = m['X_train'][0]
                df_seq = pd.DataFrame(seq_sample, columns=sel_f)
                st.dataframe(df_seq.head(10), width='stretch')
                st.caption(f"Menampilkan 10 timestep pertama dari sequence ke-0 (Tensor Shape: {m['X_train'].shape})")
            else:
                # Offer to load X_train.npy if it exists in the active folder
                x_train_path = os.path.join(cfg['paths']['processed_dir'], 'X_train.npy')
                if os.path.exists(x_train_path):
                    st.info("Preview tensor tidak di memori (Mode Hemat RAM).")
                    if st.button("📥 Load Sequence Preview"):
                        try:
                            # Load into the metadata dictionary
                            m['X_train'] = np.load(x_train_path)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Gagal memuat preview: {e}")
                else:
                    st.info("Preview tensor tidak tersedia untuk versi ini.")
    else:
        st.info("Belum ada data preprocessing. Silakan jalankan 'Step 1: Preprocessing' pada tab Runner.")


# --- TAB: TRAINING CENTER ---
with tab_train:
    st.markdown("### 🧠 Training Center")
    
    # --- DATA VERSION SELECTOR ---
    with st.expander("📦 Pilih Versi Data Preprocessed", expanded=not has_data):
        # Always use the root processed directory for listing
        if os.path.exists(proc_dir):
            versions = [f for f in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, f)) and os.path.exists(os.path.join(proc_dir, f, 'X_train.npy'))]
            versions = sorted(versions, reverse=True)
            options = ["Latest (Default)"] + versions
            
            # Select the most recent version if it was just created
            default_idx = 0
            selected_v = st.selectbox("Gunakan Versi Data untuk Training:", options, index=default_idx, key="data_version_train",
                                   format_func=lambda x: label_format_with_time(x, proc_dir))
            
            if st.button("🔄 Refresh Daftar Versi"):
                st.rerun()
            
            if selected_v == "Latest (Default)":
                active_proc_dir = proc_dir
            else:
                active_proc_dir = os.path.join(proc_dir, selected_v)
            
            cfg['paths']['processed_dir'] = active_proc_dir
            st.session_state.active_proc_dir = active_proc_dir
            
            # Check if this version is actually valid (has npy files)
            is_valid = os.path.exists(os.path.join(active_proc_dir, 'X_train.npy'))
            if is_valid:
                st.caption(f"Folder Aktif: `{os.path.basename(active_proc_dir)}` ✅")
            else:
                st.error(f"⚠️ Folder `{os.path.basename(active_proc_dir)}` tidak berisi data mapping (X_train.npy tidak ditemukan).")
                st.info("Pilih 'Latest (Default)' atau jalankan ulang Preprocessing.")
        else:
            st.warning("Folder processed belum ada. Silakan jalankan Preprocessing terlebih dahulu.")

    st.markdown("Konfigurasikan arsitektur model dan jalankan proses training di sini.")
    
    # --- Training Readiness Dashboard ---
    with st.container():
        st.markdown("#### 🚦 Training Readiness Check")
        r_col1, r_col2, r_col3 = st.columns(3)
        
        # 1. Preprocessing Readiness
        with r_col1:
            prep_ok = has_data
            status_icon = "✅" if prep_ok else "❌"
            st.markdown(f"""
            <div style="background: rgba(26, 31, 46, 0.6); padding: 15px; border-radius: 8px; border-left: 5px solid {'#22c55e' if prep_ok else '#ef4444'};">
                <div style="font-size: 0.8rem; color: #94a3b8;">Preprocessing Status</div>
                <div style="font-size: 1.1rem; font-weight: 700;">{status_icon} Data Processed</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 5px;">
                    {f"Found {st.session_state.prep_metadata['stats']['train_final']:,} samples" if st.session_state.prep_metadata else "No processed data found"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Feature Readiness
        with r_col2:
            feat_ok = st.session_state.prep_metadata is not None and len(st.session_state.prep_metadata.get('selected_features', [])) > 0
            status_icon = "✅" if feat_ok else "⚠️"
            st.markdown(f"""
            <div style="background: rgba(26, 31, 46, 0.6); padding: 15px; border-radius: 8px; border-left: 5px solid {'#818cf8' if feat_ok else '#f59e0b'};">
                <div style="font-size: 0.8rem; color: #94a3b8;">Feature Readiness</div>
                <div style="font-size: 1.1rem; font-weight: 700;">{status_icon} Vectors Ready</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 5px;">
                    {f"{len(st.session_state.prep_metadata['selected_features'])} features active" if feat_ok else "Run preprocessing first"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 3. Model Config Readiness
        with r_col3:
            model_ok = cfg['model']['hyperparameters']['lookback'] > 0
            st.markdown(f"""
            <div style="background: rgba(26, 31, 46, 0.6); padding: 15px; border-radius: 8px; border-left: 5px solid #6366f1;">
                <div style="font-size: 0.8rem; color: #94a3b8;">Target Architecture</div>
                <div style="font-size: 1.1rem; font-weight: 700;">🤖 {cfg['model'].get('architecture', 'patchtst').upper()}</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 5px;">
                    Window: {cfg['model']['hyperparameters']['lookback']}h | BS: {cfg['model']['hyperparameters']['batch_size']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # 1. Hyperparameter Configuration Area
    with st.expander("🛠️ Model Architecture & Hyperparameters", expanded=True):
        col_hp1, col_hp2 = st.columns(2)
        hp = cfg['model']['hyperparameters']
        arch = cfg['model'].get('architecture', 'patchtst').lower()
        
        def _update_architecture():
            new_a = st.session_state.arch_selector_train
            cfg['model']['architecture'] = new_a
            # Auto-update hyperparameters to defaults for the new architecture
            if new_a == 'patchtst':
                cfg['model']['hyperparameters'].update({
                    'd_model': 128, 'n_layers': 3, 'learning_rate': 0.0001, 
                    'batch_size': 32, 'dropout': 0.2, 'patch_len': 16, 
                    'stride': 8, 'n_heads': 16
                })
            elif new_a == 'timetracker':
                cfg['model']['hyperparameters'].update({
                    'd_model': 64, 'n_layers': 2, 'learning_rate': 0.0005, 
                    'batch_size': 32, 'dropout': 0.1, 'patch_len': 16, 
                    'stride': 8, 'n_heads': 8, 'n_shared_experts': 1, 
                    'n_private_experts': 4, 'top_k': 2
                })
            elif new_a == 'timeperceiver':
                cfg['model']['hyperparameters'].update({
                    'd_model': 128, 'n_layers': 2, 'learning_rate': 0.0005, 
                    'batch_size': 32, 'dropout': 0.1, 'patch_len': 16, 
                    'stride': 8, 'n_heads': 8, 'n_latent_tokens': 32
                })
            else: # gru, lstm, rnn
                cfg['model']['hyperparameters'].update({
                    'd_model': 64, 'n_layers': 2, 'learning_rate': 0.001, 
                    'batch_size': 32, 'dropout': 0.2, 'use_bidirectional': True
                })
            
            # C-Delete keys that are no longer relevant to avoid config pollution
            if new_a != 'patchtst':
                for k in ['patch_len', 'stride', 'n_heads', 'ff_dim']:
                    cfg['model']['hyperparameters'].pop(k, None)
            else:
                cfg['model']['hyperparameters'].pop('use_bidirectional', None)
            
            save_config_to_file(cfg)
            st.session_state.cfg = cfg
            st.session_state.tune_arch_selector = new_a

        with col_hp1:
            st.markdown("**Core Structure**")
            _dummy = st.selectbox("Arsitektur Model", ["patchtst", "timetracker", "timeperceiver", "gru", "lstm", "rnn"], 
                                  index=["patchtst", "timetracker", "timeperceiver", "gru", "lstm", "rnn"].index(arch),
                                  key="arch_selector_train",
                                  on_change=_update_architecture)
            
            # Rebind new_arch precisely to the confirmed architecture
            new_arch = cfg['model']['architecture']
            
            hp['lookback'] = st.select_slider("Lookback Window (h)", 
                                              options=[24, 48, 72, 96, 120, 144, 168, 192, 240, 336],
                                              value=hp.get('lookback', 72))
            
            # Adaptive Labels for Core Structure
            if new_arch == "patchtst":
                d_label = "d_model (Embedding Dimension)"
                l_label = "Transformer Blocks"
            elif new_arch == "timetracker":
                d_label = "d_model (Token Dimension)"
                l_label = "MoE Transformer Layers"
            elif new_arch == "timeperceiver":
                d_label = "d_model (Patch Embedding)"
                l_label = "Latent Self-Attention Layers (K)"
            else:
                d_label = f"Hidden Units ({new_arch.upper()} Dimension)"
                l_label = f"Stacked {new_arch.upper()} Layers"
            
            _d_model_opts = sorted(set([16, 32, 64, 128, 256, 512] + [hp.get('d_model', 128)]))
            hp['d_model'] = st.selectbox(d_label, _d_model_opts, 
                                          index=_d_model_opts.index(hp.get('d_model', 128)))
            hp['n_layers'] = st.number_input(l_label, value=hp.get('n_layers', 3), min_value=1, max_value=12)
            
        with col_hp2:
            st.markdown("**Optimization**")
            
            _loss_opts = ['mse', 'huber', 'mae']
            hp['loss_fn'] = st.selectbox("Loss Function", _loss_opts, index=_loss_opts.index(hp.get('loss_fn', 'mse')) if hp.get('loss_fn', 'mse') in _loss_opts else 0)
            
            hp['learning_rate'] = st.number_input("Learning Rate", value=hp.get('learning_rate', 0.0001),
                                                   format="%.6f", step=0.0001)
            _bs_opts = sorted(set([16, 32, 64, 128] + [hp.get('batch_size', 32)]))
            hp['batch_size'] = st.selectbox("Batch Size", _bs_opts,
                                             index=_bs_opts.index(hp.get('batch_size', 32)))
            hp['dropout'] = st.number_input("Dropout Rate", value=hp.get('dropout', 0.2), 
                                             min_value=0.0, max_value=0.9, step=0.01, format="%.2f")
            
            # --- ARCHITECTURE SPECIFIC PARAMS ---
            if new_arch == "patchtst":
                with st.expander("🧩 PatchTST Specific Params", expanded=True):
                    hp['patch_len'] = st.number_input("patch_len (P)", value=hp.get('patch_len', 16), min_value=2, step=2)
                    hp['stride'] = st.number_input("stride (S)", value=hp.get('stride', 8), min_value=1, step=1)
                    hp['ff_dim'] = st.number_input("ff_dim (F)", value=hp.get('ff_dim', hp['d_model'] * 2), min_value=32, step=32)
                    _nheads_opts = sorted(set([1, 2, 4, 8, 12, 16] + [hp.get('n_heads', 16)]))
                    hp['n_heads'] = st.selectbox("n_heads (H)", _nheads_opts, index=_nheads_opts.index(hp.get('n_heads', 16)))
            
            elif new_arch == "timetracker":
                with st.expander("⏱️ TimeTracker Specific Params", expanded=True):
                    hp['patch_len'] = st.number_input("patch_len (P)", value=hp.get('patch_len', 16), min_value=2, step=2)
                    hp['stride'] = st.number_input("stride (S)", value=hp.get('stride', 8), min_value=1, step=1)
                    _nheads_opts = sorted(set([1, 2, 4, 8, 12, 16] + [hp.get('n_heads', 8)]))
                    hp['n_heads'] = st.selectbox("n_heads (H) - Any-variate Rel", _nheads_opts, index=_nheads_opts.index(hp.get('n_heads', 8)))
                    st.markdown("**Mixture of Experts Setup**")
                    c_e1, c_e2 = st.columns(2)
                    hp['n_shared_experts'] = c_e1.number_input("Shared Experts", value=hp.get('n_shared_experts', 1), min_value=0, max_value=8)
                    hp['n_private_experts'] = c_e2.number_input("Private Experts", value=hp.get('n_private_experts', 4), min_value=1, max_value=32)
                    hp['top_k'] = st.number_input("Top-K Routing", value=hp.get('top_k', 2), min_value=1, max_value=hp['n_private_experts'], help="Berapa expert private yang aktif untuk setiap token")

            elif new_arch == "timeperceiver":
                with st.expander("👁️ TimePerceiver Specific Params", expanded=True):
                    hp['patch_len'] = st.number_input("patch_len (P)", value=hp.get('patch_len', 16), min_value=2, step=2)
                    hp['stride'] = st.number_input("stride (S)", value=hp.get('stride', 8), min_value=1, step=1)
                    _nheads_opts = sorted(set([1, 2, 4, 8, 12, 16] + [hp.get('n_heads', 8)]))
                    hp['n_heads'] = st.selectbox("n_heads (H) - Bottleneck", _nheads_opts, index=_nheads_opts.index(hp.get('n_heads', 8)))
                    hp['n_latent_tokens'] = st.number_input("Latent Tokens (M)", value=hp.get('n_latent_tokens', 32), min_value=4, max_value=256, step=4, help="Ukuran bottleneck (M) untuk attention laten")
            
            elif new_arch in ["gru", "lstm", "rnn"]:
                with st.expander(f"🔄 {new_arch.upper()} Specific Params", expanded=True):
                    st.info(f"Input 'Hidden Units' di atas menentukan kapasitas memori per {new_arch.upper()} cell.")
                    hp['use_bidirectional'] = st.checkbox("Use Bidirectional", value=hp.get('use_bidirectional', True), key=f"bi_{new_arch}")
                    hp['use_revin'] = st.checkbox("Gunakan RevIN (Reversible Instance Normalization)", value=hp.get('use_revin', False), key=f"revin_{new_arch}")
                    st.caption(f"Arsitektur {new_arch.upper()} dapat ditambah lapisan pelindung anti-anomali RevIN.")

    # 2. Training Control Center
    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        st.markdown('<div class="pipeline-step">', unsafe_allow_html=True)
        st.markdown("**Execution Control**")
        
        c1, c2 = st.columns(2)
        with c1:
            device = st.radio("Device", ["GPU", "CPU"], index=0 if st.session_state.get('execution_device', 'GPU') == 'GPU' else 1, horizontal=True)
            st.session_state.execution_device = device
            m_name_train = st.text_input("Nama Model (ID)", placeholder="misal: patchtst_v1_exp1", key="m_name_train")
        with c2:
            st.write("") # mapping
            run_train = st.button("🚀 Start Model Training", type="primary", width="stretch", disabled=not has_data)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_ctrl2:
        st.markdown("**Status**")
        if has_data:
            st.success("✅ Training Data Ready")
            st.caption(f"Sequences: {st.session_state.prep_metadata['stats']['train_final'] if st.session_state.prep_metadata else 'Loaded'}")
        else:
            st.error("❌ Data missing! Run Prep first.")

    # 3. Training Execution Logic (Consolidated here)
    if run_train:
        output_container_train = st.container()
        with output_container_train:
            st.markdown("---")
            st.markdown("### 📡 Live Training Monitor")
            
            progress_bar = st.progress(0, text="Initializing Model...")
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            epoch_display = col_info1.empty()
            loss_display = col_info2.empty()
            val_loss_display = col_info3.empty()
            eta_display = col_info4.empty()
            chart_placeholder = st.empty()
            lr_display = st.empty()
            log_expander = st.expander("Detailed Epoch Logs", expanded=False)
            log_placeholder = log_expander.empty()
            
            try:
                import tensorflow as tf
                # (Reuse the Callback logic within this block or define high up)
                class StreamlitLiveCallback(tf.keras.callbacks.Callback):
                    def __init__(self):
                        super().__init__()
                        self.epoch_data = []
                        self.start_time = None
                        self.log_lines = []
                    def on_train_begin(self, logs=None): self.start_time = time.time()
                    def on_epoch_end(self, epoch, logs=None):
                        elapsed = time.time() - self.start_time
                        total_epochs = self.params['epochs']
                        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                        loss = logs.get('loss', 0); val_loss = logs.get('val_loss', 0)
                        progress = (epoch + 1) / total_epochs
                        avg_per_epoch = elapsed / (epoch + 1)
                        eta = avg_per_epoch * (total_epochs - epoch - 1)
                        eta_str = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
                        self.epoch_data.append({'epoch': epoch+1,'loss': loss,'val_loss': val_loss,'lr': current_lr})
                        progress_bar.progress(progress, text=f"Epoch {epoch+1}/{total_epochs} | ETA: {eta_str}")
                        epoch_display.metric("Epoch", f"{epoch+1}/{total_epochs}")
                        loss_display.metric("Train Loss", f"{loss:.6f}")
                        val_loss_display.metric("Val Loss", f"{val_loss:.6f}")
                        eta_display.metric("ETA", eta_str)
                        
                        fig = go.Figure()
                        epochs_list = [d['epoch'] for d in self.epoch_data]
                        fig.add_trace(go.Scatter(x=epochs_list, y=[d['loss'] for d in self.epoch_data], name='Train', line=dict(color='#818cf8', width=2)))
                        fig.add_trace(go.Scatter(x=epochs_list, y=[d['val_loss'] for d in self.epoch_data], name='Val', line=dict(color='#f472b6', width=2)))
                        fig.update_layout(template="plotly_dark", height=350, margin=dict(t=20, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                        chart_placeholder.plotly_chart(fig, width="stretch")
                        lr_display.caption(f"LR: {current_lr:.8f} | Best Val: {min(d['val_loss'] for d in self.epoch_data):.6f}")
                        
                        # Fix: Add explicit print to CMD so user can see progress there too
                        log_msg = f"Epoch {epoch+1}/{total_epochs}: loss={loss:.6f}, val_loss={val_loss:.6f}, lr={current_lr:.8f}"
                        print(f"  > {log_msg}")
                        self.log_lines.append(log_msg)
                        log_placeholder.code('\n'.join(self.log_lines))

                # Device selection
                if device == 'CPU': os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                else: os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                
                print(f"\n[DEBUG] Entering Training block. Device: {device}")
                import sys
                sys.stdout.flush()

                tf.keras.backend.clear_session()
                live_cb = StreamlitLiveCallback()
                from src.trainer import train_model
                
                print(f"[DEBUG] Calling train_model from src.trainer...")
                sys.stdout.flush()

                custom_id = m_name_train.strip() if m_name_train.strip() else None
                model, history, meta = train_model(cfg, extra_callbacks=[live_cb], custom_model_id=custom_id, loss_fn=hp.get('loss_fn', 'mse'))
                
                # Success & Persist
                st.session_state.training_history = history.history
                st.session_state['last_training_time'] = meta.get('training_time_seconds', 0)
                save_training_history(history.history, meta['model_id'])
                
                # CRITICAL: Auto-select the newly trained model for evaluation
                st.session_state.selected_model = meta['model_id']
                
                duration_str = f"{meta['training_time_seconds']:.2f}s" if meta['training_time_seconds'] < 60 else f"{meta['training_time_seconds']/60:.2f}m"
                st.success(f"✅ Training Selesai! ({duration_str}) | Model saved: {meta['model_id']}")
                time.sleep(2) 
                st.rerun()
            except Exception as e:
                st.error(f"Training Failed: {e}")
                import traceback; st.code(traceback.format_exc())

    # 4. Results & Metrics (Static view if not training)
    if st.session_state.training_history and not run_train:
        st.markdown("---")
        st.subheader("📊 Last Training Results")
        hist = st.session_state.training_history
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(y=hist['loss'], name='Train', line=dict(color='#818cf8')))
            fig_l.add_trace(go.Scatter(y=hist['val_loss'], name='Val', line=dict(color='#f472b6')))
            fig_l.update_layout(template="plotly_dark", height=350, title="Final Loss Curves", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_l, width="stretch")
        with c2:
            st.metric("Best Val Loss", f"{min(hist['val_loss']):.6f}")
            st.metric("Total Epochs", len(hist['loss']))
            
            if 'last_training_time' in st.session_state:
                t_sec = st.session_state['last_training_time']
                t_str = f"{t_sec:.2f}s" if t_sec < 60 else f"{t_sec/60:.2f}m"
                st.metric("Time Elapsed", t_str)
            
            if st.button("🗑️ Clear History View"):
                st.session_state.training_history = None
                st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Time Series Cross-Validation (TSCV)")
    st.caption("Uji stabilitas model pada berbagai potongan waktu.")
    run_tscv = st.button("▶️ Run TSCV Evaluation", width='stretch', key="btn_tscv_tab")
    
    if run_tscv:
        with st.spinner("Running TSCV..."):
            from src.trainer import run_tscv
            res = run_tscv(cfg)
            st.dataframe(pd.DataFrame(res))

# --- TAB BATCH: SEQUENTIAL TRAINING ---
with tab_batch:
    try:
        st.markdown(f"### {gt('batch_manager_title', st.session_state.lang)}")
        st.info(gt('batch_info', st.session_state.lang))
        
        if 'batch_queue' not in st.session_state:
            st.session_state.batch_queue = []
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = []
        if 'batch_running' not in st.session_state:
            st.session_state.batch_running = False
        
        col_bq1, col_bq2 = st.columns([1, 2])
        
        with col_bq1:
            st.markdown(f"#### {gt('add_to_queue', st.session_state.lang)}")
            
            # 1. Architecture Selection
            arch_list = ["patchtst", "timetracker", "timeperceiver", "gru", "lstm", "rnn"]
            
            # Default to global active architecture if not set
            if 'batch_arch_selector' not in st.session_state:
                st.session_state.batch_arch_selector = cfg['model'].get('architecture', 'patchtst').lower()
                
            q_arch_val = st.selectbox(gt('architecture', st.session_state.lang), 
                                     arch_list, 
                                     key="batch_arch_selector")
            
            q_name = st.text_input(gt('exp_name', st.session_state.lang), 
                                  value=f"Exp_{q_arch_val}_{len(st.session_state.batch_queue)+1}",
                                  key=f"batch_exp_name_{q_arch_val}")
            
            with st.expander(gt('config_hp', st.session_state.lang), expanded=True):
                # Start with a fresh set of defaults for the selected architecture
                # instead of blindly copying from the global active model
                q_hp = {}
                
                # Base defaults
                base_defaults = {
                    'lookback': 72, 'learning_rate': 0.0001, 'batch_size': 32, 'dropout': 0.2, 'd_model': 64, 'n_layers': 2
                }
                
                if q_arch_val == 'patchtst':
                    q_hp.update(base_defaults)
                    q_hp.update({'d_model': 128, 'n_layers': 3, 'patch_len': 16, 'stride': 8, 'n_heads': 16, 'ff_dim': 256})
                    d_label, l_label = "d_model (Embedding)", "Transformer Blocks"
                elif q_arch_val == 'timetracker':
                    q_hp.update(base_defaults)
                    q_hp.update({'d_model': 64, 'n_layers': 2, 'patch_len': 16, 'stride': 8, 'n_heads': 8, 'n_shared_experts': 1, 'n_private_experts': 4, 'top_k': 2})
                    d_label, l_label = "d_model (Token Dim)", "MoE Layers"
                elif q_arch_val == 'timeperceiver':
                    q_hp.update(base_defaults)
                    q_hp.update({'d_model': 128, 'n_layers': 2, 'patch_len': 16, 'stride': 8, 'n_heads': 8, 'n_latent_tokens': 32})
                    d_label, l_label = "d_model (Patch Embed)", "Latent Attention Layers"
                else: # gru, lstm, rnn
                    q_hp.update(base_defaults)
                    q_hp.update({'learning_rate': 0.001, 'use_bidirectional': True, 'use_revin': False})
                    d_label, l_label = f"Hidden Units ({q_arch_val.upper()})", f"Stacked {q_arch_val.upper()} Layers"
                
                cq1, cq2 = st.columns(2)
                with cq1:
                    q_hp['lookback'] = st.selectbox("Lookback Window (h)", [24, 48, 72, 96, 120, 144, 168], index=2, key=f"q_lb_{q_arch_val}")
                    
                    _d_opts = [16, 32, 64, 128, 256, 512]
                    q_hp['d_model'] = st.selectbox(d_label, _d_opts, index=_d_opts.index(q_hp.get('d_model', 128)) if q_hp.get('d_model', 128) in _d_opts else 3, key=f"q_dm_{q_arch_val}")
                    q_hp['n_layers'] = st.number_input(l_label, 1, 12, q_hp.get('n_layers', 3), key=f"q_nl_{q_arch_val}")
                    
                with cq2:
                    _loss_opts = ['mse', 'huber', 'mae']
                    q_hp['loss_fn'] = st.selectbox("Loss Function", _loss_opts, index=_loss_opts.index(q_hp.get('loss_fn', 'mse')) if q_hp.get('loss_fn', 'mse') in _loss_opts else 0, key=f"q_loss_{q_arch_val}")
                    q_hp['learning_rate'] = st.number_input("Learning Rate", 0.00001, 0.01, q_hp.get('learning_rate', 0.0001), format="%.5f", key=f"q_lr_{q_arch_val}")
                    _bs_opts = [16, 32, 64, 128]
                    q_hp['batch_size'] = st.selectbox("Batch Size", _bs_opts, index=_bs_opts.index(q_hp.get('batch_size', 32)) if q_hp.get('batch_size', 32) in _bs_opts else 1, key=f"q_bs_{q_arch_val}")
                    q_hp['dropout'] = st.number_input("Dropout", 0.0, 0.9, q_hp.get('dropout', 0.2), 0.05, key=f"q_dr_{q_arch_val}")

                # --- SPECIFIC PARAMS ---
                if q_arch_val == "patchtst":
                    st.markdown("---")
                    sq1, sq2 = st.columns(2)
                    with sq1:
                        q_hp['patch_len'] = st.number_input("Patch Len", 4, 64, q_hp.get('patch_len', 16), 4, key="q_pl")
                        q_hp['stride'] = st.number_input("Stride", 2, 32, q_hp.get('stride', 8), 2, key="q_st")
                    with sq2:
                        q_hp['ff_dim'] = st.number_input("ff_dim", 32, 1024, q_hp.get('ff_dim', q_hp.get('d_model', 128)*2), 32, key="q_ff")
                        _h_opts = [1, 2, 4, 8, 12, 16]
                        q_hp['n_heads'] = st.selectbox("n_heads", _h_opts, index=_h_opts.index(q_hp.get('n_heads', 16)) if q_hp.get('n_heads', 16) in _h_opts else 5, key="q_nh")
                
                elif q_arch_val == "timetracker":
                    st.markdown("---")
                    sq1, sq2 = st.columns(2)
                    with sq1:
                        q_hp['patch_len'] = st.number_input("Patch Len", 4, 64, q_hp.get('patch_len', 16), 4, key="q_pl_tt")
                        q_hp['stride'] = st.number_input("Stride", 2, 32, q_hp.get('stride', 8), 2, key="q_st_tt")
                        _h_opts = [1, 2, 4, 8, 12, 16]
                        q_hp['n_heads'] = st.selectbox("n_heads (H)", _h_opts, index=_h_opts.index(q_hp.get('n_heads', 8)) if q_hp.get('n_heads', 8) in _h_opts else 3, key="q_nh_tt")
                    with sq2:
                        q_hp['n_shared_experts'] = st.number_input("Shared Experts", 0, 8, q_hp.get('n_shared_experts', 1), key="q_se_tt")
                        q_hp['n_private_experts'] = st.number_input("Private Experts", 1, 32, q_hp.get('n_private_experts', 4), key="q_pe_tt")
                        q_hp['top_k'] = st.number_input("Top-K Routing", 1, 32, q_hp.get('top_k', 2), key="q_tk_tt")

                elif q_arch_val == "timeperceiver":
                    st.markdown("---")
                    sq1, sq2 = st.columns(2)
                    with sq1:
                        q_hp['patch_len'] = st.number_input("Patch Len", 4, 64, q_hp.get('patch_len', 16), 4, key="q_pl_tp")
                        q_hp['stride'] = st.number_input("Stride", 2, 32, q_hp.get('stride', 8), 2, key="q_st_tp")
                    with sq2:
                        _h_opts = [1, 2, 4, 8, 12, 16]
                        q_hp['n_heads'] = st.selectbox("n_heads (H)", _h_opts, index=_h_opts.index(q_hp.get('n_heads', 8)) if q_hp.get('n_heads', 8) in _h_opts else 3, key="q_nh_tp")
                        q_hp['n_latent_tokens'] = st.number_input("Latent Tokens (M)", 4, 256, q_hp.get('n_latent_tokens', 32), 4, key="q_lt_tp")
                
                elif q_arch_val in ["gru", "lstm", "rnn"]:
                    st.markdown("---")
                    sq1, sq2 = st.columns(2)
                    with sq1:
                        q_hp['use_bidirectional'] = st.checkbox("Use Bidirectional", value=q_hp.get('use_bidirectional', True), key=f"q_bi_{q_arch_val}")
                    with sq2:
                        q_hp['use_revin'] = st.checkbox("Gunakan RevIN", value=q_hp.get('use_revin', False), key=f"q_rev_{q_arch_val}")
            
            with st.expander(gt('config_data_feat', st.session_state.lang), expanded=False):
                st.caption("Pilih versi data atau konfigurasi fitur")
                
                # List available preprocessed versions
                cur_dyn_str = gt('current_dynamic', st.session_state.lang)
                proc_versions = [cur_dyn_str]
                if os.path.exists(proc_dir):
                    dirs = [d for d in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, d)) and os.path.exists(os.path.join(proc_dir, d, 'X_train.npy'))]
                    proc_versions.extend(sorted(dirs, reverse=True))
                
                q_data_v = st.selectbox(gt('data_version_select', st.session_state.lang), proc_versions, key="q_data_v",
                                      format_func=lambda x: label_format_with_time(x, proc_dir))
                
                st.markdown("---")
                st.caption(gt('feature_groups_info', st.session_state.lang))
                q_feat = cfg['features']['groups'].copy()
                q_feat['weather'] = st.checkbox("Weather Features", value=q_feat.get('weather', True), key="q_f_w")
                q_feat['time_hour'] = st.checkbox("Hour of Day", value=q_feat.get('time_hour', True), key="q_f_h")
                q_feat['time_month'] = st.checkbox("Month of Year", value=q_feat.get('time_month', True), key="q_f_m")
                q_feat['physics'] = st.checkbox("Physics (CS Index)", value=q_feat.get('physics', False), key="q_f_p")
                q_feat_mode = st.selectbox("Selection Mode", ["auto", "manual"], key="q_f_mode")

            if st.button(gt('add_to_queue', st.session_state.lang), width="stretch", key="btn_add_batch_queue", disabled=st.session_state.batch_running):
                st.session_state.batch_queue.append({
                    "name": q_name,
                    "architecture": q_arch_val,
                    "hp": q_hp,
                    "data_version": q_data_v,
                    "features": {
                        "groups": q_feat,
                        "selection_mode": q_feat_mode
                    }
                })
                # Log to CMD for debugging
                print(f"➕ [{datetime.now().strftime('%H:%M:%S')}] Queued: {q_name} ({q_arch_val})")
                import sys
                sys.stdout.flush()
                
                st.success(gt('add_to_queue_success', st.session_state.lang))
                time.sleep(0.5)
                st.rerun()

        with col_bq2:
            st.markdown(f"#### {gt('current_queue', st.session_state.lang)}")
            if not st.session_state.batch_queue:
                st.write("Antrean kosong. Tambahkan model di sebelah kiri.")
            else:
                for i, item in enumerate(st.session_state.batch_queue):
                    col_i1, col_i2 = st.columns([4, 1])
                    col_i1.markdown(f"**{i+1}. {item['name']}** ({item['architecture']}) | LB: {item['hp']['lookback']}, D: {item['hp']['d_model']}")
                    if col_i2.button("🗑️", key=f"del_{i}"):
                        st.session_state.batch_queue.pop(i)
                        st.rerun()
                
                st.markdown("---")
                if st.button(gt('run_batch_btn', st.session_state.lang), type="primary", width="stretch", disabled=st.session_state.batch_running):
                    st.session_state.batch_running = True
                    st.session_state.batch_results = [] # Reset results for new run
                    
                    batch_progress = st.progress(0)
                    status_text = st.empty()
                    
                    # Container for live monitor (will be persistent until clear)
                    monitor_container = st.container()
                    
                    total = len(st.session_state.batch_queue)
                    for i, item in enumerate(st.session_state.batch_queue):
                        status_text.markdown(f"⏳ **Processing {i+1}/{total}:** {item['name']}...")
                        
                        with monitor_container:
                            st.markdown(f"#### 🛰️ Monitoring: {item['name']}")
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            m_epoch = mc1.empty()
                            m_loss = mc2.empty()
                            m_vloss = mc3.empty()
                            m_eta = mc4.empty()
                            m_chart = st.empty()
                        
                        # Setup config for this run
                        batch_cfg = cfg.copy()
                        batch_cfg['model']['architecture'] = item['architecture']
                        batch_cfg['model']['hyperparameters'] = item['hp']
                        
                        # --- DATA VERSION SELECTION ---
                        data_v = item.get('data_version', gt('current_dynamic', st.session_state.lang))
                        if data_v == gt('current_dynamic', st.session_state.lang):
                            if 'features' in item:
                                status_text.markdown(f"Sweep 🧹 **Preparing data for {item['name']}...**")
                                batch_cfg['features']['groups'] = item['features']['groups']
                                batch_cfg['features']['selection_mode'] = item['features']['selection_mode']
                                
                                from src.data_prep import run_preprocessing
                                run_preprocessing(batch_cfg)
                        else:
                            status_text.markdown(f"📦 **Using archived data: {data_v}**")
                            batch_cfg['paths']['processed_dir'] = os.path.join(proc_dir, data_v)
                        
                        # --- LIVE CALLBACK --- (using closure variables)
                        class BatchLiveCallback(tf.keras.callbacks.Callback):
                            def __init__(self):
                                super().__init__()
                                self.epoch_data = []
                                self.start_t = time.time()
                            def on_epoch_end(self, epoch, logs=None):
                                logs = logs or {}
                                loss, vloss = logs.get('loss', 0), logs.get('val_loss', 0)
                                total_e = self.params.get('epochs', 100)
                                elapsed = time.time() - self.start_t
                                avg = elapsed / (epoch + 1)
                                eta = avg * (total_e - epoch - 1)
                                eta_s = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
                                
                                m_epoch.metric("Epoch", f"{epoch+1}/{total_e}")
                                m_loss.metric("Loss", f"{loss:.6f}")
                                m_vloss.metric("Val Loss", f"{vloss:.6f}")
                                m_eta.metric("ETA", eta_s)
                                
                                # Fix: Add explicit print to CMD for Batch experiments
                                print(f"  > [{item['name']}] Epoch {epoch+1}/{total_e}: loss={loss:.6f}, val={vloss:.6f}")
                                
                                self.epoch_data.append({'e': epoch+1, 'l': loss, 'v': vloss})
                                fig = go.Figure()
                                ee = [d['e'] for d in self.epoch_data]
                                fig.add_trace(go.Scatter(x=ee, y=[d['l'] for d in self.epoch_data], name='Train', line=dict(color='#818cf8')))
                                fig.add_trace(go.Scatter(x=ee, y=[d['v'] for d in self.epoch_data], name='Val', line=dict(color='#f472b6')))
                                fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20,r=20,t=30,b=20),
                                                  title=f"Loss Curve: {item['name']}", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                                m_chart.plotly_chart(fig, width="stretch")

                        try:
                            from src.trainer import train_model
                            loss_function = batch_cfg['model']['hyperparameters'].get('loss_fn', 'mse')
                            model, history, meta = train_model(batch_cfg, custom_model_id=item['name'], extra_callbacks=[BatchLiveCallback()], loss_fn=loss_function)
                            st.session_state.batch_results.append({"name": item['name'], "status": "✅ Success", "loss": min(history.history['val_loss'])})
                        except Exception as e:
                            st.session_state.batch_results.append({"name": item['name'], "status": "❌ Failed", "error": str(e)})
                        
                        batch_progress.progress((i + 1) / total)
                        # Don't empty monitor_container completely, just let it stay for a bit
                    
                    st.session_state.batch_queue = []
                    st.session_state.batch_running = False
                    status_text.success("🏁 All Batch Experiments Finished!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()

            # --- RESULTS SUMMARY (Visible after batch or if queue empty) ---
            if st.session_state.batch_results:
                st.markdown("---")
                st.markdown("#### 🏁 Results Summary")
                res_df = pd.DataFrame(st.session_state.batch_results)
                st.dataframe(res_df, width="stretch")
                if st.button("🗑️ Clear Results"):
                    st.session_state.batch_results = []
                    st.rerun()
    except Exception as e:
        st.error(f"❌ Batch Tab Error: {e}")
        import traceback
        st.code(traceback.format_exc())


# --- TAB TUNING: TUNING MONITOR ---
with tab_tuning:
    st.markdown("### 🎯 Optuna Hyperparameter Tuning")
    
    # History Selector
    history_files = list_tuning_history()
    if history_files:
        c_hist1, c_hist2 = st.columns([3, 1])
        with c_hist1:
            selected_hist = st.selectbox(
                "📜 Riwayat Tuning (Pilih untuk melihat hasil sebelumnya)", 
                ["Current / Last Run"] + history_files,
                index=0
            )
        with c_hist2:
            if st.button("📂 Load History"):
                if selected_hist == "Current / Last Run":
                    st.session_state.tuning_results = load_tuning_results()
                    st.success("Loaded last run.")
                    st.session_state.tuning_results = load_specific_tuning_result(selected_hist)
                    st.success(f"Loaded: {selected_hist}")
                st.rerun()

    # Data & Fitur are now in their own tabs
    st.info("💡 **Tips**: Pengaturan dataset, cleaning, dan fitur kini dikelola di tab **Data Insights**, **Preprocessing**, dan **Feature Lab** sesuai alur pipeline.")

    # Added: Mini Analytics inside Tuning Tab for immediate feedback
    if st.session_state.prep_metadata:
        m = st.session_state.prep_metadata
        st.markdown(f"""
        <div style="background-color: rgba(129, 140, 248, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #818cf8; margin-bottom: 20px;">
            <h4 style="margin-top:0;">📊 Summary Data Preprocessing</h4>
            <div style="display: flex; justify-content: space-between;">
                <div><b>Final Rows:</b> {m['stats']['after_algorithm1']:,}</div>
                <div><b>Features:</b> {len(m['selected_features'])}</div>
                <div><b>Sequences:</b> {m['stats']['train_final']:,}</div>
            </div>
            <div style="font-size: 0.8em; color: #94a3b8; margin-top: 10px;">
                Fitur terpilih: {", ".join(m['selected_features'][:5])}... 
                (Lihat detail lengkap di tab <b>🔎 Data Insights</b>)
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- NEW: SEARCH SPACE EDITOR & EXECUTION (Always Visible) ---
    if cfg['tuning']['enabled']:
        st.markdown("#### 🚀 Konfigurasi & Jalankan Tuning")
        
        # Add model selector specifically for tuning context
        def _update_tuning_architecture():
            new_a = st.session_state.tune_arch_selector
            cfg['model']['architecture'] = new_a
            if new_a == 'patchtst':
                cfg['model']['hyperparameters'].update({
                    'd_model': 128, 'n_layers': 3, 'learning_rate': 0.0001, 
                    'batch_size': 32, 'dropout': 0.2, 'patch_len': 16, 'stride': 8, 'n_heads': 16
                })
            else:
                cfg['model']['hyperparameters'].update({
                    'd_model': 64, 'n_layers': 2, 'learning_rate': 0.001, 
                    'batch_size': 32, 'dropout': 0.2, 'use_bidirectional': True
                })
            if new_a != 'patchtst':
                for k in ['patch_len', 'stride', 'n_heads', 'ff_dim']: cfg['model']['hyperparameters'].pop(k, None)
            else:
                cfg['model']['hyperparameters'].pop('use_bidirectional', None)
            save_config_to_file(cfg)
            st.session_state.cfg = cfg
            st.session_state.arch_selector_train = new_a

        if 'tune_arch_selector' not in st.session_state:
            st.session_state.tune_arch_selector = cfg['model'].get('architecture', 'patchtst').lower()

        t_arch = st.selectbox("Arsitektur yang akan di-Tuning", ["patchtst", "timetracker", "timeperceiver", "gru", "lstm", "rnn"], 
                              index=["patchtst", "timetracker", "timeperceiver", "gru", "lstm", "rnn"].index(cfg['model'].get('architecture', 'patchtst').lower()),
                              key="tune_arch_selector",
                              on_change=_update_tuning_architecture)
        
        with st.expander("🛠️ Edit Search Space Hyperparameters", expanded=False):
            st.info("Atur range pencarian untuk setiap hyperparameter. Perubahan akan disimpan saat Anda menjalankan tuning.")
            
            space = cfg['tuning']['search_space']
            col_s1, col_s2, col_s3 = st.columns(3)
            
            with col_s1:
                if t_arch == "patchtst":
                    st.markdown("**1. Patching & Stride**")
                    p_vals = space.get('patch_len', [8, 24, 4])
                    p_min = st.number_input("Patch Min", 2, 64, p_vals[0], 2, key="p_min_new")
                    p_max = st.number_input("Patch Max", p_min, 128, p_vals[1], 2, key="p_max_new")
                    p_step = st.number_input("Patch Step", 1, 16, p_vals[2], 1, key="p_step_new")
                    space['patch_len'] = [p_min, p_max, p_step]
                    
                    s_vals = space.get('stride', [4, 12, 2])
                    s_min = st.number_input("Stride Min", 1, 32, s_vals[0], 1, key="s_min_new")
                    s_max = st.number_input("Stride Max", s_min, 64, s_vals[1], 1, key="s_max_new")
                    s_step = st.number_input("Stride Step", 1, 8, s_vals[2], 1, key="s_step_new")
                    space['stride'] = [s_min, s_max, s_step]
                else:
                    st.markdown(f"**1. {t_arch.upper()} Configuration**")
                    st.info("Parameter patching tidak tersedia untuk arsitektur GRU.")
                    # Ensure they aren't in space to avoid confusion (optional)

            with col_s2:
                # Dynamic Search Space Labels
                ss_d_label = "D_Model (Embedding)" if cfg['model']['architecture'] == 'patchtst' else f"Hidden Units ({cfg['model']['architecture'].upper()} Capacity)"
                ss_l_label = "Layers (Transformer)" if cfg['model']['architecture'] == 'patchtst' else f"Layers (Stacked {cfg['model']['architecture'].upper()})"
                
                st.markdown(f"**2. {cfg['model']['architecture'].upper()} Capacity**")
                d_vals = space.get('d_model', [64, 256])
                d_min = st.number_input(f"{ss_d_label} Min", 4, 512, d_vals[0], 4, key="d_min_new")
                d_max = st.number_input(f"{ss_d_label} Max", d_min, 1024, d_vals[1], 4, key="d_max_new")
                space['d_model'] = [d_min, d_max]
                
                l_vals = space.get('n_layers', [2, 5])
                l_min = st.number_input(f"{ss_l_label} Min", 1, 12, l_vals[0], 1, key="l_min_new")
                l_max = st.number_input(f"{ss_l_label} Max", l_min, 20, l_vals[1], 1, key="l_max_new")
                space['n_layers'] = [l_min, l_max]
                
                if t_arch == "patchtst":
                    ff_vals = space.get('ff_dim', [128, 512])
                    ff_min = st.number_input("FF_Dim Min", 4, 1024, ff_vals[0], 4, key="ff_min_new")
                    ff_max = st.number_input("FF_Dim Max", ff_min, 2048, ff_vals[1], 4, key="ff_max_new")
                    space['ff_dim'] = [ff_min, ff_max]
                    
                    h_vals = space.get('n_heads', [4, 16])
                    h_min = st.number_input("Heads Min", 1, 32, h_vals[0], 1, key="h_min_new")
                    h_max = st.number_input("Heads Max", h_min, 64, h_vals[1], 1, key="h_max_new")
                    space['n_heads'] = [h_min, h_max]

                dr_vals = space.get('dropout', [0.05, 0.3])
                dr_min = st.number_input("Dropout Min", 0.0, 0.5, dr_vals[0], 0.05, key="dr_min_new")
                dr_max = st.number_input("Dropout Max", dr_min, 0.7, dr_vals[1], 0.05, key="dr_max_new")
                space['dropout'] = [dr_min, dr_max]

            with col_s3:
                st.markdown("**3. Time & Speed**")
                loc_vals = space.get('lookback', [48, 336, 24])
                loc_min = st.number_input("Lookback Min", 24, 720, loc_vals[0], 24, key="loc_min_new")
                loc_max = st.number_input("Lookback Max", loc_min, 1440, loc_vals[1], 24, key="loc_max_new")
                loc_step = st.number_input("Lookback Step", 12, 168, loc_vals[2], 12, key="loc_step_new")
                space['lookback'] = [loc_min, loc_max, loc_step]
                
                lr_vals = space.get('learning_rate', [5e-5, 1e-3])
                lr_min = st.number_input("LR Min", 1e-6, 1e-1, lr_vals[0], format="%.6f", key="lr_min_new")
                lr_max = st.number_input("LR Max", lr_min, 1e-1, lr_vals[1], format="%.6f", key="lr_max_new")
                space['learning_rate'] = [lr_min, lr_max]

                b_vals = space.get('batch_size', [16, 64])
                b_min = st.number_input("Batch Min", 2, 256, b_vals[0], 2, key="b_min_new")
                b_max = st.number_input("Batch Max", b_min, 512, b_vals[1], 2, key="b_max_new")
                space['batch_size'] = [b_min, b_max]

            if st.button("💾 Save Search Space to Master Config", width="stretch", key="save_ss_tuning"):
                cfg['tuning']['search_space'] = space
                save_config_to_file(cfg)
                st.success("Search space berhasil disimpan ke config.yaml!")

        # Device Selector for Tuning
        tune_col_dev1, tune_col_dev2 = st.columns([1, 2])
        with tune_col_dev1:
            tune_device = st.radio("🖥️ Device untuk Tuning", ["CPU", "GPU"], index=0, 
                                   horizontal=True, key="tune_device_top",
                                   help="CPU direkomendasikan untuk menghindari OOM error pada GPU dengan VRAM terbatas.")
        with tune_col_dev2:
            run_tune = st.button("🔥 Jalankan Optuna Tuning Baru", type="primary", width="stretch", 
                                  disabled=not has_data, key="btn_tune_execute")
    else:
        st.warning("⚠️ **Optuna Tuning Belum Aktif**. Aktifkan melalui toggle 'Enable Optuna Tuning' pada sidebar di sebelah kiri.")

    st.markdown("---")
    
    if st.session_state.tuning_results:
        tr = st.session_state.tuning_results
        trials = tr['trials']
        best = tr['best_params']
        
        # Best Parameters
        st.markdown("#### Best Hyperparameters (Last Run)")
        cols = st.columns(len(best))
        for col, (k, v) in zip(cols, best.items()):
            with col:
                val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="font-size:1.4rem;">{val_str}</div>
                    <div class="metric-label">{k}</div>
                </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Optimization History
        col1, col2 = st.columns(2)
        
        with col1:
            trial_nums = list(range(1, len(trials) + 1))
            trial_values = [t['value'] for t in trials]
            best_so_far = []
            current_best = float('inf')
            for v in trial_values:
                if v < current_best:
                    current_best = v
                best_so_far.append(current_best)
            
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(
                x=trial_nums, y=trial_values,
                mode='markers', name='Trial Value',
                marker=dict(size=8, color='#818cf8', opacity=0.6)
            ))
            fig_opt.add_trace(go.Scatter(
                x=trial_nums, y=best_so_far,
                mode='lines', name='Best So Far',
                line=dict(color='#f472b6', width=2)
            ))
            fig_opt.update_layout(
                title="Optimization History",
                xaxis_title="Trial #", yaxis_title="Val Loss (MAE)",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=400,
            )
            st.plotly_chart(fig_opt, width='stretch')
        
        with col2:
            # Param Importance (bar chart of how often best value appeared)
            if len(trials) > 2:
                param_names = list(trials[0]['params'].keys())
                # Show parameter vs value scatter for top params
                fig_param = go.Figure()
                for pname in param_names[:6]:
                    pvals = [t['params'].get(pname, 0) for t in trials]
                    fig_param.add_trace(go.Scatter(
                        x=pvals, y=trial_values,
                        mode='markers', name=pname,
                        marker=dict(size=6, opacity=0.6)
                    ))
                fig_param.update_layout(
                    title="Parameter vs Objective",
                    xaxis_title="Parameter Value", yaxis_title="Val Loss",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                )
                st.plotly_chart(fig_param, width='stretch')
        
        # Trial Details Table  
        st.markdown("#### Trial Details")
        df_trials = pd.DataFrame([
            {'Trial': i+1, 'Value': t['value'], **t['params']}
            for i, t in enumerate(trials)
        ]).sort_values('Value')
        st.dataframe(df_trials, width='stretch', hide_index=True)
        
    else:
        st.info("Belum ada hasil tuning tersimpan.")
        
        # Setup Tuning Execution Section
        st.markdown('<div class="pipeline-step">', unsafe_allow_html=True)
        st.markdown("#### 🚀 Eksekusi Tuning")
        st.write("Tuning akan mencari hyperparameter terbaik berdasarkan data yang dipilih di atas.")
        
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.write(f"**Arsitektur:** {new_arch.upper()}")
        with cols[1]:
            st.write(f"**Trials:** {cfg['tuning']['n_trials']}")
        with cols[2]:
            st.write(f"**Data Status:** {'✅ Ready' if has_data else '❌ Missing'}")
        
        tune_col_d1, tune_col_loss, tune_col_d2 = st.columns([1, 1, 2])
        with tune_col_d1:
            tune_device = st.radio("🖥️ Device", ["CPU", "GPU"], index=0, 
                                   horizontal=True, key="tune_device_bottom",
                                   help="CPU direkomendasikan untuk stabilitas.")
        with tune_col_loss:
            _opt_loss = ['mse', 'huber', 'mae']
            tune_loss_fn = st.selectbox("Loss Function", _opt_loss, index=_opt_loss.index(cfg['model']['hyperparameters'].get('loss_fn', 'mse')) if cfg['model']['hyperparameters'].get('loss_fn') in _opt_loss else 0, key="tune_loss_fn")
        with tune_col_d2:
            run_tune = st.button("🔥 Jalankan Optuna Tuning", type="primary", width="stretch", 
                                  disabled=not has_data, key="btn_tune_tab")
        st.markdown('</div>', unsafe_allow_html=True)

        if cfg['tuning']['enabled']:
            st.markdown("#### 🛠️ Edit Search Space")
            st.info("Atur range pencarian untuk setiap hyperparameter. Perubahan akan disimpan saat Anda menjalankan tuning.")
            
            space = cfg['tuning']['search_space']
            
            # Create a nice editor for the search space
            with st.expander("Configure Search Space Details", expanded=True):
                col_s1, col_s2, col_s3 = st.columns(3)
                
                # Patching Params
                with col_s1:
                    st.markdown("**1. Patching & Stride**")
                    p_vals = space.get('patch_len', [8, 24, 4])
                    p_min = st.number_input("Patch Min", 2, 64, p_vals[0], 2, key="p_min")
                    p_max = st.number_input("Patch Max", p_min, 128, p_vals[1], 2, key="p_max")
                    p_step = st.number_input("Patch Step", 1, 16, p_vals[2], 1, key="p_step")
                    space['patch_len'] = [p_min, p_max, p_step]
                    
                    s_vals = space.get('stride', [4, 12, 2])
                    s_min = st.number_input("Stride Min", 1, 32, s_vals[0], 1, key="s_min")
                    s_max = st.number_input("Stride Max", s_min, 64, s_vals[1], 1, key="s_max")
                    s_step = st.number_input("Stride Step", 1, 8, s_vals[2], 1, key="s_step")
                    space['stride'] = [s_min, s_max, s_step]

                # Architecture Params
                with col_s2:
                    st.markdown("**2. Model Capacity**")
                    d_vals = space.get('d_model', [64, 256])
                    d_min = st.number_input("D_Model Min", 32, 512, d_vals[0], 32, key="d_min")
                    d_max = st.number_input("D_Model Max", d_min, 1024, d_vals[1], 32, key="d_max")
                    space['d_model'] = [d_min, d_max]
                    
                    l_vals = space.get('n_layers', [3, 8])
                    l_min = st.number_input("Layers Min", 1, 12, l_vals[0], 1, key="l_min")
                    l_max = st.number_input("Layers Max", l_min, 20, l_vals[1], 1, key="l_max")
                    space['n_layers'] = [l_min, l_max]

                # Training Params
                with col_s3:
                    st.markdown("**3. Time & Speed**")
                    loc_vals = space.get('lookback', [48, 336, 24])
                    loc_min = st.number_input("Lookback Min", 24, 720, loc_vals[0], 24, key="loc_min")
                    loc_max = st.number_input("Lookback Max", loc_min, 1440, loc_vals[1], 24, key="loc_max")
                    loc_step = st.number_input("Lookback Step", 12, 168, loc_vals[2], 12, key="loc_step")
                    space['lookback'] = [loc_min, loc_max, loc_step]
                    
                    lr_vals = space.get('learning_rate', [5e-5, 1e-3])
                    lr_min = st.number_input("LR Min", 1e-6, 1e-1, lr_vals[0], format="%.6f", key="lr_min")
                    lr_max = st.number_input("LR Max", lr_min, 1e-1, lr_vals[1], format="%.6f", key="lr_max")
                    space['learning_rate'] = [lr_min, lr_max]

            # Show active summary table
            st.markdown("#### Search Space Aktif (Hyperparameters)")
            space_df = pd.DataFrame([
                {'Parameter': k, 'Range': str(v)} for k, v in space.items()
            ])
            st.dataframe(space_df, width='stretch', hide_index=True)
            
            if st.button("💾 Save Search Space to Master Config", width="stretch"):
                cfg['tuning']['search_space'] = space
                save_config_to_file(cfg)
                st.success("Search space berhasil disimpan ke config.yaml!")
        else:
            st.warning("⚠️ **Optuna Tuning Belum Aktif**. Aktifkan melalui toggle 'Enable Optuna Tuning' pada sidebar di sebelah kiri.")
            st.info("Tuning memungkinkan sistem mencari arsitektur terbaik secara otomatis untuk mencapai R² yang lebih tinggi.")

    # --- EXECUTION LOGIC FOR TUNING ---
    if 'run_tune' in locals() and run_tune:
        st.markdown("### 🔍 Live Tuning Monitor")
        
        # Placeholders for real-time updates
        tune_progress = st.progress(0, text="Initializing Tuning...")
        col_t1, col_t2, col_t3 = st.columns(3)
        trial_display = col_t1.empty()
        best_val_display = col_t2.empty()
        last_val_display = col_t3.empty()
        tune_chart_placeholder = st.empty()
        table_placeholder = st.empty()
        
        try:
            import io, contextlib
            stdout_capture = io.StringIO()
            
            # Callback for live updates
            class TuningLiveMonitor:
                def __init__(self, n_trials):
                    self.n_trials = n_trials
                    self.history = []
                    self.trial_records = []
                
                def __call__(self, study, trial):
                    if trial.value is not None:
                        self.history.append(trial.value)
                        best_val = study.best_value
                        
                        # Add to records
                        record = {'Trial': len(study.trials), 'Loss': trial.value}
                        record.update(trial.params)
                        self.trial_records.append(record)
                        
                        # Update UI
                        progress_val = len(study.trials) / self.n_trials
                        tune_progress.progress(min(progress_val, 1.0), 
                                            text=f"Trial {len(study.trials)}/{self.n_trials}")
                        
                        trial_display.metric("Trials", f"{len(study.trials)}/{self.n_trials}")
                        best_val_display.metric("Best Val Loss", f"{best_val:.6f}")
                        last_val_display.metric("Last Trial", f"{trial.value:.6f}", 
                                             delta=f"{trial.value - best_val:.6f}", delta_color="inverse")
                        
                        # Mini chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=self.history, mode='lines+markers', name='Trial Value'))
                        fig.update_layout(height=250, margin=dict(t=10, b=10), template="plotly_dark")
                        tune_chart_placeholder.plotly_chart(fig, width='stretch')

                        # Live Table (Show latest on top)
                        df_live = pd.DataFrame(self.trial_records).sort_values('Trial', ascending=False)
                        table_placeholder.dataframe(df_live, height=300, hide_index=True)

            live_monitor = TuningLiveMonitor(cfg['tuning']['n_trials'])
            
            # Determine CPU mode from device selector
            use_cpu = tune_device == "CPU" if 'tune_device' in dir() else True
            
            with contextlib.redirect_stdout(stdout_capture):
                from src.trainer import run_optuna_tuning
                loss_function = tune_loss_fn if 'tune_loss_fn' in locals() else 'mse'
                best, study = run_optuna_tuning(cfg, extra_callbacks=[live_monitor], force_cpu=use_cpu, loss_fn=loss_function)
            
            # Save results for the Tuning Monitor tab
            trial_data = []
            for trial in study.trials:
                if trial.value is not None:
                    trial_data.append({
                        'value': trial.value,
                        'params': trial.params,
                    })
            tuning_results = {
                'best_params': best,
                'best_value': study.best_value,
                'trials': trial_data,
            }
            st.session_state.tuning_results = tuning_results
            save_tuning_results(tuning_results)
            cfg['model']['hyperparameters'].update(best)
            st.session_state.pipeline_log.append(
                f"[{datetime.now():%H:%M:%S}] Tuning selesai. "
                f"Best Val Loss: {study.best_value:.6f} | "
                f"Params: {best}"
            )
            st.success(f"Tuning selesai! Best Val Loss: {study.best_value:.6f}")
            with st.expander("Best Parameters"):
                st.json(best)
            with st.expander("Full Output"):
                st.code(stdout_capture.getvalue(), language="text")
        except Exception as e:
            st.error(f"Error: {e}")
        st.rerun()


# --- TAB: EVALUATION RESULTS ---
with tab_eval:
    st.markdown("---")
    st.markdown("#### 📈 Deep Evaluation Analysis")
    st.markdown("### Evaluation & Results")
    
    # --- MODEL SELECTOR FOR EVALUATION ---
    with st.expander("📂 Pilih Model untuk Evaluasi", expanded=not st.session_state.eval_results):
        if os.path.exists(model_dir):
            all_models = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5')) or os.path.isdir(os.path.join(model_dir, f))]
            if all_models:
                current_sel = st.session_state.get('selected_model', all_models[0])
                model_to_eval = st.selectbox("Pilih Model untuk Evaluasi (Tab):", all_models, 
                                           index=all_models.index(current_sel) if current_sel in all_models else 0,
                                           key="sel_eval_tab",
                                           format_func=lambda x: label_format_with_time(x, model_dir))
                st.session_state.selected_model = model_to_eval
                
                # Model Info Preview
                model_info_path = os.path.join(model_dir, model_to_eval, "meta.json")
                if os.path.exists(model_info_path):
                    with open(model_info_path, 'r') as f:
                        m_meta = json.load(f)
                        
                    feat_text = "N/A"
                    ds_path = m_meta.get('data_source', '').replace('\\', '/')
                    
                    summary_path = os.path.join(model_dir, model_to_eval, "prep_summary.json")
                    if not os.path.exists(summary_path) and ds_path and os.path.exists(os.path.join(ds_path, "prep_summary.json")):
                        summary_path = os.path.join(ds_path, "prep_summary.json")
                        
                    if os.path.exists(summary_path):
                        try:
                            with open(summary_path, 'r') as ff:
                                summ_data = json.load(ff)
                                if 'selected_features' in summ_data:
                                    feat_text = ", ".join(summ_data['selected_features'])
                        except: pass
                        
                    if feat_text == "N/A":
                        feat_path = os.path.join(model_dir, model_to_eval, "selected_features.json")
                        if not os.path.exists(feat_path) and ds_path and os.path.exists(os.path.join(ds_path, "selected_features.json")):
                             feat_path = os.path.join(ds_path, "selected_features.json")
                        if os.path.exists(feat_path):
                            try:
                                with open(feat_path, 'r') as ff:
                                    feat_list = json.load(ff)
                                feat_text = ", ".join(feat_list)
                            except: pass

                    st.markdown(f"""
                    <div style="background-color: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 5px; font-size: 0.9em; margin-bottom: 15px;">
                        <b>Arch:</b> {m_meta.get('architecture', 'N/A').upper()} | 
                        <b>Data Source:</b> <span style="color: #94a3b8;">{os.path.basename(m_meta.get('data_source', 'N/A'))}</span> <br>
                        <b>Features ({m_meta.get('n_features', '?')}):</b> <span style="color: #cbd5e1; font-size: 0.85em;">{feat_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.button("🔎 Run Evaluation for Selected Model", type="primary", width="stretch", key="btn_eval_tab"):
                    with st.spinner(f"Evaluating model: {model_to_eval}..."):
                        try:
                            import gc
                            tf.keras.backend.clear_session()
                            gc.collect()
                            
                            from src.model_factory import get_custom_objects, compile_model
                            from src.predictor import evaluate_model
                            
                            model_path = os.path.join(model_dir, model_to_eval)
                            model_root = model_dir
                            
                            if os.path.isdir(model_path):
                                model_root = model_path
                                for ext in ['model.keras', 'model.h5']:
                                    if os.path.exists(os.path.join(model_path, ext)):
                                        model_path = os.path.join(model_path, ext)
                                        break
                                        
                            custom_objs = get_custom_objects()
                            with tf.keras.utils.custom_object_scope(custom_objs):
                                model = tf.keras.models.load_model(model_path, compile=False)
                            
                            compile_model(model, cfg['model']['hyperparameters']['learning_rate'])
                            scaler_dir = model_root if os.path.isdir(model_root) else None
                            
                            data = st.session_state.get('prep_metadata', None)
                            results = evaluate_model(model, cfg, data=data, scaler_dir=scaler_dir)
                            results['model_id'] = model_to_eval
                            
                            st.session_state.eval_results = results
                            save_eval_results_to_disk(results)
                            st.success(f"Evaluasi berhasil untuk {model_to_eval}!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            err_msg = str(e)
                            st.error(f"Gagal mengevaluasi model: {err_msg}")
                            
                            # SMART FIX: If it's a feature mismatch, offer to switch data version
                            if "Mismatch Fitur" in err_msg or "Inkompatibilitas Fitur" in err_msg:
                                model_info_path = os.path.join(model_dir, model_to_eval, "meta.json")
                                if os.path.exists(model_info_path):
                                    with open(model_info_path, 'r') as f:
                                        m_meta = json.load(f)
                                    orig_raw = m_meta.get('data_source', '')
                                    orig_path = orig_raw.replace('\\', '/')
                                    
                                    found_path = None
                                    if orig_path and os.path.exists(orig_path):
                                        found_path = orig_path
                                    elif orig_raw:
                                        # Smart Recovery: Search for leaf folder
                                        leaf_name = os.path.basename(orig_raw.rstrip('\\/'))
                                        if leaf_name:
                                            # Find proot
                                            search_root = proc_dir # base data/processed
                                            for root, dirs, files in os.walk(search_root):
                                                if os.path.basename(root) == leaf_name:
                                                    if 'X_train.npy' in files:
                                                        found_path = root.replace('\\', '/')
                                                        break
                                    
                                    if found_path:
                                        st.info(f"💡 Model ini butuh fitur dari folder: `{os.path.basename(found_path)}`")
                                        if st.button("🔄 Switch ke Data Asli Model Ini & Jalankan Ulang"):
                                            st.session_state.cfg['paths']['processed_dir'] = found_path
                                            save_config_to_file(st.session_state.cfg)
                                            st.success("Jalur data diperbarui. Mengulangi evaluasi...")
                                            time.sleep(1)
                                            st.rerun()
                                    else:
                                        st.warning("⚠️ Data asli untuk model ini tidak ditemukan di `data/processed`. Silakan buat data dengan jumlah fitur yang sesuai di tab Preprocessing.")

                            import traceback; st.code(traceback.format_exc())
            else:
                st.warning("Belum ada model tersimpan di folder models/.")
        else:
            st.error("Folder models/ tidak ditemukan.")

    if st.session_state.eval_results:
        results = st.session_state.eval_results
        
        # Check for model consistency
        disp_model = results.get('model_id', 'Unknown')
        curr_model = st.session_state.get('selected_model', 'None')
        
        if disp_model != curr_model:
            st.warning(f"⚠️ Hasil di bawah adalah milik model **{disp_model}**, sedangkan model yang terpilih saat ini adalah **{curr_model}**. Klik tombol evaluasi di atas untuk memperbarui.")
        else:
            st.info(f"✅ Menampilkan hasil evaluasi untuk model aktif: **{disp_model}**")

        m_train = results['metrics_train']
        m_test = results['metrics_test']
        
        # ====== Fetch Training Time from meta.json if available ======
        train_time_str = "N/A"
        model_info_path = os.path.join(model_dir, disp_model, "meta.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                _meta = json.load(f)
                seconds = _meta.get('training_time_seconds', 0)
                if seconds > 0:
                    train_time_str = float(seconds)
                    
        r2_diff = m_train['r2'] - m_test['r2']
        
        # ====== ROW 1: Metric cards ======
        st.markdown(f"#### Performance Metrics (Test Set - {disp_model})")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        metrics_display = [
            ("R\u00b2", m_test['r2'], ""),
            ("MAE", m_test['mae'], " kW"),
            ("nMAE", m_test.get('norm_mae', 0) * 100, " %"),
            ("RMSE", m_test['rmse'], " kW"),
            ("nRMSE", m_test.get('norm_rmse', 0) * 100, " %"),
            ("Train Time", train_time_str, "s"),
        ]
        for col, (name, val, unit) in zip([col1, col2, col3, col4, col5, col6], metrics_display):
            with col:
                if isinstance(val, (int, float)):
                    val_str = f"{val:.4f}" if name not in ["Train Time"] else f"{val:.1f}"
                else:
                    val_str = str(val)
                    unit = "" # Clear unit if unknown/string
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{val_str}{unit}</div>
                    <div class="metric-label">{name}</div>
                </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ====== ROW 2: Train vs Test Comparison ======
        st.markdown("#### Train vs Test Comparison")
        
        # Get productive-hours metrics if available
        m_train_prod = results.get('metrics_train_productive', {})
        m_test_prod = results.get('metrics_test_productive', {})
        
        metrics_rows = [
            ['R\u00b2', f"{m_train['r2']:.4f}", f"{m_test['r2']:.4f}"],
            ['MAE (kW)', f"{m_train['mae']:.4f}", f"{m_test['mae']:.4f}"],
            ['nMAE (%)', f"{m_train.get('norm_mae', 0)*100:.2f}", f"{m_test.get('norm_mae', 0)*100:.2f}"],
            ['RMSE (kW)', f"{m_train['rmse']:.4f}", f"{m_test['rmse']:.4f}"],
            ['nRMSE (%)', f"{m_train.get('norm_rmse', 0)*100:.2f}", f"{m_test.get('norm_rmse', 0)*100:.2f}"],
        ]
        if m_test_prod:
            metrics_rows.insert(3, ['R² (Productive)', 
                                    f"{m_train_prod.get('r2', 0):.4f}", 
                                    f"{m_test_prod.get('r2', 0):.4f}"])
        
        df_metrics = pd.DataFrame(metrics_rows, columns=['Metrik', 'Train', 'Test'])
        st.dataframe(df_metrics, width='stretch', hide_index=True)
        
        # ====== Prepare common data ======
        if 'pv_test_actual' in results:
            actual_flat = results['pv_test_actual'].flatten()
            pred_flat = results['pv_test_pred'].flatten()
            ghi_flat = results['ghi_test'].flatten()
            mask_productive = ghi_flat > 50
        else:
            st.warning("Data visualisasi (arrays) belum dimuat. Jalankan 'Run Evaluation' kembali untuk melihat grafik detail.")
            st.stop()
        
        # ====== ROW 3: Scatter + Residual ======
        st.markdown("#### Actual vs Predicted")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scattergl(
                x=actual_flat[mask_productive], y=pred_flat[mask_productive],
                mode='markers', marker=dict(size=3, color='#818cf8', opacity=0.3),
                name='Predictions'
            ))
            max_val = max(actual_flat[mask_productive].max(), pred_flat[mask_productive].max())
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines', line=dict(color='#ef4444', dash='dash', width=1.5),
                name='Perfect Fit'
            ))
            fig_scatter.update_layout(
                title="Scatter: Test Set (Productive Hours, GHI > 50)",
                xaxis_title="Actual (kW)", yaxis_title="Predicted (kW)",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=450,
            )
            st.plotly_chart(fig_scatter, width='stretch')
        
        with col2:
            residuals = actual_flat[mask_productive] - pred_flat[mask_productive]
            fig_hist = go.Figure(go.Histogram(
                x=residuals, nbinsx=100,
                marker_color='#818cf8', opacity=0.7
            ))
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#ef4444", line_width=1.5)
            fig_hist.update_layout(
                title=f"Residual Distribution (Mean: {residuals.mean():.3f}, Std: {residuals.std():.3f})",
                xaxis_title="Error: Actual - Predicted (kW)", yaxis_title="Count",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=450,
            )
            st.plotly_chart(fig_hist, width='stretch')
        
        # ====== ROW 4: Time Series Sample ======
        st.markdown("#### Time Series: Actual vs Predicted (Sample)")
        
        # Use test data with indices
        try:
            df_test = results['df_test']
            test_indices = results['test_indices']
            horizon = results['pv_test_actual'].shape[1]
            
            # Show slider for selecting sample range
            total_seqs = len(test_indices)
            n_show = min(200, total_seqs)
            
            ts_start = st.slider(
                "Pilih rentang sampel (sequence index)",
                0, max(0, total_seqs - n_show), 0,
                key="ts_slider"
            )
            ts_end = min(ts_start + n_show, total_seqs)
            
            # Build time-series for the selected range (use 1st step predictions)
            sample_idx = test_indices[ts_start:ts_end]
            sample_timestamps = df_test.index[sample_idx]
            sample_actual = results['pv_test_actual'][ts_start:ts_end, 0]
            sample_pred = results['pv_test_pred'][ts_start:ts_end, 0]
            
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=sample_timestamps, y=sample_actual,
                mode='lines', name='Actual',
                line=dict(color='#6ee7b7', width=1.5)
            ))
            fig_ts.add_trace(go.Scatter(
                x=sample_timestamps, y=sample_pred,
                mode='lines', name='Predicted',
                line=dict(color='#818cf8', width=1.5)
            ))
            fig_ts.update_layout(
                title=f"h+1 Forecast: Sequences {ts_start} - {ts_end}",
                xaxis_title="Timestamp", yaxis_title="Power (kW)",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_ts, width='stretch')
        except Exception as e:
            st.caption(f"Time series plot tidak tersedia: {e}")
        
        # ====== ROW 5: Per-Step Forecast Diagnostics ======
        st.markdown("#### 📊 Diagnostik: R² per Forecast Step")
        per_step_r2 = results.get('per_step_r2', {})
        if per_step_r2:
            steps_list = sorted(per_step_r2.keys())
            r2_vals = [per_step_r2[s] for s in steps_list]
            
            fig_r2_step = go.Figure()
            fig_r2_step.add_trace(go.Scatter(
                x=steps_list, y=r2_vals,
                mode='lines+markers',
                line=dict(color='#818cf8', width=2.5),
                marker=dict(size=7, color='#818cf8'),
                name='R² per Step',
                hovertemplate='Step t+%{x}: R²=%{y:.4f}<extra></extra>'
            ))
            # Add reference lines
            fig_r2_step.add_hline(y=0.8, line_dash='dot', line_color='#6ee7b7', 
                                  annotation_text='Target R²=0.80', annotation_position='top left')
            fig_r2_step.add_hline(y=0, line_dash='dash', line_color='#ef4444', line_width=1)
            fig_r2_step.update_layout(
                title="R² Score per Forecast Step (ALL Hours)",
                xaxis_title="Forecast Step (t+n hours)", yaxis_title="R² Score",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=400,
                yaxis=dict(range=[min(min(r2_vals) - 0.05, -0.1), 1.0]),
            )
            st.plotly_chart(fig_r2_step, width='stretch')
        else:
            st.info("Per-step R² diagnostik belum tersedia. Jalankan evaluasi ulang.")
        
        # ====== ROW 6: Per-Hour-of-Day Error Analysis ======
        st.markdown("#### 🕐 Diagnostik: Error per Jam (Hour of Day)")
        hourly_metrics = results.get('hourly_metrics', {})
        if hourly_metrics:
            col1, col2 = st.columns(2)
            
            hours = list(range(24))
            h_mae = [hourly_metrics.get(h, {}).get('mae', 0) for h in hours]
            h_rmse = [hourly_metrics.get(h, {}).get('rmse', 0) for h in hours]
            h_r2 = [hourly_metrics.get(h, {}).get('r2', 0) for h in hours]
            
            with col1:
                fig_hourly = go.Figure()
                fig_hourly.add_trace(go.Bar(
                    x=hours, y=h_mae,
                    marker_color='#818cf8', opacity=0.8,
                    name='MAE (kW)',
                    hovertemplate='Hour %{x}: MAE=%{y:.3f} kW<extra></extra>'
                ))
                fig_hourly.add_trace(go.Bar(
                    x=hours, y=h_rmse,
                    marker_color='#f472b6', opacity=0.6,
                    name='RMSE (kW)',
                    hovertemplate='Hour %{x}: RMSE=%{y:.3f} kW<extra></extra>'
                ))
                fig_hourly.update_layout(
                    title="MAE & RMSE per Hour of Day (All 24h Steps)",
                    xaxis_title="Hour of Day", yaxis_title="Error (kW)",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    height=400, barmode='group',
                    xaxis=dict(dtick=1),
                )
                st.plotly_chart(fig_hourly, width='stretch')
            
            with col2:
                # Color-code R² bars: green if good, red if bad
                colors = ['#6ee7b7' if r > 0.7 else '#fbbf24' if r > 0.3 else '#ef4444' for r in h_r2]
                fig_r2h = go.Figure()
                fig_r2h.add_trace(go.Bar(
                    x=hours, y=h_r2,
                    marker_color=colors, opacity=0.85,
                    name='R² per Hour',
                    hovertemplate='Hour %{x}: R²=%{y:.4f}<extra></extra>'
                ))
                fig_r2h.add_hline(y=0.8, line_dash='dot', line_color='#6ee7b7', 
                                  annotation_text='Target', annotation_position='top left')
                fig_r2h.update_layout(
                    title="R² per Hour of Day",
                    xaxis_title="Hour of Day", yaxis_title="R²",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    xaxis=dict(dtick=1),
                )
                st.plotly_chart(fig_r2h, width='stretch')
            
            # Hourly metrics table
            with st.expander("📋 Tabel Lengkap Per-Jam", expanded=False):
                tbl_data = []
                for h in range(24):
                    m = hourly_metrics.get(h, {})
                    tbl_data.append({
                        'Jam': f"{h:02d}:00",
                        'MAE (kW)': f"{m.get('mae', 0):.4f}",
                        'RMSE (kW)': f"{m.get('rmse', 0):.4f}",
                        'R²': f"{m.get('r2', 0):.4f}",
                        'N Samples': m.get('count', 0),
                    })
                st.dataframe(pd.DataFrame(tbl_data), hide_index=True, width='stretch')
        else:
            # Fallback to old method
            try:
                df_test = results['df_test']
                test_indices = results['test_indices']
                abs_error_per_seq = np.abs(results['pv_test_actual'][:, 0] - results['pv_test_pred'][:, 0])
                hours_per_seq = df_test.index[test_indices].hour
                df_hourly = pd.DataFrame({'Hour': hours_per_seq[:len(abs_error_per_seq)], 'MAE': abs_error_per_seq[:len(hours_per_seq)]})
                hourly_stats = df_hourly.groupby('Hour')['MAE'].agg(['mean', 'std']).reset_index()
                fig_hour = go.Figure(go.Bar(x=hourly_stats['Hour'], y=hourly_stats['mean'], marker_color='#818cf8'))
                fig_hour.update_layout(title="MAE by Hour (h+1)", xaxis_title="Hour", yaxis_title="MAE (kW)", template="plotly_dark", height=400)
                st.plotly_chart(fig_hour, width='stretch')
            except Exception as e:
                st.caption(f"Hourly error chart tidak tersedia: {e}")
        
        # ====== ROW 6: Actual vs Predicted Distribution ======
        st.markdown("#### Power Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=actual_flat[mask_productive], nbinsx=80, name='Actual',
                marker_color='#6ee7b7', opacity=0.6
            ))
            fig_dist.add_trace(go.Histogram(
                x=pred_flat[mask_productive], nbinsx=80, name='Predicted',
                marker_color='#818cf8', opacity=0.6
            ))
            fig_dist.update_layout(
                title="Power Distribution (Productive Hours)",
                xaxis_title="Power (kW)", yaxis_title="Count",
                barmode='overlay',
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=400,
            )
            st.plotly_chart(fig_dist, width='stretch')
        
        with col2:
            # QQ-like: sorted actual vs sorted predicted  
            sorted_actual = np.sort(actual_flat[mask_productive])
            sorted_pred = np.sort(pred_flat[mask_productive])
            # Downsample for performance
            step = max(1, len(sorted_actual) // 2000)
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scattergl(
                x=sorted_actual[::step], y=sorted_pred[::step],
                mode='markers', marker=dict(size=3, color='#818cf8', opacity=0.5),
                name='Q-Q'
            ))
            fig_qq.add_trace(go.Scatter(
                x=[0, sorted_actual.max()], y=[0, sorted_actual.max()],
                mode='lines', line=dict(color='#ef4444', dash='dash'),
                name='Perfect'
            ))
            fig_qq.update_layout(
                title="Q-Q Plot: Actual vs Predicted Quantiles",
                xaxis_title="Actual Quantiles (kW)", yaxis_title="Predicted Quantiles (kW)",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=400,
            )
            st.plotly_chart(fig_qq, width='stretch')
        
        # ====== DOWNLOAD ======
        st.markdown("---")
        st.markdown("#### Download Data")
        col1, col2 = st.columns(2)
        with col1:
            csv_metrics = df_metrics.to_csv(index=False)
            st.download_button(
                label="Download Metrics (CSV)",
                data=csv_metrics,
                file_name="evaluation_metrics.csv",
                mime="text/csv",
                width='stretch',
            )
        with col2:
            df_preds = pd.DataFrame({
                'actual': actual_flat,
                'predicted': pred_flat,
                'ghi': ghi_flat,
            })
            csv_preds = df_preds.to_csv(index=False)
            st.download_button(
                label="Download Predictions (CSV)",
                data=csv_preds,
                file_name="test_predictions.csv",
                mime="text/csv",
                width='stretch',
            )
    else:
        st.info("Belum ada hasil evaluasi. Jalankan Evaluate atau Full Pipeline terlebih dahulu.")


# --- TAB: TARGET TESTING ---
with tab_transfer:
    st.markdown("### Target Domain Testing")
    st.markdown("Uji model terlatih pada data dari lokasi berbeda (misal: Indonesia).")
    
    os.makedirs(target_dir, exist_ok=True)
    target_files = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
    
    if not has_model:
        st.warning("Belum ada model terlatih. Jalankan Training terlebih dahulu.")
    elif not target_files:
        st.warning(f"Belum ada file CSV di `{os.path.abspath(target_dir)}/`")
        st.markdown("""
        **Langkah:**
        1. Letakkan file CSV data target di folder `data/target/`
        2. Pastikan format kolom sama dengan data training
        3. Kembali ke tab ini dan pilih file
        """)
        
        # File uploader as alternative
        uploaded = st.file_uploader("Atau upload CSV langsung:", type=['csv'])
        if uploaded:
            save_path = os.path.join(target_dir, uploaded.name)
            with open(save_path, 'wb') as f:
                f.write(uploaded.getvalue())
            st.success(f"File disimpan ke {save_path}")
            st.rerun()
    else:
        selected_target = st.selectbox("Pilih file data target:", target_files)
        
        if st.button("▶️ Run Target Testing", type="primary"):
            with st.spinner("Menjalankan Target Testing..."):
                try:
                    import io, contextlib
                    stdout_capture = io.StringIO()
                    
                    # Pick latest model
                    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5'))])
                    model_path = os.path.join(model_dir, model_files[-1])
                    target_csv = os.path.join(target_dir, selected_target)
                    
                    with contextlib.redirect_stdout(stdout_capture):
                        from src.predictor import test_on_target
                        metrics = test_on_target(model_path, target_csv, cfg)
                    
                    st.success("Target Testing selesai!")
                    col1, col2, col3, col4 = st.columns(4)
                    for col, (name, key) in zip(
                        [col1, col2, col3, col4],
                        [("MAE", "mae"), ("RMSE", "rmse"), ("R²", "r2"), ("MAPE", "mape")]
                    ):
                        with col:
                            st.metric(name, f"{metrics[key]:.4f}")
                    
                    with st.expander("📜 Output Detail"):
                        st.code(stdout_capture.getvalue(), language="text")
                except Exception as e:
                    st.error(f"Error: {e}")



# --- TAB: MODEL COMPARISON ---
with tab_compare:
    st.markdown("### 🏆 Model Comparison & Leaderboard")
    st.markdown("Bandingkan performa beberapa model secara berdampingan.")
    
    if os.path.exists(model_dir):
        all_models = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5', '.json')) or os.path.isdir(os.path.join(model_dir, f))]
        all_models = [f for f in all_models if not f.endswith('_meta.json')] # Filter meta files if any
        
        if all_models:
            st.markdown("#### 1. Pilih Model")
            selected_models = st.multiselect("Pilih model untuk dibandingkan:", all_models, 
                                            default=all_models[:min(2, len(all_models))],
                                            format_func=lambda x: label_format_with_time(x, model_dir),
                                            key="ms_comparison")
            
            if st.button("📊 Run Comparison Analysis", type="primary", use_container_width=True, key="btn_run_comp"):
                if not selected_models:
                    st.warning("Pilih minimal satu model.")
                else:
                    comparison_results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # LOG START TO CMD
                    print("\n" + "="*60)
                    print(f"📊 COMPARISON START: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"   Models to evaluate: {len(selected_models)}")
                    print("="*60)
                    sys.stdout.flush()

                    with st.spinner("🚀 Sedang menjalankan evaluasi mendalam..."):
                        for i, model_id in enumerate(selected_models):
                            msg = f"⏳ [{i+1}/{len(selected_models)}] Mengevaluasi: {model_id}"
                            status_text.text(msg)
                            print(f"   {msg}...")
                            sys.stdout.flush()
                            
                            try:
                                # 1. Clean session
                                tf.keras.backend.clear_session()
                                gc.collect()
                                
                                # 2. Get Model Path
                                model_path = os.path.join(model_dir, model_id)
                                model_root = model_dir
                                if os.path.isdir(model_path):
                                    model_root = model_path
                                    for ext in ['model.keras', 'model.h5']:
                                        if os.path.exists(os.path.join(model_path, ext)):
                                            model_path = os.path.join(model_path, ext)
                                            break
                                
                                # 3. Load Model
                                custom_objs = get_custom_objects()
                                with tf.keras.utils.custom_object_scope(custom_objs):
                                    model = tf.keras.models.load_model(model_path, compile=False)
                                
                                compile_model(model, cfg['model']['hyperparameters']['learning_rate'])
                                
                                # 4. Run Evaluation
                                res = None
                                eval_source = "Unknown"
                                
                                # Strategy 1: Load data from model's own data_source in meta.json
                                meta_path = os.path.join(model_root, "meta.json")
                                if os.path.exists(meta_path):
                                    try:
                                        with open(meta_path, 'r') as f:
                                            m_meta = json.load(f)
                                        orig_ds = m_meta.get('data_source', '').replace('\\', '/')
                                        
                                        if orig_ds and os.path.exists(orig_ds):
                                            temp_cfg = copy.deepcopy(cfg)
                                            temp_cfg['paths']['processed_dir'] = orig_ds
                                            res = evaluate_model(model, temp_cfg, data=None, scaler_dir=model_root)
                                            eval_source = "Bundled Data"
                                    except Exception as e_meta:
                                        print(f"      (!) Meta failed: {e_meta}")
                                
                                # Strategy 2: Fall back to active preprocessing data
                                if res is None:
                                    active_prep = st.session_state.get('prep_metadata', None)
                                    if active_prep is not None and active_prep.get('X_train') is not None:
                                        expected_n = model.input_shape[2]
                                        if active_prep['X_train'].shape[2] == expected_n:
                                            scaler_dir = model_root if os.path.isdir(model_root) else None
                                            res = evaluate_model(model, cfg, data=active_prep, scaler_dir=scaler_dir)
                                            eval_source = "Active Data"
                                        else:
                                            raise ValueError(f"Dim mismatch: Model={expected_n}, Prep={active_prep['X_train'].shape[2]}")
                                    else:
                                        raise ValueError("No valid data source found for this model.")

                                # 5. Extract Metrics
                                m_test = res['metrics_test']
                                m_train = res['metrics_train']
                                
                                train_time = m_meta.get('training_time_seconds', 0) if m_meta else 0
                                
                                feat_text = "N/A"
                                ds_path = m_meta.get('data_source', '').replace('\\', '/') if m_meta else ''
                                
                                summary_path = os.path.join(model_root, "prep_summary.json")
                                if not os.path.exists(summary_path) and ds_path and os.path.exists(os.path.join(ds_path, "prep_summary.json")):
                                    summary_path = os.path.join(ds_path, "prep_summary.json")
                                    
                                if os.path.exists(summary_path):
                                    try:
                                        with open(summary_path, 'r') as ff:
                                            summ_data = json.load(ff)
                                            if 'selected_features' in summ_data:
                                                feat_text = ", ".join(summ_data['selected_features'])
                                    except: pass
                                    
                                if feat_text == "N/A":
                                    feat_path = os.path.join(model_root, "selected_features.json")
                                    if not os.path.exists(feat_path) and ds_path and os.path.exists(os.path.join(ds_path, "selected_features.json")):
                                         feat_path = os.path.join(ds_path, "selected_features.json")
                                            
                                    if os.path.exists(feat_path):
                                        try:
                                            with open(feat_path, 'r') as ff:
                                                feat_list = json.load(ff)
                                            feat_text = ", ".join(feat_list)
                                        except: pass
                                
                                comparison_results.append({
                                    'Model ID': model_id,
                                    'R²': m_test['r2'],
                                    'MAE': m_test['mae'],
                                    'nMAE (%)': m_test.get('norm_mae', 0) * 100,
                                    'RMSE': m_test['rmse'],
                                    'nRMSE (%)': m_test.get('norm_rmse', 0) * 100,
                                    'Train Time (s)': train_time,
                                    'Features N': getattr(model, 'input_shape', [0,0,0])[2] if hasattr(model, 'input_shape') else '?',
                                    'Feature List': feat_text,
                                    'Lookback': getattr(model, 'input_shape', [0,0,0])[1] if hasattr(model, 'input_shape') else '?',
                                    'Verified On': eval_source
                                })
                                print(f"      ✅ Done (R²: {m_test['r2']:.4f})")
                                
                            except Exception as e:
                                err_info = str(e)
                                st.error(f"Error pada {model_id}: {err_info}")
                                print(f"      ❌ ERROR: {err_info}")
                                comparison_results.append({
                                    'Model ID': model_id,
                                    'R²': 0, 'MAE': 999, 'nMAE (%)': 999, 
                                    'RMSE': 999, 'nRMSE (%)': 999, 'Train Time (s)': 0,
                                    'Features N': 'Error', 'Feature List': 'Error',
                                    'Lookback': 'Error', 'Verified On': 'Error'
                                })
                            
                            progress_bar.progress((i + 1) / len(selected_models))
                            sys.stdout.flush()
                        
                        status_text.empty()
                        st.session_state.comparison_df = pd.DataFrame(comparison_results)
                        print(f"\n✅ COMPARISON FINISHED: {len(comparison_results)} models analyzed.")
                        print("="*60 + "\n")
                        sys.stdout.flush()
                        st.success("Analisis perbandingan selesai! Hasil tampil di bawah.")

            # Display Results if they exist in session
            if 'comparison_df' in st.session_state:
                df_comp = st.session_state.comparison_df
                
                # Metrics Table
                st.markdown("#### 2. Metrics Leaderboard")
                
                # Highlight best values
                def highlight_max(s):
                    is_max = s == s.max()
                    return ['background-color: rgba(16, 185, 129, 0.2)' if v else '' for v in is_max]
                
                def highlight_min(s):
                    is_min = s == s.min()
                    return ['background-color: rgba(16, 185, 129, 0.2)' if v else '' for v in is_min]

                styled_df = df_comp.style.format({
                    'R²': '{:.4f}', 'MAE': '{:.4f}', 'nMAE (%)': '{:.2f}',
                    'RMSE': '{:.4f}', 'nRMSE (%)': '{:.2f}', 'Train Time (s)': '{:.1f}'
                }).apply(highlight_max, subset=['R²']).apply(highlight_min, subset=['MAE', 'nMAE (%)', 'RMSE', 'nRMSE (%)'])
                
                st.dataframe(styled_df, width="stretch")
                
                # --- EXPORT TO EXCEL/CSV BUTTON ---
                st.markdown("<br>", unsafe_allow_html=True)
                export_col1, _ = st.columns([1, 3])
                with export_col1:
                    import io
                    try:
                        excel_buffer = io.BytesIO()
                        df_comp.to_excel(excel_buffer, index=False, engine='openpyxl')
                        excel_data = excel_buffer.getvalue()
                        st.download_button(
                            label="📥 Export Tabel ke Excel (.xlsx)",
                            data=excel_data,
                            file_name='model_comparison_results.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    except Exception:
                        # Fallback to CSV if openpyxl is not installed
                        csv_data = df_comp.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Export Tabel ke Excel (.csv)",
                            data=csv_data,
                            file_name='model_comparison_results.csv',
                            mime='text/csv'
                        )
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Visualization
                st.markdown("#### 3. Visual Comparison")
                
                # First Row: R2 and MAE
                c1, c2 = st.columns(2)
                with c1:
                    fig_r2 = px.bar(df_comp, x='Model ID', y='R²', color='R²', 
                                   title="R² Score (Higher is Better)",
                                   color_continuous_scale='Viridis')
                    fig_r2.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_r2, width="stretch")
                with c2:
                    fig_mae = px.bar(df_comp, x='Model ID', y='MAE', color='MAE',
                                    title="MAE Score (Lower is Better)",
                                    color_continuous_scale='Reds_r')
                    fig_mae.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_mae, width="stretch")
                
                # Second Row: Train Time and Overfitting Delta
                c3, c4 = st.columns(2)
                with c3:
                    fig_time = px.bar(df_comp, x='Model ID', y='Train Time (s)', color='Train Time (s)',
                                    title="Training Time in Seconds (Lower is Faster)",
                                    color_continuous_scale='Oranges')
                    fig_time.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_time, width="stretch")
                with c4:
                    fig_rmse = px.bar(df_comp, x='Model ID', y='RMSE', color='RMSE',
                                    title="RMSE (Lower is Safer/Better)",
                                    color_continuous_scale='Purples_r')
                    fig_rmse.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_rmse, width="stretch")
                
                
                # Radar Chart
                st.markdown("#### 4. Performance Radar")
                radar_data = df_comp.copy()
                cols_to_norm = ['R²', 'MAE', 'nMAE (%)', 'RMSE', 'nRMSE (%)']
                for col in cols_to_norm:
                    if col == 'R²':
                        radar_data[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min() + 1e-6)
                    else:
                        norm = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min() + 1e-6)
                        radar_data[col] = 1 - norm
                
                fig_radar = go.Figure()
                for i, row in radar_data.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row['R²'], row['MAE'], row['RMSE'], row['nMAE (%)'], row['nRMSE (%)']],
                        theta=['R²', 'MAE (Inverted)', 'RMSE (Inverted)', 'nMAE (Inverted)', 'nRMSE (Inverted)'],
                        fill='toself',
                        name=row['Model ID']
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=500
                )
                st.plotly_chart(fig_radar, width="stretch")
        else:
            st.warning("Belum ada model tersimpan di folder `models/`.")
    else:
        st.error("Folder models/ tidak ditemukan.")


# --- TAB: TROUBLESHOOTING & LOGS ---
with tab_troubleshoot:
    st.markdown("### Pipeline Logs")
    
    if st.session_state.pipeline_log:
        for log in reversed(st.session_state.pipeline_log):
            st.markdown(f'<div class="pipeline-step">{log}</div>', unsafe_allow_html=True)
    else:
        st.info("Belum ada aktivitas pipeline.")
    
    # Show available models
    st.markdown("---")
    st.markdown("#### Model Tersimpan")
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5', '.json'))]
        if model_files:
            for f in sorted(model_files, reverse=True):
                fpath = os.path.join(model_dir, f)
                size = os.path.getsize(fpath)
                st.markdown(f"`{f}` — {size/1024:.0f} KB")
        else:
            st.caption("Belum ada model.")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#475569; font-size:0.8rem;">'
    'PV Forecasting Pipeline v1.0 | Streamlit Dashboard'
    '</p>',
    unsafe_allow_html=True,
)
