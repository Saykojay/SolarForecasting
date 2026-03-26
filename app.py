import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Solar Forecasting Pipeline",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

"""
app.py - Streamlit Web Dashboard untuk PV Forecasting Pipeline
Run: streamlit run app.py
"""

import os
import sys

# ============================================================
# NUMPY CROSS-VERSION PICKLE COMPATIBILITY
# ============================================================
# Import safe_read_pickle from predictor to handle numpy.core <-> numpy._core
# renames between NumPy 1.x and 2.x when loading pickle files.
# This is imported early so it's available throughout the app.
from src.predictor import safe_read_pickle

# ============================================================
# PYTORCH WINDOWS DLL & INITIALIZATION FIX
# ============================================================
if os.name == 'nt':
    # Avoid DLL init failure on Windows by pre-loading torch lib if possible
    # and setting duplicate lib OK for OpenMP conflicts
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    
    # Try to add torch lib to DLL directory search
    try:
        import torch
        # Add Torch's own DLL directory to the path if it exists
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib_path):
            os.add_dll_directory(torch_lib_path)
            
        # Optional: Force CUDA initialization before TensorFlow claims the entire GPU
        if torch.cuda.is_available():
            torch.cuda.init()
            # print(f"[PyTorch] CUDA claimed: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        # Silently fail if torch isn't installed or has early errors
        pass

import json
import time
import copy
import numpy as np
import pandas as pd

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
            info = f"Running in GPU Mode. Available GPUs: {len(gpus)}"
        else:
            info = "Running in CPU Mode. No GPU detected or GPU disabled."
        
        time_str = datetime.now().strftime("%H:%M:%S")
        print(f"[{time_str}] PID:{os.getpid()} {info}")
        return info
    except Exception as e:
        return f"Error initializing GPU: {e}"

gpus_info = get_gpu_info()


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* ── System Sans-Serif Font Stack ── */
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
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    }

    /* ── Typography ── */
    .stMarkdown h1 { font-weight: 700 !important; letter-spacing: -0.02em; color: #e0e0e0; }
    .stMarkdown h2 { font-weight: 600 !important; letter-spacing: -0.01em; color: #d0d0d0; }
    .stMarkdown h3 { font-weight: 600 !important; color: #c8c8c8; }
    .stMarkdown h4 { font-weight: 500 !important; color: #b0b0b0; }
    .stMarkdown h5 { font-weight: 500 !important; color: #999; }
    .stMarkdown p, .stMarkdown li { font-weight: 400; line-height: 1.6; color: #999; }

    [data-testid="stWidgetLabel"] p {
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        color: #aaa !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        color: #777 !important;
        border-bottom: 2px solid transparent;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #e0e0e0 !important;
        border-bottom: 2px solid #ccc;
    }

    /* ── Buttons ── */
    .stButton button {
        font-weight: 500 !important;
        background: #1a1a1a !important;
        color: #ccc !important;
        border: 1px solid #333 !important;
        border-radius: 6px !important;
        transition: all 0.15s ease;
    }
    .stButton button:hover {
        background: #222 !important;
        border-color: #555 !important;
        color: #fff !important;
    }
    .stButton button[kind="primary"] {
        background: #222 !important;
        border-color: #444 !important;
        color: #e0e0e0 !important;
    }

    /* ── Background ── */
    .stApp { background-color: #0a0a0a; }

    /* ── Metric Cards ── */
    .metric-card {
        background: #0f0f0f;
        border: 1px solid #1a1a1a;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        transition: all 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.03);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e0e0e0;
    }
    .metric-value.status-yes { color: #b0b0b0; }
    .metric-value.status-no { color: #555; }
    .metric-label {
        color: #888;
        font-size: 0.8rem;
        font-weight: 400;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.03em;
        margin-top: 8px;
    }
    .status-ready { background: #1a1a1a; color: #ccc; border: 1px solid #333; }
    .status-missing { background: #111; color: #555; border: 1px solid #222; }

    /* ── Selectbox / Popover ── */
    div[data-baseweb="popover"] { z-index: 10000 !important; }
    div[data-baseweb="menu"] { max-height: 350px !important; overflow-y: auto !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d0d 0%, #111 100%);
        border-right: 1px solid #1a1a1a;
    }

    /* ── Dividers ── */
    hr, [data-testid="stDivider"] {
        border-color: #1a1a1a !important;
    }

    /* ── Pipeline Step Block ── */
    .pipeline-step {
        background: #0d0d0d;
        border-left: 3px solid #333;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 6px 6px 0;
        font-weight: 400;
    }

    /* ── Expanders ── */
    div[data-testid="stExpander"] {
        border: 1px solid #1a1a1a;
        border-radius: 6px;
    }

    /* ── Global Transitions ── */
    a, button, .metric-card, .pipeline-step, div[data-testid="stExpander"] {
        transition: all 0.15s ease;
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
    if name in ["Latest (Default)", "Current (Dynamic)", "-- Select --", "Current (Dynamic)", "Current / Last Run"]:
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

# --- ACTION FLAG SYSTEM ---
# Solves the "double-click" Streamlit bug: when widgets above a button change
# state, the first click triggers a rerun for the widget change, losing the
# button press.  Using on_click callbacks to set flags ensures the action
# fires reliably on the very next rerun.
def _set_action(flag_name):
    """on_click callback: sets a flag in session_state."""
    st.session_state[flag_name] = True

# Initialize all action flags once
_ACTION_FLAGS = [
    'action_run_prep', 'action_run_train', 'action_run_eval',
    'action_run_eval_runner', 'action_run_full', 'action_run_tune',
    'action_run_batch', 'action_run_target_test', 'action_run_comparison',
]
for _f in _ACTION_FLAGS:
    if _f not in st.session_state:
        st.session_state[_f] = False


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
    st.markdown(f"## {gt('sidebar_settings')}")
    
    # GPU status indicator
    try:
        if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
             del os.environ['CUDA_VISIBLE_DEVICES']
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            st.caption(f"GPU: Active ({gpus[0].name.split(':')[-1]})")
        else:
            st.caption("GPU: Not detected (CPU mode)")
    except Exception as e:
        st.caption(f"GPU: Error ({e})")
        
    st.markdown("---")
    
    # Device Selector
    st.markdown("##### Device Acceleration")
    device_options = ["GPU", "CPU"]
    actual_gpus = tf.config.list_physical_devices('GPU')
    default_device = "GPU" if actual_gpus else "CPU"
    
    selected_device = st.radio(
        "Hardware:",
        device_options,
        index=device_options.index(st.session_state.get('execution_device', default_device)),
        horizontal=True,
        help="GPU = Fast, CPU = Stable"
    )
    st.session_state.execution_device = selected_device
    
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
            
            selected = st.selectbox("Select Model for Evaluation", model_options, index=curr_idx,
                                   format_func=lambda x: label_format_with_time(x, model_dir))
            st.session_state.selected_model = selected
            st.caption(f"Active: `{selected}`")
        else:
            st.warning("No saved models found.")
    else:
        st.info("Models folder not found.")
    
    # Actions
    st.markdown("---")
    if st.button("Stop All Processes", use_container_width=True, type="secondary"):
        import subprocess
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe', '/FI', 'MEMUSAGE gt 500000'], 
                      capture_output=True, text=True)
        st.session_state.is_running = False
        st.warning("Processes stopped.")
    
    if st.button("Save Config", use_container_width=True):
        save_config_to_file(cfg)
        st.success("Config saved.")
        st.cache_data.clear()




# ============================================================
# MAIN AREA - HEADER
# ============================================================
# Status Cards & Definitions
new_arch = cfg['model'].get('architecture', 'patchtst')
has_data = os.path.exists(os.path.join(proc_dir, 'X_train.npy'))
has_model = any(f.endswith(('.keras', '.h5', '.json')) for f in os.listdir(model_dir)) or any(os.path.isdir(os.path.join(model_dir, f)) for f in os.listdir(model_dir) if not f.startswith('.')) if os.path.exists(model_dir) else False
has_target = any(f.endswith('.csv') for f in os.listdir(target_dir)) if os.path.exists(target_dir) else False

st.markdown(f"""
<div style="text-align:center; margin-bottom: 1.5rem;">
    <h1 style="color: #e0e0e0; font-size: 1.8rem; font-weight: 700;
               letter-spacing: -0.02em; margin-bottom: 0.25rem;">
        {gt('page_title')}
    </h1>
    <p style="color: #666; font-size: 0.95rem; font-weight: 400;">
        Architecture: {new_arch.upper()}
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    status = "ready" if has_data else "missing"
    val_cls = "status-yes" if has_data else "status-no"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value {val_cls}">{"YES" if has_data else "NO"}</div>
        <div class="metric-label">{gt('data_preprocessed')}</div>
        <span class="status-badge status-{status}">{gt('status_ready') if has_data else gt('status_missing')}</span>
    </div>""", unsafe_allow_html=True)
with col2:
    status = "ready" if has_model else "missing"
    val_cls = "status-yes" if has_model else "status-no"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value {val_cls}">{"YES" if has_model else "NO"}</div>
        <div class="metric-label">{gt('model_trained')}</div>
        <span class="status-badge status-{status}">{gt('status_ready') if has_model else gt('status_missing')}</span>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{new_arch.upper()}</div>
        <div class="metric-label">{gt('active_arch')}</div>
        <span class="status-badge status-ready">{gt('status_ready')}</span>
    </div>""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab_prep_features, tab_data, tab_baseline, tab_train, tab_batch, tab_tuning, tab_eval, tab_compare, tab_transfer = st.tabs([
    gt('data_prep'),
    gt('data_insights'),
    "Baseline & Physics",
    gt('training_center'),
    gt('batch_experiments'),
    gt('optuna_tuning'),
    gt('prediction_eval'),
    gt('model_comparison'),
    gt('target_testing'),
])

# --- TAB: DATA PREP & FEATURES ---
with tab_prep_features:
    st.markdown("### Data Preparation & Feature Engineering")
    st.markdown("Transform raw CSV data into training-ready tensors with optimal features.")


    # --- SUB-SECTION 2: PREPROCESSING ---
    st.markdown("#### Pipeline Preprocessing")
    st.markdown("### Data Preprocessing")
    st.markdown("Transform raw CSV data into training-ready tensors.")
    
    # --- LAST LOG PERSISTENCE ---
    if st.session_state.get('last_prep_log'):
        with st.expander("Last Preprocessing Log", expanded=False):
            st.code(st.session_state.last_prep_log, language="text")
            if st.button("Clear Log", key="clear_log_prep"):
                st.session_state.last_prep_log = None
                st.rerun()

    # --- DATASET SELECTION (Moved from Data Insight) ---
    with st.expander("Select Source Dataset", expanded=not has_data):
        raw_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
        if os.path.exists(raw_dir):
            raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
            current_csv_name = os.path.basename(cfg['data']['csv_path'])
            selected_file_p = st.selectbox("Select CSV File to Process:", raw_files, 
                                         index=raw_files.index(current_csv_name) if current_csv_name in raw_files else 0,
                                         key="dataset_select_prep")
            cfg['data']['csv_path'] = f"data/raw/{selected_file_p}"
            
            if st.button("Preview Raw Data", key="btn_preview_prep"):
                df_prev = pd.read_csv(cfg['data']['csv_path'], sep=cfg['data']['csv_separator'], nrows=5)
                st.dataframe(df_prev)
        else:
            st.warning("Folder data/raw not found.")

    # --- NEW SECTION: TARGET CONFIGURATION ---
    with st.expander("Step 1: Target Variable & Physical Normalization", expanded=True):
        st.markdown("##### Target Prediction Mode")
        st.info("Tentukan variabel apa yang ingin diprediksi oleh model AI.")
        
        target_mode_options = ["Clear-Sky Index (CSI) - Recommended", "Raw PV Output (kW)"]
        target_mode_idx = 0 if cfg['target'].get('use_csi', True) else 1
        
        target_selection = st.radio(
            "Select Target Mode:",
            target_mode_options,
            index=target_mode_idx,
            help="CSI menormalisasi cuaca ekstrem agar model lebih fokus pada pola awan. Raw PV memprediksi nilai kW langsung.",
            key="target_mode_radio"
        )
        
        use_csi = (target_selection == target_mode_options[0])
        cfg['target']['use_csi'] = use_csi
        
        if use_csi:
            cc1, cc2 = st.columns(2)
            with cc1:
                cfg['target']['csi_ghi_threshold'] = st.number_input(
                    "GHI Threshold for CSI (W/m²)", 
                    value=cfg['target'].get('csi_ghi_threshold', 50),
                    min_value=0, max_value=500, 
                    help="Data di bawah GHI ini akan dianggap 0 (malam/gelap) untuk menghindari noise pembagian nol.",
                    key="p_csi_th"
                )
            with cc2:
                cfg['target']['csi_max'] = st.number_input(
                    "CSI Max Clipping", 
                    value=cfg['target'].get('csi_max', 1.2),
                    min_value=1.0, max_value=2.0, step=0.1,
                    help="Membatasi nilai CSI (biasanya max 1.2) untuk menghindari spike akibat refleksi awan yang tidak wajar.",
                    key="p_csi_max"
                )
        else:
            st.warning("⚠️ Memprediksi Raw PV Output (kW) secara langsung mungkin lebih sulit bagi model tanpa normalisasi cuaca.")

    # --- CONFIGURATION SECTION ---
    with st.expander("Step 2: Preprocessing & Features Configuration", expanded=not has_data):
        c1, c2 = st.columns(2)
        with c1:
            # --- CLEAR SEPARATION: INPUT FEATURES ---
            st.markdown("##### 1. Historical Input Features (Past Data)")
            st.caption("Select which historical metrics (X) are used by the model to forecast the target.")
            g = cfg['features'].get('groups', {})
            # time features
            st.markdown("---")
            st.markdown("**Cyclical Time Features (Sin/Cos)**")
            if True: # Simulating the block that was inside expander

                g['time_hour'] = st.checkbox("Hourly (Hour Sin/Cos)", value=g.get('time_hour', True), 
                                            help="Captures daily patterns (morning-noon-night).", key="p_time_h")
                g['time_day'] = st.checkbox("Daily (Day of Month Sin/Cos)", value=g.get('time_day', True), 
                                             help="Captures day-of-month patterns.", key="p_time_day")
                g['time_month'] = st.checkbox("Monthly (Month Sin/Cos)", value=g.get('time_month', True), 
                                             help="Captures monthly seasonal patterns.", key="p_time_m")
                g['time_doy'] = st.checkbox("Seasonal (DOY Sin/Cos)", value=g.get('time_doy', True), 
                                           help="Day of Year: High-resolution seasonal dynamics.", key="p_time_d")
                g['time_year'] = st.checkbox("Yearly (Linear)", value=g.get('time_year', False), 
                                            help="Captures long-term/yearly trends.", key="p_time_y")
            g['weather'] = st.checkbox("Weather (Temperature, GHI, etc.)", value=g.get('weather', True), key="p_weather")
            g['lags'] = st.checkbox("Time Lags (Lookback History)", value=g.get('lags', True), key="p_lags")
            g['rolling'] = st.checkbox("Moving Average", value=g.get('rolling', True), key="p_roll")
            g['physics'] = st.checkbox("Physics-based (Target as historical input)", value=g.get('physics', True), key="p_phys")
            cfg['features']['groups'] = g
            
            st.markdown("---")
            st.markdown("##### 2. Feature Selection Mode")
            sel_mode = st.radio(
                "Feature Selection Mode",
                ["auto", "manual"],
                index=0 if cfg['features'].get('selection_mode', 'auto') == 'auto' else 1,
                horizontal=True,
                help="**Auto**: Automatic correlation-based selection. **Manual**: Choose features manually.",
                key="p_sel_mode"
            )
            cfg['features']['selection_mode'] = sel_mode
            
            if sel_mode == 'manual':
                st.caption("Select input features manually. Auxiliary features (pv_clear_sky, pv_output_dc_kw) are automatically blocked.")
                try:
                    # Try to load ACTUAL feature names from last preprocessing run
                    feats_pkl_path = os.path.join(proc_dir, 'df_train_feats.pkl')
                    blocked_cols = {'pv_clear_sky', 'pv_cs_normalized', 'pv_output_dc_kw', 
                                   cfg['data']['time_col'], 'timestamp_col'}
                    
                    if os.path.exists(feats_pkl_path):
                        df_feats_sample = safe_read_pickle(feats_pkl_path)
                        all_available = sorted([c for c in df_feats_sample.columns 
                                              if c not in blocked_cols])
                        st.info(f"Loading {len(all_available)} features from last preprocessing")
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
                        st.warning("No preprocessing data available. Feature names are estimated. "
                                  "Run Preprocessing (Auto mode) first for an accurate feature list.")
                    
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
                        "Select Historical Input Features (X):",
                        options=all_available,
                        default=valid_defaults,
                        help="NOTE: These are NOT targets. If you select 'csi_target' here, the model will use 72h of historical 'csi_target' to forecast the future.",
                        key="p_manual_feats"
                    )
                    cfg['features']['manual_features'] = selected_manual
                    
                    if selected_manual:
                        st.success(f" {len(selected_manual)} features selected.")
                    else:
                        st.warning("No features selected. Select at least 1 feature.")
                except Exception as e:
                    st.error(f"Failed to read features: {e}")

            else:
                st.caption("Features will be auto-selected based on correlation with target.")
                corr_th = st.slider("Correlation Threshold", 0.01, 0.5, 
                                   cfg['features'].get('corr_threshold', 0.1), 0.01,
                                   help="Features with correlation below threshold will be dropped.",
                                   key="p_corr_th")
                cfg['features']['corr_threshold'] = corr_th
                
                multicol_th = st.slider("Multicollinearity Threshold", 0.7, 1.0, 
                                       cfg['features'].get('multicol_threshold', 0.95), 0.01,
                                       help="Features with inter-feature correlation above threshold will be dropped (prevents redundancy).",
                                       key="p_multicol_th")
                cfg['features']['multicol_threshold'] = multicol_th

            st.markdown("##### Data Split & Scaling")
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                cap_p = st.number_input("Capacity (kW)", value=cfg['pv_system']['nameplate_capacity_kw'],
                                       min_value=0.1, step=0.5, key="p_cap")
                cfg['pv_system']['nameplate_capacity_kw'] = cap_p
                
                # Split Mode Selection
                split_mode_ui = st.radio("Splitting Method:", ["Standard (Temporal)", "Tropical Seasonal"], 
                                         index=0 if cfg['splitting'].get('split_mode', 'standard') == 'standard' else 1,
                                         key="p_split_mode_radio")
                split_mode = 'seasonal' if "Season" in split_mode_ui else 'standard'
                cfg['splitting']['split_mode'] = split_mode
                
                if split_mode == 'standard':
                    train_ratio_p = st.slider("Train Ratio", 0.5, 0.95, cfg['splitting'].get('train_ratio', 0.8), 0.05, key="p_split")
                    cfg['splitting']['train_ratio'] = train_ratio_p
                    cfg['splitting']['test_ratio'] = round(1 - train_ratio_p, 2)
                else:
                    month_names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agt", "Sep", "Okt", "Nov", "Des"]
                    curr_months = cfg['splitting'].get('test_months', [12]) 
                    default_m = [month_names[m-1] for m in curr_months if 1 <= m <= 12]
                    sel_m_names = st.multiselect("Test Months:", month_names, default=default_m, key="p_sel_months")
                    cfg['splitting']['test_months'] = [month_names.index(m)+1 for m in sel_m_names]

            with c_s2:
                horizon_p = st.number_input("Horizon (hours)", value=cfg['forecasting']['horizon'],
                                           min_value=1, max_value=168, step=1, key="p_hor")
                cfg['forecasting']['horizon'] = horizon_p
                
                scaler_options = ["minmax", "standard"]
                current_scaler = cfg['features'].get('scaler_type', 'minmax').lower()
                selected_scaler = st.selectbox("Scaling Method (Prep)", scaler_options,
                                             index=scaler_options.index(current_scaler) if current_scaler in scaler_options else 0,
                                             key="p_scale")
                cfg['features']['scaler_type'] = selected_scaler

            st.markdown("---")
            # Target transform moved to the top of c1

        with c2:
            st.markdown("##### Cleaning (Algorithm 1)")
            pcfg = cfg.get('preprocessing', {})
            pcfg['resample_1h'] = st.checkbox("Resample Hourly", value=pcfg.get('resample_1h', True), key="p_res")
            pcfg['remove_outliers'] = st.checkbox("Remove Outliers", value=pcfg.get('remove_outliers', True), key="p_out")
            
            if pcfg['remove_outliers']:
                st.caption("Outlier Rules:")
                pcfg['ghi_high_pv_zero'] = st.checkbox("PV=0 when GHI is bright", value=pcfg.get('ghi_high_pv_zero', True), key="p_ghi_pv")
                pcfg['ghi_dark_pv_high'] = st.checkbox("High PV when GHI is dark", value=pcfg.get('ghi_dark_pv_high', True), key="p_dark_pv")
            
            st.markdown("---")
            st.markdown("##### Advanced Cleaning")
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
            st.markdown("##### Limit Dataset (Subset)")
            subset_mode = pcfg.get('subset_mode', 'semua_data' if not pcfg.get('trim_rows') else 'baris')
            
            smode_list = ["All Data", "Limit Row Count", "Date Range"]
            smode_idx = 0
            if subset_mode == 'baris': smode_idx = 1
            elif subset_mode == 'tanggal': smode_idx = 2
            
            sel_smode = st.radio("Data Subsetting Method:", smode_list, index=smode_idx, key="p_smode_radio")
            
            if sel_smode == "All Data":
                pcfg['subset_mode'] = 'semua_data'
                pcfg['trim_rows'] = False
            elif sel_smode == "Limit Row Count":
                pcfg['subset_mode'] = 'baris'
                pcfg['trim_rows'] = st.number_input("Take first X rows:", 
                                                   min_value=100, 
                                                   value=int(pcfg.get('trim_rows', 5000)) if pcfg.get('trim_rows') else 5000,
                                                   step=1000,
                                                   key="p_trim_val")
            elif sel_smode == "Date Range":
                pcfg['subset_mode'] = 'tanggal'
                pcfg['trim_rows'] = False
                
                # Default dates fallback
                default_start = datetime(2021, 1, 1).date()
                default_end = datetime(2021, 12, 31).date()
                
                st.caption("Select the date range for the dataset.")
                c_d1, c_d2 = st.columns(2)
                
                with c_d1:
                    curr_start = pcfg.get('start_date', default_start)
                    if isinstance(curr_start, str):
                        try: curr_start = datetime.strptime(curr_start, "%Y-%m-%d").date()
                        except: curr_start = default_start
                    d_start = st.date_input("Start:", value=curr_start, key="p_dstart")
                    pcfg['start_date'] = str(d_start)
                    
                with c_d2:
                    curr_end = pcfg.get('end_date', default_end)
                    if isinstance(curr_end, str):
                        try: curr_end = datetime.strptime(curr_end, "%Y-%m-%d").date()
                        except: curr_end = default_end
                    d_end = st.date_input("End:", value=curr_end, key="p_dend")
                    pcfg['end_date'] = str(d_end)

            cfg['preprocessing'] = pcfg


    st.markdown("---")
    st.info(" **Pipeline**: Raw Data → Cleaning → Feature Engineering → Scaling → Sequencing")
    
    col_prep_l, col_prep_r = st.columns([2, 1])
    with col_prep_l:
        st.markdown('<div class="pipeline-step">', unsafe_allow_html=True)
        st.markdown("**Run Preprocessing Pipeline**")
        v_name_prep = st.text_input("Version Name (Optional)", placeholder="e.g. v1_weather_only", key="v_name_prep")
        
        # NEW: Lookback Method Selection
        prep_method_ui = st.radio(
            "Preprocessing Type:",
            ["Fixed Sequence (Tensor .npy)", "Lookback Agnostic (Clean Table)"],
            index=0,
            horizontal=True,
            help="Fixed: Creates NPY tensors with fixed lookback (required for training). Agnostic: Data cleaning and features only (for flexible data exploration)."
        )
        prep_method = 'fixed' if "Fixed" in prep_method_ui else 'agnostic'
        
        if prep_method == 'fixed':
            st.caption("This will generate .npy tensor files and scalers in data/processed.")
            lookback_p = st.number_input("Lookback (hours)", value=cfg['model']['hyperparameters']['lookback'],
                                         min_value=6, max_value=720, step=6, key="p_lookback_final",
                                         help="Pilih ukuran window history data. Hanya diperlukan untuk mode Fixed Sequence.")
            cfg['model']['hyperparameters']['lookback'] = lookback_p
        else:
            st.caption("This will only perform data cleaning, feature engineering, and scaling without sequence slicing. Saves disk space and is flexible.")
            
        st.button("Start Preprocessing", type="primary", use_container_width=True, key="btn_prep_main", on_click=_set_action, args=("action_run_prep",))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_prep_r:
        st.markdown("**Dataset Status**")
        if st.session_state.get('prep_metadata'):
            st.success("Data Ready")
            # Show condensed stats
            stats = st.session_state.prep_metadata.get('stats', {})
            st.metric("Final Train Rows", stats.get('train_final', 0))
            st.metric("Final Test Rows", stats.get('test_final', 0))
        else:
            st.warning("Data Not Processed")

    if st.session_state.get("action_run_prep"):
        st.session_state.action_run_prep = False
        with st.spinner("Running preprocessing..."):
            import io, contextlib
            try:
                from src.data_prep import run_preprocessing
                stdout_capture = io.StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    # Pass the custom version name if provided
                    v_name = v_name_prep.strip() if v_name_prep.strip() else None
                    metadata = run_preprocessing(cfg, version_name=v_name, method=prep_method)
                st.session_state.prep_metadata = metadata
                st.session_state.last_prep_log = stdout_capture.getvalue()
                st.session_state.roll_corr_fig = None # Clear old chart
                st.session_state.pipeline_log.append(
                    f"[{datetime.now():%H:%M:%S}] Preprocessing ({prep_method}) selesai. "
                    f"Features: {metadata['n_features']}"
                )
                st.success(f"Preprocessing complete!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# --- TAB: EVALUATION RUNNER ---
with tab_eval:

    
    # Show last prep log if available
    if st.session_state.get('last_prep_log'):
        with st.expander("Last Preprocessing Log", expanded=False):
            st.code(st.session_state.last_prep_log, language="text")
            if st.button("Clear Log"):
                st.session_state.last_prep_log = None
                st.rerun()



# --- TAB: DATA INSIGHTS ---
with tab_data:
    st.markdown(f"### {gt('data_insights')}")
    st.caption("Details of data transformation from raw CSV to model-ready tensors.")
    
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
                
            sel_v = st.selectbox("Load Archived Data Version:", options, index=default_idx, key="sel_v_insight_global")
            
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
            st.warning("Folder 'data/processed' not found.")

    with v_col2:
        st.write("") # padding
        st.write("") 
        if st.button("Refresh Versions", use_container_width=True):
            st.rerun()

    st.markdown("---")
    m = st.session_state.get('prep_metadata')
    if m:
        stats = m['stats']
        sel_f = m['selected_features']
        all_f = m['all_features']
        
        # --- PHASE 1: DATA CLEANING (ALGORITHM 1) ---
        st.markdown("#### Phase 1: Cleaning & Integrity (Algorithm 1)")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Data", f"{stats['original_rows']:,}", help="Total baris awal dari file CSV.")
        c2.metric("Post-Cleaning", f"{stats['after_algorithm1']:,}", 
                  delta=f"{stats['after_algorithm1'] - stats['original_rows']:,}",
                  help="Baris tersisa setelah Filter Fisika & Outlier.")
        c3.metric("NaN Dropped", f"{stats['dropped_missing']:,}", delta_color="inverse",
                  help="Baris yang dibuang karena memiliki nilai kosong (NaN).")
        c4.metric("Valid Sequences", f"{stats['train_final']:,}",
                  help="Total sequences (X, y) created after temporal continuity checks.")

        # Visual Flow of Data Reduction
        flow_data = pd.DataFrame({
            'Stage': ['Original', 'After Cleaning', 'After NaN Drop', 'Final Sequences'],
            'Rows': [stats['original_rows'], stats['after_algorithm1'], 
                     stats['after_algorithm1'] - stats['dropped_missing'], stats['train_final']]
        })
        fig_flow = px.area(flow_data, x='Stage', y='Rows', title="Data Pipeline Flow (Volume Retention)")
        fig_flow.update_layout(template="plotly_dark", height=300, 
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_flow, use_container_width=True)

        st.markdown("---")
        
        # --- PHASE 2: FEATURE ENGINEERING & SELECTION ---
        st.markdown("#### Phase 2: Feature Engineering & Selection")
        
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
            st.plotly_chart(fig_types, use_container_width=True)
            
            st.markdown(f"**Features Selected by Algorithm:** `{len(sel_f)}`")
            with st.expander("View Final Feature List"):
                for i, f in enumerate(sel_f):
                    st.markdown(f"{i+1}. `{f}`")
        
        with col_f2:
            st.markdown("**Pearson Correlation Heatmap (Features ↔ Target)**")
            # Try to load the feature table from disk
            _feats_pkl = os.path.join(cfg['paths']['processed_dir'], 'df_train_feats.pkl')
            _corr_loaded = False
            if os.path.exists(_feats_pkl):
                try:
                    _df_corr = safe_read_pickle(_feats_pkl)
                    # Only keep selected features + target columns that exist in df
                    _target_col = None
                    for _tc in ['csi_target', 'pv_output_kw', 'pv_output_dc_kw']:
                        if _tc in _df_corr.columns:
                            _target_col = _tc
                            break
                    _keep_cols = [f for f in sel_f if f in _df_corr.columns]
                    if _target_col and _target_col not in _keep_cols:
                        _keep_cols = [_target_col] + _keep_cols
                    elif not _target_col and _keep_cols:
                        _target_col = _keep_cols[0]
                    
                    if len(_keep_cols) >= 2:
                        _df_sub = _df_corr[_keep_cols].dropna()
                        _corr_matrix = _df_sub.corr(method='pearson')
                        
                        # Sort rows/cols by correlation to target (descending absolute value)
                        if _target_col in _corr_matrix.columns:
                            _sort_order = _corr_matrix[_target_col].abs().sort_values(ascending=True).index.tolist()
                            _corr_sorted = _corr_matrix.loc[_sort_order, _sort_order]
                        else:
                            _corr_sorted = _corr_matrix
                        
                        # Build custom hover text: "x: ...\ny: ...\nr: ..."
                        _z = _corr_sorted.values
                        _labels_x = _corr_sorted.columns.tolist()
                        _labels_y = _corr_sorted.index.tolist()
                        _hover = [[
                            f"x: {_labels_x[j]}<br>y: {_labels_y[i]}<br>r: {_z[i][j]:.4f}"
                            for j in range(len(_labels_x))]
                            for i in range(len(_labels_y))]
                        
                        import plotly.graph_objects as go
                        _fig_corr = go.Figure(data=go.Heatmap(
                            z=_z,
                            x=_labels_x,
                            y=_labels_y,
                            text=_hover,
                            hovertemplate="%{text}<extra></extra>",
                            colorscale=[
                                [0.0,  "#d73027"],
                                [0.25, "#f46d43"],
                                [0.45, "#fdae61"],
                                [0.5,  "#f7f7f7"],
                                [0.55, "#abd9e9"],
                                [0.75, "#4575b4"],
                                [1.0,  "#053061"],
                            ],
                            zmid=0,
                            zmin=-1,
                            zmax=1,
                            colorbar=dict(
                                thickness=12,
                                len=0.9,
                                tickfont=dict(color="#94a3b8", size=10),
                                title=dict(text="r", font=dict(color="#94a3b8"))
                            ),
                        ))
                        _n = len(_labels_x)
                        _cell_size = max(16, min(32, 600 // _n))
                        _fig_corr.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=max(350, _n * _cell_size + 80),
                            margin=dict(t=10, b=80, l=120, r=20),
                            xaxis=dict(
                                tickfont=dict(size=9, color="#94a3b8"),
                                tickangle=-45,
                                side="bottom",
                            ),
                            yaxis=dict(
                                tickfont=dict(size=9, color="#94a3b8"),
                                autorange="reversed",
                            ),
                        )
                        st.plotly_chart(_fig_corr, use_container_width=True)
                        st.caption(
                            f"Pearson r antara {len(_keep_cols)} fitur yang dipilih. "
                            f"Hover sel untuk melihat nilai r persis. "
                            f"Merah = korelasi negatif kuat, Biru = positif kuat."
                        )
                        _corr_loaded = True
                    else:
                        st.warning("Tidak cukup kolom untuk membuat heatmap korelasi.")
                except Exception as _e:
                    st.error(f"Gagal membuat heatmap korelasi: {_e}")
            
            if not _corr_loaded and not os.path.exists(_feats_pkl):
                st.info("Jalankan Preprocessing terlebih dahulu untuk melihat heatmap korelasi Pearson.")

        st.markdown("---")
        
        # --- PHASE 3: DATA SPLITTING & SEQUENCING ---
        st.markdown("#### Phase 3: Dataset Splitting & Scaling")
        
        # Display the distribution chart
        split_data = pd.DataFrame({
            'Set': ['Training', 'Validation'],
            'Sequences': [stats['train_final'], stats['test_final']]
        })
        fig_split = px.bar(split_data, x='Set', y='Sequences', color='Set',
                            color_discrete_map={'Training': '#818cf8', 'Validation': '#555555'})
        fig_split.update_layout(template="plotly_dark", height=350, showlegend=False,
                                title="Train/Val Distribution")
        st.plotly_chart(fig_split, use_container_width=True)
        st.caption("Distribusi data training dan validasi (sequences/baris) berdasarkan splitting method yang dipilih.")
    else:
        st.info("No preprocessing data. Run 'Step 1: Preprocessing' in the Runner tab.")

# --- TAB: TRAINING CENTER ---
with tab_train:
    st.markdown("### Training Center")
    
    # --- DATA VERSION SELECTOR ---
    with st.expander("Select Preprocessed Data Version", expanded=not has_data):
        # Always use the root processed directory for listing
        if os.path.exists(proc_dir):
            # Show ALL folders that have at least a prep_summary.json (not just X_train.npy)
            all_versions = [f for f in os.listdir(proc_dir) 
                           if os.path.isdir(os.path.join(proc_dir, f)) 
                           and os.path.exists(os.path.join(proc_dir, f, 'prep_summary.json'))]
            all_versions = sorted(all_versions, reverse=True)
            
            # Annotate which ones have tensors ready (X_train.npy)
            def _train_label(x):
                if x == "Latest (Default)":
                    return x
                has_tensors = os.path.exists(os.path.join(proc_dir, x, 'X_train.npy'))
                base = label_format_with_time(x, proc_dir)
                return base if has_tensors else f"⚠️ {base} [Agnostic - perlu Fixed Sequence]"
            
            options = ["Latest (Default)"] + all_versions
            
            # Select the most recent version if it was just created
            default_idx = 0
            selected_v = st.selectbox("Data Version for Training:", options, index=default_idx, key="data_version_train",
                                   format_func=_train_label)
            
            if st.button("Refresh Daftar Versi"):
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
                st.caption(f"Folder Active: `{os.path.basename(active_proc_dir)}` ✅")
            else:
                st.warning(
                    f"⚠️ Folder **`{os.path.basename(active_proc_dir)}`** belum memiliki data sekuens (`X_train.npy`). "
                    f"Dataset ini dibuat dengan mode **Lookback Agnostic** dan tidak bisa langsung dipakai untuk training. "
                    f"Silakan kembali ke tab **Preprocessing** dan jalankan ulang dengan mode **Fixed Sequence (Tensor .npy)** "
                    f"menggunakan nama yang sama."
                )
        else:
            st.warning("Processed folder not found. Run Preprocessing first.")


    st.markdown("Configure model architecture and run training here.")
    
    # --- Training Readiness Dashboard ---
    with st.container():
        st.markdown("#### Training Readiness Check")
        r_col1, r_col2, r_col3 = st.columns(3)
        
        # 1. Preprocessing Readiness
        with r_col1:
            prep_ok = has_data
            status_icon = "" if prep_ok else ""
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
            status_icon = "" if feat_ok else ""
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
            <div style="background: rgba(26, 31, 46, 0.6); padding: 15px; border-radius: 8px; border-left: 5px solid #555555;">
                <div style="font-size: 0.8rem; color: #94a3b8;">Target Architecture</div>
                <div style="font-size: 1.1rem; font-weight: 700;"> {cfg['model'].get('architecture', 'patchtst').upper()}</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 5px;">
                    Window: {cfg['model']['hyperparameters']['lookback']}h | BS: {cfg['model']['hyperparameters']['batch_size']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # 1. Hyperparameter Configuration Area
    with st.expander("Model Architecture & Hyperparameters", expanded=True):
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
            elif new_a in ['patchtst_hf', 'autoformer_hf', 'causal_transformer_hf', 'autoformer']: # New HF Models
                cfg['model']['hyperparameters'].update({
                    'd_model': 128, 'n_layers': 3, 'learning_rate': 0.0001, 
                    'batch_size': 32, 'dropout': 0.2, 'patch_len': 16, 
                    'stride': 8, 'n_heads': 16, 'ff_dim': 256, 'moving_avg': 25
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
            if new_a not in ['patchtst', 'patchtst_hf', 'autoformer_hf', 'causal_transformer_hf', 'timetracker', 'timeperceiver', 'autoformer']:
                for k in ['patch_len', 'stride', 'n_heads', 'ff_dim', 'n_shared_experts', 'n_private_experts', 'top_k', 'n_latent_tokens']:
                    cfg['model']['hyperparameters'].pop(k, None)
            elif new_a not in ['gru', 'lstm', 'rnn']:
                for k in ['use_bidirectional', 'use_revin']:
                    cfg['model']['hyperparameters'].pop(k, None)
            
            save_config_to_file(cfg)
            st.session_state.cfg = cfg
            st.session_state.tune_arch_selector = new_a

        with col_hp1:
            st.markdown("**Core Structure**")
            _valid_archs = ["patchtst", "patchtst_hf", "autoformer_hf", "causal_transformer_hf", "timetracker", "timeperceiver", "autoformer", "gru", "lstm", "rnn"]
            _dummy = st.selectbox("Model Architecture", _valid_archs, 
                                  index=_valid_archs.index(arch) if arch in _valid_archs else 0,
                                  key="arch_selector_train",
                                  on_change=_update_architecture)
            
            # Rebind new_arch precisely to the confirmed architecture
            new_arch = cfg['model']['architecture']
            
            hp['lookback'] = st.select_slider("Lookback Window (h)", 
                                              options=[24, 48, 72, 96, 120, 144, 168, 192, 240, 336, 504, 720],
                                              value=hp.get('lookback', 72))
            
            # Adaptive Labels for Core Structure
            if new_arch in ["patchtst", "patchtst_hf", "autoformer_hf", "causal_transformer_hf", "autoformer"]:
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
            _bs_opts = sorted(set([16, 32, 64, 128, 256, 512] + [hp.get('batch_size', 32)]))
            hp['batch_size'] = st.selectbox("Batch Size", _bs_opts,
                                             index=_bs_opts.index(hp.get('batch_size', 32)))
            

            # --- NEW TACTIC 3 SCHEDULING ---
            cfg['training']['use_batch_scheduling'] = st.checkbox(
                "Batch Size Scheduling", 
                value=cfg['training'].get('use_batch_scheduling', False),
                key="cb_batch_scheduling",
                help="Tactic 3: Start training with a small Batch Size, then double it periodically. Makes searcrian akurasi lebih baik di awal dan eksekusi lebih cepat di akhir."
            )
            if cfg['training']['use_batch_scheduling']:
                cfg['training']['max_batch_size'] = st.selectbox(
                    "Batas Maksimal Batch Size (Limit)", 
                    _bs_opts,
                    index=_bs_opts.index(cfg['training'].get('max_batch_size', 512)) if cfg['training'].get('max_batch_size', 512) in _bs_opts else len(_bs_opts)-1,
                    key="sb_max_batch_size",
                    help="Stop doubling batch size when it reaches this limit to prevent OOM errors."
                )

            hp['dropout'] = st.number_input("Dropout Rate", value=hp.get('dropout', 0.2), 
                                             min_value=0.0, max_value=0.9, step=0.01, format="%.2f")
            
            # --- ARCHITECTURE SPECIFIC PARAMS ---
            if new_arch in ["patchtst", "patchtst_hf", "autoformer_hf", "causal_transformer_hf", "autoformer", "timetracker", "timeperceiver"]:
                # Dynamic Labeling & Content
                _is_patch = new_arch in ["patchtst", "patchtst_hf", "timetracker", "timeperceiver"]
                _is_autoformer = new_arch in ["autoformer", "autoformer_hf"]
                
                st.markdown("---")
                st.markdown(f"**{exp_title}**")
                if True: # Simulating the block

                    # 1. Patching (Only for models that support Patching)
                    if _is_patch:
                        hp['patch_len'] = st.number_input("patch_len (P)", value=hp.get('patch_len', 16), min_value=2, step=2, key=f"hp_pl_{new_arch}")
                        hp['stride'] = st.number_input("stride (S)", value=hp.get('stride', 8), min_value=1, step=1, key=f"hp_st_{new_arch}")
                    
                    # 2. Moving Average (Khusus Autoformer)
                    if _is_autoformer:
                        hp['moving_avg'] = st.number_input("Moving Average Kernel", value=hp.get('moving_avg', 25), min_value=1, step=2, help="Kernel size for trend-seasonal decomposition (Must be odd)", key=f"hp_ma_{new_arch}")

                    # 3. Transformer/Attention Commons
                    hp['ff_dim'] = st.number_input("ff_dim (Feed-Forward dimension)", value=hp.get('ff_dim', hp['d_model'] * 2), min_value=32, step=32, key=f"hp_ff_{new_arch}")
                    _nheads_opts = sorted(set([1, 2, 4, 8, 12, 16, 32] + [hp.get('n_heads', 16)]))
                    hp['n_heads'] = st.selectbox("n_heads (Attention Heads)", _nheads_opts, index=_nheads_opts.index(hp.get('n_heads', 16)), key=f"hp_nh_{new_arch}")
            
            elif new_arch == "timetracker":
                st.markdown("---")
                st.markdown("**TimeTracker Specific Params**")
                if True:

                    hp['patch_len'] = st.number_input("patch_len (P)", value=hp.get('patch_len', 16), min_value=2, step=2)
                    hp['stride'] = st.number_input("stride (S)", value=hp.get('stride', 8), min_value=1, step=1)
                    _nheads_opts = sorted(set([1, 2, 4, 8, 12, 16] + [hp.get('n_heads', 8)]))
                    hp['n_heads'] = st.selectbox("n_heads (H) - Any-variate Rel", _nheads_opts, index=_nheads_opts.index(hp.get('n_heads', 8)))
                    st.markdown("**Mixture of Experts Setup**")
                    c_e1, c_e2 = st.columns(2)
                    hp['n_shared_experts'] = c_e1.number_input("Shared Experts", value=hp.get('n_shared_experts', 1), min_value=0, max_value=8)
                    hp['n_private_experts'] = c_e2.number_input("Private Experts", value=hp.get('n_private_experts', 4), min_value=1, max_value=32)
                    hp['top_k'] = st.number_input("Top-K Routing", value=hp.get('top_k', 2), min_value=1, max_value=hp['n_private_experts'], help="How many private experts are active per token")

            elif new_arch == "timeperceiver":
                st.markdown("---")
                st.markdown("**TimePerceiver Specific Params**")
                if True:

                    hp['patch_len'] = st.number_input("patch_len (P)", value=hp.get('patch_len', 16), min_value=2, step=2)
                    hp['stride'] = st.number_input("stride (S)", value=hp.get('stride', 8), min_value=1, step=1)
                    _nheads_opts = sorted(set([1, 2, 4, 8, 12, 16] + [hp.get('n_heads', 8)]))
                    hp['n_heads'] = st.selectbox("n_heads (H) - Bottleneck", _nheads_opts, index=_nheads_opts.index(hp.get('n_heads', 8)))
                    hp['n_latent_tokens'] = st.number_input("Latent Tokens (M)", value=hp.get('n_latent_tokens', 32), min_value=4, max_value=256, step=4, help="Ukuran bottleneck (M) untuk attention laten")
            
            elif new_arch in ["gru", "lstm", "rnn"]:
                st.markdown("---")
                st.markdown(f"**{new_arch.upper()} Specific Params**")
                if True:

                    st.info(f"Input 'Hidden Units' di atas menentukan kapasitas memori per {new_arch.upper()} cell.")
                    hp['use_bidirectional'] = st.checkbox("Use Bidirectional", value=hp.get('use_bidirectional', True), key=f"bi_{new_arch}")
                    hp['use_revin'] = st.checkbox("Use RevIN (Reversible Instance Normalization)", value=hp.get('use_revin', False), key=f"revin_{new_arch}")
                    st.caption(f"Architecture {new_arch.upper()} can have RevIN anti-anomaly normalization layer added.")

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
            st.button("Start Model Training", type="primary", use_container_width=True, disabled=not has_data, on_click=_set_action, args=("action_run_train",))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_ctrl2:
        st.markdown("**Status**")
        if has_data:
            st.success("Training Data Ready")
            st.caption(f"Sequences: {st.session_state.prep_metadata['stats']['train_final'] if st.session_state.prep_metadata else 'Loaded'}")
        else:
            st.error("Data missing! Run Prep first.")

    # 3. Training Execution Logic (Consolidated here)
    if st.session_state.get("action_run_train"):
        st.session_state.action_run_train = False
        output_container_train = st.container()
        with output_container_train:
            st.markdown("---")
            st.markdown("### Live Training Monitor")
            
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
                    def __init__(self, total_epochs=None):
                        super().__init__()
                        self.epoch_data = []
                        self.start_time = None
                        self.log_lines = []
                        self.forced_total_epochs = total_epochs
                    def on_train_begin(self, logs=None): 
                        if self.start_time is None:
                            self.start_time = time.time()
                    def on_epoch_end(self, epoch, logs=None):
                        elapsed = time.time() - self.start_time
                        total_epochs = self.forced_total_epochs if self.forced_total_epochs else self.params['epochs']
                        try:
                            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                        except Exception:
                            try:
                                current_lr = float(cfg['model']['hyperparameters'].get('learning_rate', 0.0001))
                            except Exception:
                                current_lr = 0.0001
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
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
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
                live_cb = StreamlitLiveCallback(cfg['training']['epochs'])
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
                st.success(f"Training Complete! ({duration_str}) | Model saved: {meta['model_id']}")
                time.sleep(2) 
                st.rerun()
            except Exception as e:
                st.error(f"Training Failed: {e}")
                import traceback; st.code(traceback.format_exc())

    # 4. Results & Metrics (Static view if not training)
    if st.session_state.training_history and not st.session_state.get('action_run_train'):
        st.markdown("---")
        st.subheader("Last Training Results")
        hist = st.session_state.training_history
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(y=hist['loss'], name='Train', line=dict(color='#818cf8')))
            fig_l.add_trace(go.Scatter(y=hist['val_loss'], name='Val', line=dict(color='#f472b6')))
            fig_l.update_layout(template="plotly_dark", height=350, title="Final Loss Curves", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_l, use_container_width=True)
        with c2:
            st.metric("Best Val Loss", f"{min(hist['val_loss']):.6f}")
            st.metric("Total Epochs", len(hist['loss']))
            
            if 'last_training_time' in st.session_state:
                t_sec = st.session_state['last_training_time']
                t_str = f"{t_sec:.2f}s" if t_sec < 60 else f"{t_sec/60:.2f}m"
                st.metric("Time Elapsed", t_str)
            
            if st.button("Clear History View"):
                st.session_state.training_history = None
                st.rerun()

    st.markdown("---")
    st.markdown("### Time Series Cross-Validation (TSCV)")
    st.caption("Test model stability across different time segments.")
    run_tscv = st.button("Run TSCV Evaluation", use_container_width=True, key="btn_tscv_tab")
    
    if run_tscv:
        with st.spinner("Running TSCV..."):
            from src.trainer import run_tscv
            res = run_tscv(cfg)
            st.dataframe(pd.DataFrame(res))

# --- TAB BATCH: SEQUENTIAL TRAINING ---
with tab_batch:
    try:
        st.markdown(f"### {gt('batch_manager_title')}")
        st.info(gt('batch_info'))
        
        if 'batch_queue' not in st.session_state:
            st.session_state.batch_queue = []
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = []
        if 'batch_running' not in st.session_state:
            st.session_state.batch_running = False
        
        col_bq1, col_bq2 = st.columns([1, 2])
        
        with col_bq1:
            st.markdown(f"#### {gt('add_to_queue')}")
            
            arch_list = ["patchtst", "patchtst_hf", "autoformer_hf", "causal_transformer_hf", "timetracker", "timeperceiver", "autoformer", "gru", "lstm", "rnn"]
            
            # Default to global active architecture if not set
            if 'batch_arch_selector' not in st.session_state:
                st.session_state.batch_arch_selector = cfg['model'].get('architecture', 'patchtst').lower()
                
            # Callback to handle dropdown changes cleanly
            def _update_batch_arch():
                st.session_state.batch_arch_selector = st.session_state.temp_batch_arch_selector

            q_arch_val = st.selectbox(gt('architecture'), 
                                     arch_list, 
                                     index=arch_list.index(st.session_state.batch_arch_selector) if st.session_state.batch_arch_selector in arch_list else 0,
                                     key="temp_batch_arch_selector",
                                     on_change=_update_batch_arch)
            
            q_name = st.text_input(gt('exp_name'), 
                                  value=f"Exp_{q_arch_val}_{len(st.session_state.batch_queue)+1}",
                                  key=f"batch_exp_name_{q_arch_val}")
            
            with st.expander(gt('config_hp'), expanded=True):
                # Start with a fresh set of defaults for the selected architecture
                # instead of blindly copying from the global active model
                q_hp = {}
                
                # Base defaults
                base_defaults = {
                    'lookback': 72, 'learning_rate': 0.0001, 'batch_size': 32, 'dropout': 0.2, 'd_model': 64, 'n_layers': 2
                }
                
                if q_arch_val in ['patchtst', 'patchtst_hf', 'autoformer_hf', 'causal_transformer_hf', 'autoformer']:
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
                    q_hp['lookback'] = st.selectbox("Lookback Window (h)", [24, 48, 72, 96, 120, 144, 168, 192, 240, 336, 504, 720], index=2, key=f"q_lb_{q_arch_val}")
                    
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
                if q_arch_val in ["patchtst", "patchtst_hf", "autoformer_hf", "causal_transformer_hf", "autoformer"]:
                    st.markdown("---")
                    sq1, sq2 = st.columns(2)
                    with sq1:
                        q_hp['patch_len'] = st.number_input("Patch Len", 4, 64, q_hp.get('patch_len', 16), 4, key=f"q_pl_{q_arch_val}")
                        q_hp['stride'] = st.number_input("Stride", 2, 32, q_hp.get('stride', 8), 2, key=f"q_st_{q_arch_val}")
                    with sq2:
                        q_hp['ff_dim'] = st.number_input("ff_dim", 32, 1024, q_hp.get('ff_dim', q_hp.get('d_model', 128)*2), 32, key=f"q_ff_{q_arch_val}")
                        _h_opts = [1, 2, 4, 8, 12, 16]
                        q_hp['n_heads'] = st.selectbox("n_heads", _h_opts, index=_h_opts.index(q_hp.get('n_heads', 16)) if q_hp.get('n_heads', 16) in _h_opts else 5, key=f"q_nh_{q_arch_val}")
                
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
                        q_hp['use_revin'] = st.checkbox("Use RevIN", value=q_hp.get('use_revin', False), key=f"q_rev_{q_arch_val}")
            
            with st.expander(gt('config_data_feat'), expanded=False):
                st.caption("Select data version or feature configuration")
                
                # List available preprocessed versions
                cur_dyn_str = gt('current_dynamic')
                proc_versions = [cur_dyn_str]
                if os.path.exists(proc_dir):
                    dirs = [d for d in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, d)) and os.path.exists(os.path.join(proc_dir, d, 'X_train.npy'))]
                    proc_versions.extend(sorted(dirs, reverse=True))
                
                q_data_v = st.selectbox(gt('data_version_select'), proc_versions, key="q_data_v",
                                      format_func=lambda x: label_format_with_time(x, proc_dir))
                
                st.markdown("---")
                st.caption(gt('feature_groups_info'))
                q_feat = cfg['features']['groups'].copy()
                q_feat['weather'] = st.checkbox("Weather Features", value=q_feat.get('weather', True), key="q_f_w")
                q_feat['time_hour'] = st.checkbox("Hour of Day", value=q_feat.get('time_hour', True), key="q_f_h")
                q_feat['time_month'] = st.checkbox("Month of Year", value=q_feat.get('time_month', True), key="q_f_m")
                q_feat['physics'] = st.checkbox("Physics (CS Index)", value=q_feat.get('physics', False), key="q_f_p")
                q_feat_mode = st.selectbox("Selection Mode", ["auto", "manual"], key="q_f_mode")

            if st.button(gt('add_to_queue'), use_container_width=True, key="btn_add_batch_queue", disabled=st.session_state.batch_running):
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
                print(f" [{datetime.now().strftime('%H:%M:%S')}] Queued: {q_name} ({q_arch_val})")
                import sys
                sys.stdout.flush()
                
                st.success(gt('add_to_queue_success'))
                time.sleep(0.5)
                st.rerun()

        with col_bq2:
            st.markdown(f"#### {gt('current_queue')}")
            if not st.session_state.batch_queue:
                st.write("Queue is empty. Add models on the left.")
            else:
                for i, item in enumerate(st.session_state.batch_queue):
                    col_i1, col_i2 = st.columns([4, 1])
                    col_i1.markdown(f"**{i+1}. {item['name']}** ({item['architecture']}) | LB: {item['hp']['lookback']}, D: {item['hp']['d_model']}")
                    if col_i2.button("", key=f"del_{i}"):
                        st.session_state.batch_queue.pop(i)
                        st.rerun()
                
                st.markdown("---")
                st.button(gt('run_batch_btn'), type="primary", use_container_width=True, disabled=st.session_state.batch_running, on_click=_set_action, args=("action_run_batch",))
                if st.session_state.get("action_run_batch"):
                    st.session_state.action_run_batch = False
                    st.session_state.batch_running = True
                    st.session_state.batch_results = [] # Reset results for new run
                    
                    batch_progress = st.progress(0)
                    status_text = st.empty()
                    
                    # Container for live monitor (will be persistent until clear)
                    monitor_container = st.container()
                    
                    total = len(st.session_state.batch_queue)
                    for i, item in enumerate(st.session_state.batch_queue):
                        status_text.markdown(f" **Processing {i+1}/{total}:** {item['name']}...")
                        
                        with monitor_container:
                            st.markdown(f"#### Monitoring: {item['name']}")
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
                        data_v = item.get('data_version', gt('current_dynamic'))
                        if data_v == gt('current_dynamic'):
                            if 'features' in item:
                                status_text.markdown(f"Sweep  **Preparing data for {item['name']}...**")
                                batch_cfg['features']['groups'] = item['features']['groups']
                                batch_cfg['features']['selection_mode'] = item['features']['selection_mode']
                                
                                from src.data_prep import run_preprocessing
                                run_preprocessing(batch_cfg)
                        else:
                            status_text.markdown(f" **Using archived data: {data_v}**")
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
                                m_chart.plotly_chart(fig, use_container_width=True)

                        try:
                            from src.trainer import train_model
                            loss_function = batch_cfg['model']['hyperparameters'].get('loss_fn', 'mse')
                            model, history, meta = train_model(batch_cfg, custom_model_id=item['name'], extra_callbacks=[BatchLiveCallback()], loss_fn=loss_function)
                            st.session_state.batch_results.append({"name": item['name'], "status": "Success", "loss": min(history.history['val_loss'])})
                        except Exception as e:
                            st.session_state.batch_results.append({"name": item['name'], "status": "Failed", "error": str(e)})
                        
                        batch_progress.progress((i + 1) / total)
                        # Don't empty monitor_container completely, just let it stay for a bit
                    
                    st.session_state.batch_queue = []
                    st.session_state.batch_running = False
                    status_text.success("All Batch Experiments Finished!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()

            # --- RESULTS SUMMARY (Visible after batch or if queue empty) ---
            if st.session_state.batch_results:
                st.markdown("---")
                st.markdown("#### Results Summary")
                res_df = pd.DataFrame(st.session_state.batch_results)
                st.dataframe(res_df, use_container_width=True)
                if st.button("Clear Results"):
                    st.session_state.batch_results = []
                    st.rerun()
    except Exception as e:
        st.error(f"Batch Tab Error: {e}")
        import traceback
        st.code(traceback.format_exc())

# --- TAB BASELINE & PHYSICS ---
with tab_baseline:
    st.markdown("### Baseline & Physics Experiments")
    st.markdown("Test and set benchmarks using classical algorithms or pure-physics models (PVWatts/Single Diode) without Deep Learning.")
    
    st.info("These algorithms are much faster to validate and serve as a 'Minimum Viable Performance' benchmark for Deep Learning to surpass.")
    
    c_b1, c_b2 = st.columns([1, 1])
    
    with c_b1:
        baseline_group = st.selectbox("Select Baseline Category", ["Classical Machine Learning", "Physics Models (PVLib)"], key="baseline_group_sel")
        
        b_options = ["Linear Regression", "Ridge Regression", "Random Forest"] if baseline_group == "Classical Machine Learning" else ["PVWatts", "Single Diode (SAPM)"]
        b_label = "Select Algorithm" if baseline_group == "Classical Machine Learning" else "Select Physics Model"
        
        b_model = st.selectbox(b_label, b_options, key="b_model_sel")
            
    with c_b2:
        st.markdown("**Pemilihan Data**")
        proc_dir = cfg['paths'].get('processed_dir', 'data/processed')
        versions = []
        if os.path.exists(proc_dir):
            versions = [d for d in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, d)) and os.path.exists(os.path.join(proc_dir, d, 'X_train.npy'))]
            versions.sort(key=lambda x: os.path.getmtime(os.path.join(proc_dir, x)), reverse=True)
            
        b_data_opts = ["Latest (Default)"] + versions
        b_data_sel = st.selectbox("Test Data Version:", b_data_opts, format_func=lambda x: label_format_with_time(x, proc_dir) if x != "Latest (Default)" else x)
        
        active_b_dir = proc_dir if b_data_sel == "Latest (Default)" else os.path.join(proc_dir, b_data_sel)
        
        if baseline_group == "Physics Models (PVLib)":
            b_capacity = st.number_input("PV Capacity (kWDC)", value=1000, step=100, min_value=10)
    
    st.markdown("---")
    if st.button(f"Run Evaluation {b_model}", type="primary", use_container_width=True):
        if not os.path.exists(os.path.join(active_b_dir, 'X_train.npy')) and baseline_group == "Classical Machine Learning":
            st.error("X_train.npy not found. Create dataset in Feature Lab tab.")
        elif not os.path.exists(os.path.join(active_b_dir, 'df_test_feats.pkl')) and baseline_group == "Physics Models (PVLib)":
             st.error("df_test_feats.pkl not found.")
        else:
            with st.spinner(f"Running {b_model}..."):
                from src.baseline_models import evaluate_ml_baseline, evaluate_physics_baseline
                try:
                    if baseline_group == "Classical Machine Learning":
                        X_train = np.load(os.path.join(active_b_dir, 'X_train.npy'))
                        y_train = np.load(os.path.join(active_b_dir, 'y_train.npy'))
                        X_test = np.load(os.path.join(active_b_dir, 'X_test.npy'))
                        y_test = np.load(os.path.join(active_b_dir, 'y_test.npy'))
                        
                        y_scaler_path = os.path.join(active_b_dir, 'y_scaler.pkl')
                        if not os.path.exists(y_scaler_path):
                            y_scaler_path = os.path.join(proc_dir, 'y_scaler.pkl')
                            
                        import joblib
                        y_scaler = joblib.load(y_scaler_path) if os.path.exists(y_scaler_path) else None
                        
                        res = evaluate_ml_baseline(b_model, X_train, y_train, X_test, y_test, y_scaler=y_scaler)
                    else:
                        df_test = safe_read_pickle(os.path.join(active_b_dir, 'df_test_feats.pkl'))
                        res = evaluate_physics_baseline(b_model, df_test, capacity_kw=b_capacity)
                    
                    st.success("Evaluasi Selesai!")
                    
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("R2 Score", f"{res['metrics']['R2']:.4f}")
                    rc2.metric("MAE", f"{res['metrics']['MAE']:.4f}")
                    rc3.metric("RMSE", f"{res['metrics']['RMSE']:.4f}")
                    st.info(f"Train/Execution Time: {res['train_time']:.2f} sec")
                    
                    import plotly.graph_objects as go
                    try:
                        act = res['y_actual'].flatten() if res['y_actual'] is not None else []
                        prd = res['y_pred'].flatten()
                        
                        # Ambil 500 sampel terakhir maksimal
                        viz_limit = 500
                        if len(act) > viz_limit:
                            act = act[-viz_limit:]
                            prd = prd[-viz_limit:]
                            
                        fig = go.Figure()
                        if len(act) > 0:
                            fig.add_trace(go.Scatter(y=act, mode='lines', name='Actual', line=dict(color='gray', dash='dot')))
                        fig.add_trace(go.Scatter(y=prd, mode='lines', name=f'Predicted ({b_model})', line=dict(color='cyan')))
                        fig.update_layout(title="Grafik Cuplikan Prediksi vs Aktual", template="plotly_dark", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Gagal me-render chart: {e}")
                        
                except Exception as e:
                    import traceback
                    st.error(f"Gagal Evaluasi: {e}")
                    st.code(traceback.format_exc(), language="python")

# --- TAB TUNING: TUNING MONITOR ---
with tab_tuning:
    st.markdown("### Optuna Hyperparameter Tuning")
    
    # History Selector
    history_files = list_tuning_history()
    if history_files:
        c_hist1, c_hist2 = st.columns([3, 1])
        with c_hist1:
            selected_hist = st.selectbox(
                "Tuning History (Select to view previous results)", 
                ["Current / Last Run"] + history_files,
                index=0
            )
        with c_hist2:
            if st.button("Load History"):
                if selected_hist == "Current / Last Run":
                    st.session_state.tuning_results = load_tuning_results()
                    st.success("Loaded last run.")
                    st.session_state.tuning_results = load_specific_tuning_result(selected_hist)
                    st.success(f"Loaded: {selected_hist}")
                st.rerun()

    # Data & Features are now in their own tabs
    st.info(" **Tips**: Dataset, cleaning, and feature settings are now managed in **Data Insights**, **Preprocessing**, and **Feature Lab** tabs pipeline.")

    # Added: Data Selection and Mini Analytics inside Tuning Tab for immediate feedback
    st.markdown("**Select Dataset Version for Tuning**")
    
    versions = []
    if os.path.exists(proc_dir):
        versions = [d for d in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, d)) and os.path.exists(os.path.join(proc_dir, d, 'X_train.npy'))]
        versions.sort(key=lambda x: os.path.getmtime(os.path.join(proc_dir, x)), reverse=True)
    options = ["Latest (Default)"] + versions
    
    selected_v_tune = st.selectbox("Data Version for Tuning:", options, index=0, key="data_version_tune",
                           format_func=lambda x: label_format_with_time(x, proc_dir) if x != "Latest (Default)" else x)
                           
    if selected_v_tune == "Latest (Default)":
        active_tune_dir = proc_dir
    else:
        active_tune_dir = os.path.join(proc_dir, selected_v_tune)
        
    cfg['paths']['processed_dir'] = active_tune_dir
    has_data = os.path.exists(os.path.join(active_tune_dir, 'X_train.npy'))

    if has_data:
        try:
            with open(os.path.join(active_tune_dir, 'meta.json'), 'r') as f:
                m = json.load(f)
            
            st.markdown(f"""
            <div style="background-color: rgba(129, 140, 248, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #818cf8; margin-bottom: 20px;">
                <h4 style="margin-top:0;"> Selected Dataset: {selected_v_tune if selected_v_tune != "Latest (Default)" else "Default"}</h4>
                <div style="display: flex; justify-content: space-between;">
                    <div><b>Final Rows:</b> {m.get('stats', {}).get('after_algorithm1', 0):,}</div>
                    <div><b>Features:</b> {len(m.get('selected_features', []))}</div>
                    <div><b>Sequences:</b> {m.get('stats', {}).get('train_final', 0):,}</div>
                </div>
                <div style="font-size: 0.8em; color: #94a3b8; margin-top: 10px;">
                    Selected features: {", ".join(m.get('selected_features', [])[:5])}... 
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass
    else:
        st.warning(f"Dataset not ready at ({active_tune_dir}). Run preprocessing first in the Feature Lab.", icon="")

    # --- NEW: SEARCH SPACE EDITOR & EXECUTION (Always Visible) ---
    if cfg['tuning']['enabled']:
        st.markdown("#### Configure & Run Tuning")
        
        # Add model selector specifically for tuning context
        def _update_tuning_architecture():
            new_a = st.session_state.tune_arch_selector
            cfg['model']['architecture'] = new_a
            if new_a in ['patchtst', 'patchtst_hf', 'timetracker']:
                # Architectures that DO use patching
                cfg['model']['hyperparameters'].update({
                    'd_model': 128, 'n_layers': 3, 'learning_rate': 0.0001, 
                    'batch_size': 32, 'dropout': 0.2, 'patch_len': 16, 'stride': 8, 'n_heads': 16
                })
                for k in ['moving_avg', 'use_bidirectional']: cfg['model']['hyperparameters'].pop(k, None)
            elif new_a in ['causal_transformer_hf']:
                # Decoder-only: NO patching, NO moving avg
                cfg['model']['hyperparameters'].update({
                    'd_model': 128, 'n_layers': 3, 'learning_rate': 0.0001, 
                    'batch_size': 32, 'dropout': 0.2, 'n_heads': 16
                })
                for k in ['moving_avg', 'use_bidirectional', 'patch_len', 'stride']: cfg['model']['hyperparameters'].pop(k, None)
            elif new_a in ['autoformer_hf', 'autoformer']:
                # Autoformer: uses moving_avg, NO patch_len/stride
                cfg['model']['hyperparameters'].update({
                    'd_model': 128, 'n_layers': 3, 'learning_rate': 0.0001, 
                    'batch_size': 32, 'dropout': 0.2, 'n_heads': 16, 'moving_avg': 25
                })
                for k in ['patch_len', 'stride', 'use_bidirectional']: cfg['model']['hyperparameters'].pop(k, None)
            else:
                # RNN/GRU/LSTM: no transformer params
                cfg['model']['hyperparameters'].update({
                    'd_model': 64, 'n_layers': 2, 'learning_rate': 0.001, 
                    'batch_size': 32, 'dropout': 0.2, 'use_bidirectional': True
                })
                for k in ['patch_len', 'stride', 'n_heads', 'ff_dim', 'moving_avg']: cfg['model']['hyperparameters'].pop(k, None)
            save_config_to_file(cfg)
            st.session_state.cfg = cfg
            st.session_state.arch_selector_train = new_a

        if 'tune_arch_selector' not in st.session_state:
            st.session_state.tune_arch_selector = cfg['model'].get('architecture', 'patchtst').lower()

        _valid_tune_archs = ["patchtst", "patchtst_hf", "autoformer_hf", "causal_transformer_hf", "timetracker", "timeperceiver", "autoformer", "gru", "lstm", "rnn"]
        _current_tune_arch = cfg['model'].get('architecture', 'patchtst').lower()
        t_arch = st.selectbox("Architecture to Tune", _valid_tune_archs, 
                              index=_valid_tune_archs.index(_current_tune_arch) if _current_tune_arch in _valid_tune_archs else 0,
                              key="tune_arch_selector",
                              on_change=_update_tuning_architecture)
                              
        n_trials_input = st.number_input("Optuna Trial Count", min_value=1, max_value=5000, value=cfg['tuning'].get('n_trials', 50), step=10, 
                                        help="More trials take longer, but Optuna has a better chance of finding the best converging parameters.")
        
        # --- NEW TACTIC 1: SUBSAMPLING ---
        cfg['tuning']['use_subsampling'] = st.checkbox(
            "Use Tactic 1: Data Subsampling", 
            value=cfg['tuning'].get('use_subsampling', False),
            key="cb_use_subsampling_tuning",
            help="Tactic 1: Use only a small portion of data (e.g. 10-20%) during tuning for 5-10x faster runs."
        )
        if cfg['tuning']['use_subsampling']:
            cfg['tuning']['subsample_ratio'] = st.slider(
                "Data Percentage for Tuning", 
                min_value=0.05, max_value=0.80, 
                value=cfg['tuning'].get('subsample_ratio', 0.20), step=0.05,
                key="sl_subsample_ratio_tuning",
                help="Proportion of initial data used to find the best parameters."
            )
        
        with st.expander("Edit Search Space Hyperparameters", expanded=False):
            st.info("Set search ranges for each hyperparameter. Changes are saved when you run tuning.")
            
            space = cfg['tuning']['search_space']
            col_s1, col_s2, col_s3 = st.columns(3)
            
            with col_s1:
                if t_arch in ["patchtst", "patchtst_hf", "timetracker"]:
                    st.markdown("**1. Patching & Stride**")
                    p_vals = space.get('patch_len', [8, 24, 4])
                    p_min = st.number_input("Patch Min", 2, 64, p_vals[0], 2, key=f"p_min_{t_arch}")
                    p_max = st.number_input("Patch Max", p_min, 128, p_vals[1], 2, key=f"p_max_{t_arch}")
                    p_step = st.number_input("Patch Step", 1, 16, p_vals[2], 1, key=f"p_step_{t_arch}")
                    space['patch_len'] = [p_min, p_max, p_step]
                    
                    s_vals = space.get('stride', [4, 12, 2])
                    s_min = st.number_input("Stride Min", 1, 32, s_vals[0], 1, key=f"s_min_{t_arch}")
                    s_max = st.number_input("Stride Max", s_min, 64, s_vals[1], 1, key=f"s_max_{t_arch}")
                    s_step = st.number_input("Stride Step", 1, 8, s_vals[2], 1, key=f"s_step_{t_arch}")
                    space['stride'] = [s_min, s_max, s_step]
                elif t_arch in ["autoformer", "autoformer_hf"]:
                    st.markdown("**1. Moving Avg (Decomposition)**")
                    st.info("Parameter wajib bernilai ganjil.")
                    m_vals = space.get('moving_avg', [25, 49])
                    m_min_val = m_vals[0] if len(m_vals) > 0 else 25
                    m_max_val = m_vals[1] if len(m_vals) > 1 else 49
                    m_min = st.number_input("Moving Avg Min (Odd)", 3, 99, m_min_val, 2, key=f"m_min_{t_arch}")
                    m_max = st.number_input("Moving Avg Max (Odd)", m_min, 99, m_max_val, 2, key=f"m_max_{t_arch}")
                    space['moving_avg'] = [m_min, m_max]
                elif t_arch == "causal_transformer_hf":
                    st.markdown("**1. Causal Decoder**")
                    st.info("Pure Decoder-Only architecture. Does not use Patching or Decomposition.")
                    for k in ['patch_len', 'stride', 'moving_avg', 'top_k', 'n_shared_experts', 'n_private_experts', 'use_bidirectional']:
                        space.pop(k, None)
                else:
                    st.markdown(f"**1. {t_arch.upper()} Configuration**")
                    st.info("Patching parameters are not available for this architecture.")
                    
                    # Bidirectional tuning space
                    bi_vals = space.get('use_bidirectional', [True, False])
                    if not isinstance(bi_vals, list):
                        bi_vals = [bi_vals]
                    
                    bi_options = ["True", "False"]
                    default_vals = ["True"] if True in bi_vals else []
                    if False in bi_vals:
                        default_vals.append("False")
                        
                    bi_sel = st.multiselect("Bandingkan Mode Bidirectional:", bi_options, default=default_vals, key=f"bi_sel_{t_arch}",
                                            help="Select options for Optuna to try. If both are selected, Optuna will determine which is better.")
                    
                    if not bi_sel:
                        st.warning("Select at least one mode (True or False).")
                        bi_sel = ["True"] # fail-safe
                    
                    space['use_bidirectional'] = [True if v == "True" else False for v in bi_sel]
                    
                    # Clean up unused patch params from space if transitioning from PatchTST
                    for k in ['patch_len', 'stride', 'n_heads', 'ff_dim', 'top_k', 'n_shared_experts', 'n_private_experts', 'moving_avg']:
                        space.pop(k, None)

            with col_s2:
                # Dynamic Search Space Labels
                ss_d_label = "D_Model (Embedding)" if t_arch in ['patchtst', 'patchtst_hf', 'autoformer_hf', 'causal_transformer_hf', 'autoformer', 'timetracker'] else f"Hidden Units ({t_arch.upper()} Capacity)"
                ss_l_label = "Layers (Transformer)" if t_arch in ['patchtst', 'patchtst_hf', 'autoformer_hf', 'causal_transformer_hf', 'autoformer', 'timetracker'] else f"Layers (Stacked {t_arch.upper()})"
                
                st.markdown(f"**2. {t_arch.upper()} Capacity**")
                d_vals = space.get('d_model', [64, 256])
                d_min = st.number_input(f"{ss_d_label} Min", 4, 512, d_vals[0], 4, key=f"d_min_{t_arch}")
                d_max = st.number_input(f"{ss_d_label} Max", d_min, 1024, d_vals[1], 4, key=f"d_max_{t_arch}")
                space['d_model'] = [d_min, d_max]
                
                l_vals = space.get('n_layers', [2, 5])
                l_min = st.number_input(f"{ss_l_label} Min", 1, 12, l_vals[0], 1, key=f"l_min_{t_arch}")
                l_max = st.number_input(f"{ss_l_label} Max", l_min, 20, l_vals[1], 1, key=f"l_max_{t_arch}")
                space['n_layers'] = [l_min, l_max]
                
                if t_arch in ["patchtst", "patchtst_hf", "autoformer_hf", "autoformer", "causal_transformer_hf", "timetracker"]:
                    if t_arch in ["patchtst", "patchtst_hf", "autoformer_hf", "autoformer", "causal_transformer_hf"]:
                        ff_vals = space.get('ff_dim', [128, 512])
                        ff_min = st.number_input("FF_Dim Min", 4, 1024, ff_vals[0], 4, key=f"ff_min_{t_arch}")
                        ff_max = st.number_input("FF_Dim Max", ff_min, 2048, ff_vals[1], 4, key=f"ff_max_{t_arch}")
                        space['ff_dim'] = [ff_min, ff_max]
                    
                    h_vals = space.get('n_heads', [4, 16])
                    h_min = st.number_input("Heads Min", 1, 32, h_vals[0], 1, key=f"h_min_{t_arch}")
                    h_max = st.number_input("Heads Max", h_min, 64, h_vals[1], 1, key=f"h_max_{t_arch}")
                    space['n_heads'] = [h_min, h_max]

                if t_arch == "timetracker":
                    se_vals = space.get('n_shared_experts', [1, 2])
                    se_min = st.number_input("Shared Exp Min", 0, 8, se_vals[0], 1, key=f"se_min_{t_arch}")
                    se_max = st.number_input("Shared Exp Max", se_min, 8, se_vals[1], 1, key=f"se_max_{t_arch}")
                    space['n_shared_experts'] = [se_min, se_max]

                    pe_vals = space.get('n_private_experts', [2, 8])
                    pe_min = st.number_input("Priv Exp Min", 1, 32, pe_vals[0], 1, key=f"pe_min_{t_arch}")
                    pe_max = st.number_input("Priv Exp Max", pe_min, 32, pe_vals[1], 1, key=f"pe_max_{t_arch}")
                    space['n_private_experts'] = [pe_min, pe_max]

                    tk_vals = space.get('top_k', [1, 2])
                    tk_min = st.number_input("Top-K Min", 1, 8, tk_vals[0], 1, key=f"tk_min_{t_arch}")
                    tk_max = st.number_input("Top-K Max", tk_min, 8, tk_vals[1], 1, key=f"tk_max_{t_arch}")
                    space['top_k'] = [tk_min, tk_max]

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

            if st.button("Save Search Space to Master Config", use_container_width=True, key="save_ss_tuning"):
                cfg['tuning']['search_space'] = space
                cfg['tuning']['n_trials'] = n_trials_input
                save_config_to_file(cfg)
                st.success(f"Search space & ({n_trials_input} Trials) saved to config.yaml!")

        # Device Selector for Tuning
        tune_col_dev1, tune_col_loss, tune_col_dev2 = st.columns([1, 1, 2])
        with tune_col_dev1:
            tune_device = st.radio("Device for Tuning", ["CPU", "GPU"], index=0, 
                                   horizontal=True, key="tune_device_top",
                                   help="CPU is recommended to avoid OOM errors on GPUs with limited VRAM.")
        with tune_col_loss:
            _opt_loss = ['mse', 'huber', 'mae']
            tune_loss_fn = st.selectbox("Loss Function", _opt_loss, index=_opt_loss.index(cfg['model']['hyperparameters'].get('loss_fn', 'mse')) if cfg['model']['hyperparameters'].get('loss_fn') in _opt_loss else 0, key="tune_loss_fn_top")
        with tune_col_dev2:
            st.button("Run New Optuna Tuning", type="primary", use_container_width=True, 
                                  disabled=not has_data, key="btn_tune_execute", on_click=_set_action, args=("action_run_tune",))
    else:
        st.warning(" **Optuna Tuning is not active**. Enable it via the 'Enable Optuna Tuning' toggle in the sidebar.")

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
                marker=dict(size=8, color='#818cf8', opacity=0.3)
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
            st.plotly_chart(fig_opt, use_container_width=True)
        
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
                st.plotly_chart(fig_param, use_container_width=True)
        
        # Trial Details Table  
        st.markdown("#### Trial Details")
        df_trials = pd.DataFrame([
            {'Trial': i+1, 'Value': t['value'], **t['params']}
            for i, t in enumerate(trials)
        ]).sort_values('Value')
        st.dataframe(df_trials, use_container_width=True)
        
    else:
        st.info("No tuning results saved yet.")

    # --- EXECUTION LOGIC FOR TUNING ---
    is_run_tune = st.session_state.get('action_run_tune', False)
    if is_run_tune:
        st.session_state.action_run_tune = False
        st.markdown("### Live Tuning Monitor")
        
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
                        tune_chart_placeholder.plotly_chart(fig, use_container_width=True)

                        # Live Table (Show latest on top)
                        df_live = pd.DataFrame(self.trial_records).sort_values('Trial', ascending=False)
                        table_placeholder.dataframe(df_live, height=300)

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
                f"[{datetime.now():%H:%M:%S}] Tuning complete. "
                f"Best Val Loss: {study.best_value:.6f} | "
                f"Params: {best}"
            )
            st.success(f"Tuning complete! Best Val Loss: {study.best_value:.6f}")
            with st.expander("Best Parameters"):
                st.json(best)
            with st.expander("Full Output"):
                st.code(stdout_capture.getvalue(), language="text")
        except Exception as e:
            import traceback
            st.error(f"Error during Tuning execution: {str(e)}")
            st.code(traceback.format_exc(), language="python")


# --- TAB: EVALUATION RESULTS ---
with tab_eval:
    st.markdown("---")
    st.markdown("#### Deep Evaluation Analysis")
    st.markdown("### Evaluation & Results")
    
    # --- MODEL SELECTOR FOR EVALUATION ---
    with st.expander("Select Model for Evaluation", expanded=not st.session_state.eval_results):
        if os.path.exists(model_dir):
            all_models = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5')) or os.path.isdir(os.path.join(model_dir, f))]
            if all_models:
                current_sel = st.session_state.get('selected_model', all_models[0])
                model_to_eval = st.selectbox("Select Model for Evaluation (Tab):", all_models, 
                                           index=all_models.index(current_sel) if current_sel in all_models else 0,
                                           key="sel_eval_tab",
                                           format_func=lambda x: label_format_with_time(x, model_dir))
                st.session_state.selected_model = model_to_eval
                
                # Model Info Preview
                model_info_path = os.path.join(model_dir, model_to_eval, "meta.json")
                m_meta = {}
                if os.path.exists(model_info_path):
                    try:
                        with open(model_info_path, 'r', encoding='utf-8') as f:
                            m_meta = json.load(f)
                    except Exception:
                        pass # Handle likely OneDrive online-only or locked file
                        
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
                    <div style="background-color: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 5px; font-size: 0.9em; margin-bottom: 5px;">
                        <b>Arch:</b> {m_meta.get('architecture', 'N/A').upper()} | 
                        <b>Data Source:</b> <span style="color: #94a3b8;">{os.path.basename(m_meta.get('data_source', 'N/A'))}</span> <br>
                        <b>Features ({m_meta.get('n_features', '?')}):</b> <span style="color: #cbd5e1; font-size: 0.85em;">{feat_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 'hyperparameters' in m_meta:
                        st.markdown("---")
                        st.markdown("**View Hyperparameters**")
                        if True:

                            hp = m_meta['hyperparameters']
                            cols = st.columns(3)
                            for i, (k, v) in enumerate(hp.items()):
                                with cols[i % 3]:
                                    st.markdown(f"**{k}:** `{v}`")
                    
                    st.markdown("---")
                    st.markdown("** Evaluation Data Source:**")
                    use_active_data = st.checkbox(
                        "Test this model using **Active Data** from the Preprocessing tab",
                        value=False,
                        help="Default (Unchecked): Model is evaluated using its original training data. Check this to test the model on a new/different dataset being prepared."
                    )
                
                st.button("Run Evaluation for Selected Model", type="primary", use_container_width=True, key="btn_eval_tab", on_click=_set_action, args=("action_run_eval",))
                if st.session_state.get("action_run_eval"):
                    st.session_state.action_run_eval = False
                    with st.spinner(f"Evaluating model: {model_to_eval}..."):
                        try:
                            import gc
                            tf.keras.backend.clear_session()
                            gc.collect()
                            
                            from src.model_factory import get_custom_objects, compile_model
                            from src.predictor import evaluate_model
                            
                            model_path = os.path.join(model_dir, model_to_eval)
                            model_root = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
                            
                            is_hf = False
                            if os.path.exists(model_path) and os.path.isdir(model_path):
                                if os.path.exists(os.path.join(model_path, 'pytorch_model.bin')) or \
                                   os.path.exists(os.path.join(model_path, 'config.json')) or \
                                   os.path.exists(os.path.join(model_path, 'model_hf')):
                                    is_hf = True
                                    if os.path.exists(os.path.join(model_path, 'model_hf')):
                                        model_path = os.path.join(model_path, 'model_hf')
                                else:
                                    # It's a directory but not HF. Find the actual Keras file inside.
                                    found_k = False
                                    for ext in ['.keras', '.h5']:
                                        cand = os.path.join(model_path, f'model{ext}')
                                        if os.path.exists(cand):
                                            model_path = cand
                                            found_k = True
                                            break
                                    if not found_k:
                                        # Fallback to first available keras/h5 file
                                        for f in os.listdir(model_path):
                                            if f.endswith(('.keras', '.h5')):
                                                model_path = os.path.join(model_path, f)
                                                break

                            custom_objs = get_custom_objects()
                            with tf.keras.utils.custom_object_scope(custom_objs):
                                if is_hf:
                                    from src.model_hf import load_hf_wrapper
                                    model = load_hf_wrapper(model_path)
                                else:
                                    import zipfile
                                    if model_path.endswith('.keras') and not zipfile.is_zipfile(model_path):
                                        h5_path = model_path.replace('.keras', '.h5')
                                        if not os.path.exists(h5_path):
                                            import shutil
                                            shutil.copy(model_path, h5_path)
                                        model_path = h5_path
                                    try:
                                        try:
                                            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
                                        except TypeError:
                                            model = tf.keras.models.load_model(model_path, compile=False)
                                    except OSError as ke3_err:
                                        # Keras 3 ZIP format on TF 2.x
                                        if model_path.endswith('.keras') and zipfile.is_zipfile(model_path):
                                            extract_dir = os.path.join(model_root, '_extracted_k3')
                                            os.makedirs(extract_dir, exist_ok=True)
                                            with zipfile.ZipFile(model_path, 'r') as zf:
                                                zf.extractall(extract_dir)
                                            weights_h5 = os.path.join(extract_dir, 'model.weights.h5')
                                            
                                            m_meta_r = {}
                                            mp = os.path.join(model_root, 'meta.json')
                                            if os.path.exists(mp):
                                                try:
                                                    with open(mp, 'r', encoding='utf-8') as ff: m_meta_r = json.load(ff)
                                                except: pass
                                            
                                            arch = m_meta_r.get('architecture', cfg['model']['architecture'])
                                            hp_r = m_meta_r.get('hyperparameters', cfg['model']['hyperparameters'])
                                            lb = m_meta_r.get('lookback', cfg['model']['hyperparameters']['lookback'])
                                            nf = m_meta_r.get('n_features', 0)
                                            hz = m_meta_r.get('horizon', cfg['forecasting']['horizon'])
                                            
                                            if nf == 0:
                                                prep_p = os.path.join(model_root, 'prep_summary.json')
                                                if os.path.exists(prep_p):
                                                    try:
                                                        with open(prep_p, 'r') as f:
                                                            p_m = json.load(f)
                                                            if n_f := p_m.get('n_features'): nf = n_f
                                                            if l_b := p_m.get('lookback'): lb = l_b
                                                            if h_z := p_m.get('horizon'): hz = h_z
                                                    except: pass
                                            
                                            from src.model_factory import build_model as bm, manual_load_k3_weights
                                            model = bm(arch, lb, nf, hz, hp_r)
                                            manual_load_k3_weights(model, weights_h5)
                                            st.toast(f"Model loaded via Keras 3 ZIP recovery ({arch})")
                                        else:
                                            raise ke3_err
                                    from src.model_factory import fix_lambda_tf_refs
                                    fix_lambda_tf_refs(model)
                                    compile_model(model, cfg['model']['hyperparameters']['learning_rate'])
                            
                            scaler_dir = model_root if os.path.isdir(model_root) else None
                            
                            temp_cfg = copy.deepcopy(cfg)
                            # Decide whether to use active or bundled data
                            if use_active_data:
                                eval_data = st.session_state.get('prep_metadata', None)
                                if not eval_data:
                                    st.warning("Active Data is empty. Using original model data...")
                                    eval_data = None
                                    use_active_data = False # fall down to next block
                                    
                            if not use_active_data:
                                eval_data = None # Strategy 1: Original Model Data called inside predictor.py
                                meta_path = os.path.join(model_root, "meta.json")
                                if os.path.exists(meta_path):
                                    try:
                                        with open(meta_path, 'r') as f:
                                            m_meta = json.load(f)
                                        orig_ds = m_meta.get('data_source', '').replace('\\', '/')
                                        if orig_ds and os.path.exists(orig_ds):
                                            temp_cfg['paths']['processed_dir'] = orig_ds
                                    except: pass
                                
                            results = evaluate_model(model, temp_cfg, data=eval_data, scaler_dir=scaler_dir)
                            results['model_id'] = model_to_eval
                            
                            st.session_state.eval_results = results
                            save_eval_results_to_disk(results)
                            st.success(f"Evaluation complete for {model_to_eval}!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            err_msg = str(e)
                            st.error(f"Gagal mengevaluasi model: {err_msg}")
                            
                            # SMART FIX: If it's a feature mismatch, offer to switch data version
                            if "Mismatch Fitur" in err_msg or "Inkompatibilitas Fitur" in err_msg:
                                model_info_path = os.path.join(model_dir, model_to_eval, "meta.json")
                                m_meta = {}
                                if os.path.exists(model_info_path):
                                    try:
                                        with open(model_info_path, 'r', encoding='utf-8') as f:
                                            m_meta = json.load(f)
                                    except: pass
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
                                        st.info(f"This model requires features from folder: `{os.path.basename(found_path)}`")
                                        if st.button("Switch to Original Model Data & Re-evaluate"):
                                            st.session_state.cfg['paths']['processed_dir'] = found_path
                                            save_config_to_file(st.session_state.cfg)
                                            st.success("Data path updated. Re-running evaluation...")
                                            time.sleep(1)
                                            st.rerun()
                                    else:
                                        st.warning("Original data for this model not found in `data/processed`. Create data with matching feature count in the tab Preprocessing.")

                            import traceback; st.code(traceback.format_exc())
            else:
                st.warning("No saved models in models/ folder.")
        else:
            st.error("Folder models/ not found.")

    if st.session_state.eval_results:
        results = st.session_state.eval_results
        
        # Check for model consistency
        disp_model = results.get('model_id', 'Unknown')
        curr_model = st.session_state.get('selected_model', 'None')
        
        if disp_model != curr_model:
            st.warning(f"Results below belong to model **{disp_model}**, while the currently selected model is **{curr_model}**. Click the evaluate button above tok memperbarui.")
        else:
            st.info(f"Showing evaluation results for active model: **{disp_model}**")

        m_train = results['metrics_train']
        m_test = results['metrics_test']
        
        # ====== Fetch Training Time from meta.json if available ======
        train_time_str = "N/A"
        model_info_path = os.path.join(model_dir, disp_model, "meta.json")
        if os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    _meta = json.load(f)
                    seconds = _meta.get('training_time_seconds', 0)
                    if seconds > 0:
                        train_time_str = float(seconds)
            except: pass
                    
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
            ("Inference Time", results.get('inference_time_ms', m_test.get('inference_time_ms', 0)), "ms"),
        ]
        for col, (name, val, unit) in zip([col1, col2, col3, col4, col5, col6], metrics_display):
            with col:
                if isinstance(val, (int, float)):
                    val_str = f"{val:.4f}" if name not in ["Inference Time"] else f"{val:.1f}"
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
            metrics_rows.insert(3, ['R2 (Productive)', 
                                    f"{m_train_prod.get('r2', 0):.4f}", 
                                    f"{m_test_prod.get('r2', 0):.4f}"])
        
        df_metrics = pd.DataFrame(metrics_rows, columns=['Metrik', 'Train', 'Test'])
        st.dataframe(df_metrics, use_container_width=True)
        
        # ====== Prepare common data ======
        if 'pv_test_actual' in results:
            actual_flat = results['pv_test_actual'].flatten()
            pred_flat = results['pv_test_pred'].flatten()
            ghi_flat = results['ghi_test'].flatten()
            mask_productive = ghi_flat > 50
        else:
            st.warning("Visualization data (arrays) not loaded. Run 'Run Evaluation' again to see detailed charts.")
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
            st.plotly_chart(fig_scatter, use_container_width=True)
        
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
            st.plotly_chart(fig_hist, use_container_width=True)
        
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
                "Select sample range (sequence index)",
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
                line=dict(color='#ffffff', width=1.5)
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
            st.plotly_chart(fig_ts, use_container_width=True)
        except Exception as e:
            st.caption(f"Time series plot not available: {e}")
        
        # ====== ROW 5: Per-Step Forecast Diagnostics ======
        st.markdown("#### Diagnostics: R2 per Forecast Step")
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
                name='R2 per Step',
                hovertemplate='Step t+%{x}: R2=%{y:.4f}<extra></extra>'
            ))
            # Add reference lines
            fig_r2_step.add_hline(y=0.8, line_dash='dot', line_color='#ffffff', 
                                  annotation_text='Target R2=0.80', annotation_position='top left')
            fig_r2_step.add_hline(y=0, line_dash='dash', line_color='#ef4444', line_width=1)
            fig_r2_step.update_layout(
                title="R2 Score per Forecast Step (ALL Hours)",
                xaxis_title="Forecast Step (t+n hours)", yaxis_title="R2 Score",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=400,
                yaxis=dict(range=[min(min(r2_vals) - 0.05, -0.1), 1.0]),
            )
            st.plotly_chart(fig_r2_step, use_container_width=True)
        else:
            st.info("Per-step R2 diagnostics not available. Re-run evaluation.")
        
        # ====== ROW 6: Per-Hour-of-Day Error Analysis ======
        st.markdown("#### Diagnostics: Error per Hour of Day")
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
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with col2:
                # Color-code R2 bars: green if good, red if bad
                colors = ['#ffffff' if r > 0.7 else '#fbbf24' if r > 0.3 else '#ef4444' for r in h_r2]
                fig_r2h = go.Figure()
                fig_r2h.add_trace(go.Bar(
                    x=hours, y=h_r2,
                    marker_color=colors, opacity=0.85,
                    name='R2 per Hour',
                    hovertemplate='Hour %{x}: R2=%{y:.4f}<extra></extra>'
                ))
                fig_r2h.add_hline(y=0.8, line_dash='dot', line_color='#ffffff', 
                                  annotation_text='Target', annotation_position='top left')
                fig_r2h.update_layout(
                    title="R2 per Hour of Day",
                    xaxis_title="Hour of Day", yaxis_title="R2",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    xaxis=dict(dtick=1),
                )
                st.plotly_chart(fig_r2h, use_container_width=True)
            
            # Hourly metrics table
            with st.expander("Full Hourly Table", expanded=False):
                tbl_data = []
                for h in range(24):
                    m = hourly_metrics.get(h, {})
                    tbl_data.append({
                        'Jam': f"{h:02d}:00",
                        'MAE (kW)': f"{m.get('mae', 0):.4f}",
                        'RMSE (kW)': f"{m.get('rmse', 0):.4f}",
                        'R2': f"{m.get('r2', 0):.4f}",
                        'N Samples': m.get('count', 0),
                    })
                st.dataframe(pd.DataFrame(tbl_data), use_container_width=True)
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
                st.plotly_chart(fig_hour, use_container_width=True)
            except Exception as e:
                st.caption(f"Hourly error chart not available: {e}")
        
        # ====== ROW 6: Actual vs Predicted Distribution ======
        st.markdown("#### Power Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=actual_flat[mask_productive], nbinsx=80, name='Actual',
                marker_color='#ffffff', opacity=0.6
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
            st.plotly_chart(fig_dist, use_container_width=True)
        
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
            st.plotly_chart(fig_qq, use_container_width=True)
        
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
                use_container_width=True,            )
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
                use_container_width=True,            )
    else:
        st.info("No evaluation results yet. Run Evaluate or Full Pipeline first.")


# --- TAB: TARGET TESTING ---
with tab_transfer:
    st.markdown("### Target Domain Testing")
    st.markdown("Test trained models on data from different locations.")
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
    if not has_model:
        st.warning("No trained models found. Run Training first.")
    else:
        model_list = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5', '.json')) or (os.path.isdir(os.path.join(model_dir, f)) and not f.startswith('.'))]
        selected_model = st.selectbox("1. Select Model to Test:", model_list, format_func=lambda x: label_format_with_time(x, model_dir), key="target_model_sel")
        
        # Model Info Preview in Target Testing
        m_info_path = os.path.join(model_dir, selected_model, "meta.json")
        m_meta = {}
        if os.path.exists(m_info_path):
            try:
                with open(m_info_path, 'r', encoding='utf-8') as f:
                    m_meta = json.load(f)
                
                st.markdown(f"""
                <div style="background-color: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 5px; font-size: 0.9em; margin-bottom: 5px;">
                    <b>Model Arch:</b> {m_meta.get('architecture', 'N/A').upper()} | 
                    <b>Original Data:</b> <span style="color: #94a3b8;">{os.path.basename(m_meta.get('data_source', 'N/A'))}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if 'hyperparameters' in m_meta:
                    with st.expander("View Model Hyperparameters"):
                        hp = m_meta['hyperparameters']
                        cols = st.columns(3)
                        for i, (k, v) in enumerate(hp.items()):
                            with cols[i % 3]:
                                st.markdown(f"**{k}:** `{v}`")
            except: pass

        st.markdown("---")
        st.markdown("---")
        st.markdown("#### 2. Select Target Data (Preprocessed)")
        st.info("Select folder from **1. Preprocessing** results (e.g. `v10_...`) containing target location data.")
        
        processed_base_dir = cfg['paths']['processed_dir']
        # Dapatkan parent directory yang berisi folder-folder preprocessed ("data/processed")
        processed_root = os.path.dirname(processed_base_dir) if os.path.basename(processed_base_dir).startswith(('v', 'version')) else processed_base_dir
        
        if not os.path.exists(processed_root):
            st.warning("No preprocessed data available.")
        else:
            processed_folders = [f for f in os.listdir(processed_root) if os.path.isdir(os.path.join(processed_root, f)) and not f.startswith('.')]
            
            if not processed_folders:
                st.info("No preprocessed folders found. Go to **Data Prep** tab first.")
            else:
                selected_target = st.selectbox("Select Processed Data Folder:", sorted(processed_folders, reverse=True), key="target_folder_sel")
                
                # TSCV Options
                c_t1, c_t2 = st.columns([1, 1])
                with c_t1:
                    use_tscv_eval = st.checkbox("Use Time-Series CV (Evaluation)", value=False, 
                                               help="Split target data into chronological segments to evaluate model stability.")
                with c_t2:
                    n_folds_eval = st.slider("Number of Folds:", 2, 12, 5, disabled=not use_tscv_eval)

                # Container for results
                if 'target_eval' not in st.session_state:
                    st.session_state.target_eval = None

                st.button("Run Target Testing", type="primary", use_container_width=True, key="run_target_test_btn", on_click=_set_action, args=("action_run_target_test",))
                if st.session_state.get("action_run_target_test"):
                    st.session_state.action_run_target_test = False
                    with st.spinner("Running inference on target data..."):
                        try:
                            import io, contextlib
                            from datetime import datetime
                            stdout_capture = io.StringIO()
                            
                            model_path = os.path.join(model_dir, selected_model)
                            target_data_path = os.path.join(processed_root, selected_target)
                            
                            from src.predictor import test_on_preprocessed_target, calculate_full_metrics
                            
                            with contextlib.redirect_stdout(stdout_capture):
                                result = test_on_preprocessed_target(model_path, target_data_path, cfg)
                            
                            # Standard results
                            eval_data = {
                                'metrics': result['metrics'],
                                'inference_time': result['inference_time'],
                                'timestamps': result['timestamps'],
                                'actual_full': result['actual_full'],
                                'pred_full': result['pred_full'],
                                'horizon': result['horizon'],
                                'output': stdout_capture.getvalue(),
                                'target_folder': selected_target,
                                'model_id': selected_model,
                                'timestamp': datetime.now().strftime("%H:%M:%S"),
                                'use_tscv': use_tscv_eval
                            }

                            # TSCV Logic: Segment-based evaluation
                            if use_tscv_eval:
                                n_total = len(result['actual_full'])
                                fold_size = n_total // n_folds_eval
                                fold_results = []
                                
                                capacity = cfg['pv_system']['nameplate_capacity_kw']
                                
                                for i in range(n_folds_eval):
                                    start_idx = i * fold_size
                                    # Last fold takes all remaining
                                    end_idx = (i + 1) * fold_size if i < n_folds_eval - 1 else n_total
                                    
                                    act_fold = result['actual_full'][start_idx:end_idx]
                                    pred_fold = result['pred_full'][start_idx:end_idx]
                                    ts_fold = result['timestamps'][start_idx:end_idx]
                                    
                                    if len(act_fold) > 0:
                                        m_fold = calculate_full_metrics(act_fold, pred_fold, None, f"Fold {i+1}", capacity)
                                        fold_results.append({
                                            'fold': i + 1,
                                            'start': ts_fold[0].strftime("%Y-%m-%d") if hasattr(ts_fold[0], 'strftime') else str(ts_fold[0]),
                                            'end': ts_fold[-1].strftime("%Y-%m-%d") if hasattr(ts_fold[-1], 'strftime') else str(ts_fold[-1]),
                                            'mae': m_fold['mae'],
                                            'rmse': m_fold['rmse'],
                                            'r2': m_fold['r2'],
                                            'nmae': m_fold['norm_mae'] * 100,
                                            'nrmse': m_fold['norm_rmse'] * 100
                                        })
                                eval_data['fold_results'] = fold_results
                            
                            st.session_state.target_eval = eval_data
                            st.rerun()
                        except Exception as e:
                            st.error(f"Inference Error: {e}")
                            print(f"ERROR during target testing: {e}")

                # ALWAYS RENDER RESULTS IF THEY EXIST IN SESSION STATE
                if st.session_state.target_eval:
                    eval_data = st.session_state.target_eval
                    
                    st.success(f"Evaluation Results: **{eval_data['model_id']}** on data **{eval_data['target_folder']}**")
                    
                    # --- SHOW HYPERPARAMETERS ---
                    m_eval_info_path = os.path.join(model_dir, eval_data.get('model_id', ''), "meta.json")
                    if os.path.exists(m_eval_info_path):
                        try:
                            with open(m_eval_info_path, 'r', encoding='utf-8') as f:
                                m_eval_meta = json.load(f)
                            if 'hyperparameters' in m_eval_meta:
                                with st.expander("View Model Hyperparameters & Config", expanded=False):
                                    hp = m_eval_meta['hyperparameters']
                                    st.markdown(f"**Arch:** `{m_eval_meta.get('architecture', 'N/A').upper()}` | **Train Data:** `{os.path.basename(m_eval_meta.get('data_source', 'N/A'))}`")
                                    if m_eval_meta.get('fine_tuned'):
                                        st.success(" **Fine-Tuned / Target-Transformed Model**")
                                        ft = m_eval_meta.get('ft_config', {})
                                        if ft:
                                            st.markdown("##### Fine-Tuning Configuration:")
                                            f_cols = st.columns(4)
                                            f_cols[0].markdown(f"**FT Epochs:** `{ft.get('epochs', 'N/A')}`")
                                            f_cols[1].markdown(f"**FT LR:** `{ft.get('learning_rate', 'N/A')}`")
                                            f_cols[2].markdown(f"**Trainable N-Layers:** `{ft.get('trainable_last_n', 'All')}`" if ft.get('freeze_backbone') else "**Freezing:** `Disabled (All Layers)`")
                                            f_cols[3].markdown(f"**Reset Weights:** `{'Yes' if ft.get('reset_weights') else 'No'}`")
                                            
                                            bz = m_eval_meta.get('base_model', 'N/A')
                                            if ft.get('reset_weights'):
                                                st.info(f"This model mimics architecture `{bz}` but re-initialized from scratch (pre-trained weights discarded) for pure training on source Target Data.")
                                            else:
                                                st.info(f"Using Transfer Learning weights from base model: `{bz}`")
                                                
                                    st.markdown("---")
                                    st.markdown("##### Parameter Original Model Base:")
                                    cols = st.columns(3)
                                    for i, (k, v) in enumerate(hp.items()):
                                        with cols[i % 3]:
                                            st.markdown(f"**{k}:** `{v}`")
                        except Exception:
                            pass

                    m = eval_data['metrics']
                    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                    col1.metric("MAE", f"{m.get('mae', 0):.4f}")
                    col2.metric("RMSE", f"{m.get('rmse', 0):.4f}")
                    col3.metric("nMAE", f"{m.get('norm_mae', 0):.4f}")
                    col4.metric("nRMSE", f"{m.get('norm_rmse', 0):.4f}")
                    col5.metric("R2", f"{m.get('r2', 0):.4f}")
                    col6.metric("MAPE", f"{m.get('mape', 0):.2f}%")
                    col7.metric("Inf. Time", f"{eval_data['inference_time']:.3f}s")
                    
                    # ==========================================================
                    # 4. VISUALIZATION & ANALYSIS (COMPLETE REWRITE)
                    # ==========================================================
                    st.markdown("---")
                    st.markdown("#### Visualisasi & Analysis Hasil Target Testing")
                    
                    import plotly.graph_objects as go
                    import plotly.express as px

                    # 1. Prediction Step Selection
                    horizon = eval_data.get('horizon', 24)
                    st.markdown("##### Prediction Controls")
                    selected_step = st.slider("Select Forecast Step (T+n):", 1, horizon, 1, key="res_step_slider_vfinal")
                    step_idx = selected_step - 1
                    st.caption(f"Showing visualization for hour T+{selected_step} after input on all testing data (no time filter).")
                    
                    # 2. Build Base Data (Sorted & Clean)
                    df_res = pd.DataFrame({
                        'AnchorTime': pd.to_datetime(eval_data['timestamps']),
                        'Actual_kW': eval_data['actual_full'][:, step_idx],
                        'Predicted_kW': eval_data['pred_full'][:, step_idx],
                    })
                    # Target Time is when the power actually happened (Anchor + Step)
                    df_res['TargetTime'] = df_res['AnchorTime'] + pd.Timedelta(hours=selected_step)
                    df_res['Date'] = df_res['TargetTime'].dt.date
                    df_res['Error_kW'] = df_res['Actual_kW'] - df_res['Predicted_kW']
                    df_res = df_res.sort_values('TargetTime')

                    # 3. Stats (Immediate Feedback)
                    if not df_res.empty:
                        f_mae = np.mean(np.abs(df_res['Error_kW']))
                        f_rmse = np.sqrt(np.mean(df_res['Error_kW']**2))
                        f_ss_res = np.sum(df_res['Error_kW']**2)
                        f_ss_tot = np.sum((df_res['Actual_kW'] - np.mean(df_res['Actual_kW']))**2)
                        f_r2 = 1 - (f_ss_res / f_ss_tot) if f_ss_tot > 0 else 0
                        
                        st.info(f" **Statistik Keseluruhan (T+{selected_step})**: MAE=`{f_mae:.4f}` | RMSE=`{f_rmse:.4f}` | R2=`{f_r2:.4f}` | Total Data: `{len(df_res)}` baris")
                    else:
                        st.warning(f"Data is empty.")

                    # 5. Result Tabs
                    tab_labels = ["Scatter Plot", "Line Chart Harian"]
                    if eval_data.get('use_tscv'):
                        tab_labels.append("Stability Analysis (TSCV)")
                    
                    tabs = st.tabs(tab_labels)
                    
                    with tabs[0]: # Scatter Plot
                        fig_s = px.scatter(
                            df_res, x='Actual_kW', y='Predicted_kW', 
                            title=f"Correlation Plot (Step T+{selected_step})",
                            template="plotly_dark", opacity=0.5, color_discrete_sequence=['#FFC107'],
                            labels={'Actual_kW': 'Actual (kW)', 'Predicted_kW': 'Predicted (kW)'}
                        )
                        if not df_res.empty:
                            limit = max(df_res['Actual_kW'].max(), df_res['Predicted_kW'].max())
                            fig_s.add_shape(type='line', x0=0, y0=0, x1=limit, y1=limit, line=dict(color='white', dash='dash'), name="Ideal (y=x)")
                        fig_s.update_layout(height=450)
                        st.plotly_chart(fig_s, use_container_width=True)

                    with tabs[1]: # Line Chart Harian
                        st.markdown("##### Analysis Fluktuasi Energi (Rising/Falling Graph)")
                        if not df_res.empty:
                            all_dates = sorted(df_res['Date'].unique())
                            min_d, max_d = all_dates[0], all_dates[-1]
                            
                            st.info(f"You can select a date range. The chart will show energy fluctuation trends (especially during daytime).")
                            
                            c_date1, c_date2 = st.columns([1, 1])
                            with c_date1:
                                start_d = st.date_input("Start Date:", min_d, min_value=min_d, max_value=max_d, key=f"ds_start_{eval_data.get('timestamp')}")
                            with c_date2:
                                end_d = st.date_input("Sampai Tanggal:", max_d, min_value=min_d, max_value=max_d, key=f"ds_end_{eval_data.get('timestamp')}")
                            
                            # Filter data by range
                            df_plot = df_res[(df_res['Date'] >= start_d) & (df_res['Date'] <= end_d)].copy()
                            
                            if not df_plot.empty:
                                fig_line = go.Figure()
                                
                                # ACTUAL - Blueish line, thick, with area fill for "Rising/Falling" look
                                fig_line.add_trace(go.Scatter(
                                    x=df_plot['TargetTime'], y=df_plot['Actual_kW'], 
                                    mode='lines', name='Actual Energy', 
                                    line=dict(color='#3b82f6', width=3, shape='spline'), # Spline for smooth curve
                                    fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)', # Subtle fill
                                    hovertemplate="Waktu: %{x}<br>Actual: %{y:.2f} kW<extra></extra>"
                                ))
                                
                                # PREDICTED - Orange/Amber line, dashed
                                fig_line.add_trace(go.Scatter(
                                    x=df_plot['TargetTime'], y=df_plot['Predicted_kW'], 
                                    mode='lines', name='Predicted (AI)', 
                                    line=dict(color='#f59e0b', width=3, dash='dash', shape='spline'),
                                    hovertemplate="Waktu: %{x}<br>Predicted: %{y:.2f} kW<extra></extra>"
                                ))
                                
                                # Layout tuning to match reference aesthetic
                                title_text = f"Kurva Energi: {start_d} s/d {end_d} (T+{selected_step})" if start_d != end_d else f"Kurva Energi: {start_d} (T+{selected_step})"
                                
                                fig_line.update_layout(
                                    title=dict(text=title_text, font=dict(size=20, family="Manrope", color="#f8fafc")),
                                    xaxis=dict(
                                        title="Timeline", 
                                        type='date',
                                        gridcolor='rgba(255,255,255,0.05)',
                                        rangeslider=dict(visible=True), # Range slider for easy navigation
                                        rangeselector=dict(
                                            buttons=list([
                                                dict(count=1, label="1d", step="day", stepmode="backward"),
                                                dict(count=3, label="3d", step="day", stepmode="backward"),
                                                dict(step="all")
                                            ]),
                                            bgcolor="rgba(30, 41, 59, 0.8)"
                                        )
                                    ),
                                    yaxis=dict(
                                        title="Power (kW)",
                                        gridcolor='rgba(255,255,255,0.1)',
                                        zeroline=False
                                    ),
                                    template="plotly_dark",
                                    hovermode="x unified",
                                    height=600,
                                    margin=dict(l=50, r=50, t=80, b=50),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                st.plotly_chart(fig_line, use_container_width=True)
                                
                                # Stats for the selected range
                                r_mae = np.mean(np.abs(df_plot['Error_kW']))
                                r_rmse = np.sqrt(np.mean(df_plot['Error_kW']**2))
                                st.info(f" **Selected Period Statistics**: MAE=`{r_mae:.4f}` | RMSE=`{r_rmse:.4f}` | Total Data Points: `{len(df_plot)}` jam")
                            else:
                                st.warning("No data available for the selected date range.")
                        else:
                            st.warning("Testing results not available. Run 'Run Target Testing' first.")

                    if eval_data.get('use_tscv') and len(tabs) > 2:
                        with tabs[2]:
                            st.markdown("##### Stabilitas Performa Geografis/Kronologis")
                            df_f = pd.DataFrame(eval_data['fold_results'])
                            st.dataframe(df_f.style.format({
                                'mae': '{:.4f}', 'rmse': '{:.4f}', 'r2': '{:.4f}', 'nmae': '{:.2f}%'
                            }).background_gradient(cmap='YlGnBu_r', subset=['mae', 'rmse']), use_container_width=True)
                            
                            fig_t = go.Figure()
                            fig_t.add_trace(go.Bar(x=df_f['fold'], y=df_f['mae'], name="MAE (Fold)", marker_color='#2196F3'))
                            fig_t.add_trace(go.Scatter(x=df_f['fold'], y=df_f['r2'], name="R2 Score", yaxis="y2", line=dict(color='#FFEB3B', width=3)))
                            fig_t.update_layout(
                                title="TSCV Fold Results", template="plotly_dark", height=400,
                                yaxis=dict(title="MAE (kW)"),
                                yaxis2=dict(title="R2 Score", overlaying='y', side='right', range=[0, 1]),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig_t, use_container_width=True)
                            st.info(f" **Analysis**: Highest R2 at Fold {df_f.loc[df_f['r2'].idxmax(), 'fold']} ({df_f['r2'].max():.4f}).")

                    # --- NEW: EXPORT ---
                    with st.expander("Export & Data Details"):
                        st.info(f"Showing data for prediction step **T+{selected_step}**")
                        st.dataframe(df_res.head(100), use_container_width=True)
                        
                        # Export button
                        try:
                            import io
                            excel_buffer = io.BytesIO()
                            df_res.to_excel(excel_buffer, index=False)
                            st.download_button(
                                label=f"Download Hasil Prediksi T+{selected_step} (.xlsx)",
                                data=excel_buffer.getvalue(),
                                file_name=f"prediksi_T{selected_step}_{eval_data['model_id']}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        except Exception as e_exp:
                            st.warning(f"Gagal menyiapkan export: {e_exp}")
                    
                    with st.expander("Log Konsol Detail"):
                        st.code(eval_data['output'], language="text")

                    # --- ZEROSHOT SAVING SECTION ---
                    st.markdown("---")
                    st.markdown("#### Save Model with New Data (Zero-Shot)")
                    st.markdown(f"Export model `{eval_data['model_id']}` using this data configuration set (Scalers & Features) from **{eval_data['target_folder']}** without Fine-Tuning? The model will be saved as a new entity.")

                    col_zs1, col_zs2 = st.columns([3, 1])
                    with col_zs1:
                        zs_name = st.text_input("New Model Name (Zero-Shot)", placeholder=f"e.g. {eval_data['model_id']}_on_{eval_data['target_folder']}", key="zs_model_name_input")
                    with col_zs2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Save Zero-Shot Model", type="primary", use_container_width=True):
                            try:
                                from datetime import datetime
                                import shutil
                                
                                m_src_dir = os.path.join(model_dir, eval_data['model_id'])
                                
                                if not zs_name.strip():
                                    zs_name = f"{eval_data['model_id']}_on_{eval_data['target_folder']}"
                                else:
                                    import re
                                    zs_name = re.sub(r'[^\w\-_\.]', '_', zs_name.strip())
                                
                                m_dst_dir = os.path.join(model_dir, zs_name)
                                
                                if os.path.exists(m_dst_dir):
                                    st.error("Model name already exists. Use a different name.")
                                else:
                                    # 1. Copy entire model source folder first (weights, config, etc)
                                    shutil.copytree(m_src_dir, m_dst_dir)
                                    
                                    # 2. Overwrite Scalers & Prep Summary from the evaluated Target Data!
                                    eval_proc_dir = os.path.join(processed_root, eval_data['target_folder'])
                                    for f in ['X_scaler.pkl', 'y_scaler.pkl', 'prep_summary.json', 'selected_features.json']:
                                        f_src = os.path.join(eval_proc_dir, f)
                                        if os.path.exists(f_src):
                                            shutil.copy(f_src, os.path.join(m_dst_dir, f))
                                            
                                    # 3. Modify meta.json to point to the new data naturally
                                    meta_dst_path = os.path.join(m_dst_dir, 'meta.json')
                                    if os.path.exists(meta_dst_path):
                                        with open(meta_dst_path, 'r') as mf:
                                            zs_meta = json.load(mf)
                                        
                                        zs_meta['model_id'] = zs_name
                                        zs_meta['base_model'] = eval_data['model_id']
                                        zs_meta['data_source'] = eval_proc_dir
                                        zs_meta['zero_shot_transfer'] = True
                                        zs_meta['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M")
                                        
                                        with open(meta_dst_path, 'w') as mf:
                                            json.dump(zs_meta, mf, indent=2)
                                            
                                    st.success(f"Success! Model **{zs_name}** saved and bound to evaluation data **{eval_data['target_folder']}**.")
                                    st.balloons()
                            except Exception as e_zs:
                                st.error(f"Gagal menyimpan: {e_zs}")

                    # --- FINE-TUNING SECTION ---
                    st.markdown("---")
                    st.markdown("#### Fine-Tuning (Transfer Learning)")
                    st.markdown(f"""
                    If the results above are unsatisfactory, you can fine-tune model `{eval_data['model_id']}` 
                    khusus di data `{eval_data['target_folder']}`.
                    """)
                    
                    # NEW: Fine-tuning Configuration UI (Adaptive Layer Control)
                    with st.expander("Fine-tuning Configuration (Advanced Layer Control)", expanded=False):
                        ft_name = st.text_input("New Model Name (Optional)", placeholder="e.g. model_tangerang_v1", help="If left empty, name will auto-nerate a timestamp-based name.", key="ft_name_input")
                        
                        # Dynamic Layer Analysis
                        try:
                            import tensorflow as tf
                            from src.model_factory import get_custom_objects
                            # Robust path finding: check if folder contains model.keras or .h5
                            m_id = eval_data['model_id']
                            m_root = os.path.join(model_dir, m_id)
                            m_path = m_root
                            
                            # Detection
                            is_hf = False
                            meta_p = os.path.join(m_root, 'meta.json')
                            if os.path.exists(meta_p):
                                try:
                                    with open(meta_p, 'r') as f:
                                        meta_data = json.load(f)
                                        if 'hf' in meta_data.get('architecture', '').lower() or 'causal' in meta_data.get('architecture', '').lower():
                                            is_hf = True
                                except: pass
                            
                            if not is_hf:
                                if os.path.isdir(m_root):
                                    for ext in ['.keras', '.h5']:
                                        cand = os.path.join(m_root, f'model{ext}')
                                        if os.path.exists(cand):
                                            m_path = cand
                                            break
                            
                            if is_hf:
                                import torch
                                from src.model_hf import load_hf_wrapper
                                hf_wrapper = load_hf_wrapper(m_root)
                                torch_model = hf_wrapper.model
                                # List top-level modules or critical components
                                layer_names = []
                                for i, (name, module) in enumerate(torch_model.named_modules()):
                                    if i < 50: # Limit output
                                        layer_names.append(f"[{i}] {name} ({type(module).__name__})")
                                total_layers = len(layer_names)
                                st.markdown(f"**Struktur Model (HF/PyTorch):** `{m_id}` memiliki **{total_layers}** sub-modules.")
                            else:
                                # Minimize memory by not loading full weights if possible, but we need the structure
                                tmp_model = tf.keras.models.load_model(m_path, custom_objects=get_custom_objects(), compile=False)
                                layer_names = [f"[{i}] {l.name} ({type(l).__name__})" for i, l in enumerate(tmp_model.layers)]
                                total_layers = len(layer_names)
                                st.markdown(f"**Struktur Model (Keras):** `{m_id}` memiliki **{total_layers}** total layers.")
                            
                            st.markdown("---")
                            st.markdown("**Daftar Layer (Layer Browser)**")
                            st.code("\n".join(layer_names), language="text")
                            
                            c_f1, c_f2 = st.columns(2)
                            with c_f1:
                                ft_epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10, key="ft_epoch_input")
                                current_lr = cfg['model']['hyperparameters'].get('learning_rate', 0.001)
                                ft_lr = st.number_input("Learning Rate", min_value=0.000001, max_value=0.1, value=current_lr*0.1, format="%.6f", key="ft_lr_input")
                            
                            with c_f2:
                                ft_reset_weights = st.checkbox("Train from Scratch (Reset Weights)", value=False, help="Ignore pre-trained weights, initialize new weights from scratch with the exact same architecture on this target data.", key="ft_reset_weights_check")
                                ft_freeze = st.checkbox("Enable Layer Freezing", value=True, help="Freeze early layers to preserve pretrained knowledge.", key="ft_freeze_check", disabled=ft_reset_weights)
                                
                                if ft_reset_weights:
                                    ft_last_n = total_layers
                                    st.info("Training new model from scratch. Previous weights will be discarded.")
                                elif ft_freeze:
                                    # Use a slider for the split point
                                    # Layer 0 to X-1 are frozen. Layer X to End are trainable.
                                    split_point = st.slider(
                                        "Titik Beku (Freeze Point)", 
                                        min_value=0, max_value=total_layers-1, 
                                        value=max(0, total_layers-2),
                                        help="Layers BEFORE this point will be frozen. Layers AFTER this point will be retrained."
                                    )
                                    ft_last_n = total_layers - split_point
                                    st.warning(f" **{split_point}** Layers Awal Beku |  **{ft_last_n}** Layers Akhir Dilatih.")
                                    if ft_last_n <= 1:
                                        st.caption("You are only training the output layer. Very stable but slow to adapt.")
                                    elif ft_last_n > total_layers * 0.5:
                                        st.caption("You are training more than 50% of the model. Prior knowledge may be lost quickly (Catastrophic Forgetting).")
                                else:
                                    ft_last_n = total_layers # Train everything
                                    st.info("Melatih **Seluruh** Layer Model (No Freezing).")
                                    
                        except Exception as e_layers:
                            st.error(f"Gagal menganalisis layer: {e_layers}")
                            st.caption("Use manual configuration below:")
                            c_err1, c_err2 = st.columns(2)
                            with c_err1:
                                ft_epochs = st.number_input("Epochs", 1, 100, 10, key="ft_err_e")
                                ft_lr = st.number_input("Learning Rate", 0.000001, 0.1, 0.0001, format="%.6f", key="ft_err_lr")
                            with c_err2:
                                ft_freeze = st.checkbox("Freeze Backbone", True, key="ft_err_frz")
                                ft_last_n = st.number_input("Last N Layers Trainable", 1, 50, 2, key="ft_err_n")

                    if st.button("Start Fine-Tuning on Target Data", 
                                 type="secondary", use_container_width=True, key="run_fine_tune_btn"):
                        # Progress logging area
                        progress_container = st.container()
                        with progress_container:
                            st.markdown("---")
                            st.markdown("##### Fine-tuning Progress")
                            ft_progress_bar = st.progress(0, text="Initializing...")
                            col_f1, col_f2, col_f3 = st.columns(3)
                            epoch_disp = col_f1.empty()
                            loss_disp = col_f2.empty()
                            vloss_disp = col_f3.empty()
                            log_disp = st.empty()

                        try:
                            import tensorflow as tf
                            from src.trainer import fine_tune_model
                            from datetime import datetime
                            import time

                            class FTStreamlitCallback(tf.keras.callbacks.Callback):
                                def __init__(self, total_epochs):
                                    super().__init__()
                                    self.total_epochs = total_epochs
                                    self.logs = []
                                def on_epoch_end(self, epoch, logs=None):
                                    logs = logs or {}
                                    loss = logs.get('loss', 0)
                                    v_loss = logs.get('val_loss', 0)
                                    prog = (epoch + 1) / self.total_epochs
                                    
                                    ft_progress_bar.progress(prog, text=f"Epoch {epoch+1}/{self.total_epochs}")
                                    epoch_disp.metric("Epoch", f"{epoch+1}/{self.total_epochs}")
                                    loss_disp.metric("Loss", f"{loss:.6f}")
                                    vloss_disp.metric("Val Loss", f"{v_loss:.6f}")
                                    
                                    msg = f"[{datetime.now():%H:%M:%S}] Epoch {epoch+1}/{self.total_epochs}: loss={loss:.6f}, val_loss={v_loss:.6f}"
                                    self.logs.append(msg)
                                    log_disp.code("\n".join(self.logs[-10:])) # Show last 10 lines

                            model_path = os.path.join(model_dir, eval_data['model_id'])
                            
                            cfg_target = cfg.copy()
                            cfg_target['paths']['processed_dir'] = os.path.join(processed_root, eval_data['target_folder'])
                            
                            # Prepare ft_config
                            ft_config = {
                                'epochs': ft_epochs,
                                'learning_rate': ft_lr,
                                'freeze_backbone': ft_freeze,
                                'trainable_last_n': ft_last_n,
                                'custom_name': ft_name,
                                'reset_weights': ft_reset_weights if 'ft_reset_weights' in locals() else False
                            }
                            
                            # Start fine-tuning with callback
                            ft_cb = FTStreamlitCallback(ft_epochs)
                            ft_model, ft_history, ft_id = fine_tune_model(
                                cfg_target, model_path, ft_config=ft_config, extra_callbacks=[ft_cb]
                            )
                            
                            st.success(f"Fine-tuning complete! New model: **{ft_id}**")
                            st.session_state.pipeline_log.append(f"[{datetime.now():%H:%M:%S}] Fine-tuned {eval_data['model_id']} -> {ft_id}")
                            st.info("Select this new model in the dropdown above to test its improvement!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"Fine-tuning Error: {e}")


                    st.markdown("---")
                    st.markdown("#### Freezing Point Sweep (Optimization)")
                    st.markdown("Automatically find the optimal number of layers to unfreeze. The system will test various layer depths and compare results.")
                    
                    sweep_col1, sweep_col2 = st.columns(2)
                    with sweep_col1:
                        sweep_ep = st.slider("Epochs per Point", 1, 10, 3, help="Use small epochs (e.g. 3) to speed up searching for optimal trends.")
                    with sweep_col2:
                        sweep_lr = st.number_input("Sweep LR", 0.00001, 0.01, 0.0001, format="%.5f")
                    
                    if st.button("Run Optimization Sweep", type="secondary", use_container_width=True):
                        try:
                            from src.trainer import run_freezing_sweep
                            progress_ph = st.empty()
                            plot_ph = st.empty()
                            
                            model_path_ft = os.path.join(model_dir, eval_data['model_id'])
                            cfg_sweep = cfg.copy()
                            cfg_sweep['paths']['processed_dir'] = os.path.join(processed_root, eval_data['target_folder'])
                            
                            # Streamlit callback for sweep
                            class SweepProgressCallback(tf.keras.callbacks.Callback):
                                def on_epoch_end(self, epoch, logs=None):
                                    pass # Silent for sweep points
                            
                            with st.spinner("Running unfreeze point optimization..."):
                                sweep_results = run_freezing_sweep(
                                    cfg_sweep, model_path_ft, 
                                    ft_config={'sweep_epochs': sweep_ep, 'learning_rate': sweep_lr},
                                    callbacks=[SweepProgressCallback()]
                                )
                            
                            if sweep_results:
                                st.session_state.sweep_data = sweep_results
                                st.success(f"Sweep complete! {len(sweep_results)} configurations tested.")
                                
                                df_sweep = pd.DataFrame(sweep_results)
                                
                                # Visualization
                                fig_sweep = go.Figure()
                                fig_sweep.add_trace(go.Scatter(
                                    x=df_sweep['trainable_layers'], y=df_sweep['r2'],
                                    mode='lines+markers', name='R2 Score',
                                    line=dict(color='#818cf8', width=3),
                                    marker=dict(size=8)
                                ))
                                fig_sweep.update_layout(
                                    title="Freezing Point Sweep: Trainable Layers vs Performance",
                                    xaxis_title="Trainable Layers (from Output)",
                                    yaxis_title="R2 Score",
                                    template="plotly_dark",
                                    height=400
                                )
                                st.plotly_chart(fig_sweep, use_container_width=True)
                                
                                # Recommendation
                                best_pt = df_sweep.loc[df_sweep['r2'].idxmax()]
                                st.success(f" **Rekomendasi**: Unfreeze **{best_pt['trainable_layers']:.0f} layers** (R2={best_pt['r2']:.4f}).")
                                
                                st.markdown("---")
                                st.markdown("**Sweep Results Table**")
                                st.table(df_sweep)
                        except Exception as e_sweep:
                            st.error(f"Sweep Error: {e_sweep}")



# --- TAB: MODEL COMPARISON ---
with tab_compare:
    st.markdown("### Model Comparison & Leaderboard")
    st.markdown("Bandingkan performa beberapa model secara berdampingan.")
    
    if os.path.exists(model_dir):
        all_models = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5', '.json')) or os.path.isdir(os.path.join(model_dir, f))]
        all_models = [f for f in all_models if not f.endswith('_meta.json')] # Filter meta files if any
        
        if all_models:
            st.markdown("#### 1. Select Models")
            selected_models = st.multiselect("Select models to compare:", all_models, 
                                            default=all_models[:min(2, len(all_models))],
                                            format_func=lambda x: label_format_with_time(x, model_dir),
                                            key="ms_comparison")
            
            comp_eval_mode = st.radio("Evaluation Method (Dataset):", 
                                 ["Based on Original Dataset (Model Default)", "Cross-Test on Active Target Data (Cross-Domain)"], 
                                 help="Choose whether to compare built-in metrics from original training data, or cross-test against active target data.",
                                 key="comp_eval_mode")
            
            st.button("Run Comparison Analysis", type="primary", use_container_width=True, key="btn_run_comp", on_click=_set_action, args=("action_run_comparison",))
            if st.session_state.get("action_run_comparison"):
                st.session_state.action_run_comparison = False
                if not selected_models:
                    st.warning("Select at least one model.")
                else:
                    comparison_results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # LOG START TO CMD
                    print("\n" + "="*60)
                    print(f"COMPARISON START: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"   Models to evaluate: {len(selected_models)}")
                    print("="*60)
                    sys.stdout.flush()

                    with st.spinner("Running in-depth evaluation..."):
                        for i, model_id in enumerate(selected_models):
                            msg = f"[{i+1}/{len(selected_models)}] Mengevaluasi: {model_id}"
                            status_text.text(" " + msg)
                            print(f"   {msg}...")
                            sys.stdout.flush()
                            
                            try:
                                import io, contextlib
                                # Mute stdout here to avoid UnicodeEncodeErrors in Windows consoles caused by emojis inside evaluate_model
                                dummy_out = io.StringIO()
                                
                                # 1. Clean session
                                tf.keras.backend.clear_session()
                                gc.collect()
                                
                                # 2. Get Model Path & Root
                                model_id_clean = model_id
                                model_path = os.path.join(model_dir, model_id)
                                model_root = model_path if os.path.isdir(model_path) else model_dir
                                
                                # Robust Detection (Matches trainer.py)
                                is_hf = False
                                meta_p = os.path.join(model_root, 'meta.json')
                                m_meta_r = {}
                                if os.path.exists(meta_p):
                                    try:
                                        with open(meta_p, 'r') as f:
                                            m_meta_r = json.load(f)
                                            arch_low = m_meta_r.get('architecture', '').lower()
                                            if 'hf' in arch_low or 'causal' in arch_low:
                                                is_hf = True
                                    except: pass
                                
                                if not is_hf and os.path.isdir(model_root):
                                    if os.path.exists(os.path.join(model_root, 'pytorch_model.bin')) or \
                                       os.path.exists(os.path.join(model_root, 'config.json')):
                                        is_hf = True

                                if os.path.isdir(model_root) and not is_hf:
                                    # Find Keras file within directory
                                    found_k = False
                                    for ext in ['.keras', '.h5']:
                                        cand = os.path.join(model_root, f'model{ext}')
                                        if os.path.exists(cand):
                                            model_path = cand
                                            found_k = True
                                            break
                                    if not found_k:
                                        # Default to first available file that's Keras
                                        for f in os.listdir(model_root):
                                            if f.endswith(('.keras', '.h5')):
                                                model_path = os.path.join(model_root, f)
                                                break
                                
                                # 3. Load Model
                                custom_objs = get_custom_objects()
                                with tf.keras.utils.custom_object_scope(custom_objs):
                                    with contextlib.redirect_stdout(dummy_out):
                                        if is_hf:
                                            from src.model_hf import load_hf_wrapper
                                            model = load_hf_wrapper(model_root)
                                        else:
                                            import zipfile
                                            # Keras 3 ZIP detection
                                            is_k3_zip = model_path.endswith('.keras') and zipfile.is_zipfile(model_path)
                                            
                                            try:
                                                if is_k3_zip:
                                                    raise OSError("Trigger Keras 3 Recovery") # Jump to recovery
                                                try:
                                                    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
                                                except TypeError:
                                                    model = tf.keras.models.load_model(model_path, compile=False)
                                            except (OSError, Exception) as k_err:
                                                if model_path.endswith('.keras') or is_k3_zip:
                                                    print(f"      [RECOVER] K3 ZIP or Corrupt Keras detected, attempting rebuild...")
                                                    extract_dir = os.path.join(model_root, '_extracted_k3_eval')
                                                    os.makedirs(extract_dir, exist_ok=True)
                                                    if zipfile.is_zipfile(model_path):
                                                        with zipfile.ZipFile(model_path, 'r') as zf:
                                                            zf.extractall(extract_dir)
                                                    
                                                    weights_h5 = os.path.join(extract_dir, 'model.weights.h5')
                                                    arch = m_meta_r.get('architecture', cfg['model']['architecture'])
                                                    hp_r = m_meta_r.get('hyperparameters', cfg['model']['hyperparameters'])
                                                    lb = m_meta_r.get('lookback', cfg['model']['hyperparameters']['lookback'])
                                                    nf = m_meta_r.get('n_features', 0)
                                                    hz = m_meta_r.get('horizon', m_meta_r.get('forecast_horizon', cfg['forecasting']['horizon']))
                                                    
                                                    if nf == 0:
                                                        prep_p = os.path.join(model_root, 'prep_summary.json')
                                                        if os.path.exists(prep_p):
                                                            try:
                                                                with open(prep_p, 'r') as f:
                                                                    p_m = json.load(f)
                                                                    if n_f := p_m.get('n_features'): nf = n_f
                                                                    if l_b := p_m.get('lookback'): lb = l_b
                                                                    if h_z := p_m.get('horizon'): hz = h_z
                                                            except: pass
                                                    
                                                    from src.model_factory import build_model as bm, manual_load_k3_weights
                                                    model = bm(arch, lb, nf, hz, hp_r)
                                                    manual_load_k3_weights(model, weights_h5)
                                                else:
                                                    raise ke3_err
                                            from src.model_factory import fix_lambda_tf_refs
                                            fix_lambda_tf_refs(model)
                                            compile_model(model, cfg['model']['hyperparameters']['learning_rate'])
                                    
                                # 4. Run Evaluation
                                res = None
                                eval_source = "Unknown"
                                m_meta = {}
                                
                                # Pre-load metadata for UI display
                                meta_path = os.path.join(model_root, "meta.json")
                                if os.path.exists(meta_path):
                                    try:
                                        with open(meta_path, 'r') as f:
                                            m_meta = json.load(f)
                                    except: pass
                                
                                # --- REFACTORED EVALUATION STRATEGY ---
                                # Strategy 0: Active Target Testing Folder (Highly Priority if set AND user selected cross-domain mode)
                                target_sel = st.session_state.get('target_folder_sel')
                                if target_sel and comp_eval_mode == "Cross-Test on Active Target Data (Cross-Domain)":
                                    try:
                                        # Construct path from processed root
                                        processed_base = cfg['paths']['processed_dir']
                                        processed_root = os.path.dirname(processed_base) if os.path.basename(processed_base).startswith(('v', 'version')) else processed_base
                                        target_path = os.path.join(processed_root, target_sel).replace('\\', '/')
                                        
                                        if os.path.exists(target_path):
                                            temp_cfg = copy.deepcopy(cfg)
                                            temp_cfg['paths']['processed_dir'] = target_path
                                            scaler_dir = model_root if os.path.isdir(model_root) else None
                                            with contextlib.redirect_stdout(dummy_out):
                                                res = evaluate_model(model, temp_cfg, data=None, scaler_dir=scaler_dir)
                                            eval_source = f"Active Target ({target_sel})"
                                            print(f"      Resolved via Active Target: {target_path}")
                                    except Exception as e_active:
                                        print(f"      (!) Active Target strategy failed: {e_active}")

                                # Strategy 1: Smart Evaluation (delegates data discovery to predictor.py)
                                if res is None:
                                    try:
                                        scaler_dir = model_root if os.path.isdir(model_root) else None
                                        temp_cfg = copy.deepcopy(cfg)
                                        
                                        # FORCE evaluation on original dataset if requested
                                        if comp_eval_mode == "Based on Original Dataset (Model Default)":
                                            if m_meta and m_meta.get('data_source'):
                                                orig_ds = m_meta.get('data_source', '').replace('\\', '/')
                                                # Use relative fallback if absolute path moved
                                                if not os.path.exists(orig_ds):
                                                    d_name = os.path.basename(orig_ds)
                                                    rel_ds = os.path.join('data', 'processed', d_name).replace('\\', '/')
                                                    if os.path.exists(rel_ds): orig_ds = rel_ds
                                                
                                                if os.path.exists(orig_ds):
                                                    temp_cfg['paths']['processed_dir'] = orig_ds
                                                    print(f"      [Eval] Forced to model's original dataset: {orig_ds}")
                                                    
                                        with contextlib.redirect_stdout(dummy_out):
                                            res = evaluate_model(model, temp_cfg, data=None, scaler_dir=scaler_dir)
                                        eval_source = "Bundled/Resolved Data"
                                    except Exception as e_smart:
                                        print(f"      (!) Smart Eval failed: {e_smart}")
                                        res = None
                                
                                # Strategy 2: Active Preprocessing Data (Fallback)
                                if res is None:
                                    active_prep = st.session_state.get('prep_metadata', None)
                                    if active_prep is not None and active_prep.get('X_train') is not None:
                                        print(f"      Checking Active Data compatibility...")
                                        expected_n = model.input_shape[2] if hasattr(model, 'input_shape') else active_prep['X_train'].shape[2]
                                        if active_prep['X_train'].shape[2] == expected_n:
                                            scaler_dir = model_root if os.path.isdir(model_root) else None
                                            with contextlib.redirect_stdout(dummy_out):
                                                res = evaluate_model(model, cfg, data=active_prep, scaler_dir=scaler_dir)
                                            eval_source = "Active Data"
                                        else:
                                            print(f"      (!) Active Data mismatch: {active_prep['X_train'].shape[2]} features != {expected_n}")
                                
                                if res is None:
                                    raise ValueError("No valid data source found or recovered for this model.")

                                # 5. Extract Metrics and Source Information
                                m_test = res['metrics_test']
                                m_train = res['metrics_train']
                                res_path = res.get('data_path', '')
                                
                                train_time = m_meta.get('training_time_seconds', 0)
                                if train_time is None: train_time = 0
                                
                                # Original data source from metadata
                                ds_path = m_meta.get('data_source', '').replace('\\', '/') if m_meta else ''
                                
                                # Robust Feature Extraction for UI
                                feat_text = "N/A"
                                feat_count = 0
                                
                                # Search order: 1. Model's evaluate result path, 2. Model Bundle folder, 3. Original source path
                                summary_paths = [
                                    os.path.join(model_root, "prep_summary.json"),
                                    os.path.join(res_path, "prep_summary.json") if res_path else None,
                                    os.path.join(ds_path, "prep_summary.json") if ds_path else None
                                ]
                                
                                for sp in summary_paths:
                                    if sp and os.path.exists(sp):
                                        try:
                                            with open(sp, 'r') as ff:
                                                summ_data = json.load(ff)
                                                if 'selected_features' in summ_data:
                                                    feat_text = ", ".join(summ_data['selected_features'])
                                                    feat_count = len(summ_data['selected_features'])
                                                    break
                                        except: pass
                                
                                if feat_count == 0:
                                    # Fallback to model architecture shape
                                    if hasattr(model, 'input_shape') and len(model.input_shape) >= 3:
                                        feat_count = model.input_shape[2]
                                    else:
                                        feat_count = '?'
                                
                                # Final verification source string
                                verified_on = os.path.basename(res_path) if res_path else eval_source

                                comparison_results.append({
                                    'Model ID': model_id,
                                    'R2 Test': m_test['r2'],
                                    'R2 Train': m_train['r2'],
                                    'MAE': m_test['mae'],
                                    'nMAE (%)': m_test.get('norm_mae', 0) * 100,
                                    'RMSE': m_test['rmse'],
                                    'nRMSE (%)': m_test.get('norm_rmse', 0) * 100,
                                    'Inference Time (ms)': res.get('inference_time_ms', 0),
                                    'Features N': feat_count,
                                    'Feature List': feat_text,
                                    'Lookback': model.input_shape[1] if hasattr(model, 'input_shape') else '?',
                                    'Verified On': verified_on
                                })
                                print(f"      OK (R2: {m_test['r2']:.4f})")
                                
                            except Exception as e:
                                err_info = str(e)
                                st.error(f"Error for {model_id}: {err_info}")
                                print(f"      ERROR: {err_info}")
                                comparison_results.append({
                                    'Model ID': model_id,
                                    'R2 Test': 0, 'R2 Train': 0, 'MAE': 999, 'nMAE (%)': 999, 
                                    'RMSE': 999, 'nRMSE (%)': 999, 'Inference Time (ms)': 0,
                                    'Features N': 'Error', 'Feature List': 'Error',
                                    'Lookback': 'Error', 'Verified On': 'Error'
                                })
                            
                            progress_bar.progress((i + 1) / len(selected_models))
                            sys.stdout.flush()
                        
                        status_text.empty()
                        st.session_state.comparison_df = pd.DataFrame(comparison_results)
                        print(f"\nCOMPARISON FINISHED: {len(comparison_results)} models analyzed.")
                        print("="*60 + "\n")
                        sys.stdout.flush()
                        st.success("Comparison analysis complete! Results shown below.")

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

                # Build format dict and highlight subsets dynamically
                fmt = {}
                r2_cols = []
                for col in ['R2 Test', 'R2 Train', 'R2']:
                    if col in df_comp.columns:
                        fmt[col] = '{:.4f}'
                        r2_cols.append(col)
                for col in ['MAE', 'RMSE']:
                    if col in df_comp.columns: fmt[col] = '{:.4f}'
                for col in ['nMAE (%)', 'nRMSE (%)']:
                    if col in df_comp.columns: fmt[col] = '{:.2f}'
                # Add formatting for time column if it exists
                if 'Inference Time (ms)' in df_comp.columns:
                    fmt['Inference Time (ms)'] = '{:.1f}'
                
                err_cols = [c for c in ['MAE', 'nMAE (%)', 'RMSE', 'nRMSE (%)'] if c in df_comp.columns]
                
                styled_df = df_comp.style.format(fmt)
                if r2_cols:
                    styled_df = styled_df.apply(highlight_max, subset=r2_cols)
                if err_cols:
                    styled_df = styled_df.apply(highlight_min, subset=err_cols)
                
                st.dataframe(styled_df, use_container_width=True)
                
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
                            label="Export Table to Excel (.xlsx)",
                            data=excel_data,
                            file_name='model_comparison_results.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    except Exception:
                        # Fallback to CSV if openpyxl is not installed
                        csv_data = df_comp.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Export Table to CSV (.csv)",
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
                    fig_r2 = px.bar(df_comp, x='Model ID', y='R2 Test', color='R2 Test', 
                                   title="R2 Score (Higher is Better)",
                                   color_continuous_scale='Viridis')
                    fig_r2.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_r2, use_container_width=True)
                with c2:
                    fig_mae = px.bar(df_comp, x='Model ID', y='MAE', color='MAE',
                                    title="MAE Score (Lower is Better)",
                                    color_continuous_scale='Reds_r')
                    fig_mae.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_mae, use_container_width=True)
                
                # Second Row: Inference Time and Overfitting Delta
                c3, c4 = st.columns(2)
                with c3:
                    fig_time = px.bar(df_comp, x='Model ID', y='Inference Time (ms)', color='Inference Time (ms)',
                                    title="Inference Time per sample (Lower is Faster)",
                                    color_continuous_scale='Oranges')
                    fig_time.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_time, use_container_width=True)
                with c4:
                    fig_rmse = px.bar(df_comp, x='Model ID', y='RMSE', color='RMSE',
                                    title="RMSE (Lower is Safer/Better)",
                                    color_continuous_scale='Purples_r')
                    fig_rmse.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                
                # Trade-off: Inference Time vs nRMSE
                st.markdown("#### 4. Trade-off: Speed vs Error")
                if 'Inference Time (ms)' in df_comp.columns and 'nRMSE (%)' in df_comp.columns:
                    fig_scatter = px.scatter(
                        df_comp, 
                        x='Inference Time (ms)', 
                        y='nRMSE (%)', 
                        color='Model ID',
                        hover_name='Model ID',
                        title="Trade-off: Inference Time vs nRMSE"
                    )
                    fig_scatter.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='White')))
                    fig_scatter.update_layout(
                        template="plotly_dark", 
                        height=500,
                        xaxis_title="Inference Time (ms)",
                        yaxis_title="nRMSE (%)"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("Inference Time (ms) or nRMSE (%) data not available for trade-off chart.")
                
                # Radar Chart
                st.markdown("#### 5. Performance Radar")
                radar_data = df_comp.copy()
                cols_to_norm = ['R2 Test', 'MAE', 'nMAE (%)', 'RMSE', 'nRMSE (%)']
                for col in cols_to_norm:
                    if col not in radar_data.columns: continue
                    if col == 'R2 Test':
                        radar_data[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min() + 1e-6)
                    else:
                        norm = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min() + 1e-6)
                        radar_data[col] = 1 - norm
                
                fig_radar = go.Figure()
                for i, row in radar_data.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row.get('R2 Test', 0), row['MAE'], row['RMSE'], row['nMAE (%)'], row['nRMSE (%)']],
                        theta=['R2 Test', 'MAE (Inverted)', 'RMSE (Inverted)', 'nMAE (Inverted)', 'nRMSE (Inverted)'],
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
                st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("No saved models in `models/` folder.")
    else:
        st.error("Folder models/ not found.")



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
