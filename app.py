"""
app.py - Streamlit Web Dashboard untuk PV Forecasting Pipeline
Jalankan: streamlit run app.py
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

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
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Using a friendlier print for the terminal, Streamlit will show this in the CLI logs
        print(f"✅ Running in GPU Mode. Available GPUs: {len(gpus)}")
    else:
        print("💡 Running in CPU Mode. No GPU detected or GPU disabled.")
except Exception as e:
    print(f"Error initializing GPU: {e}")

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
PERSIST_DIR = 'logs/session'
PRESETS_DIR = 'configs/presets'

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

cfg = st.session_state.cfg
proc_dir = cfg['paths']['processed_dir']
model_dir = cfg['paths']['models_dir']
target_dir = cfg['paths']['target_data_dir']


# ============================================================
# SIDEBAR - CONFIGURATION
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi Pipeline")
    
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
            
            selected = st.selectbox("Pilih Model untuk Evaluasi", model_options, index=curr_idx)
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


# ============================================================
# MAIN AREA - HEADER
# ============================================================
st.markdown("""
<div style="text-align:center; margin-bottom: 2rem;">
    <h1 style="font-family: 'Manrope', sans-serif;
               background: linear-gradient(135deg, #818cf8, #6366f1, #4f46e5); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em;
               margin-bottom: 0.5rem;">
        PV Forecasting Pipeline
    </h1>
    <p style="font-family: 'Manrope', sans-serif; color: #64748b; 
              font-size: 1.1rem; font-weight: 400;">
        Universal Dashboard &mdash; PatchTST, GRU, dan lainnya
    </p>
</div>
""", unsafe_allow_html=True)

# Status Cards
new_arch = cfg['model'].get('architecture', 'patchtst')
has_data = os.path.exists(os.path.join(proc_dir, 'X_train.npy'))
has_model = any(f.endswith(('.keras', '.h5')) for f in os.listdir(model_dir)) if os.path.exists(model_dir) else False
has_target = any(f.endswith('.csv') for f in os.listdir(target_dir)) if os.path.exists(target_dir) else False

col1, col2, col3, col4 = st.columns(4)
with col1:
    status = "ready" if has_data else "missing"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{"✅" if has_data else "⏳"}</div>
        <div class="metric-label">Data Preprocessed</div>
        <span class="status-badge status-{status}">{"Ready" if has_data else "Belum"}</span>
    </div>""", unsafe_allow_html=True)
with col2:
    status = "ready" if has_model else "missing"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{"✅" if has_model else "⏳"}</div>
        <div class="metric-label">Model Terlatih</div>
        <span class="status-badge status-{status}">{"Ready" if has_model else "Belum"}</span>
    </div>""", unsafe_allow_html=True)
with col3:
    status = "ready" if has_target else "missing"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{"📂" if has_target else "⏳"}</div>
        <div class="metric-label">Data Target</div>
        <span class="status-badge status-{status}">{"Ada" if has_target else "Belum"}</span>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{new_arch.upper()}</div>
        <div class="metric-label">Arsitektur Aktif</div>
        <span class="status-badge status-ready">Selected</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab_lab, tab_prep, tab_data, tab_tuning, tab2, tab3, tab4, tab1, tab5 = st.tabs([
    "🧪 Feature Lab",
    "📥 Preprocessing",
    "🔎 Data Insights",
    "🎯 Tuning",
    "🧠 Training Center",
    "📈 Evaluation",
    "📦 Target Test",
    "🚀 Runner",
    "📋 Logs",
])

# --- NEW TAB: FEATURE LAB ---
with tab_lab:
    st.markdown("### 🧪 Feature Engineering Lab")
    st.markdown("Eksperimen dengan kombinasi fitur untuk meningkatkan performa model.")
    
    # --- Preset Manager ---
    with st.expander("📂 Preset Manager (Save/Log Templates)", expanded=True):
        p1, p2 = st.columns([2, 1])
        with p1:
            presets = list_feature_presets()
            selected_preset = st.selectbox("Pilih Preset untuk Dimuat", ["-- Silakan Pilih --"] + presets)
            if selected_preset != "-- Silakan Pilih --":
                if st.button("📥 Load Preset"):
                    loaded = load_feature_preset(selected_preset)
                    if loaded:
                        # Update the session config directly
                        cfg['features'].update(loaded.get('features', {}))
                        cfg['target'].update(loaded.get('target', {}))
                        st.success(f"Preset '{selected_preset}' berhasil dimuat!")
                        st.session_state.cfg = cfg
                        st.rerun()
        with p2:
            new_preset_name = st.text_input("Simpan sebagai Preset Baru", placeholder="v1_lags_only")
            if st.button("💾 Save to Log/Presets"):
                if new_preset_name.strip():
                    save_feature_preset(new_preset_name.strip(), cfg)
                    st.success(f"Konfigurasi '{new_preset_name}' telah di-log dan disimpan!")
                    st.rerun()
                else:
                    st.error("Nama preset tidak boleh kosong!")
    
    st.markdown("---")
    
    col_l, col_r = st.columns([1, 1.5])
    
    with col_l:
        st.subheader("1. Preset Overrides")
        st.info("Gunakan Preset Manager di atas untuk memuat konfigurasi cepat. Detail fitur dapat disesuaikan di tab Preprocessing.")
        
    with col_r:
        st.subheader("2. Selection Strategy")
        s_mode = st.radio("Mode Seleksi", ["auto", "manual"], 
                          index=0 if cfg['features'].get('selection_mode', 'auto') == 'auto' else 1,
                          help="Auto: Menggunakan korelasi & statistik. Manual: User memilih daftar fitur.")
        cfg['features']['selection_mode'] = s_mode
        
        if s_mode == "auto":
            st.info("Sistem akan memilih fitur yang memiliki korelasi tinggi ke target dan membuang fitur yang duplikat.")
            cfg['features']['corr_threshold'] = st.slider(
                "Minimum Korelasi (Pearson)", 0.0, 0.5, 
                cfg['features'].get('corr_threshold', 0.05), 0.05
            )
            cfg['features']['multicol_threshold'] = st.slider(
                "Maksimum Multikolinearitas", 0.7, 1.0, 
                cfg['features'].get('multicol_threshold', 0.95), 0.05
            )
        else:
            candidate_features = []
            if st.session_state.get('prep_metadata'):
                candidate_features = st.session_state.prep_metadata.get('all_features', [])
            
            if candidate_features:
                selected_man = st.multiselect(
                    "Daftar Fitur yang Tersedia", 
                    options=candidate_features,
                    default=[f for f in cfg['features'].get('manual_features', []) if f in candidate_features]
                )
                cfg['features']['manual_features'] = selected_man
                st.success(f"Daftar manual: {len(selected_man)} fitur aktif.")
            else:
                st.warning("⚠️ Belum ada metadata fitur. Jalankan 'Run Preprocessing' sekali dengan mode Auto terlebih dahulu.")

    st.warning("⚠️ **Penting**: Setelah mengubah pengaturan di sini, Anda WAJIB menjalankan ulang **Step 1 (Preprocessing)** di tab 'Runner' sebelum melakukan Training.")
    
    if st.button("💾 Simpan Konfigurasi Fitur"):
        save_config_to_file(cfg)
        st.toast("Konfigurasi fitur berhasil disimpan!", icon="✅")

# --- NEW TAB: PREPROCESSING ---
with tab_prep:
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
            st.markdown("##### 📦 Feature Groups")
            g = cfg['features'].get('groups', {})
            # time features
            with st.expander("🕒 Cyclical Time Features (Sin/Cos)", expanded=False):
                g['time_hour'] = st.checkbox("Hourly (Hour Sin/Cos)", value=g.get('time_hour', True), 
                                            help="Menangkap pola harian (pagi-siang-malam).", key="p_time_h")
                g['time_month'] = st.checkbox("Monthly (Month Sin/Cos)", value=g.get('time_month', True), 
                                             help="Menangkap pola musiman bulanan.", key="p_time_m")
                g['time_doy'] = st.checkbox("Seasonal (DOY Sin/Cos)", value=g.get('time_doy', True), 
                                           help="Day of Year: Resolusi tinggi untuk dinamika musiman.", key="p_time_d")
                g['time_year'] = st.checkbox("Yearly (Linear)", value=g.get('time_year', False), 
                                            help="Menangkap tren jangka panjang/tahunan.", key="p_time_y")
            g['weather'] = st.checkbox("Weather (Raw)", value=g.get('weather', True), key="p_weather")
            g['lags'] = st.checkbox("Time Lags (History)", value=g.get('lags', True), key="p_lags")
            g['rolling'] = st.checkbox("Moving Average", value=g.get('rolling', True), key="p_roll")
            g['physics'] = st.checkbox("Physics-based (CSI)", value=g.get('physics', True), key="p_phys")
            cfg['features']['groups'] = g
            
            st.markdown("---")
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
            st.markdown("##### 🎯 Target Transform")
            use_csi = st.checkbox("Gunakan CSI Normalization", 
                                   value=cfg['target'].get('use_csi', True),
                                   help="True: Prediksi rasio (CSI), False: Prediksi daya langsung (kW)",
                                   key="p_csi")
            cfg['target']['use_csi'] = use_csi
            if use_csi:
                cfg['target']['csi_ghi_threshold'] = st.number_input(
                    "GHI Threshold (W/m²)", 
                    value=cfg['target'].get('csi_ghi_threshold', 50),
                    min_value=0, max_value=500, key="p_csi_th"
                )

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
        run_preprocess = st.button("▶️ Start Preprocessing", type="primary", use_container_width=True, key="btn_prep_main")
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
                    metadata = run_preprocessing(cfg)
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

# --- TAB 1: PIPELINE RUNNER ---
with tab1:
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
                    st.session_state.eval_results = results
                    save_eval_results_to_disk(results)
                    st.session_state.pipeline_log.append(
                        f"[{datetime.now():%H:%M:%S}] Evaluasi selesai. "
                        f"Test R²: {results['metrics_test']['r2']:.4f}"
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
    st.markdown("### 🔎 Data Transformation Insights")
    st.caption("Detail metamorfosis data dari CSV mentah menjadi tensor yang siap dilatih oleh model.")
    
    if st.session_state.prep_metadata:
        m = st.session_state.prep_metadata
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
            if 'X_train' in m:
                # Show a sample of the first sequence 
                seq_sample = m['X_train'][0] # (lookback, n_features)
                df_seq = pd.DataFrame(seq_sample, columns=sel_f)
                st.dataframe(df_seq.head(10), width='stretch')
                st.caption(f"Menampilkan 10 timestep pertama dari sequence ke-0 (Tensor Shape: {m['X_train'].shape})")
            else:
                st.info("Preview tensor tidak tersedia di memori (silakan jalankan ulang Step 1 jika ingin melihat detail ini).")
        
    else:
        st.info("Belum ada data preprocessing. Silakan jalankan 'Step 1: Preprocessing' pada tab Runner.")


# --- TAB 2: TRAINING CENTER ---
with tab2:
    st.markdown("### 🧠 Training Center")
    
    # --- DATA VERSION SELECTOR ---
    with st.expander("📦 Pilih Versi Data Preprocessed", expanded=not has_data):
        if os.path.exists(proc_dir):
            versions = [f for f in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, f)) and (f.startswith('version_') or f.startswith('v_'))]
            versions = sorted(versions, reverse=True)
            options = ["Latest (Default)"] + versions
            
            # Select the most recent version if it was just created
            default_idx = 0
            selected_v = st.selectbox("Gunakan Versi Data untuk Training:", options, index=default_idx, key="data_version_train")
            
            if st.button("🔄 Refresh Daftar Versi"):
                st.rerun()
            
            if selected_v == "Latest (Default)":
                active_proc_dir = proc_dir
            else:
                active_proc_dir = os.path.join(proc_dir, selected_v)
            
            cfg['paths']['processed_dir'] = active_proc_dir
            st.session_state.active_proc_dir = active_proc_dir
            st.caption(f"Folder Aktif: `{os.path.basename(active_proc_dir)}`")
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
        
        with col_hp1:
            st.markdown("**Core Structure**")
            new_arch = st.selectbox("Arsitektur Model", ["patchtst", "gru", "lstm", "mlp"], 
                                    index=["patchtst", "gru", "lstm", "mlp"].index(cfg['model'].get('architecture', 'patchtst').lower()))
            cfg['model']['architecture'] = new_arch
            
            hp['lookback'] = st.select_slider("Lookback Window (h)", 
                                              options=[24, 48, 72, 96, 120, 144, 168, 192, 240, 336],
                                              value=hp['lookback'])
            hp['d_model'] = st.selectbox("d_model (Hidden Dim)", [32, 64, 128, 256], 
                                          index=[32,64,128,256].index(hp['d_model']))
            hp['n_layers'] = st.number_input("n_layers", value=hp['n_layers'], min_value=1, max_value=12)
            
        with col_hp2:
            st.markdown("**Optimization**")
            hp['learning_rate'] = st.number_input("Learning Rate", value=hp['learning_rate'],
                                                   format="%.6f", step=0.0001)
            hp['batch_size'] = st.selectbox("Batch Size", [16, 32, 64, 128],
                                             index=[16,32,64,128].index(hp['batch_size']))
            hp['dropout'] = st.slider("Dropout Rate", 0.0, 0.5, hp['dropout'], 0.05)
            
            # Adv. Params expander inside center
            with st.expander("Advanced Params (Patch/Stride)"):
                hp['patch_len'] = st.number_input("patch_len", value=hp['patch_len'], min_value=4, step=4)
                hp['stride'] = st.number_input("stride", value=hp['stride'], min_value=4, step=4)
                hp['n_heads'] = st.selectbox("n_heads", [4, 8, 12], index=[4,8,12].index(hp['n_heads']))

    # 2. Training Control Center
    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        st.markdown('<div class="pipeline-step">', unsafe_allow_html=True)
        st.markdown("**Execution Control**")
        
        c1, c2 = st.columns(2)
        with c1:
            device = st.radio("Device", ["GPU", "CPU"], index=0 if st.session_state.get('execution_device', 'GPU') == 'GPU' else 1, horizontal=True)
            st.session_state.execution_device = device
        with c2:
            st.write("") # mapping
            run_train = st.button("🚀 Start Model Training", type="primary", use_container_width=True, disabled=not has_data)
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
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                        lr_display.caption(f"LR: {current_lr:.8f} | Best Val: {min(d['val_loss'] for d in self.epoch_data):.6f}")
                        self.log_lines.append(f"Epoch {epoch+1:3d}: loss={loss:.6f}, val={val_loss:.6f}")
                        log_placeholder.code('\n'.join(self.log_lines))

                # Device selection
                if device == 'CPU': os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                else: os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                
                tf.keras.backend.clear_session()
                live_cb = StreamlitLiveCallback()
                from src.trainer import train_model
                model, history, meta = train_model(cfg, extra_callbacks=[live_cb])
                
                # Success & Persist
                st.session_state.training_history = history.history
                save_training_history(history.history, meta['model_id'])
                st.success(f"Training Selesai! Model saved as: {meta['model_id']}")
                time.sleep(1); st.rerun()
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
            st.plotly_chart(fig_l, use_container_width=True)
        with c2:
            st.metric("Best Val Loss", f"{min(hist['val_loss']):.6f}")
            st.metric("Total Epochs", len(hist['loss']))
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

    st.markdown("---")
    
    if st.session_state.tuning_results:
        tr = st.session_state.tuning_results
        trials = tr['trials']
        best = tr['best_params']
        
        # Best Parameters
        st.markdown("#### Best Hyperparameters")
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
            
        run_tune = st.button("🔥 Jalankan Optuna Tuning", type="primary", width='stretch', 
                              disabled=not has_data, key="btn_tune_tab")
        st.markdown('</div>', unsafe_allow_html=True)

        if cfg['tuning']['enabled']:
            st.markdown("#### Search Space Aktif (Hyperparameters)")
            space = cfg['tuning']['search_space']
            space_df = pd.DataFrame([
                {'Parameter': k, 'Range': str(v)} for k, v in space.items()
            ])
            st.dataframe(space_df, width='stretch', hide_index=True)
        else:
            st.warning("Optuna Tuning belum diaktifkan di sidebar.")

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
            
            with contextlib.redirect_stdout(stdout_capture):
                from src.trainer import run_optuna_tuning
                best, study = run_optuna_tuning(cfg, extra_callbacks=[live_monitor])
            
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


# --- TAB 3: EVALUATION & RESULTS ---
with tab3:
    st.markdown("### Evaluation & Results")
    
    # --- MODEL SELECTOR FOR EVALUATION ---
    with st.expander("📂 Pilih Model untuk Evaluasi", expanded=not st.session_state.eval_results):
        if os.path.exists(model_dir):
            all_models = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5')) or os.path.isdir(os.path.join(model_dir, f))]
            if all_models:
                current_sel = st.session_state.get('selected_model', all_models[0])
                model_to_eval = st.selectbox("Pilih Model:", all_models, 
                                           index=all_models.index(current_sel) if current_sel in all_models else 0)
                st.session_state.selected_model = model_to_eval
                
                if st.button("🔎 Run Evaluation for Selected Model", type="primary", use_container_width=True):
                    # Logic to run evaluation (roughly same as Runner/tab1)
                    st.rerun() # Just rerun to trigger the logic if integrated or call it directly
            else:
                st.warning("Belum ada model tersimpan di folder models/.")
        else:
            st.error("Folder models/ tidak ditemukan.")

    if st.session_state.eval_results:
        results = st.session_state.eval_results
        m_train = results['metrics_train']
        m_test = results['metrics_test']
        
        # ====== ROW 1: Metric cards ======
        st.markdown("#### Performance Metrics (Test Set)")
        col1, col2, col3, col4 = st.columns(4)
        
        metrics_display = [
            ("MAE", m_test['mae'], " kW"),
            ("RMSE", m_test['rmse'], " kW"),
            ("R\u00b2", m_test['r2'], ""),
            ("MAPE", m_test['mape'], "%"),
        ]
        for col, (name, val, unit) in zip([col1, col2, col3, col4], metrics_display):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{val:.4f}{unit}</div>
                    <div class="metric-label">Test {name}</div>
                </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ====== ROW 2: Train vs Test Comparison ======
        st.markdown("#### Train vs Test Comparison")
        df_metrics = pd.DataFrame({
            'Metrik': ['MAE (kW)', 'RMSE (kW)', 'R\u00b2', 'MAPE (%)', 'NormMAE (%)'],
            'Train': [f"{m_train['mae']:.4f}", f"{m_train['rmse']:.4f}", 
                      f"{m_train['r2']:.4f}", f"{m_train['mape']:.2f}",
                      f"{m_train.get('norm_mae', 0)*100:.2f}"],
            'Test': [f"{m_test['mae']:.4f}", f"{m_test['rmse']:.4f}",
                     f"{m_test['r2']:.4f}", f"{m_test['mape']:.2f}",
                     f"{m_test.get('norm_mae', 0)*100:.2f}"],
        })
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
        
        # ====== ROW 5: Error Analysis ======
        st.markdown("#### Error Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Error by Hour of Day
            try:
                df_test = results['df_test']
                test_indices = results['test_indices']
                abs_error_per_seq = np.abs(results['pv_test_actual'][:, 0] - results['pv_test_pred'][:, 0])
                hours_per_seq = df_test.index[test_indices].hour
                
                df_hourly = pd.DataFrame({
                    'Hour': hours_per_seq[:len(abs_error_per_seq)],
                    'MAE': abs_error_per_seq[:len(hours_per_seq)]
                })
                hourly_stats = df_hourly.groupby('Hour')['MAE'].agg(['mean', 'std']).reset_index()
                
                fig_hour = go.Figure()
                fig_hour.add_trace(go.Bar(
                    x=hourly_stats['Hour'], y=hourly_stats['mean'],
                    marker_color='#818cf8', opacity=0.8,
                    name='Mean |Error|',
                    error_y=dict(type='data', array=hourly_stats['std'].fillna(0), visible=True,
                                 color='#f472b6')
                ))
                fig_hour.update_layout(
                    title="MAE by Hour of Day (h+1)",
                    xaxis_title="Hour", yaxis_title="MAE (kW)",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    xaxis=dict(dtick=1),
                )
                st.plotly_chart(fig_hour, width='stretch')
            except Exception as e:
                st.caption(f"Hourly error chart tidak tersedia: {e}")
        
        with col2:
            # Error by Forecast Horizon Step
            try:
                horizon = results['pv_test_actual'].shape[1]
                mae_per_step = []
                for h in range(horizon):
                    step_actual = results['pv_test_actual'][:, h]
                    step_pred = results['pv_test_pred'][:, h]
                    # Filter productive hours
                    step_ghi = results['ghi_test'][:, h]
                    mask_h = step_ghi > 50
                    if mask_h.sum() > 0:
                        mae_h = np.mean(np.abs(step_actual[mask_h] - step_pred[mask_h]))
                    else:
                        mae_h = 0
                    mae_per_step.append(mae_h)
                
                fig_step = go.Figure()
                fig_step.add_trace(go.Scatter(
                    x=list(range(1, horizon + 1)), y=mae_per_step,
                    mode='lines+markers',
                    line=dict(color='#f472b6', width=2),
                    marker=dict(size=6, color='#f472b6'),
                    name='MAE'
                ))
                fig_step.update_layout(
                    title="MAE by Forecast Step (Productive Hours)",
                    xaxis_title="Forecast Step (h+n)", yaxis_title="MAE (kW)",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                )
                st.plotly_chart(fig_step, width='stretch')
            except Exception as e:
                st.caption(f"Forecast step chart tidak tersedia: {e}")
        
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


# --- TAB 4: TARGET TESTING ---
with tab4:
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


# --- TAB 5: LOGS ---
with tab5:
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
