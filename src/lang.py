
# src/lang.py - Labels for PV Forecasting Dashboard (English only)

TR = {
    'page_title': "PV Forecasting Pipeline",
    'sidebar_settings': "Global Settings",
    'data_insights': "Data Insights",
    'data_prep': "Data Prep & Features",
    'training_center': "Training Center",
    'batch_experiments': "Batch Experiments",
    'optuna_tuning': "Optuna Tuning",
    'prediction_eval': "Prediction / Eval",
    'model_comparison': "Model Comparison",
    'target_testing': "Target Testing",
    'status_ready': "Ready",
    'status_missing': "Missing",
    'status_present': "Present",
    'data_preprocessed': "Data Preprocessed",
    'model_trained': "Model Trained",
    'target_data': "Target Data",
    'active_arch': "Active Architecture",
    'not_ready': "Not Ready",
    'start_prep': "Start Preprocessing",
    'start_train': "Start Training",
    'run_eval': "Run Evaluation",
    'run_batch': "Run Batch Training Now",
    'add_to_queue': "Add to Queue",
    'current_queue': "Current Queue",
    'hp_config': "Model Architecture & Hyperparameters",
    'batch_manager_title': "Batch Experiment Manager",
    'batch_info': "Build a training queue for various models and run them all automatically.",
    'add_exp': "Add to Queue",
    'exp_name': "Experiment Name",
    'architecture': "Architecture",
    'config_hp': "Configure Hyperparameters",
    'config_data_feat': "Configure Data & Features",
    'data_version_select': "Preprocessed Data Version",
    'feature_groups_info': "Feature Groups (Only if 'Current (Dynamic)' is selected)",
    'current_dynamic': "Current (Dynamic)",
    'run_batch_btn': "Run Batch Now",
    'add_to_queue_success': "Added to queue!",
}

def gt(key, lang=None):
    """Get label text for a key (English only)."""
    return TR.get(key, key)
