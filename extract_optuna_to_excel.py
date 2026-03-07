"""
Extract Optuna tuning_gru study dari database SQLite ke file Excel.
"""
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = 'optuna_history.db'
conn = sqlite3.connect(DB_PATH)

# ============================================================
# 1. Ambil semua trials
# ============================================================
trials = pd.read_sql_query("""
    SELECT 
        t.trial_id,
        t.number AS trial_number,
        t.state,
        t.datetime_start,
        t.datetime_complete
    FROM trials t
    JOIN studies s ON t.study_id = s.study_id
    WHERE s.study_name = 'tuning_gru'
    ORDER BY t.number ASC
""", conn)

# ============================================================
# 2. Ambil values (objective value / val_loss) per trial
# ============================================================
values = pd.read_sql_query("""
    SELECT 
        tv.trial_id,
        tv.value AS val_loss
    FROM trial_values tv
""", conn)

# ============================================================
# 3. Ambil semua hyperparameters per trial (pivot)
# ============================================================
params = pd.read_sql_query("""
    SELECT 
        tp.trial_id,
        tp.param_name,
        tp.param_value
    FROM trial_params tp
    JOIN trials t ON tp.trial_id = t.trial_id
    JOIN studies s ON t.study_id = s.study_id
    WHERE s.study_name = 'tuning_gru'
""", conn)

# Pivot params: each param_name becomes a column
params_pivot = params.pivot(index='trial_id', columns='param_name', values='param_value').reset_index()

# ============================================================
# 4. Merge everything
# ============================================================
df = trials.merge(values, on='trial_id', how='left')
df = df.merge(params_pivot, on='trial_id', how='left')

# Calculate duration in minutes
df['datetime_start'] = pd.to_datetime(df['datetime_start'])
df['datetime_complete'] = pd.to_datetime(df['datetime_complete'])
df['duration_minutes'] = (df['datetime_complete'] - df['datetime_start']).dt.total_seconds() / 60.0
df['duration_minutes'] = df['duration_minutes'].round(2)

# Reorder columns nicely
id_cols = ['trial_number', 'state', 'val_loss', 'duration_minutes', 'datetime_start', 'datetime_complete']
param_cols = [c for c in df.columns if c not in id_cols + ['trial_id']]
df = df[id_cols + sorted(param_cols)]

# Sort: best val_loss first for COMPLETE trials
df_complete = df[df['state'] == 'COMPLETE'].sort_values('val_loss', ascending=True)
df_pruned = df[df['state'] == 'PRUNED'].sort_values('trial_number', ascending=True)
df_other = df[~df['state'].isin(['COMPLETE', 'PRUNED'])].sort_values('trial_number', ascending=True)

# ============================================================
# 5. Save to Excel with multiple sheets
# ============================================================
timestamp = datetime.now().strftime('%y%m%d_%H%M')
excel_file = f'tuning_gru_history_{timestamp}.xlsx'

with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    # Sheet 1: All trials sorted by trial number
    df.sort_values('trial_number').to_excel(writer, sheet_name='All Trials', index=False)
    
    # Sheet 2: Only COMPLETE trials, sorted by best val_loss
    df_complete.to_excel(writer, sheet_name='Best Trials (Complete)', index=False)
    
    # Sheet 3: Summary statistics
    summary_data = {
        'Metric': [
            'Total Trials',
            'Completed Trials',
            'Pruned Trials',
            'Best val_loss',
            'Best Trial #',
            'Avg val_loss (Complete)',
            'Std val_loss (Complete)',
            'Avg Duration (min, Complete)',
            'Avg Duration (min, Pruned)',
            'Total Duration (hours)',
        ],
        'Value': [
            len(df),
            len(df_complete),
            len(df_pruned),
            df_complete['val_loss'].min() if not df_complete.empty else 'N/A',
            int(df_complete['trial_number'].iloc[0]) if not df_complete.empty else 'N/A',
            round(df_complete['val_loss'].mean(), 6) if not df_complete.empty else 'N/A',
            round(df_complete['val_loss'].std(), 6) if not df_complete.empty else 'N/A',
            round(df_complete['duration_minutes'].mean(), 2) if not df_complete.empty else 'N/A',
            round(df_pruned['duration_minutes'].mean(), 2) if not df_pruned.empty else 'N/A',
            round(df['duration_minutes'].sum() / 60.0, 2),
        ]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

conn.close()

print(f"[DONE] Database berhasil diekstrak ke: {excel_file}")
print(f"  - Sheet 'All Trials': {len(df)} trials")
print(f"  - Sheet 'Best Trials (Complete)': {len(df_complete)} trials")
print(f"  - Sheet 'Summary': Ringkasan statistik")
