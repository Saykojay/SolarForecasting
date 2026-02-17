import pandas as pd
import numpy as np
import os
import json

proc_dir = 'data/processed'
# Find the latest version folder
versions = [f for f in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, f)) and f.startswith('version_')]
if versions:
    proc_dir = os.path.join(proc_dir, sorted(versions, reverse=True)[0])

print(f"--- Diagnosing Data Learnability in: {proc_dir} ---")

# Load features and targets
X_train = np.load(os.path.join(proc_dir, 'X_train.npy'))
y_train = np.load(os.path.join(proc_dir, 'y_train.npy'))
X_test = np.load(os.path.join(proc_dir, 'X_test.npy'))
y_test = np.load(os.path.join(proc_dir, 'y_test.npy'))

with open(os.path.join(proc_dir, 'prep_summary.json'), 'r') as f:
    summary = json.load(f)

# Flatten X to correlate last step of history with next 1h target
# Assuming y_train is (N, horizon), we correlate with y_train[:, 0]
features = summary['selected_features']
target_1h = y_train[:, 0]

# Take the last time step in the lookback window as the 'current' state
X_current = X_train[:, -1, :]

print("\nKorelasi Fitur 'Saat Ini' terhadap Target (Next 1 Hour):")
corrs = []
for i, feat in enumerate(features):
    c = np.corrcoef(X_current[:, i], target_1h)[0, 1]
    corrs.append((feat, c))

corrs.sort(key=lambda x: abs(x[1]), reverse=True)
for f, c in corrs[:15]:
    print(f"  {f:25}: {c:.4f}")

# Check Distribution Shift
print("\nAnalisis Stabilitas Data (Train vs Test):")
y_train_mean, y_train_std = y_train.mean(), y_train.std()
y_test_mean, y_test_std = y_test.mean(), y_test.std()

print(f"  Target Mean (Train): {y_train_mean:.4f} | (Test): {y_test_mean:.4f}")
print(f"  Target Std  (Train): {y_train_std:.4f} | (Test): {y_test_std:.4f}")

diff_p = abs(y_train_mean - y_test_mean) / (y_train_mean + 1e-6) * 100
print(f"  Perbedaan Mean: {diff_p:.2f}%")

if diff_p > 20:
    print("\n⚠️ PERINGATAN: Ada perbedaan distribusi besar antara Train & Test. Ini sebabnya val_loss tinggi!")
else:
    print("\n✅ Distribusi Stabil. Data secara teori bisa dipelajari.")

if abs(corrs[0][1]) < 0.5:
    print("⚠️ PERINGATAN: Korelasi fitur terkuat rendah. Model akan kesulitan memprediksi.")
else:
    print(f"✅ Sinyal Kuat: Fitur '{corrs[0][0]}' punya korelasi {corrs[0][1]:.4f}")
