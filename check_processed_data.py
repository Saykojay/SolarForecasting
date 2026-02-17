import numpy as np
import os
import json
import joblib

proc_dir = 'data/processed'
# Find the latest version folder
versions = [f for f in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, f)) and f.startswith('version_')]
if versions:
    proc_dir = os.path.join(proc_dir, sorted(versions, reverse=True)[0])

print(f"Checking data in: {proc_dir}")

X_train = np.load(os.path.join(proc_dir, 'X_train.npy'))
y_train = np.load(os.path.join(proc_dir, 'y_train.npy'))
with open(os.path.join(proc_dir, 'prep_summary.json'), 'r') as f:
    summary = json.load(f)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_train NaN count: {np.isnan(X_train).sum()}")
print(f"y_train NaN count: {np.isnan(y_train).sum()}")
print(f"y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
print(f"y_train mean: {y_train.mean():.4f}")

# Check features
selected_features = summary['selected_features']
print(f"Selected features ({len(selected_features)}): {selected_features}")

# Check first 5 samples of y_train
print(f"First 10 y_train values: {y_train[:10]}")

# Check if target is all zeros
if np.all(y_train == 0):
    print("!!! ERROR: Target is all zeros !!!")

# Check GHI feature (if exists)
if 'ghi_wm2' in selected_features:
    idx = selected_features.index('ghi_wm2')
    ghi_slice = X_train[:, :, idx]
    print(f"GHI range in X_train: [{ghi_slice.min():.4f}, {ghi_slice.max():.4f}]")
    print(f"GHI mean: {ghi_slice.mean():.4f}")
