import os, json
model_dir = 'models'
for d in sorted(os.listdir(model_dir)):
    dpath = os.path.join(model_dir, d)
    if not os.path.isdir(dpath): continue
    meta = os.path.join(dpath, 'meta.json')
    if not os.path.exists(meta): continue
    m = json.load(open(meta))
    hp = m.get('hyperparameters', {})
    print(f"{d}: feat={m.get('n_features','?')}, val_loss={m.get('val_loss',999):.6f}, "
          f"d_model={hp.get('d_model','?')}, heads={hp.get('n_heads','?')}, "
          f"layers={hp.get('n_layers','?')}, patch={hp.get('patch_len','?')}, "
          f"stride={hp.get('stride','?')}, lb={hp.get('lookback','?')}, "
          f"bs={hp.get('batch_size','?')}, drop={hp.get('dropout','?')}, "
          f"lr={hp.get('learning_rate','?')}, epochs={m.get('epochs_trained',0)}")
