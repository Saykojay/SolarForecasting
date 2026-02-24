import numpy as np
import logging
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import PatchTSTConfig, PatchTSTForPrediction, AutoformerConfig, AutoformerModel
import torch.nn as nn

logger = logging.getLogger(__name__)

def build_patchtst_hf(lookback, n_features, forecast_horizon, hp: dict):
    """
    Builds the original PatchTST model using Hugging Face Transformers.
    """
    
    d_model = hp.get('d_model', 64)
    n_layers = hp.get('n_layers', 2)
    n_heads = hp.get('n_heads', 4)
    patch_len = hp.get('patch_len', 16)
    stride = hp.get('stride', 8)
    dropout = hp.get('dropout', 0.2)
    
    # We must construct a config mapping our inputs to HF schema
    config = PatchTSTConfig(
        context_length=lookback,
        prediction_length=forecast_horizon,
        num_input_channels=n_features,
        patch_length=patch_len,
        patch_stride=stride,
        d_model=d_model,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        ffn_dim=hp.get('ff_dim', d_model*2),
        attention_dropout=dropout,
        ff_dropout=dropout,
        head_dropout=dropout,
        use_cache=False
    )
    
    model = PatchTSTForPrediction(config)
    return model

class CustomAutoformerForPrediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.autoformer = AutoformerModel(config)
        self.head = nn.Linear(config.d_model * config.context_length, config.prediction_length * config.input_size)
        
    def forward(self, past_values):
        # Generate dummy masks and time features to satisfy AutoformerModel signature
        batch_size = past_values.shape[0]
        device = past_values.device
        
        past_time_features = torch.zeros(batch_size, self.config.context_length, 0, device=device)
        future_time_features = torch.zeros(batch_size, self.config.prediction_length, 0, device=device)
        past_observed_mask = torch.ones(batch_size, self.config.context_length, device=device)
        
        # pass through Autoformer
        # cuFFT requires power-of-two sizes for half precision. 
        # We force float32 for the FFT part to support arbitrary lookback (like 72).
        with torch.amp.autocast('cuda', enabled=False):
            out = self.autoformer(
                past_values=past_values.float(),
                past_time_features=past_time_features.float(),
                past_observed_mask=past_observed_mask.float(),
                future_time_features=future_time_features.float()
            )
        
        hidden = out.encoder_last_hidden_state  # [B, T, d_model]
        hidden = hidden.contiguous().view(batch_size, -1)  # [B, T * d_model]
        
        preds = self.head(hidden)  # [B, Horizon * input_size]
        preds = preds.contiguous().view(batch_size, self.config.prediction_length, self.config.input_size)
        
        # Return object that mocks 'prediction_outputs' field of PatchTSTForPrediction
        class DummyOutput:
            def __init__(self, p):
                self.prediction_outputs = p
        return DummyOutput(preds)

def build_autoformer_hf(lookback, n_features, forecast_horizon, hp: dict):
    """
    Builds Autoformer with deterministic output head using Hugging Face AutoformerModel.
    """
    d_model = hp.get('d_model', 64)
    n_layers = hp.get('n_layers', 2)
    n_heads = hp.get('n_heads', 4)
    dropout = hp.get('dropout', 0.1)
    
    config = AutoformerConfig(
        context_length=lookback,
        prediction_length=forecast_horizon,
        input_size=n_features,
        num_time_features=0,
        d_model=d_model,
        encoder_layers=n_layers,
        decoder_layers=n_layers,
        encoder_attention_heads=n_heads,
        decoder_attention_heads=n_heads,
        encoder_ffn_dim=hp.get('ff_dim', d_model*2),
        decoder_ffn_dim=hp.get('ff_dim', d_model*2),
        dropout=dropout,
        attention_dropout=dropout,
        activation_dropout=dropout,
        moving_avg=hp.get('moving_avg', 25),
        lags_sequence=[0],
        label_length=0,
        scaling=False,
    )
    model = CustomAutoformerForPrediction(config)
    return model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x

class CausalTransformerForPrediction(nn.Module):
    def __init__(self, lookback, n_features, forecast_horizon, d_model=64, n_heads=4, n_layers=2, dropout=0.1, ff_dim=128):
        super().__init__()
        self.lookback = lookback
        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        
        # 1. Project input features to d_model space
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=lookback)
        
        # 2. Decoder-Only Transformer Structure
        # Note: PyTorch TransformerEncoder with a Causal Mask acts as a Decoder-Only model (like GPT).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. Prediction Head
        # Flattens the causal hidden states and predicts the horizon
        self.head = nn.Linear(lookback * d_model, forecast_horizon * n_features)

    def _generate_square_subsequent_mask(self, sz):
        # Generates a causal mask for the transformer
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, past_values):
        # past_values: [Batch, Lookback, Features]
        batch_size = past_values.shape[0]
        device = past_values.device
        
        # Linear map + Positional Encoding
        x = self.input_projection(past_values)  # [B, L, d_model]
        
        # PositionalEncoding expects [Seq, Batch, d_model]
        x = x.transpose(0, 1) # [L, B, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # [B, L, d_model]
        
        # Generate causal mask to prevent looking into the future mathematically
        causal_mask = self._generate_square_subsequent_mask(self.lookback).to(device)
        
        # Pass through Transformer blocks
        hidden_states = self.transformer(x, mask=causal_mask, is_causal=True) # [B, L, d_model]
        
        # Flatten and predict
        hidden_states = hidden_states.contiguous().view(batch_size, -1) # [B, L * d_model]
        preds = self.head(hidden_states) # [B, Horizon * Features]
        
        # Reshape to expected [B, Horizon, Features]
        preds = preds.contiguous().view(batch_size, self.forecast_horizon, self.n_features)
        
        # Return mocked HF Output object
        class DummyOutput:
            def __init__(self, p):
                self.prediction_outputs = p
        return DummyOutput(preds)

def build_causal_transformer_hf(lookback, n_features, forecast_horizon, hp: dict):
    """
    Builds the PyTorch Causal / Decoder-Only Transformer that is trainable from scratch.
    """
    d_model = hp.get('d_model', 64)
    n_layers = hp.get('n_layers', 2)
    n_heads = hp.get('n_heads', 4)
    dropout = hp.get('dropout', 0.1)
    ff_dim = hp.get('ff_dim', d_model*2)
    
    model = CausalTransformerForPrediction(
        lookback=lookback,
        n_features=n_features,
        forecast_horizon=forecast_horizon,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        ff_dim=ff_dim
    )
    return model

def train_eval_pytorch_model(model, X_train, y_train, X_val, y_val, hp, patience=10, callbacks=None, trial=None, verbose=True):
    """
    Custom PyTorch training loop to replicate Keras-like training process.
    Included Optuna pruning and EarlyStopping logic.
    Returns a dummy Keras-like History object containing 'val_loss'.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if len(y_train.shape) == 2:
        # Multi-to-one (Target is not 3D). PyTorch needs matching dimensions.
        y_train_torch = np.expand_dims(y_train, axis=-1)
        y_val_torch = np.expand_dims(y_val, axis=-1)
    else:
        y_train_torch = y_train
        y_val_torch = y_val
        
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_torch, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_torch, dtype=torch.float32)
    
    batch_size = hp.get('batch_size', 32)
    epochs = hp.get('epochs', 50)
    lr = hp.get('learning_rate', 0.0001)
    patience = hp.get('patience', 10)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Mixed Precision Setup for massive speedup and memory efficiency
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    class DummyHistory:
        def __init__(self):
            self.history = {'val_loss': []}
            
    history = DummyHistory()
    best_val_loss = float('inf')
    patience_counter = 0
    
    if callbacks:
        for cb in callbacks:
            if hasattr(cb, 'on_train_begin'):
                cb.on_train_begin()
                
    for epoch in range(epochs):
        model.train()
        train_losses = []
        total_steps = len(train_dl)
        
        # Cetak info mulai epoch
        if verbose:
            print(f"\n▶️ Dimulai: Epoch {epoch+1}/{epochs} (Total Batches: {total_steps})", flush=True)
        
        for step, (xb, yb) in enumerate(train_dl):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            # Predict using Mixed Precision Autocast
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(past_values=xb)
                preds = outputs.prediction_outputs # shape (B, Horizon, Channels)
                
                # Sub-slice channel target logic if model outputs all channels but supervision is Multi-to-one
                if yb.shape[-1] == 1 and preds.shape[-1] > 1:
                    preds = preds[:, :, 0:1] # Assumes Target variable is always channel 0 in dataset
                    
                loss = torch.nn.functional.huber_loss(preds, yb) if hp.get('loss') == 'huber' else torch.nn.functional.mse_loss(preds, yb)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            
            # Print update every 20 steps or at the end
            if verbose and ((step + 1) % 20 == 0 or (step + 1) == total_steps):
                print(f"   [Batch {step+1:03d}/{total_steps:03d}] Loss sementara: {loss.item():.5f}", flush=True)
            
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(past_values=xb)
                preds = outputs.prediction_outputs
                
                if yb.shape[-1] == 1 and preds.shape[-1] > 1:
                    preds = preds[:, :, 0:1] 
                    
                loss = torch.nn.functional.huber_loss(preds, yb) if hp.get('loss') == 'huber' else torch.nn.functional.mse_loss(preds, yb)
                val_losses.append(loss.item())
                
        val_loss = np.mean(val_losses)
        train_loss_mean = np.mean(train_losses)
        history.history['val_loss'].append(val_loss)
        
        if 'loss' not in history.history:
            history.history['loss'] = []
        history.history['loss'].append(train_loss_mean)
        
        # Real-time console log printed to terminal
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss_mean:.4f} - val_loss: {val_loss:.4f}", flush=True)
        
        # Trigger Keras-like Callbacks to update Streamlit UI
        if callbacks:
            logs = {'loss': train_loss_mean, 'val_loss': val_loss}
            for cb in callbacks:
                if hasattr(cb, 'on_epoch_end'):
                    cb.on_epoch_end(epoch, logs)
                    
        # Clear VRAM cache slightly to prevent fragmentation OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.exceptions.TrialPruned()
                
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[PyTorch] EarlyStopping triggers at epoch {epoch+1}")
                break
                
    return history, model


class HFModelWrapper:
    def __init__(self, model):
        self.model = model
        # Expose properties to mock Keras model
        if hasattr(model, 'config'):
            self.output_shape = (None, model.config.prediction_length)
            self.input_shape = (None, model.config.context_length, getattr(model.config, 'num_input_channels', getattr(model.config, 'input_size', 1)))
        else:
            # For custom models like CausalTransformer
            self.output_shape = (None, model.forecast_horizon)
            self.input_shape = (None, model.lookback, model.n_features)
        self.name = "hf_model_wrapper"

    def predict(self, dataset_or_array, verbose=0):
        import torch
        import numpy as np
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        preds = []
        
        # Check if it's a tf.data.Dataset
        if hasattr(dataset_or_array, 'take'):
            for batch in dataset_or_array:
                # batch is a tf tensor
                xb = torch.tensor(batch.numpy(), dtype=torch.float32).to(device)
                with torch.no_grad():
                    out = self.model(past_values=xb).prediction_outputs
                    if out.shape[-1] > 1:
                        out = out[:,:,0:1] # Default assume multi-to-one
                preds.append(out.cpu().numpy())
            return np.vstack(preds)
        else:
            # Process numpy array in batches to prevent Memory OOM and CUBLAS execution errors
            batch_size = 32
            out_list = []
            for i in range(0, len(dataset_or_array), batch_size):
                xb = torch.tensor(dataset_or_array[i:i+batch_size], dtype=torch.float32).to(device)
                with torch.no_grad():
                    out = self.model(past_values=xb).prediction_outputs
                    if hasattr(out, 'prediction_outputs'):
                         out = out.prediction_outputs
                    if out.shape[-1] > 1:
                        out = out[:,:,0:1]
                out_list.append(out.cpu().numpy())
                
                del xb, out
            
            return np.vstack(out_list) if out_list else np.array([])

def load_hf_wrapper(model_path):
    """
    Robust loader for PyTorch-based models (PatchTST, Autoformer, CausalTransformer).
    """
    import os
    import json
    import torch
    
    # meta.json is usually inside the bundle folder or the parent of model_hf
    arch = "unknown"
    meta_candidates = [
        os.path.join(model_path, "meta.json"),
        os.path.join(os.path.dirname(model_path), "meta.json")
    ]
    meta_content = {}
    for mc in meta_candidates:
        if os.path.exists(mc):
            try:
                with open(mc, 'r') as f:
                    meta_content = json.load(f)
                    arch = meta_content.get('architecture', 'unknown').lower()
                    break
            except Exception: pass
    
    # 1. HF Native models (PatchTST)
    if arch == 'patchtst_hf' or (arch == 'unknown' and os.path.exists(os.path.join(model_path, "config.json"))):
        try:
            from transformers import PatchTSTForPrediction
            model = PatchTSTForPrediction.from_pretrained(model_path)
        except Exception as e:
            # If standard from_pretrained fails (e.g. torch vulnerability check), try manual reconstruction if we have meta
            if arch == 'patchtst_hf' and meta_content:
                from transformers import PatchTSTConfig
                hp = meta_content.get('hyperparameters', {})
                config = PatchTSTConfig(
                    context_length=meta_content.get('lookback', hp.get('lookback', 72)),
                    prediction_length=meta_content.get('forecast_horizon', 24),
                    num_input_channels=meta_content.get('n_features', 1),
                    patch_length=hp.get('patch_len', 16),
                    patch_stride=hp.get('stride', 8),
                    d_model=hp.get('d_model', 64),
                    num_hidden_layers=hp.get('n_layers', 2),
                    num_attention_heads=hp.get('n_heads', 4),
                    ffn_dim=hp.get('ff_dim', 128),
                    use_cache=False
                )
                model = PatchTSTForPrediction(config)
                bin_path = os.path.join(model_path, "pytorch_model.bin")
                if not os.path.exists(bin_path):
                    bin_path = os.path.join(os.path.dirname(model_path), "pytorch_model.bin")
                state_dict = torch.load(bin_path, map_location='cpu')
                model.load_state_dict(state_dict)
            else:
                raise e
    
    # 2. Custom HF Wrappers (Autoformer)
    elif arch == 'autoformer_hf':
        from transformers import AutoformerConfig
        hp = meta_content.get('hyperparameters', {})
        lookback = meta_content.get('lookback', hp.get('lookback', 72))
        horizon = meta_content.get('forecast_horizon', 24)
        n_features = meta_content.get('n_features', 1)
        
        # MUST match the build_autoformer_hf exactly
        config = AutoformerConfig(
            context_length=lookback,
            prediction_length=horizon,
            input_size=n_features,
            num_time_features=0,
            d_model=hp.get('d_model', 128),
            encoder_layers=hp.get('n_layers', 3),
            decoder_layers=hp.get('n_layers', 3),
            encoder_attention_heads=hp.get('n_heads', 16),
            decoder_attention_heads=hp.get('n_heads', 16),
            encoder_ffn_dim=hp.get('ff_dim', 256),
            decoder_ffn_dim=hp.get('ff_dim', 256),
            dropout=hp.get('dropout', 0.2),
            moving_avg=hp.get('moving_avg', 25),
            lags_sequence=[0], # CRITICAL to match dimension 3*input_size
            label_length=0,
            scaling=False,
        )
        
        model = CustomAutoformerForPrediction(config)
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(bin_path):
            bin_path = os.path.join(os.path.dirname(model_path), "pytorch_model.bin")
        
        state_dict = torch.load(bin_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    # 2. Custom PyTorch models (CausalTransformer)
    elif arch == 'causal_transformer_hf' or 'causal' in arch:
        checkpoint_candidates = [
            os.path.join(model_path, "pytorch_model.bin"),
            os.path.join(os.path.dirname(model_path), "pytorch_model.bin"),
            os.path.join(model_path, "model.pt")
        ]
        checkpoint_path = None
        for cc in checkpoint_candidates:
            if os.path.exists(cc):
                checkpoint_path = cc
                break
                
        if checkpoint_path and meta_content:
            hp = meta_content.get('hyperparameters', {})
            lookback = meta_content.get('lookback', hp.get('lookback', 336))
            n_features = meta_content.get('n_features', 1)
            horizon = meta_content.get('forecast_horizon', 24)
            
            model = build_causal_transformer_hf(lookback, n_features, horizon, hp)
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        else:
            raise ValueError(f"Missing checkpoint or metadata for CausalTransformer at {model_path}")
    else:
        raise ValueError(f"Cannot determine how to load HF model at {model_path} (Arch: {arch})")
            
    return HFModelWrapper(model)
