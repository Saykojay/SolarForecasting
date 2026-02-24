import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_patchtst_hf(lookback, n_features, forecast_horizon, hp: dict):
    """
    Builds the original PatchTST model using Hugging Face Transformers.
    """
    import torch
    from transformers import PatchTSTConfig, PatchTSTForPrediction
    
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

def train_eval_pytorch_model(model, X_train, y_train, X_val, y_val, hp, trial=None):
    """
    Custom PyTorch training loop to replicate Keras-like training process.
    Included Optuna pruning and EarlyStopping logic.
    Returns a dummy Keras-like History object containing 'val_loss'.
    """
    import torch
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
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
    
    class DummyHistory:
        def __init__(self):
            self.history = {'val_loss': []}
            
    history = DummyHistory()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            # Predict
            outputs = model(past_values=xb)
            preds = outputs.prediction_outputs # shape (B, Horizon, Channels)
            
            # Sub-slice channel target logic if model outputs all channels but supervision is Multi-to-one
            if yb.shape[-1] == 1 and preds.shape[-1] > 1:
                preds = preds[:, :, 0:1] # Assumes Target variable is always channel 0 in dataset
                
            loss = torch.nn.functional.huber_loss(preds, yb) if hp.get('loss') == 'huber' else torch.nn.functional.mse_loss(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
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
        history.history['val_loss'].append(val_loss)
        
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
        self.output_shape = (None, model.config.prediction_length)
        self.input_shape = (None, model.config.context_length, model.config.num_input_channels)
        self.name = "patchtst_hf"

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
            xb = torch.tensor(dataset_or_array, dtype=torch.float32).to(device)
            with torch.no_grad():
                out = self.model(past_values=xb).prediction_outputs
                if out.shape[-1] > 1:
                    out = out[:,:,0:1]
            return out.cpu().numpy()

def load_hf_wrapper(model_path):
    from transformers import PatchTSTForPrediction
    model = PatchTSTForPrediction.from_pretrained(model_path)
    return HFModelWrapper(model)
