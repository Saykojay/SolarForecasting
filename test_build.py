import os
import tensorflow as tf
from src.model_factory import build_model, get_custom_objects

def test_models():
    # Mock parameters
    lookback = 72
    n_features = 5
    forecast_horizon = 24

    hp_tracker = {
        'patch_len': 16,
        'stride': 8,
        'd_model': 32,
        'n_heads': 4,
        'n_layers': 2,
        'dropout': 0.1,
        'n_shared_experts': 1,
        'n_private_experts': 4,
        'top_k': 2
    }

    hp_perceiver = {
        'patch_len': 16,
        'stride': 8,
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'dropout': 0.1,
        'n_latent_tokens': 16
    }
    
    print("Testing TimeTracker Build...")
    try:
        model_tt = build_model('timetracker', lookback, n_features, forecast_horizon, hp_tracker)
        print(f"TimeTracker built successfully. Params: {model_tt.count_params()}")
    except Exception as e:
        print(f"TimeTracker error: {e}")
        
    print("\nTesting TimePerceiver Build...")
    try:
        model_tp = build_model('timeperceiver', lookback, n_features, forecast_horizon, hp_perceiver)
        print(f"TimePerceiver built successfully. Params: {model_tp.count_params()}")
    except Exception as e:
        print(f"TimePerceiver error: {e}")

if __name__ == "__main__":
    # Ensure custom objects are registered just in case tf needs it
    tf.keras.utils.get_custom_objects().update(get_custom_objects())
    test_models()
