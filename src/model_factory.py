"""
model_factory.py - Factory untuk membangun arsitektur model (PatchTST, GRU, dll).
Mendukung multiple architectures agar bisa dipilih dari config.yaml.
"""
import tensorflow as tf
from keras.layers import (Dense, Dropout, Input, LayerNormalization,
                          MultiHeadAttention, GRU as GRULayer,
                          GlobalAveragePooling1D, Bidirectional)
from keras.optimizers import Adam
import logging

logger = logging.getLogger(__name__)


# ============================================================
# PATCHTST CUSTOM LAYERS
# ============================================================
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_len, stride, d_model, **kwargs):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.patch_conv = tf.keras.layers.Conv1D(
            filters=d_model, kernel_size=patch_len,
            strides=stride, padding='valid', name='patch_projection'
        )

    def call(self, x):
        return self.patch_conv(x)

    def get_config(self):
        config = super().get_config()
        config.update({"patch_len": self.patch_len, "stride": self.stride, "d_model": self.d_model})
        return config


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name='pos_embedding', shape=(self.max_len, self.d_model),
            initializer='uniform', trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_embedding[:seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.mha = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='gelu'),
            Dense(d_model)
        ])
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, x, training=False):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.ln1(x + attn_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.ln2(x + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_heads": self.n_heads,
                        "ff_dim": self.ff_dim, "dropout": self.dropout_rate})
        return config


# ============================================================
# MODEL BUILDERS
# ============================================================
def build_patchtst(lookback, n_features, forecast_horizon, hp: dict):
    """Membangun model PatchTST."""
    patch_len = hp['patch_len']
    stride = hp['stride']
    d_model = hp['d_model']
    n_heads = hp['n_heads']
    n_layers = hp['n_layers']
    dropout = hp['dropout']

    inputs = Input(shape=(lookback, n_features), name='main_input')
    x = PatchEmbedding(patch_len, stride, d_model, name='patch_embedding')(inputs)
    num_patches = (lookback - patch_len) // stride + 1
    x = PositionalEncoding(max_len=num_patches + 10, d_model=d_model, name='positional_encoding')(x)
    for i in range(n_layers):
        x = TransformerEncoderBlock(d_model, n_heads, ff_dim=d_model * 4,
                                     dropout=dropout, name=f'encoder_block_{i}')(x)
    x = tf.keras.layers.Flatten(name='flatten_head')(x)
    x = Dense(128, activation='relu', name='dense_head')(x)
    x = Dropout(dropout, name='dropout_head')(x)
    outputs = Dense(forecast_horizon, activation='linear', name='output_layer')(x)
    return tf.keras.Model(inputs, outputs, name='PatchTST')


def build_gru(lookback, n_features, forecast_horizon, hp: dict):
    """Membangun model GRU (Gated Recurrent Unit)."""
    d_model = hp.get('d_model', 64)
    n_layers = hp.get('n_layers', 2)
    dropout = hp.get('dropout', 0.2)

    inputs = Input(shape=(lookback, n_features), name='main_input')
    x = inputs
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        x = Bidirectional(GRULayer(d_model, return_sequences=return_seq,
                                    dropout=dropout, name=f'gru_{i}'), name=f'bi_gru_{i}')(x)
    x = Dense(128, activation='relu', name='dense_head')(x)
    x = Dropout(dropout, name='dropout_head')(x)
    outputs = Dense(forecast_horizon, activation='linear', name='output_layer')(x)
    return tf.keras.Model(inputs, outputs, name='GRU_Model')


# ============================================================
# FACTORY FUNCTION
# ============================================================
MODEL_REGISTRY = {
    'patchtst': build_patchtst,
    'gru': build_gru,
}


def build_model(architecture: str, lookback: int, n_features: int,
                forecast_horizon: int, hp: dict) -> tf.keras.Model:
    """
    Factory function — membangun model sesuai arsitektur yang dipilih.
    Pilihan arsitektur: 'patchtst', 'gru'.
    """
    arch = architecture.lower()
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Arsitektur '{arch}' tidak dikenal. Pilihan: {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[arch](lookback, n_features, forecast_horizon, hp)
    logger.info(f"Model '{arch}' dibangun: params={model.count_params():,}")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float):
    """Compile model dengan optimizer dan loss standard."""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.MeanAbsoluteError()
    )
    return model


def get_custom_objects():
    """Mengembalikan custom objects untuk loading model yang disimpan."""
    return {
        'PatchEmbedding': PatchEmbedding,
        'PositionalEncoding': PositionalEncoding,
        'TransformerEncoderBlock': TransformerEncoderBlock,
    }
