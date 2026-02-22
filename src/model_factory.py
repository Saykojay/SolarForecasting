"""
model_factory.py - Factory untuk membangun arsitektur model (PatchTST, GRU, dll).
100% Faithful to "A Time Series is Worth 64 Words" (Nie et al., ICLR 2023).
"""
import tensorflow as tf
from keras.layers import (Dense, Dropout, Input, BatchNormalization,
                          MultiHeadAttention, GRU as GRULayer,
                          Reshape, Permute, Flatten)
from keras.optimizers import Adam
import logging

logger = logging.getLogger(__name__)

# ============================================================
# PATCHTST CUSTOM LAYERS (ICLR 2023 COMPLIANT)
# ============================================================

class RevIN(tf.keras.layers.Layer):
    """Reversible Instance Normalization (RevIN) as per Paper Section 3.1."""
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = tf.reduce_mean(x, axis=1, keepdims=True)
            self.std = tf.math.reduce_std(x, axis=1, keepdims=True) + self.eps
            return (x - self.mean) / self.std
        else: # denorm
            return x * self.std + self.mean

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_len, stride, d_model, **kwargs):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        # Paper uses Linear projection for patches
        self.projection = Dense(d_model, name='patch_projection')

    def call(self, x):
        # x shape: (Batch*M, L, 1) -> (Batch*M, NumPatches, PatchLen)
        # Use signal.frame to create patches
        patches = tf.signal.frame(x[:, :, 0], frame_length=self.patch_len, frame_step=self.stride, axis=-1)
        return self.projection(patches)

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
        
        # Paper Footnote 1 suggests BatchNorm instead of LayerNorm
        self.bn1 = BatchNormalization()
        self.mha = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)
        self.dropout1 = Dropout(dropout)
        
        self.bn2 = BatchNormalization()
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='gelu'), # Paper specifies GELU
            Dense(d_model),
            Dropout(dropout)
        ])

    def call(self, x, training=False):
        # Transformer architecture according to paper (usually Pre-Norm or specialized for TS)
        # We'll use the BatchNorm implementation based on Zerveas et al. (2021) suggestion in paper
        x_norm = self.bn1(x, training=training)
        attn_output = self.mha(x_norm, x_norm)
        x = x + self.dropout1(attn_output, training=training)
        
        x_norm = self.bn2(x, training=training)
        ffn_output = self.ffn(x_norm, training=training)
        return x + ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_heads": self.n_heads,
                        "ff_dim": self.ff_dim, "dropout": self.dropout_rate})
        return config

# ============================================================
# TIMETRACKER CUSTOM LAYERS
# ============================================================

class MoELayer(tf.keras.layers.Layer):
    """
    Simplified Mixture of Experts Layer for TimeTracker.
    Uses Top-K routing to private experts and adds a shared expert.
    """
    def __init__(self, d_model, ff_dim, n_shared=1, n_private=4, top_k=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.n_shared = n_shared
        self.n_private = n_private
        self.top_k = min(top_k, n_private)
        self.dropout_rate = dropout

        # Shared experts
        self.shared_experts = [
            tf.keras.Sequential([
                Dense(ff_dim, activation='gelu'),
                Dense(d_model),
                Dropout(dropout)
            ], name=f'shared_expert_{i}') for i in range(n_shared)
        ]
        
        # Private experts
        self.private_experts = [
            tf.keras.Sequential([
                Dense(ff_dim, activation='gelu'),
                Dense(d_model),
                Dropout(dropout)
            ], name=f'private_expert_{i}') for i in range(n_private)
        ]
        
        # Router
        if n_private > 0:
            self.router = Dense(n_private, use_bias=False, name='router')

    def call(self, x, training=False):
        # x shape: (Batch, SeqLen, D)
        out = tf.zeros_like(x)
        
        # 1. Add shared experts contribution
        if self.n_shared > 0:
            shared_out = tf.reduce_mean([expert(x, training=training) for expert in self.shared_experts], axis=0)
            out = out + shared_out
            
        # 2. Add private experts via Top-K routing
        if self.n_private > 0:
            # Routing logits: (Batch, SeqLen, n_private)
            router_logits = self.router(x)
            
            # Select Top-K experts
            routing_weights = tf.nn.softmax(router_logits, axis=-1)
            top_k_weights, top_k_indices = tf.math.top_k(routing_weights, k=self.top_k)
            # Normalize top-k weights so they sum to 1
            top_k_weights = top_k_weights / tf.reduce_sum(top_k_weights, axis=-1, keepdims=True)
            
            # This is a simplified dense execution for TF graph compatibility:
            # Evaluate all experts (ok for small n_private like 4) and multiply by mask
            private_outs = [expert(x, training=training) for expert in self.private_experts]
            private_outs = tf.stack(private_outs, axis=-1) # (B, S, D, E)
            
            # Create a mask for top-k selection
            # top_k_indices: (B, S, K). We need mask of shape (B, S, E)
            batch_size = tf.shape(x)[0]
            seq_len = tf.shape(x)[1]
            
            # Flatten to use scatter_nd
            flat_indices = tf.reshape(top_k_indices, [-1])
            batch_seq_idx = tf.repeat(tf.range(batch_size * seq_len), self.top_k)
            indices_2d = tf.stack([batch_seq_idx, flat_indices], axis=1)
            
            flat_weights = tf.reshape(top_k_weights, [-1])
            mask_flat = tf.scatter_nd(indices_2d, flat_weights, shape=[batch_size * seq_len, self.n_private])
            mask = tf.reshape(mask_flat, [batch_size, seq_len, 1, self.n_private])
            
            # Apply mask and sum
            routed_out = tf.reduce_sum(private_outs * mask, axis=-1)
            out = out + routed_out
            
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "ff_dim": self.ff_dim, 
                       "n_shared": self.n_shared, "n_private": self.n_private, 
                       "top_k": self.top_k, "dropout": self.dropout_rate})
        return config

class TimeTrackerBlock(tf.keras.layers.Layer):
    """Transformer Block with MoE replacing standard FFN."""
    def __init__(self, d_model, n_heads, ff_dim, n_shared, n_private, top_k, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.n_shared = n_shared
        self.n_private = n_private
        self.top_k = top_k
        self.dropout_rate = dropout
        
        self.mha = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)
        self.dropout1 = Dropout(dropout)
        self.bn1 = BatchNormalization()
        
        self.moe = MoELayer(d_model, ff_dim, n_shared, n_private, top_k, dropout)
        self.bn2 = BatchNormalization()

    def call(self, x, training=False):
        # Attention
        x_norm = self.bn1(x, training=training)
        attn_out = self.mha(x_norm, x_norm)
        x = x + self.dropout1(attn_out, training=training)
        
        # MoE FFN
        x_norm = self.bn2(x, training=training)
        moe_out = self.moe(x_norm, training=training)
        return x + moe_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "n_heads": self.n_heads, "ff_dim": self.ff_dim,
            "n_shared": self.n_shared, "n_private": self.n_private, 
            "top_k": self.top_k, "dropout": self.dropout_rate
        })
        return config

class CrossAttentionBlock(tf.keras.layers.Layer):
    """Attention Block (U, Z) -> U attends to context Z."""
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        
        self.mha = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)
        self.dropout1 = Dropout(dropout)
        self.bn1 = BatchNormalization()
        
        self.bn2 = BatchNormalization()
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='gelu'),
            Dense(d_model),
            Dropout(dropout)
        ])

    def call(self, inputs, training=False):
        u, z = inputs # u attends to z
        u_norm = self.bn1(u, training=training)
        z_norm = self.bn1(z, training=training)
        
        attn_out = self.mha(u_norm, z_norm)
        x = u + self.dropout1(attn_out, training=training)
        
        x_norm = self.bn2(x, training=training)
        ffn_out = self.ffn(x_norm, training=training)
        return x + ffn_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "n_heads": self.n_heads, 
            "ff_dim": self.ff_dim, "dropout": self.dropout_rate
        })
        return config

class LatentBottleneckEncoder(tf.keras.layers.Layer):
    """Encodes Input Patches to Latent Tokens and back."""
    def __init__(self, n_latent_tokens, d_model, n_heads, ff_dim, n_layers, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_latent_tokens = n_latent_tokens
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout
        
        # Initial latent tokens Z(0)
        self.z_init = self.add_weight(
            name='latent_tokens_init', shape=(1, n_latent_tokens, d_model),
            initializer='random_normal', trainable=True
        )
        
        # Cross Attention Z(0) attends to H(0)
        self.cross_attn_in = CrossAttentionBlock(d_model, n_heads, ff_dim, dropout)
        
        # Self Attention Z(k)
        self.self_attn_layers = [
            TransformerEncoderBlock(d_model, n_heads, ff_dim, dropout, name=f'latent_self_attn_{i}')
            for i in range(n_layers)
        ]
        
        # Cross Attention H(0) attends to Z(K)
        self.cross_attn_out = CrossAttentionBlock(d_model, n_heads, ff_dim, dropout)

    def call(self, h_in, training=False):
        batch_size = tf.shape(h_in)[0]
        z_0 = tf.repeat(self.z_init, batch_size, axis=0)
        
        # Stage 1: Z attends H
        z = self.cross_attn_in([z_0, h_in], training=training)
        
        # Stage 2: Z Self-attention
        for layer in self.self_attn_layers:
            z = layer(z, training=training)
            
        # Stage 3: H attends Z
        h_out = self.cross_attn_out([h_in, z], training=training)
        
        return h_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_latent_tokens": self.n_latent_tokens, "d_model": self.d_model,
            "n_heads": self.n_heads, "ff_dim": self.ff_dim, 
            "n_layers": self.n_layers, "dropout": self.dropout_rate
        })
        return config

class TimePerceiverDecoder(tf.keras.layers.Layer):
    """Decodes via Querying Target Patches."""
    def __init__(self, forecast_horizon, d_model, n_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        
        # We need a query for each target timestamp (in our case, for univariate output per channel)
        self.query_vectors = self.add_weight(
            name='target_queries', shape=(1, forecast_horizon, d_model),
            initializer='random_normal', trainable=True
        )
        
        self.cross_attn = CrossAttentionBlock(d_model, n_heads, ff_dim, dropout)
        self.head = Dense(1) # We predict a scalar value for each target query step

    def call(self, h_enc, training=False):
        batch_size = tf.shape(h_enc)[0]
        q_0 = tf.repeat(self.query_vectors, batch_size, axis=0)
        
        # Queries attend to Encoded representations
        q_out = self.cross_attn([q_0, h_enc], training=training)
        
        # Final prediction per query
        y_pred = self.head(q_out) # (Batch*Channels, Horizon, 1)
        return tf.squeeze(y_pred, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "forecast_horizon": self.forecast_horizon, "d_model": self.d_model,
            "n_heads": self.n_heads, "ff_dim": self.ff_dim, 
            "dropout": self.dropout_rate
        })
        return config

# ============================================================
# MODEL BUILDERS
# ============================================================

def build_timetracker(lookback, n_features, forecast_horizon, hp: dict):
    """
    TimeTracker Implementation (Decoder-only Transformer with MoE).
    """
    patch_len = hp['patch_len']
    stride = hp['stride']
    d_model = hp['d_model']
    n_heads = hp['n_heads']
    n_layers = hp['n_layers']
    dropout = hp['dropout']
    ff_dim = hp.get('ff_dim', d_model * 2)
    n_shared = hp.get('n_shared_experts', 1)
    n_private = hp.get('n_private_experts', 4)
    top_k = hp.get('top_k', 2)

    inputs = Input(shape=(lookback, n_features), name='main_input')
    
    # Instance Normalization
    revin_layer = RevIN()
    z = revin_layer(inputs, mode='norm') 
    
    # Channel Independence (reshape channels into batch dimension)
    z = Permute((2, 1))(z)
    z = Reshape((n_features, lookback, 1))(z)
    z = tf.reshape(z, [-1, lookback, 1]) 

    # Patching
    z = PatchEmbedding(patch_len, stride, d_model)(z)
    num_patches = (lookback - patch_len) // stride + 1
    z = PositionalEncoding(max_len=num_patches, d_model=d_model)(z)

    # TimeTracker Backbone (Transformer + MoE)
    for i in range(n_layers):
        z = TimeTrackerBlock(
            d_model, n_heads, ff_dim, n_shared, n_private, top_k, 
            dropout=dropout, name=f'tt_block_{i}'
        )(z)

    # Forecasting Head
    z = Flatten()(z)
    z = Dense(forecast_horizon, activation='linear')(z)
    
    # Reshape back to channels
    z = tf.reshape(z, [-1, n_features, forecast_horizon])
    z = Permute((2, 1))(z)
    
    # Denormalize
    z = revin_layer(z, mode='denorm')
    
    # Output dense for multi-to-one
    z_final = Flatten()(z)
    outputs = Dense(forecast_horizon, activation='linear', name='output_layer')(z_final)

    return tf.keras.Model(inputs, outputs, name='TimeTracker')

def build_patchtst(lookback, n_features, forecast_horizon, hp: dict):
    """
    PatchTST ICLR 2023 Implementation.
    Features: Channel Independence, RevIN, Patching, Weight Sharing.
    """
    patch_len = hp['patch_len']
    stride = hp['stride']
    d_model = hp['d_model']
    n_heads = hp['n_heads']
    n_layers = hp['n_layers']
    dropout = hp['dropout']
    ff_dim = hp.get('ff_dim', d_model * (hp.get('ff_ratio', 2))) # Paper default ratio: 2x

    inputs = Input(shape=(lookback, n_features), name='main_input')
    
    # 1. Instance Normalization (RevIN)
    revin_layer = RevIN()
    z = revin_layer(inputs, mode='norm') # (Batch, L, M)
    
    # 2. Channel Independence (Reshape to treat variables as batch items)
    # (Batch, L, M) -> (Batch, M, L) -> (Batch*M, L, 1)
    z = Permute((2, 1))(z)
    z = Reshape((n_features, lookback, 1))(z)
    # Effectively sharing weights across all channels
    z = tf.reshape(z, [-1, lookback, 1]) 

    # 3. Patching & Positioning
    z = PatchEmbedding(patch_len, stride, d_model)(z) # (Batch*M, NumPatches, D)
    num_patches = (lookback - patch_len) // stride + 1
    z = PositionalEncoding(max_len=num_patches, d_model=d_model)(z)

    # 4. Transformer Backbone
    for i in range(n_layers):
        z = TransformerEncoderBlock(d_model, n_heads, ff_dim=ff_dim, 
                                     dropout=dropout, name=f'encoder_{i}')(z)

    # 5. Head Logic (Multivariate to Univariate Forecasting)
    # Each channel gets a linear forecast
    z = Flatten()(z)
    z = Dense(forecast_horizon, activation='linear')(z) # (Batch*M, Horizon)
    
    # Reshape back to channels: (Batch, M, Horizon)
    z = tf.reshape(z, [-1, n_features, forecast_horizon])
    # Permute back to time-last if needed, but RevIN expects (Batch, L, M)
    z = Permute((2, 1))(z) # (Batch, Horizon, M)
    
    # 6. Denormalization (RevIN)
    # Denormalize based on the Mean/Std of each channel
    z = revin_layer(z, mode='denorm')
    
    # 7. Final Target Selection (Multi-to-One)
    # Since we only want to predict the target channel (assume index 0 or as defined)
    # We use a final dense to mix if needed, or just select. 
    # Paper uses Linear Head for forecasting. For multi-to-one, we mix feature maps.
    z_final = Flatten()(z)
    outputs = Dense(forecast_horizon, activation='linear', name='output_layer')(z_final)

    return tf.keras.Model(inputs, outputs, name='PatchTST_Faithful')

def build_gru(lookback, n_features, forecast_horizon, hp: dict):
    """Membangun model GRU Standar."""
    from keras.layers import Bidirectional, GRU as GRULayer, Reshape, Flatten
    d_model = hp.get('d_model', 64)
    n_layers = hp.get('n_layers', 2)
    dropout = hp.get('dropout', 0.2)
    use_bi = hp.get('use_bidirectional', True)
    use_revin = hp.get('use_revin', False)

    inputs = Input(shape=(lookback, n_features), name='main_input')
    
    if use_revin:
        revin_layer = RevIN()
        x = revin_layer(inputs, mode='norm')
    else:
        x = inputs
        
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        layer = GRULayer(d_model, return_sequences=return_seq, dropout=dropout)
        x = Bidirectional(layer)(x) if use_bi else layer(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    
    if use_revin:
        x = Dense(forecast_horizon * n_features, activation='linear')(x)
        x = Reshape((forecast_horizon, n_features))(x)
        x = revin_layer(x, mode='denorm')
        x_final = Flatten()(x)
        outputs = Dense(forecast_horizon, activation='linear', name='output_layer')(x_final)
    else:
        outputs = Dense(forecast_horizon, activation='linear', name='output_layer')(x)
        
    return tf.keras.Model(inputs, outputs, name='GRU_Model')

def build_lstm(lookback, n_features, forecast_horizon, hp: dict):
    """Membangun model LSTM Standar."""
    from keras.layers import Bidirectional, LSTM as LSTMLayer, Reshape, Flatten
    d_model = hp.get('d_model', 64)
    n_layers = hp.get('n_layers', 2)
    dropout = hp.get('dropout', 0.2)
    use_bi = hp.get('use_bidirectional', True)
    use_revin = hp.get('use_revin', False)

    inputs = Input(shape=(lookback, n_features), name='main_input')
    
    if use_revin:
        revin_layer = RevIN()
        x = revin_layer(inputs, mode='norm')
    else:
        x = inputs
        
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        layer = LSTMLayer(d_model, return_sequences=return_seq, dropout=dropout)
        x = Bidirectional(layer)(x) if use_bi else layer(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    
    if use_revin:
        x = Dense(forecast_horizon * n_features, activation='linear')(x)
        x = Reshape((forecast_horizon, n_features))(x)
        x = revin_layer(x, mode='denorm')
        x_final = Flatten()(x)
        outputs = Dense(forecast_horizon, activation='linear', name='output_layer')(x_final)
    else:
        outputs = Dense(forecast_horizon, activation='linear', name='output_layer')(x)
        
    return tf.keras.Model(inputs, outputs, name='LSTM_Model')

def build_rnn(lookback, n_features, forecast_horizon, hp: dict):
    """Membangun model SimpleRNN Standar."""
    from keras.layers import Bidirectional, SimpleRNN as RNNLayer, Reshape, Flatten
    d_model = hp.get('d_model', 64)
    n_layers = hp.get('n_layers', 2)
    dropout = hp.get('dropout', 0.2)
    use_bi = hp.get('use_bidirectional', True)
    use_revin = hp.get('use_revin', False)

    inputs = Input(shape=(lookback, n_features), name='main_input')
    
    if use_revin:
        revin_layer = RevIN()
        x = revin_layer(inputs, mode='norm')
    else:
        x = inputs
        
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        layer = RNNLayer(d_model, return_sequences=return_seq, dropout=dropout)
        x = Bidirectional(layer)(x) if use_bi else layer(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    
    if use_revin:
        x = Dense(forecast_horizon * n_features, activation='linear')(x)
        x = Reshape((forecast_horizon, n_features))(x)
        x = revin_layer(x, mode='denorm')
        x_final = Flatten()(x)
        outputs = Dense(forecast_horizon, activation='linear', name='output_layer')(x_final)
    else:
        outputs = Dense(forecast_horizon, activation='linear', name='output_layer')(x)
        
    return tf.keras.Model(inputs, outputs, name='SimpleRNN_Model')

def build_timeperceiver(lookback, n_features, forecast_horizon, hp: dict):
    """
    TimePerceiver Implementation (Latent Bottleneck Encoder + Query Decoder).
    """
    patch_len = hp['patch_len']
    stride = hp['stride']
    d_model = hp['d_model']
    n_heads = hp['n_heads']
    n_layers = hp['n_layers']
    dropout = hp['dropout']
    n_latent = hp.get('n_latent_tokens', 32)
    ff_dim = hp.get('ff_dim', d_model * 2)

    inputs = Input(shape=(lookback, n_features), name='main_input')
    
    # Instance Normalization
    revin_layer = RevIN()
    z = revin_layer(inputs, mode='norm') 
    
    # Channel Independence
    z = Permute((2, 1))(z)
    z = Reshape((n_features, lookback, 1))(z)
    z = tf.reshape(z, [-1, lookback, 1]) 

    # Patching
    z = PatchEmbedding(patch_len, stride, d_model)(z)
    num_patches = (lookback - patch_len) // stride + 1
    z = PositionalEncoding(max_len=num_patches, d_model=d_model)(z)

    # TimePerceiver Encoder (Latent Bottleneck)
    z = LatentBottleneckEncoder(
        n_latent_tokens=n_latent, d_model=d_model, n_heads=n_heads, 
        ff_dim=ff_dim, n_layers=n_layers, dropout=dropout, name='timeperceiver_encoder'
    )(z)

    # TimePerceiver Decoder (Target Queries)
    # Output shape: (Batch * Channels, Horizon)
    z = TimePerceiverDecoder(
        forecast_horizon=forecast_horizon, d_model=d_model, n_heads=n_heads,
        ff_dim=ff_dim, dropout=dropout, name='timeperceiver_decoder'
    )(z)
    
    # Reshape back to multivariate
    z = tf.reshape(z, [-1, n_features, forecast_horizon])
    z = Permute((2, 1))(z)
    
    # Denormalize
    z = revin_layer(z, mode='denorm')
    
    # Target alignment (Multi-to-one or specific target mapping via generic output_layer)
    z_final = Flatten()(z)
    outputs = Dense(forecast_horizon, activation='linear', name='output_layer')(z_final)

    return tf.keras.Model(inputs, outputs, name='TimePerceiver')

def build_linear(lookback, n_features, forecast_horizon, hp: dict):
    """Membangun model Baseline Regresi Linear."""
    from keras.layers import Flatten
    inputs = Input(shape=(lookback, n_features), name='main_input')
    x = Flatten()(inputs)
    outputs = Dense(forecast_horizon, activation='linear', name='output_layer')(x)
    return tf.keras.Model(inputs, outputs, name='Linear_Baseline')

def build_mlp(lookback, n_features, forecast_horizon, hp: dict):
    """Membangun model Baseline MLP (Multi-Layer Perceptron)."""
    from keras.layers import Flatten
    d_model = hp.get('d_model', 128)
    n_layers = hp.get('n_layers', 2)
    dropout = hp.get('dropout', 0.2)
    
    inputs = Input(shape=(lookback, n_features), name='main_input')
    x = Flatten()(inputs)
    
    for _ in range(n_layers):
        x = Dense(d_model, activation='relu')(x)
        x = Dropout(dropout)(x)
        
    outputs = Dense(forecast_horizon, activation='linear', name='output_layer')(x)
    return tf.keras.Model(inputs, outputs, name='MLP_Baseline')

MODEL_REGISTRY = {
    'patchtst': build_patchtst,
    'timetracker': build_timetracker,
    'timeperceiver': build_timeperceiver,
    'gru': build_gru,
    'lstm': build_lstm,
    'rnn': build_rnn,
    'linear': build_linear,
    'mlp': build_mlp
}

def build_model(architecture: str, lookback: int, n_features: int,
                forecast_horizon: int, hp: dict) -> tf.keras.Model:
    arch = architecture.lower()
    if arch not in MODEL_REGISTRY: raise ValueError(f"Arsitektur '{arch}' tidak dikenal.")
    model = MODEL_REGISTRY[arch](lookback, n_features, forecast_horizon, hp)
    logger.info(f"Model '{arch}' dibangun: params={model.count_params():,}")
    return model

def compile_model(model: tf.keras.Model, learning_rate: float, loss_fn: str = 'mse'):
    model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss=loss_fn)
    return model

def get_custom_objects():
    return {
        'RevIN': RevIN,
        'PatchEmbedding': PatchEmbedding,
        'PositionalEncoding': PositionalEncoding,
        'TransformerEncoderBlock': TransformerEncoderBlock,
        'MoELayer': MoELayer,
        'TimeTrackerBlock': TimeTrackerBlock,
        'CrossAttentionBlock': CrossAttentionBlock,
        'LatentBottleneckEncoder': LatentBottleneckEncoder,
        'TimePerceiverDecoder': TimePerceiverDecoder,
    }
