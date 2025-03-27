import numpy as np
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Add, Dropout
import tensorflow as tf


def positional_encoding(position, d_model):
    """
    Generate positional encoding for transformer

    Args:
        position: sequence length
        d_model: dimension of the model

    Returns:
        Tensor of shape (1, position, d_model)
    """
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )

    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(pos, i, d_model):
    """
    Helper function for positional encoding
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def transformer_encoder_block(inputs, num_heads, key_dim, dropout_rate):
    # Add positional encoding to the input first
    seq_length = inputs.shape[1]
    pos_enc = positional_encoding(seq_length, key_dim)
    pos_enc = tf.cast(pos_enc, dtype=inputs.dtype)
    inputs_with_pos = inputs + pos_enc

    # First sub-layer: Multi-head attention with residual and normalization
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(inputs_with_pos, inputs_with_pos)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = Add()([inputs_with_pos, attention_output])
    normalized_attention = LayerNormalization()(attention_output)

    # Second sub-layer: FFN with residual and normalization
    ffn_output = Dense(key_dim * 4, activation='relu')(normalized_attention)
    ffn_output = Dense(key_dim)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    ffn_output = Add()([normalized_attention, ffn_output])
    output = LayerNormalization()(ffn_output)

    return output
