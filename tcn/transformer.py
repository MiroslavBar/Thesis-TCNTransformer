import numpy as np
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Add, Dropout
import tensorflow as tf


def positional_encoding(position: int, d_model: int) -> tf.Tensor:
    """
    Generate positional encoding for transformer model.

    This function creates a positional encoding matrix that helps the model
    understand the relative or absolute position of tokens in a sequence.
    It uses sine and cosine functions of different frequencies to generate
    unique encodings for each position.

    Args:
        position: The length of the sequence (number of tokens)
        d_model: The dimensionality of the model's embeddings

    Returns:
        A tensor of shape (1, position, d_model) containing positional encodings
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


def get_angles(pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def transformer_encoder_block(
    inputs: tf.Tensor,
    num_heads: int,
    key_dim: int,
    dropout_rate: float
) -> tf.Tensor:
    """
    Create a single transformer encoder block.

    This function implements a standard transformer encoder block with:
    1. Positional encoding
    2. Multi-head self-attention mechanism
    3. Feed-forward neural network
    4. Residual connections
    5. Layer normalization

    Args:
        inputs: Input tensor of shape (batch_size, sequence_length, embedding_dim)
        num_heads: Number of attention heads
        key_dim: Dimensionality of the key and query spaces
        dropout_rate: Dropout rate for regularization

    Returns:
        Processed tensor after passing through the encoder block
    """

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
