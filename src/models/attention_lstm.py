"""Attention-LSTM model definition (TensorFlow/Keras)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("vtech.models.attention_lstm")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class AttentionLayer(layers.Layer):
    """Multi-head self-attention layer."""

    def __init__(self, units: int, num_heads: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.head_dim = units // num_heads

    def build(self, input_shape: tf.TensorShape) -> None:
        d_in = int(input_shape[-1])
        self.W_q = self.add_weight(name="W_q", shape=(d_in, self.units), initializer="glorot_uniform")
        self.W_k = self.add_weight(name="W_k", shape=(d_in, self.units), initializer="glorot_uniform")
        self.W_v = self.add_weight(name="W_v", shape=(d_in, self.units), initializer="glorot_uniform")
        self.W_o = self.add_weight(name="W_o", shape=(self.units, self.units), initializer="glorot_uniform")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        Q = tf.reshape(tf.matmul(inputs, self.W_q), (batch_size, seq_len, self.num_heads, self.head_dim))
        K = tf.reshape(tf.matmul(inputs, self.W_k), (batch_size, seq_len, self.num_heads, self.head_dim))
        V = tf.reshape(tf.matmul(inputs, self.W_v), (batch_size, seq_len, self.num_heads, self.head_dim))

        Q, K, V = (tf.transpose(t, [0, 2, 1, 3]) for t in (Q, K, V))

        scale = tf.sqrt(tf.cast(self.head_dim, tf.float32))
        scores = tf.matmul(Q, K, transpose_b=True) / scale
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, V)

        context = tf.reshape(tf.transpose(context, [0, 2, 1, 3]), (batch_size, seq_len, self.units))
        return tf.matmul(context, self.W_o)

    def get_config(self) -> dict:
        return {**super().get_config(), "units": self.units, "num_heads": self.num_heads}


def build_attention_lstm(
    input_shape: tuple[int, int],
    num_classes: int = 3,
    lstm_units: list[int] | None = None,
    attention_heads: int = 4,
    dense_units: list[int] | None = None,
    dropout: float = 0.3,
    use_attention: bool = True,
    learning_rate: float = 0.001,
) -> Model:
    """Build and compile an Attention-LSTM classifier.

    Architecture
    ------------
    Input → [LSTM layers] → [Attention] → LayerNorm → GlobalPool → Dense → Softmax
    """
    if lstm_units is None:
        lstm_units = [64, 32]
    if dense_units is None:
        dense_units = [64, 32]

    inputs = layers.Input(shape=input_shape, name="input")
    x = inputs

    # Stacked LSTM layers — all but last return sequences
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1) or use_attention
        x = layers.LSTM(
            units,
            return_sequences=return_seq,
            dropout=dropout,
            name=f"lstm_{i}",
        )(x)

    if use_attention:
        att_units = lstm_units[-1]
        # Round up to nearest multiple of attention_heads
        att_units = ((att_units + attention_heads - 1) // attention_heads) * attention_heads
        x = AttentionLayer(att_units, attention_heads, name="attention")(x)
        x = layers.LayerNormalization(name="layer_norm")(x)
        x = layers.GlobalAveragePooling1D(name="global_pool")(x)

    # Dense head
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = layers.Dropout(dropout, name=f"drop_{i}")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="attention_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("model_built", extra={"input": input_shape, "params": model.count_params()})
    return model
