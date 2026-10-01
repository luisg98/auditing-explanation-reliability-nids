"""Keras LSTM model for feature-sequence IDS experiments."""

from __future__ import annotations

from typing import Any

def build_lstm(n_features: int, n_classes: int, params: dict[str, Any]):
    """Build an LSTM baseline treating ordered features as a short sequence."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    units = int(params.get("units", 64))
    dense_units = int(params.get("dense_units", 64))
    dropout = float(params.get("dropout", 0.3))
    learning_rate = float(params.get("learning_rate", 0.001))
    binary = n_classes <= 2

    model = models.Sequential(name="lstm")
    model.add(layers.Input(shape=(n_features, 1)))
    model.add(layers.LSTM(units, dropout=dropout))
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout))
    if binary:
        model.add(layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(layers.Dense(n_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
    metrics = ["accuracy"]
    if binary:
        from src.metrics.keras_metrics import binary_classification_metrics

        metrics = binary_classification_metrics()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model
