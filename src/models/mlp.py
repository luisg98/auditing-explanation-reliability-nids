"""Keras MLP model."""

from __future__ import annotations

from typing import Any

def build_mlp(n_features: int, n_classes: int, params: dict[str, Any]):
    """Build the MLP architecture used in the original experiments."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    hidden_units = params.get("hidden_units", [128, 64, 32])
    dropout = float(params.get("dropout", 0.3))
    learning_rate = float(params.get("learning_rate", 0.001))
    binary = n_classes <= 2

    model = models.Sequential(name="mlp")
    model.add(layers.Input(shape=(n_features,)))
    for units in hidden_units:
        model.add(layers.Dense(int(units), activation="relu"))
        if dropout > 0:
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
