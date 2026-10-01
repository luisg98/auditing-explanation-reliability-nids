"""Keras 1D CNN model for tabular IDS features."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _parsed_cnn_params(n_classes: int, params: dict[str, Any]):
    filters = params.get("filters", [64, 128])
    if isinstance(filters, (int, float)):
        filters = [int(filters)]
    elif isinstance(filters, Iterable) and not isinstance(filters, (str, bytes)):
        filters = [int(f) for f in filters]
    else:
        raise TypeError(f"Unsupported filters value for CNN: {filters!r}")
    kernel_size = int(params.get("kernel_size", 3))
    dense_units = int(params.get("dense_units", 64))
    dropout = float(params.get("dropout", 0.3))
    learning_rate = float(params.get("learning_rate", 0.001))
    binary = n_classes <= 2
    return filters, kernel_size, dense_units, dropout, learning_rate, binary


def _compile_head(model, n_classes: int, binary: bool, learning_rate: float):
    import tensorflow as tf
    from tensorflow.keras import layers

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


def build_cnn(n_features: int, n_classes: int, params: dict[str, Any]):
    """Build a compact 1D CNN following the old repository's successful setup.

    Expects pre-reshaped 3D input (N, n_features, 1) — callers must reshape
    externally (see src.models.factory.reshape_for_model). Kept unchanged for
    existing callers (pipeline_v3, pipeline_v5, etc).
    """
    from tensorflow.keras import layers, models

    filters, kernel_size, dense_units, dropout, learning_rate, binary = _parsed_cnn_params(
        n_classes, params
    )
    model = models.Sequential(name="cnn")
    model.add(layers.Input(shape=(n_features, 1)))
    for f in filters:
        model.add(layers.Conv1D(int(f), kernel_size, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout))
    return _compile_head(model, n_classes, binary, learning_rate)


def build_cnn_flat_input(n_features: int, n_classes: int, params: dict[str, Any]):
    """Same 1D CNN architecture as build_cnn, but accepts flat 2D input
    (N, n_features) directly via an internal Reshape layer, so callers that
    treat every architecture uniformly (no external reshape_for_model step)
    can use it as a drop-in alternative to build_mlp/build_tabular_resnet.

    Used by src.leakage_free_tabular (v2), which feeds every model the same
    2D (N, n_features) tensors throughout training, diagnostics, and the XAI
    epistemic audit.
    """
    from tensorflow.keras import layers, models

    filters, kernel_size, dense_units, dropout, learning_rate, binary = _parsed_cnn_params(
        n_classes, params
    )
    model = models.Sequential(name="cnn")
    model.add(layers.Input(shape=(n_features,)))
    model.add(layers.Reshape((n_features, 1)))
    for f in filters:
        model.add(layers.Conv1D(int(f), kernel_size, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout))
    return _compile_head(model, n_classes, binary, learning_rate)
