"""Residual fully connected network for tabular inputs.

The architecture follows the tabular ResNet pattern evaluated by Gorishniy
et al. (NeurIPS 2021): a dense input projection, pre-normalised residual MLP
blocks, and a normalised output head.  Unlike the retired 1-D CNN audit case,
it makes no assumption that adjacent feature columns form a spatial sequence.
"""

from __future__ import annotations

from typing import Any


def build_tabular_resnet(
    n_features: int,
    n_classes: int,
    params: dict[str, Any],
):
    """Build a compact ResNet-like model for continuous tabular features."""

    import tensorflow as tf
    from tensorflow.keras import layers, models

    d_main = int(params.get("d_main", 192))
    d_hidden = int(params.get("d_hidden", 384))
    n_blocks = int(params.get("n_blocks", 2))
    dropout_first = float(params.get("dropout_first", params.get("dropout", 0.1)))
    dropout_second = float(params.get("dropout_second", params.get("dropout", 0.1)))
    learning_rate = float(params.get("learning_rate", 0.001))
    if min(d_main, d_hidden, n_blocks) <= 0:
        raise ValueError("d_main, d_hidden, and n_blocks must be positive")
    if not 0.0 <= dropout_first < 1.0 or not 0.0 <= dropout_second < 1.0:
        raise ValueError("dropout rates must be in [0, 1)")

    inputs = layers.Input(shape=(int(n_features),), name="features")
    x = layers.Dense(d_main, name="input_projection")(inputs)
    for block_index in range(n_blocks):
        residual = x
        x = layers.BatchNormalization(name=f"block_{block_index}_normalization")(x)
        x = layers.Dense(d_hidden, name=f"block_{block_index}_dense_in")(x)
        x = layers.Activation("relu", name=f"block_{block_index}_activation")(x)
        if dropout_first:
            x = layers.Dropout(
                dropout_first,
                name=f"block_{block_index}_dropout_in",
            )(x)
        x = layers.Dense(d_main, name=f"block_{block_index}_dense_out")(x)
        if dropout_second:
            x = layers.Dropout(
                dropout_second,
                name=f"block_{block_index}_dropout_out",
            )(x)
        x = layers.Add(name=f"block_{block_index}_residual")([residual, x])

    x = layers.BatchNormalization(name="output_normalization")(x)
    x = layers.Activation("relu", name="output_activation")(x)
    binary = int(n_classes) <= 2
    if binary:
        outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(
            int(n_classes),
            activation="softmax",
            name="prediction",
        )(x)
        loss = "sparse_categorical_crossentropy"

    model = models.Model(inputs=inputs, outputs=outputs, name="tabular_resnet")
    metrics: list[Any] = ["accuracy"]
    if binary:
        from src.metrics.keras_metrics import binary_classification_metrics

        metrics = binary_classification_metrics()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model
