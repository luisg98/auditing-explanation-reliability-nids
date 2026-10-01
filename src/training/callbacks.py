"""Keras callback helpers."""

from __future__ import annotations

from pathlib import Path


def build_callbacks(
    model_path: str | Path,
    patience: int,
    monitor: str = "val_loss",
    mode: str = "min",
    min_delta: float = 0.0,
):
    """Create early stopping and checkpoint callbacks."""
    import tensorflow as tf

    model_path = str(model_path)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=int(patience),
            min_delta=float(min_delta),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
        ),
    ]
