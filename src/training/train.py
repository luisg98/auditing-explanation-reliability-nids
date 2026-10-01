"""Training orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics.performance_metrics import tune_binary_threshold
from src.models.factory import build_model, predict_scores, reshape_for_model
from src.training.callbacks import build_callbacks


def train_keras_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_params: dict[str, Any],
    fit_params: dict[str, Any],
    output_dir: Path,
    debug: bool = False,
) -> tuple[object, pd.DataFrame, float, float, dict[str, float | str]]:
    """Train one Keras model and return model, history, best threshold, and runtime."""
    n_features = X_train.shape[1]
    n_classes = int(len(np.unique(np.concatenate([y_train, y_val]))))
    model = build_model(model_name, n_features, n_classes, model_params)

    epochs = int(fit_params.get("epochs", 50))
    batch_size = int(fit_params.get("batch_size", 128))
    patience = int(fit_params.get("early_stopping_patience", 5))
    min_delta = float(fit_params.get("early_stopping_min_delta", 0.0))
    monitor = str(fit_params.get("early_stopping_monitor", "val_loss"))
    monitor_mode = str(fit_params.get("early_stopping_mode", "min"))
    if debug:
        epochs = min(epochs, int(fit_params.get("debug_epochs", 2)))
        batch_size = min(batch_size, int(fit_params.get("debug_batch_size", batch_size)))

    model_path = output_dir / "model.keras"
    callbacks = build_callbacks(model_path, patience, monitor, monitor_mode, min_delta)
    start = time.time()
    history = model.fit(
        reshape_for_model(X_train, model_name),
        y_train,
        validation_data=(reshape_for_model(X_val, model_name), y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=int(fit_params.get("verbose", 1)),
    )
    runtime = time.time() - start

    hist_df = pd.DataFrame(history.history)
    monitored = [c for c in ("loss", "val_loss") if c in hist_df.columns]
    if monitored and not np.isfinite(hist_df[monitored].to_numpy(dtype=float)).all():
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(
            "Training produced non-finite loss values. This usually means the "
            "preprocessed features contain extreme values or NaNs. Fix preprocessing "
            "before saving model outputs."
        )

    val_scores = predict_scores(model, X_val, model_name)
    if not np.isfinite(np.asarray(val_scores, dtype=float)).all():
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError("Model produced non-finite validation scores after training.")
    if np.asarray(val_scores).ndim == 1:
        threshold, threshold_info = tune_binary_threshold(
            y_val,
            val_scores,
            strategy=str(fit_params.get("threshold_strategy", "best_f1")),
            max_fpr=float(fit_params.get("max_fpr", 0.05)),
            beta=float(fit_params.get("threshold_beta", 2.0)),
        )
    else:
        threshold = 0.5
        threshold_info = {"threshold_strategy": "fixed_multiclass"}

    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    hist_df["training_time_seconds"] = runtime
    return model, hist_df, threshold, runtime, threshold_info
