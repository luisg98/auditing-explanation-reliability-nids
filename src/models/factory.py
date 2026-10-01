"""Model construction and prediction adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models.cnn import build_cnn
from src.models.lstm import build_lstm
from src.models.mlp import build_mlp
from src.models.tabular_resnet import build_tabular_resnet


SEQUENCE_MODELS = {"cnn", "lstm"}
PREDICT_BATCH_SIZE = 1024


def reshape_for_model(X: np.ndarray, model_name: str) -> np.ndarray:
    """Reshape tabular arrays for the requested model."""
    if model_name.lower() in SEQUENCE_MODELS and X.ndim == 2:
        return X.reshape((X.shape[0], X.shape[1], 1))
    return X


def build_model(model_name: str, n_features: int, n_classes: int, params: dict[str, Any]):
    """Build a Keras model by name."""
    name = model_name.lower()
    if name == "mlp":
        return build_mlp(n_features, n_classes, params)
    if name == "cnn":
        return build_cnn(n_features, n_classes, params)
    if name == "lstm":
        return build_lstm(n_features, n_classes, params)
    if name == "tabular_resnet":
        return build_tabular_resnet(n_features, n_classes, params)
    raise ValueError(f"Unsupported model '{model_name}'.")


def predict_scores(model, X: np.ndarray, model_name: str, batch_size: int = PREDICT_BATCH_SIZE) -> np.ndarray:
    """Return positive-class scores for binary models or probabilities for multiclass."""
    X_model = reshape_for_model(X, model_name)
    rows = []
    for start in range(0, len(X_model), int(batch_size)):
        out = model(X_model[start : start + int(batch_size)], training=False)
        if hasattr(out, "numpy"):
            out = out.numpy()
        rows.append(np.asarray(out))
    pred = np.concatenate(rows, axis=0) if rows else np.asarray([])
    pred = np.asarray(pred)
    if pred.ndim == 1:
        return pred
    if pred.shape[1] == 1:
        return pred.ravel()
    return pred


def predict_labels(model, X: np.ndarray, model_name: str, threshold: float = 0.5) -> np.ndarray:
    """Convert model scores to class labels."""
    scores = predict_scores(model, X, model_name)
    if scores.ndim == 1:
        return (scores >= threshold).astype(int)
    return np.argmax(scores, axis=1).astype(int)


def predict_proba_2d(model, X: np.ndarray, model_name: str) -> np.ndarray:
    """Return a 2D probability matrix required by LIME."""
    scores = predict_scores(model, X, model_name)
    if scores.ndim == 1:
        return np.column_stack([1.0 - scores, scores])
    return scores
