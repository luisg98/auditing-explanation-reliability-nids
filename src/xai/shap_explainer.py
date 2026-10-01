"""SHAP explanation generation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.models.factory import predict_scores, reshape_for_model


def normalize_shap_values(values: Any) -> np.ndarray:
    """Normalize SHAP outputs to an n_samples x n_features matrix."""
    if isinstance(values, list):
        values = values[-1]
    arr = np.asarray(values)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        else:
            arr = arr[:, :, -1]
    return arr.astype("float32")


def explain_shap(
    model,
    X_background: np.ndarray,
    X_samples: np.ndarray,
    model_name: str,
    method: str = "kernel",
    nsamples: int = 200,
) -> tuple[np.ndarray, float]:
    """Generate SHAP values. KernelExplainer is the safe default."""
    import shap

    start = time.time()

    def predict_positive(data):
        scores = predict_scores(model, np.asarray(data, dtype="float32"), model_name)
        if np.asarray(scores).ndim == 1:
            return scores
        return scores[:, -1]

    method = method.lower()
    if method == "deep":
        explainer = shap.DeepExplainer(model, reshape_for_model(X_background, model_name))
    elif method == "gradient":
        explainer = shap.GradientExplainer(model, reshape_for_model(X_background, model_name))
    else:
        explainer = shap.KernelExplainer(predict_positive, X_background)

    values = shap_values_with_explainer(explainer, X_samples, model_name, method, nsamples)

    return normalize_shap_values(values), time.time() - start


def build_shap_explainer(model, X_background: np.ndarray, model_name: str, method: str = "kernel"):
    """Build a reusable SHAP explainer."""
    import shap

    def predict_positive(data):
        scores = predict_scores(model, np.asarray(data, dtype="float32"), model_name)
        if np.asarray(scores).ndim == 1:
            return scores
        return scores[:, -1]

    method = method.lower()
    if method == "deep":
        return shap.DeepExplainer(model, reshape_for_model(X_background, model_name))
    if method == "gradient":
        return shap.GradientExplainer(model, reshape_for_model(X_background, model_name))
    return shap.KernelExplainer(predict_positive, X_background)


def shap_values_with_explainer(
    explainer,
    X_samples: np.ndarray,
    model_name: str,
    method: str = "kernel",
    nsamples: int = 200,
) -> np.ndarray:
    """Generate SHAP values using a pre-built explainer."""
    method = method.lower()
    if method in {"deep", "gradient"}:
        values = explainer.shap_values(reshape_for_model(X_samples, model_name))
    else:
        values = explainer.shap_values(X_samples, nsamples=nsamples)
    return normalize_shap_values(values)
