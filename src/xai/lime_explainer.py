"""LIME explanation generation."""

from __future__ import annotations

import time

import numpy as np

from src.models.factory import predict_proba_2d


def build_lime_explainer(
    X_train: np.ndarray,
    feature_names: list[str],
    class_names: list[str] | None = None,
    random_state: int = 42,
):
    """Build a reusable LIME tabular explainer."""
    from lime.lime_tabular import LimeTabularExplainer

    return LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names or ["benign", "attack"],
        mode="classification",
        discretize_continuous=False,
        random_state=int(random_state),
    )


def explain_lime_with_explainer(
    explainer,
    model,
    X_samples: np.ndarray,
    feature_names: list[str],
    model_name: str,
    num_features: int | None = None,
) -> tuple[np.ndarray, float]:
    """Generate dense LIME explanations using a pre-built explainer."""
    start = time.time()
    num_features = num_features or len(feature_names)
    dense = np.zeros((len(X_samples), len(feature_names)), dtype="float32")

    def predict_fn(data):
        return predict_proba_2d(model, np.asarray(data, dtype="float32"), model_name)

    for i, row in enumerate(X_samples):
        exp = explainer.explain_instance(row, predict_fn, num_features=num_features, labels=[1])
        for feature_idx, weight in exp.as_map().get(1, []):
            dense[i, int(feature_idx)] = float(weight)

    return dense, time.time() - start


def explain_lime(
    model,
    X_train: np.ndarray,
    X_samples: np.ndarray,
    feature_names: list[str],
    model_name: str,
    class_names: list[str] | None = None,
    num_features: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """Generate dense LIME local explanations for selected samples."""
    explainer = build_lime_explainer(X_train, feature_names, class_names, random_state)
    return explain_lime_with_explainer(explainer, model, X_samples, feature_names, model_name, num_features)
