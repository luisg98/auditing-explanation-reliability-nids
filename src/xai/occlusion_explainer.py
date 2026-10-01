"""Feature occlusion explanations for tabular IDS models."""

from __future__ import annotations

import time

import numpy as np

from src.models.factory import predict_scores


def _positive_scores(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores)
    if arr.ndim == 1:
        return arr.astype("float32")
    return arr[:, -1].astype("float32")


def explain_occlusion(
    model,
    baseline: np.ndarray,
    X_samples: np.ndarray,
    model_name: str,
    batch_size: int = 512,
) -> tuple[np.ndarray, float]:
    """Generate local feature occlusion attributions.

    Attribution is score(x) - score(x with feature j replaced by baseline_j).
    Positive values mean the feature supports the positive attack score.
    """
    start = time.time()
    samples = np.asarray(X_samples, dtype="float32")
    baseline = np.asarray(baseline, dtype="float32").reshape(-1)
    if baseline.shape[0] != samples.shape[1]:
        raise ValueError(
            f"Occlusion baseline has {baseline.shape[0]} features, "
            f"but samples have {samples.shape[1]}."
        )

    batch_size = max(1, int(batch_size))
    n_samples, n_features = samples.shape
    clean_scores = _positive_scores(predict_scores(model, samples, model_name))
    explanations = np.zeros((n_samples, n_features), dtype="float32")

    for sample_idx, sample in enumerate(samples):
        occluded = np.repeat(sample[None, :], n_features, axis=0)
        occluded[np.arange(n_features), np.arange(n_features)] = baseline
        occluded_scores = []
        for start_idx in range(0, n_features, batch_size):
            batch = occluded[start_idx : start_idx + batch_size]
            occluded_scores.append(_positive_scores(predict_scores(model, batch, model_name)))
        scores = np.concatenate(occluded_scores, axis=0)
        explanations[sample_idx] = clean_scores[sample_idx] - scores

    return explanations, time.time() - start
