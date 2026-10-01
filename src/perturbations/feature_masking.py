"""Feature masking perturbations."""

from __future__ import annotations

import numpy as np


def replacement_values(X_reference: np.ndarray, replacement: str) -> np.ndarray:
    """Compute replacement values from a reference matrix."""
    if replacement == "median":
        return np.median(X_reference, axis=0)
    if replacement == "zero":
        return np.zeros(X_reference.shape[1])
    return np.mean(X_reference, axis=0)


def select_feature_indices(
    n_features: int,
    k: int,
    mode: str,
    top_features: list[int] | None,
    seed: int,
) -> np.ndarray:
    """Select top-k or random-k feature indices."""
    if mode == "top-k":
        if not top_features:
            raise ValueError("top_features are required for top-k perturbations.")
        return np.asarray(top_features[:k], dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_features, size=min(k, n_features), replace=False))


def apply_feature_masking(
    X: np.ndarray,
    X_reference: np.ndarray,
    k: int,
    mode: str,
    replacement: str,
    top_features: list[int] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask selected features with mean, median, or zero values."""
    cols = select_feature_indices(X.shape[1], k, mode, top_features, seed)
    values = replacement_values(X_reference, replacement)
    Xp = X.copy()
    Xp[:, cols] = values[cols]
    return Xp.astype("float32"), cols
