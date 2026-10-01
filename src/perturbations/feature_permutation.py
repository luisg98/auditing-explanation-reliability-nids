"""Feature permutation perturbations."""

from __future__ import annotations

import numpy as np

from src.perturbations.feature_masking import select_feature_indices


def apply_feature_permutation(
    X: np.ndarray,
    k: int,
    mode: str,
    top_features: list[int] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute selected features across selected samples."""
    rng = np.random.default_rng(seed)
    cols = select_feature_indices(X.shape[1], k, mode, top_features, seed)
    Xp = X.copy()
    for col in cols:
        Xp[:, col] = rng.permutation(Xp[:, col])
    return Xp.astype("float32"), cols
