"""Gaussian noise perturbations."""

from __future__ import annotations

import numpy as np


def apply_gaussian_noise(X: np.ndarray, level: float, seed: int = 42) -> np.ndarray:
    """Add feature-wise Gaussian noise scaled by each feature's sample std."""
    rng = np.random.default_rng(seed)
    scale = np.nanstd(X, axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return (X + rng.normal(0.0, level * scale, size=X.shape)).astype("float32")
