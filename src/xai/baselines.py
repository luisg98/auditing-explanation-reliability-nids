"""Baseline helpers shared by gradient and occlusion explainers."""

from __future__ import annotations

import numpy as np


def explanation_baseline(X_reference: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """Compute a one-row feature baseline from reference data."""
    strategy = str(strategy).lower()
    if strategy == "median":
        baseline = np.median(X_reference, axis=0)
    elif strategy == "zero":
        baseline = np.zeros(X_reference.shape[1], dtype="float32")
    else:
        baseline = np.mean(X_reference, axis=0)
    return np.asarray(baseline, dtype="float32")
