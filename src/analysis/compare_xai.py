"""SHAP vs LIME agreement analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics.stability_metrics import compute_pair_metrics, summarize_stability


def compare_explanation_matrices(
    shap_values: np.ndarray,
    lime_values: np.ndarray,
    ks: list[int],
    dataset: str,
    model: str,
    method_a: str | None = None,
    method_b: str | None = None,
) -> pd.DataFrame:
    """Compare two local explanation matrices on the same samples."""
    n = min(len(shap_values), len(lime_values))
    rows = []
    for i in range(n):
        row = {"sample_order": i}
        row.update(compute_pair_metrics(shap_values[i], lime_values[i], ks))
        rows.append(row)
    summary = summarize_stability(pd.DataFrame(rows))
    summary.insert(0, "model", model)
    summary.insert(0, "dataset", dataset)
    if method_a is not None and method_b is not None:
        summary.insert(2, "method_a", method_a)
        summary.insert(3, "method_b", method_b)
    return summary
