"""Explanation stability and agreement metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import adjusted_rand_score


NON_METRIC_COLUMNS = {
    "sample_order",
    "test_sample_order",
    "clean_prediction",
    "perturbed_prediction",
    "clean_score",
    "perturbed_score",
    "prediction_changed",
    "y_true",
    "y_pred",
    "outcome",
    "confidence",
    "confidence_bin",
}


def _safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    a_constant = np.allclose(a, a[0])
    b_constant = np.allclose(b, b[0])
    if a_constant and b_constant:
        return 1.0 if np.allclose(a, b) else 0.0
    if a_constant or b_constant:
        return 0.0
    try:
        value = fn(a, b)[0]
        return float(value) if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def topk_indices(values: np.ndarray, k: int) -> set[int]:
    """Return indices of the k largest absolute importance values."""
    k = min(k, len(values))
    return set(np.argsort(np.abs(values))[-k:])


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Top-k overlap ratio."""
    effective_k = min(k, len(a), len(b))
    ta, tb = topk_indices(a, k), topk_indices(b, k)
    return len(ta & tb) / max(effective_k, 1)


def jaccard_at_k(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Jaccard similarity of top-k feature sets."""
    ta, tb = topk_indices(a, k), topk_indices(b, k)
    union = ta | tb
    return len(ta & tb) / len(union) if union else np.nan


def importance_groups(values: np.ndarray) -> np.ndarray:
    """Map absolute importances to low/medium/high groups by tertiles."""
    scores = np.abs(values)
    if np.allclose(scores, scores[0]):
        return np.zeros_like(scores, dtype=int)
    q1, q2 = np.quantile(scores, [1 / 3, 2 / 3])
    return np.digitize(scores, [q1, q2], right=True)


def compute_pair_metrics(clean: np.ndarray, perturbed: np.ndarray, ks: list[int]) -> dict[str, float]:
    """Compute all per-sample stability metrics for one clean/perturbed pair."""
    row: dict[str, float] = {
        "pearson": _safe_corr(pearsonr, clean, perturbed),
        "spearman": _safe_corr(spearmanr, clean, perturbed),
        "kendall_tau": _safe_corr(kendalltau, clean, perturbed),
        "emd": float(wasserstein_distance(clean, perturbed)),
        "ari": float(adjusted_rand_score(importance_groups(clean), importance_groups(perturbed))),
    }
    for k in ks:
        row[f"topk_overlap_{k}"] = topk_overlap(clean, perturbed, k)
        row[f"jaccard_{k}"] = jaccard_at_k(clean, perturbed, k)
    return row


def per_sample_stability(
    clean: np.ndarray,
    perturbed: np.ndarray,
    ks: list[int],
    prediction_changed: np.ndarray | None = None,
    sample_order: np.ndarray | None = None,
    prediction_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-sample stability dataframe."""
    n = min(len(clean), len(perturbed))
    if sample_order is None and prediction_metadata is not None and "sample_order" in prediction_metadata.columns:
        sample_order = prediction_metadata["sample_order"].to_numpy()
    metadata = prediction_metadata.reset_index(drop=True) if prediction_metadata is not None else None
    rows = []
    for i in range(n):
        row = {"sample_order": int(sample_order[i]) if sample_order is not None and i < len(sample_order) else i}
        row.update(compute_pair_metrics(clean[i], perturbed[i], ks))
        if metadata is not None and i < len(metadata):
            for column in [
                "y_true",
                "y_pred",
                "outcome",
                "confidence",
                "confidence_bin",
                "clean_prediction",
                "perturbed_prediction",
                "clean_score",
                "perturbed_score",
                "prediction_changed",
            ]:
                if column in metadata.columns:
                    row[column] = metadata.at[i, column]
        elif prediction_changed is not None and i < len(prediction_changed):
            row["prediction_changed"] = bool(prediction_changed[i])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-sample metrics with mean/std/median/IQR, split by prediction status when present."""
    metrics = [c for c in df.columns if c not in NON_METRIC_COLUMNS]
    groups = [("all", df)]
    if "prediction_changed" in df.columns:
        groups.extend(
            [
                ("prediction_unchanged", df[df["prediction_changed"] == False]),
                ("prediction_changed", df[df["prediction_changed"] == True]),
            ]
        )
    rows = []
    for group_name, frame in groups:
        for metric in metrics:
            series = pd.to_numeric(frame[metric], errors="coerce").dropna()
            if series.empty:
                rows.append(
                    {
                        "group": group_name,
                        "metric": metric,
                        "mean": np.nan,
                        "std": np.nan,
                        "median": np.nan,
                        "iqr": np.nan,
                        "n": 0,
                        "n_missing": int(len(frame)),
                    }
                )
            else:
                rows.append(
                    {
                        "group": group_name,
                        "metric": metric,
                        "mean": series.mean(),
                        "std": series.std(ddof=0),
                        "median": series.median(),
                        "iqr": series.quantile(0.75) - series.quantile(0.25),
                        "n": int(series.size),
                        "n_missing": int(len(frame) - series.size),
                    }
                )
    return pd.DataFrame(rows)
