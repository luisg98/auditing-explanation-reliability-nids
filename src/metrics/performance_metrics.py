"""Predictive performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score


def _binary_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    return {
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def _threshold_table(y_true: np.ndarray, scores: np.ndarray, beta: float) -> pd.DataFrame:
    rows = []
    thresholds = np.linspace(0.0, 1.0, 1001)
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        rates = _binary_rates(y_true, y_pred)
        rows.append(
            {
                "threshold": float(thr),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "f2": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
                **rates,
            }
        )
    return pd.DataFrame(rows)


def tune_binary_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    strategy: str = "best_f1",
    max_fpr: float = 0.05,
    beta: float = 2.0,
) -> tuple[float, dict[str, float | str]]:
    """Select a binary decision threshold on validation scores."""
    table = _threshold_table(y_true, scores, beta)
    strategy = str(strategy)
    if strategy == "min_fnr_at_max_fpr":
        candidates = table[table["fpr"] <= float(max_fpr)].copy()
        if candidates.empty:
            candidates = table.copy()
            selected_strategy = "fallback_min_fpr"
            sort_cols = ["fpr", "fnr", "threshold"]
            ascending = [True, True, False]
        else:
            selected_strategy = strategy
            sort_cols = ["fnr", "fpr", "f2", "threshold"]
            ascending = [True, True, False, False]
        best = candidates.sort_values(sort_cols, ascending=ascending).iloc[0]
    elif strategy == "best_f2":
        selected_strategy = strategy
        best = table.sort_values(["f2", "fnr", "fpr"], ascending=[False, True, True]).iloc[0]
    else:
        selected_strategy = "best_f1"
        best = table.sort_values(["f1", "fnr", "fpr"], ascending=[False, True, True]).iloc[0]
    info = {
        "threshold_strategy": selected_strategy,
        "threshold_max_fpr": float(max_fpr),
        "threshold_validation_precision": float(best["precision"]),
        "threshold_validation_recall": float(best["recall"]),
        "threshold_validation_f1": float(best["f1"]),
        "threshold_validation_f2": float(best["f2"]),
        "threshold_validation_fpr": float(best["fpr"]),
        "threshold_validation_fnr": float(best["fnr"]),
    }
    return float(best["threshold"]), info


def compute_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> pd.DataFrame:
    """Compute a one-row performance metrics table."""
    average = "binary" if len(np.unique(y_true)) <= 2 else "macro"
    data = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if len(np.unique(np.concatenate([y_true, y_pred]))) <= 2:
        data.update({"f2": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)})
        data.update({key: value for key, value in _binary_rates(y_true, y_pred).items() if key in {"fpr", "fnr"}})
    if scores is not None:
        try:
            if np.asarray(scores).ndim == 1:
                data["roc_auc"] = roc_auc_score(y_true, scores)
            else:
                data["roc_auc"] = roc_auc_score(y_true, scores, multi_class="ovr")
        except Exception:
            data["roc_auc"] = np.nan
    return pd.DataFrame([data])


def confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return a labeled confusion matrix dataframe."""
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
