"""Model evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics.performance_metrics import compute_performance_metrics, confusion_matrix_df
from src.models.factory import predict_labels, predict_scores


def evaluate_model(model, X, y, model_name: str, threshold: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return predictions, metrics, and confusion matrix dataframes."""
    scores = predict_scores(model, X, model_name)
    if not np.isfinite(np.asarray(scores, dtype=float)).all():
        raise RuntimeError("Model produced non-finite test scores; refusing to save invalid predictions.")
    preds = predict_labels(model, X, model_name, threshold)
    score_col = "score" if getattr(scores, "ndim", 1) == 1 else "probabilities"
    pred_df = pd.DataFrame({"y_true": y, "y_pred": preds})
    if score_col == "score":
        pred_df["score"] = scores
    else:
        pred_df["probabilities"] = [list(row) for row in scores]
    metrics_df = compute_performance_metrics(y, preds, scores)
    cm_df = confusion_matrix_df(y, preds)
    return pred_df, metrics_df, cm_df
