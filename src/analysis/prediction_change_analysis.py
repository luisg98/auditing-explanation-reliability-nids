"""Analysis helpers for same-prediction explanation drift."""

from __future__ import annotations

import pandas as pd


def same_prediction_different_explanation(
    stability_df: pd.DataFrame,
    dataset: str,
    model: str,
    xai: str,
    perturbation_id: str,
    spearman_threshold: float = 0.5,
    jaccard10_threshold: float = 0.5,
) -> pd.DataFrame:
    """Return rows where prediction is unchanged but explanation similarity is low."""
    if "prediction_changed" not in stability_df.columns:
        return pd.DataFrame()
    frame = stability_df[stability_df["prediction_changed"] == False].copy()
    if frame.empty:
        return frame
    mask = (frame["spearman"].fillna(-1) < spearman_threshold)
    if "jaccard_10" in frame.columns:
        mask |= frame["jaccard_10"].fillna(-1) < jaccard10_threshold
    frame = frame[mask].copy()
    frame.insert(0, "perturbation", perturbation_id)
    frame.insert(0, "xai", xai)
    frame.insert(0, "model", model)
    frame.insert(0, "dataset", dataset)
    return frame
