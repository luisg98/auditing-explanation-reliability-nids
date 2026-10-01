"""Tabular cleaning and label preparation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _sanitize_numeric_extremes(X: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Replace impossible numeric extremes and optionally clip heavy tails."""
    max_abs_value = config.get("max_abs_value")
    if max_abs_value is not None:
        threshold = float(max_abs_value)
        if threshold <= 0:
            raise ValueError("max_abs_value must be positive when provided.")
        X = X.mask(X.abs() > threshold)

    clip_cfg = config.get("clip_quantiles")
    if clip_cfg:
        lower_q = float(clip_cfg.get("lower", 0.0))
        upper_q = float(clip_cfg.get("upper", 1.0))
        if not 0.0 <= lower_q < upper_q <= 1.0:
            raise ValueError("clip_quantiles must satisfy 0 <= lower < upper <= 1.")
        lower = X.quantile(lower_q, numeric_only=True)
        upper = X.quantile(upper_q, numeric_only=True)
        X = X.clip(lower=lower, upper=upper, axis=1)

    return X


def clean_features(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Drop configured columns, coerce features to numeric, and encode labels."""
    label_col = config.get("label_column", "Label")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' was not found. Available columns: {list(df.columns)[:20]}")

    drop_columns = [c for c in config.get("columns_to_drop", []) if c in df.columns and c != label_col]
    df = df.drop(columns=drop_columns)

    y_raw = df[label_col].astype(str).str.strip()
    X = df.drop(columns=[label_col]).copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = _sanitize_numeric_extremes(X, config)
    if config.get("drop_all_nan_features", True):
        X = X.dropna(axis=1, how="all")

    missing_strategy = config.get("missing_strategy", "mean")
    if missing_strategy == "drop":
        keep = ~X.isna().any(axis=1)
        X = X.loc[keep]
        y_raw = y_raw.loc[keep]
    elif missing_strategy == "median":
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
    elif missing_strategy == "zero":
        X = X.fillna(0)
    else:
        X = X.fillna(X.mean(numeric_only=True)).fillna(0)

    if config.get("binary", True):
        benign_labels = {str(v).lower() for v in config.get("benign_labels", ["BENIGN", "Benign", "normal", "Normal"])}
        y = (~y_raw.str.lower().isin(benign_labels)).astype(int)
        label_mapping = {"benign": 0, "attack": 1}
    else:
        labels = sorted(y_raw.dropna().unique().tolist())
        label_mapping = {label: i for i, label in enumerate(labels)}
        y = y_raw.map(label_mapping).astype(int)

    return X, y, label_mapping


def dataset_summary(df: pd.DataFrame, dataset: str, label_column: str) -> pd.DataFrame:
    """Build a one-row dataset summary table."""
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "n_rows": int(len(df)),
                "n_columns": int(df.shape[1]),
                "n_features_before_drop": int(max(df.shape[1] - 1, 0)),
                "label_column": label_column,
                "n_classes_raw": int(df[label_column].nunique()) if label_column in df.columns else 0,
            }
        ]
    )


def class_distribution(y: pd.Series, dataset: str, split: str) -> pd.DataFrame:
    """Build a class distribution table for one split."""
    counts = y.value_counts().sort_index()
    total = max(int(counts.sum()), 1)
    return pd.DataFrame(
        {
            "dataset": dataset,
            "split": split,
            "class_id": counts.index.astype(int),
            "count": counts.values.astype(int),
            "proportion": counts.values / total,
        }
    )
