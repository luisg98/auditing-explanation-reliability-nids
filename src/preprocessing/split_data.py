"""Train/validation/test splitting."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data with stratification when every class has enough samples."""
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0; got {total}")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=val_size + test_size,
        stratify=stratify,
        random_state=seed,
    )
    relative_test = test_size / (val_size + test_size)
    stratify_temp = y_temp if y_temp.value_counts().min() >= 2 else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test,
        stratify=stratify_temp,
        random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
