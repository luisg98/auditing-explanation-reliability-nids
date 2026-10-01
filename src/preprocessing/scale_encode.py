"""Feature scaling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def build_scaler(scaler_type: str):
    """Create a scaler from a config string."""
    scaler_type = scaler_type.lower()
    if scaler_type == "minmax":
        return MinMaxScaler()
    if scaler_type == "standard":
        return StandardScaler()
    raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Use 'standard' or 'minmax'.")


def fit_transform_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Fit scaler on train and transform all splits."""
    scaler = build_scaler(scaler_type)
    X_train_s = scaler.fit_transform(X_train).astype("float32")
    X_val_s = scaler.transform(X_val).astype("float32")
    X_test_s = scaler.transform(X_test).astype("float32")
    for split_name, values in {
        "train": X_train_s,
        "validation": X_val_s,
        "test": X_test_s,
    }.items():
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values found after scaling the {split_name} split. "
                "Check the dataset config for extreme values; set max_abs_value "
                "or clip_quantiles before fitting the scaler."
            )
    return X_train_s, X_val_s, X_test_s, scaler
