"""Shared XAI utilities."""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd


def global_importance(explanations: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean absolute global feature importance."""
    importance = np.mean(np.abs(explanations), axis=0)
    return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    )


def save_explanation_outputs(
    explanations: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    xai: str,
    suffix: str,
    runtime_seconds: float,
) -> None:
    """Save local explanations, global importance, top features, and runtime."""
    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / f"{xai}_{suffix}.npy"
    tmp_array_path = output_dir / f".{array_path.name}.{os.getpid()}.tmp.npy"
    try:
        np.save(tmp_array_path, explanations)
        os.replace(tmp_array_path, array_path)
    finally:
        if tmp_array_path.exists():
            tmp_array_path.unlink()
    imp = global_importance(explanations, feature_names)
    imp.to_csv(output_dir / f"{xai}_global_importance_{suffix}.csv", index=False)
    imp.head(20).to_csv(output_dir / f"{xai}_top_features_{suffix}.csv", index=False)
    pd.DataFrame([{"xai": xai, "artifact": suffix, "runtime_seconds": runtime_seconds}]).to_csv(
        output_dir / f"{xai}_runtime_{suffix}.csv", index=False
    )
