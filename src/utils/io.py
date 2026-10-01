"""Input/output helpers for the research pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def project_root() -> Path:
    """Return the repository root from any module under src/."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root when it is not absolute."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_root() -> Path:
    """Return the active results root, optionally overridden for seed runs."""
    return ensure_dir(os.environ.get("RESULTS_ROOT", "results"))


def processed_root() -> Path:
    """Return the active processed-data root, optionally overridden for seed runs."""
    return resolve_path(os.environ.get("PROCESSED_ROOT", "data/processed"))


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file."""
    with resolve_path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save a YAML file."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_json(path: str | Path) -> Any:
    """Load JSON."""
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_array(array: np.ndarray, path: str | Path) -> None:
    """Save a NumPy array."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, array)


def load_array(path: str | Path) -> np.ndarray:
    """Load a NumPy array with a clearer error message."""
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing array: {p}")
    return np.load(p, allow_pickle=False)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a dataframe as CSV."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataframe."""
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing table: {p}")
    return pd.read_csv(p)


def save_joblib(obj: Any, path: str | Path) -> None:
    """Persist a Python object with joblib."""
    import joblib

    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)


def load_joblib(path: str | Path) -> Any:
    """Load a joblib object."""
    import joblib

    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing joblib artifact: {p}")
    return joblib.load(p)


def dataset_config_path(dataset: str) -> Path:
    """Return the conventional config path for a dataset id."""
    return resolve_path("configs") / f"{dataset}.yaml"


def processed_dir(dataset: str) -> Path:
    """Return processed data directory for a dataset id."""
    return processed_root() / dataset


def performance_dir(dataset: str, model_name: str) -> Path:
    """Return model performance artifact directory."""
    return results_root() / "performance" / dataset / model_name


def explanation_dir(dataset: str, model_name: str) -> Path:
    """Return XAI artifact directory."""
    return results_root() / "explanations" / dataset / model_name


def perturbation_dir(dataset: str, model_name: str) -> Path:
    """Return perturbation artifact directory."""
    return results_root() / "perturbations" / dataset / model_name


def stability_dir(dataset: str, model_name: str) -> Path:
    """Return stability artifact directory."""
    return results_root() / "stability" / dataset / model_name
