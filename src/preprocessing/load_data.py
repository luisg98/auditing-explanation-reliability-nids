"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import resolve_path


def _read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported data file type: {path}")


def load_tabular_data(raw_data_path: str | Path, glob_pattern: str | None = None) -> pd.DataFrame:
    """Load CSV or Parquet data from a file or directory."""
    path = resolve_path(raw_data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data path does not exist: {path}. Update the dataset YAML raw_data_path."
        )

    if path.is_file():
        df = _read_file(path)
    else:
        pattern = glob_pattern or "*"
        files = sorted(p for p in path.glob(pattern) if p.is_file())
        if not files:
            files = sorted([*path.glob("*.parquet"), *path.glob("*.csv")])
        if not files:
            raise FileNotFoundError(f"No CSV or Parquet files found under {path}")
        frames = [_read_file(p) for p in files]
        df = pd.concat(frames, ignore_index=True, sort=False)

    df.columns = df.columns.astype(str).str.strip()
    return df
