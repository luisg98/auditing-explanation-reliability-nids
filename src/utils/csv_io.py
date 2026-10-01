"""CSV helpers for concurrent pipeline writers."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback.
    fcntl = None


def read_csv_or_empty(path: str | Path) -> pd.DataFrame:
    """Read a CSV, treating missing or zero-column files as empty."""
    p = Path(path)
    try:
        return pd.read_csv(p)
    except (FileNotFoundError, EmptyDataError):
        return pd.DataFrame()


def write_csv_atomic(df: pd.DataFrame, path: str | Path) -> None:
    """Write a CSV via os.replace so readers never observe a half-written file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if df.empty and len(df.columns) == 0:
        if p.exists():
            p.unlink()
        return

    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


@contextmanager
def csv_file_lock(path: str | Path):
    """Serialize read-modify-write updates for one CSV path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_path = p.with_suffix(f"{p.suffix}.lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)


def replace_matching_rows(
    path: str | Path,
    new_rows: pd.DataFrame,
    match_values: dict[str, Any],
) -> None:
    """Atomically replace rows matching match_values, then append new_rows."""
    with csv_file_lock(path):
        existing = read_csv_or_empty(path)
        if not existing.empty and all(column in existing.columns for column in match_values):
            mask = pd.Series(True, index=existing.index)
            for column, value in match_values.items():
                mask &= existing[column].eq(value)
            existing = existing[~mask]

        if new_rows.empty:
            output = existing
        elif existing.empty and len(existing.columns) == 0:
            output = new_rows
        else:
            output = pd.concat([existing, new_rows], ignore_index=True)
        write_csv_atomic(output, path)
