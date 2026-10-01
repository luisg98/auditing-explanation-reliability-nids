from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import ensure_dir, resolve_path
from src.utils.logging import get_logger


def safe_stem(path: Path) -> str:
    """Return a stable filesystem-safe stem."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_")


def prepare_cicids(force: bool = False) -> None:
    """Convert CICIDS2017 raw CSV files into reusable Parquet parts."""
    logger = get_logger("source_parquet")
    src_dir = resolve_path("data/raw/cic_ids")
    out_dir = ensure_dir("data/processed/source_parquet/cicids2017")
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing CICIDS raw directory: {src_dir}")

    csv_files = sorted(src_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CICIDS CSV files found under {src_dir}")

    for csv_path in csv_files:
        out_path = out_dir / f"{safe_stem(csv_path)}.parquet"
        if out_path.exists() and not force:
            logger.info("Keeping existing %s", out_path)
            continue
        logger.info("Converting %s -> %s", csv_path, out_path)
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.astype(str).str.strip()
        df.to_parquet(out_path, index=False)


def link_or_copy(src: Path, dst: Path, force: bool = False) -> None:
    """Create a hard link when possible, otherwise copy the source file."""
    if dst.exists():
        if not force:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_ton_iot(force: bool = False) -> None:
    """Expose the cleaned NF-ToN-IoT v2 Parquet under data/processed/source_parquet."""
    src = resolve_path("data/raw/ton_iot/NF-ToN-IoT-V2.parquet")
    if not src.exists():
        raise FileNotFoundError(f"Missing ToN-IoT v2 Parquet: {src}")
    out_dir = ensure_dir("data/processed/source_parquet/ton_iot")
    link_or_copy(src, out_dir / src.name, force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare stable cleaned Parquet sources.")
    parser.add_argument(
        "--dataset",
        choices=["cicids2017", "ton_iot", "all"],
        default="all",
        help="Dataset source to prepare.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing Parquet sources.")
    args = parser.parse_args()

    if args.dataset in {"cicids2017", "all"}:
        prepare_cicids(args.force)
    if args.dataset in {"ton_iot", "all"}:
        prepare_ton_iot(args.force)
    get_logger("source_parquet").info("Prepared source Parquet files for %s", args.dataset)


if __name__ == "__main__":
    main()
