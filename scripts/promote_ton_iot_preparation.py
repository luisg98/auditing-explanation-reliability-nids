#!/usr/bin/env python3
"""Promote the validated ToN-IoT preparation into the shared training tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "results/leakage_free_tabular_ton_iot_prep/processed/ton_iot_binary"
)
DEFAULT_TARGET = Path("results/leakage_free_tabular/processed/ton_iot_binary")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()

    source = _resolve(args.source)
    target = _resolve(args.target)
    decision_path = source.parents[1] / "decision.json"
    if not source.is_dir():
        raise FileNotFoundError(f"Missing staged preparation: {source}")
    if not decision_path.is_file():
        raise FileNotFoundError(f"Missing staged preparation decision: {decision_path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "complete":
        raise RuntimeError(f"Staged preparation is not complete: {decision_path}")
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing promoted preparation: {target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, copy_function=shutil.copy2)
    print(f"promoted {source.relative_to(ROOT)} -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
