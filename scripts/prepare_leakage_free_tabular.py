#!/usr/bin/env python3
"""Build or validate the leakage-free CICIDS2017 tabular tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.leakage_free_tabular.prepare import (  # noqa: E402
    DEFAULT_CONFIG,
    prepare_all,
    validate_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-cache", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if args.validate_cache:
        raise SystemExit(0 if validate_cache(config_path) else 1)
    decision = prepare_all(config_path)
    print(json.dumps({key: decision[key] for key in ("status", "signature")}, indent=2))


if __name__ == "__main__":
    main()
