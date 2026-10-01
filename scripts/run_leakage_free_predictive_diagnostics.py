#!/usr/bin/env python3
"""Build diagnostics from a complete v2 predictive cache without retraining."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.leakage_free_tabular.predictive_diagnostics import (  # noqa: E402
    run_cached_predictive_diagnostics,
)

# ton_iot_binary is prepared by its own, isolated prepare_all() run (see
# amendment ton_iot_retrained_via_v2_pipeline_20260829 in
# configs/leakage_free_epistemic_audit.yaml) rather than by the shared
# CICIDS2017 preparation config, so its provenance must be validated against
# its own config/decision instead of the default --preparation-config.
TASK_PREPARATION_OVERRIDES = {
    "ton_iot_binary": ROOT / "configs/leakage_free_tabular_ton_iot.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the complete leakage-free preparation/training caches and "
            "derive predictive diagnostics without loading or retraining models."
        )
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/leakage_free_tabular_training.yaml"),
    )
    parser.add_argument(
        "--preparation-config",
        type=Path,
        default=Path("configs/leakage_free_tabular.yaml"),
    )
    args = parser.parse_args()
    training_config = (
        args.training_config
        if args.training_config.is_absolute()
        else ROOT / args.training_config
    )
    preparation_config = (
        args.preparation_config
        if args.preparation_config.is_absolute()
        else ROOT / args.preparation_config
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("leakage_free_predictive_diagnostics")
    started = time.perf_counter()
    logger.info("reading predictive caches and deriving diagnostics (no retraining)")
    outputs = run_cached_predictive_diagnostics(
        project_root=ROOT,
        training_config_path=training_config,
        preparation_config_path=preparation_config,
        task_preparation_overrides=TASK_PREPARATION_OVERRIDES,
    )
    for name, path in sorted(outputs.items()):
        logger.info("wrote %s: %s", name, path)
    logger.info(
        "predictive diagnostics finished in %.1fs (%d tables)",
        time.perf_counter() - started,
        len(outputs),
    )


if __name__ == "__main__":
    main()

