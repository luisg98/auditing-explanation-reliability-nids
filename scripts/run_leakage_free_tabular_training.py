#!/usr/bin/env python3
"""Run validation-only selection and leakage-controlled final training."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Must precede `import pandas`: importing pandas before tensorflow has been
# touched causes tf.data's worker-thread pool to deadlock on its first use in
# this environment (verified by process-level bisection). Importing
# tensorflow first sidesteps it.
import tensorflow as _tensorflow_import_order_guard  # noqa: F401,E402

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.leakage_free_tabular.cache import (  # noqa: E402
    artifact_records,
    sha256_file,
    signature,
    write_cache_metadata,
)
from src.leakage_free_tabular.training import (  # noqa: E402
    ALLOWED_MODELS,
    FINAL_SEEDS,
    implementation_records,
    load_config,
    run_final_condition,
    run_model_selection,
    write_summary_tables,
)


def _parse_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-controlled selection and MLP/tabular-ResNet training."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/leakage_free_tabular_training.yaml"),
    )
    parser.add_argument("--stage", choices=["selection", "final", "all"], default="all")
    parser.add_argument("--tasks", help="Comma-separated task IDs; default: all frozen tasks")
    parser.add_argument("--models", help="Comma-separated models; default: both frozen models")
    parser.add_argument("--seeds", help="Comma-separated final seeds; default: 42--46")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    task_ids = _parse_csv(args.tasks, list(config["tasks"]))
    model_names = _parse_csv(args.models, list(ALLOWED_MODELS))
    seeds = [int(value) for value in _parse_csv(args.seeds, [str(seed) for seed in FINAL_SEEDS])]
    unknown_tasks = sorted(set(task_ids) - set(config["tasks"]))
    unknown_models = sorted(set(model_names) - set(ALLOWED_MODELS))
    unknown_seeds = sorted(set(seeds) - set(FINAL_SEEDS))
    if unknown_tasks or unknown_models or unknown_seeds:
        raise ValueError(
            f"Outside frozen scope: tasks={unknown_tasks}, models={unknown_models}, seeds={unknown_seeds}"
        )
    full_scope = (
        task_ids == list(config["tasks"])
        and model_names == list(ALLOWED_MODELS)
        and seeds == list(FINAL_SEEDS)
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("leakage_free_tabular_training")
    started = time.perf_counter()
    selection_tables: list[pd.DataFrame] = []
    metric_tables: list[pd.DataFrame] = []

    if args.stage in {"selection", "all"}:
        selection_conditions = [(t, m) for t in task_ids for m in model_names]
        logger.info("selection stage: %d task/model conditions queued", len(selection_conditions))
        for task_id, model_name in tqdm(
            selection_conditions, desc="model selection", unit="condition"
        ):
            condition_start = time.perf_counter()
            table, lock = run_model_selection(
                config,
                project_root=ROOT,
                config_path=config_path,
                task_id=task_id,
                model_name=model_name,
                force=args.force,
            )
            selection_tables.append(table)
            logger.info(
                "selection complete %s/%s: candidate=%s, elapsed=%.1fs",
                task_id,
                model_name,
                lock["selected_candidate"]["candidate_id"],
                time.perf_counter() - condition_start,
            )
    else:
        for task_id in task_ids:
            for model_name in model_names:
                path = (
                    ROOT
                    / str(config["paths"]["results_root"])
                    / "model_selection"
                    / task_id
                    / model_name
                    / "candidate_results.csv"
                )
                if not path.is_file():
                    raise FileNotFoundError(f"Missing model-selection table: {path}")
                selection_tables.append(pd.read_csv(path))

    if args.stage in {"final", "all"}:
        final_conditions = [(t, m, s) for t in task_ids for m in model_names for s in seeds]
        logger.info("final stage: %d task/model/seed conditions queued", len(final_conditions))
        for task_id, model_name, seed in tqdm(
            final_conditions, desc="final fit", unit="condition"
        ):
            condition_start = time.perf_counter()
            result = run_final_condition(
                config,
                project_root=ROOT,
                config_path=config_path,
                task_id=task_id,
                model_name=model_name,
                seed=seed,
                force=args.force,
            )
            metric_tables.append(result["metrics"])
            logger.info(
                "final complete %s/%s/seed_%s, elapsed=%.1fs",
                task_id,
                model_name,
                seed,
                time.perf_counter() - condition_start,
            )
    tables: dict[str, Path] = {}
    expected_final_conditions = len(config["tasks"]) * len(ALLOWED_MODELS) * len(FINAL_SEEDS)
    final_coverage_complete = (
        args.stage in {"final", "all"}
        and full_scope
        and len(metric_tables) == expected_final_conditions
    )
    if selection_tables and final_coverage_complete:
        tables = write_summary_tables(
            config,
            project_root=ROOT,
            selection_tables=selection_tables,
            final_metric_tables=metric_tables,
        )
        table_metadata = ROOT / str(config["paths"]["tables_root"]) / "cache_metadata.json"
        config_hash = sha256_file(config_path)
        implementations = implementation_records(ROOT)
        implementation_hash = signature({"files": implementations})
        condition_metadata = sorted(
            (ROOT / str(config["paths"]["results_root"])).glob("**/cache_metadata.json")
        )
        inputs = artifact_records(condition_metadata)
        table_signature = signature(
            {
                "stage": "summary_tables",
                "config_hash": config_hash,
                "implementation_hash": implementation_hash,
                "inputs": inputs,
                "tasks": task_ids,
                "models": model_names,
                "seeds": seeds,
            }
        )
        write_cache_metadata(
            table_metadata,
            stage="summary_tables",
            signature_value=table_signature,
            config_hash=config_hash,
            implementation_hash=implementation_hash,
            inputs=inputs,
            outputs=tables.values(),
        )

    results_root = ROOT / str(config["paths"]["results_root"])
    if final_coverage_complete:
        decision_path = results_root / "training_decision.json"
        status = "complete"
    elif args.stage == "selection" and full_scope:
        decision_path = results_root / "selection_decision.json"
        status = "selection_complete"
    else:
        scope_id = signature(
            {"stage": args.stage, "tasks": task_ids, "models": model_names, "seeds": seeds}
        )[:12]
        decision_path = results_root / "decisions" / f"{args.stage}_{scope_id}.json"
        status = "partial_stage_complete"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision = {
        "pipeline_version": "leakage_free_tabular_v2",
        "status": status,
        "tasks": task_ids,
        "models": model_names,
        "selection_seed": int(config["selection_seed"]),
        "final_seeds": seeds,
        "validation_only_model_selection": True,
        "fixed_processed_split_across_seeds": True,
        "test_used_for_model_or_threshold_selection": False,
        "full_frozen_scope": full_scope,
        "final_condition_count": len(metric_tables),
        "expected_final_condition_count": expected_final_conditions,
        "runtime_seconds": float(time.perf_counter() - started),
        "config_path": config_path.resolve().as_posix(),
        "config_sha256": sha256_file(config_path),
        "implementation_hash": signature({"files": implementation_records(ROOT)}),
        "summary_tables": {key: value.resolve().as_posix() for key, value in tables.items()},
    }
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("pipeline stage %s finished in %.1fs", args.stage, decision["runtime_seconds"])


if __name__ == "__main__":
    main()
