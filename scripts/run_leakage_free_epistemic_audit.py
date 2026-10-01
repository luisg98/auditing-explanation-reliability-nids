#!/usr/bin/env python3
"""Run the frozen, bounded decision-account audit for clean tabular models."""

from __future__ import annotations

import argparse
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

from src.leakage_free_tabular.epistemic_audit import (  # noqa: E402
    ALLOWED_MODELS,
    ALLOWED_PROBES,
    ALLOWED_TASKS,
    estimate_functional_workload,
    estimate_lime_workload,
    load_context,
    probe_specs,
    run_audit_summaries,
    run_functional_condition,
    run_lime_cost_benchmark,
    run_panel_selection,
    run_parameter_sanity_condition,
    run_probe_condition,
)
from src.leakage_free_tabular.training import FINAL_SEEDS  # noqa: E402


def _parse_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


def _validate_scope(
    tasks: list[str],
    models: list[str],
    seeds: list[int],
    methods: list[str],
) -> None:
    unknown = {
        "tasks": sorted(set(tasks) - set(ALLOWED_TASKS)),
        "models": sorted(set(models) - set(ALLOWED_MODELS)),
        "seeds": sorted(set(seeds) - set(FINAL_SEEDS)),
        "methods": sorted(set(methods) - set(ALLOWED_PROBES)),
    }
    if any(unknown.values()):
        raise ValueError(f"Requested scope is outside the frozen protocol: {unknown}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Attribution-blind panel, bounded probes, functional checks, and "
            "seed-balanced audit summaries."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/leakage_free_epistemic_audit.yaml"),
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/leakage_free_tabular_training.yaml"),
    )
    parser.add_argument(
        "--stage",
        choices=[
            "panel",
            "probes",
            "functional",
            "parameter-sanity",
            "summarize",
            "estimate-lime",
            "estimate-functional",
            "benchmark-lime",
            "all",
        ],
        default="panel",
    )
    parser.add_argument("--tasks", help="Comma-separated frozen task IDs")
    parser.add_argument("--models", help="Comma-separated frozen model families")
    parser.add_argument("--seeds", help="Comma-separated final fitted-model seeds")
    parser.add_argument("--methods", help="Comma-separated registered probes")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--accept-preregistered-lime-cost",
        action="store_true",
        help=(
            "Required before confirmatory stochastic-probe generation. The launcher "
            "writes a workload estimate before checking this flag."
        ),
    )
    parser.add_argument("--benchmark-flows-per-class", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=1)
    args = parser.parse_args()

    config, training_config, paths = load_context(
        ROOT,
        args.config,
        args.training_config,
    )
    tasks = _parse_csv(args.tasks, list(ALLOWED_TASKS))
    models = _parse_csv(args.models, list(ALLOWED_MODELS))
    seeds = [int(value) for value in _parse_csv(args.seeds, [str(v) for v in FINAL_SEEDS])]
    methods = _parse_csv(args.methods, list(ALLOWED_PROBES))
    _validate_scope(tasks, models, seeds, methods)
    full_scope = (
        tasks == list(ALLOWED_TASKS)
        and models == list(ALLOWED_MODELS)
        and seeds == list(FINAL_SEEDS)
        and methods == list(ALLOWED_PROBES)
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # shap's KernelExplainer logs per-explanation internals (weight vectors,
    # phi arrays) at INFO level; at thousands of explanations this drowns out
    # our own per-condition progress logging.
    logging.getLogger("shap").setLevel(logging.WARNING)
    logger = logging.getLogger("leakage_free_epistemic_audit")
    started = time.perf_counter()

    if args.stage in {"panel", "probes", "parameter-sanity", "all"}:
        panel = run_panel_selection(
            config,
            training_config,
            paths,
            force=args.force and args.stage in {"panel", "all"},
        )
        logger.info(
            "attribution-blind panel ready: central=%d, stochastic=%d, outcomes=%d",
            len(panel["central"]),
            int(panel["central"]["in_stochastic_subset"].astype(bool).sum()),
            len(panel["outcome"]),
        )

    if args.stage in {
        "estimate-lime",
        "estimate-functional",
        "benchmark-lime",
    } and not paths.audit_units.is_file():
        raise FileNotFoundError("Run --stage panel before estimating or benchmarking cost")

    if args.stage == "estimate-lime":
        estimate = estimate_lime_workload(pd.read_csv(paths.audit_units), config)
        output_dir = paths.audit_root / "cost_estimate"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "registered_lime_workload.csv"
        estimate.to_csv(output, index=False)
        total = estimate.iloc[-1]
        logger.info(
            "registered stochastic workload: %d explanations, %d requested model evaluations; %s",
            int(total["explanation_calls"]),
            int(total["model_evaluations_requested"]),
            output,
        )
        return

    if args.stage == "estimate-functional":
        estimate = estimate_functional_workload(pd.read_csv(paths.audit_units), config)
        output_dir = paths.audit_root / "cost_estimate"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "registered_functional_workload.csv"
        estimate.to_csv(output, index=False)
        total = estimate.iloc[-1]
        logger.info(
            "registered functional workload: %d model-row evaluations; %s",
            int(total["requested_model_row_evaluations"]),
            output,
        )
        return

    if args.stage == "benchmark-lime":
        for task_id in tasks:
            for model in models:
                for seed in seeds:
                    result = run_lime_cost_benchmark(
                        config,
                        training_config,
                        paths,
                        task_id=task_id,
                        model=model,
                        seed=int(seed),
                        flows_per_class=int(args.benchmark_flows_per_class),
                        runs=int(args.benchmark_runs),
                    )
                    logger.info(
                        "non-confirmatory cost benchmark %s/%s/seed_%d: %.3fs/explanation",
                        task_id,
                        model,
                        seed,
                        float(result["seconds_per_explanation"].mean()),
                    )
        return

    if args.stage in {"probes", "all"}:
        if "lime" in methods:
            estimate = estimate_lime_workload(pd.read_csv(paths.audit_units), config)
            estimate_dir = paths.audit_root / "cost_estimate"
            estimate_dir.mkdir(parents=True, exist_ok=True)
            estimate.to_csv(estimate_dir / "registered_lime_workload.csv", index=False)
            if not args.accept_preregistered_lime_cost:
                total = estimate.iloc[-1]
                raise RuntimeError(
                    "Stochastic confirmatory computation requires "
                    "--accept-preregistered-lime-cost after reviewing "
                    f"{int(total['explanation_calls'])} explanation calls in "
                    f"{estimate_dir / 'registered_lime_workload.csv'}"
                )
        probe_conditions = [
            (task_id, model, seed, method, spec)
            for task_id in tasks
            for model in models
            for seed in seeds
            for method in methods
            for spec in probe_specs(config, method)
        ]
        logger.info("probes stage: %d probe conditions queued", len(probe_conditions))
        for task_id, model, seed, method, spec in tqdm(
            probe_conditions, desc="probe conditions", unit="condition"
        ):
            condition_start = time.perf_counter()
            result = run_probe_condition(
                config,
                training_config,
                paths,
                task_id=task_id,
                model=model,
                seed=int(seed),
                spec=spec,
                force=args.force,
            )
            logger.info(
                "probe %s/%s/seed_%d/%s/%s: rows=%d, elapsed=%.1fs",
                task_id,
                model,
                seed,
                method,
                spec.spec_id,
                len(result.index),
                time.perf_counter() - condition_start,
            )

    if args.stage in {"functional", "all"}:
        functional_estimate = estimate_functional_workload(
            pd.read_csv(paths.audit_units), config
        )
        estimate_dir = paths.audit_root / "cost_estimate"
        estimate_dir.mkdir(parents=True, exist_ok=True)
        functional_estimate.to_csv(
            estimate_dir / "registered_functional_workload.csv", index=False
        )
        functional_conditions = [
            (task_id, model, seed, method)
            for task_id in tasks
            for model in models
            for seed in seeds
            for method in methods
        ]
        logger.info("functional stage: %d conditions queued", len(functional_conditions))
        for task_id, model, seed, method in tqdm(
            functional_conditions, desc="functional conditions", unit="condition"
        ):
            condition_start = time.perf_counter()
            result = run_functional_condition(
                config,
                training_config,
                paths,
                task_id=task_id,
                model=model,
                seed=int(seed),
                method=method,
                force=args.force,
            )
            logger.info(
                "functional %s/%s/seed_%d/%s: flows=%d, elapsed=%.1fs",
                task_id,
                model,
                seed,
                method,
                result["aopc"]["flow_id"].nunique(),
                time.perf_counter() - condition_start,
            )

    if args.stage in {"parameter-sanity", "all"}:
        sanity_methods = [
            method
            for method in methods
            if method in set(config["parameter_dependence_sanity"]["probes"])
        ]
        if not sanity_methods:
            raise ValueError("No registered parameter-sanity probe is in the requested scope")
        sanity_conditions = [
            (task_id, model, seed, method)
            for task_id in tasks
            for model in models
            for seed in seeds
            for method in sanity_methods
        ]
        logger.info("parameter-sanity stage: %d conditions queued", len(sanity_conditions))
        for task_id, model, seed, method in tqdm(
            sanity_conditions, desc="parameter sanity", unit="condition"
        ):
            condition_start = time.perf_counter()
            detail = run_parameter_sanity_condition(
                config,
                training_config,
                paths,
                task_id=task_id,
                model=model,
                seed=int(seed),
                method=method,
                force=args.force,
            )
            logger.info(
                "parameter sanity %s/%s/seed_%d/%s: flows=%d, elapsed=%.1fs",
                task_id,
                model,
                seed,
                method,
                len(detail),
                time.perf_counter() - condition_start,
            )

    if args.stage in {"summarize", "all"}:
        if not full_scope:
            raise ValueError("Confirmatory summaries require the complete frozen scope")
        tables = run_audit_summaries(config, training_config, paths)
        logger.info("audit summaries written: %d tables", len(tables))

    logger.info("stage %s finished in %.1fs", args.stage, time.perf_counter() - started)


if __name__ == "__main__":
    main()
