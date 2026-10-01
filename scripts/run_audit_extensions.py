#!/usr/bin/env python3
"""Run the research extensions declared in configs/audit_extensions.yaml.

The camera-ready protocol is the baseline and is never modified: this launcher
only reads the frozen caches and writes under
results/leakage_free_tabular/audit_extensions/ and
tables/leakage_free_tabular/audit_extensions/.

Examples
--------
    python scripts/run_audit_extensions.py --study 5 --stage thresholds
    python scripts/run_audit_extensions.py --study 4 --stage all
    python scripts/run_audit_extensions.py --study 1 --stage metadata
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Must precede `import pandas`; see src/leakage_free_tabular/epistemic_audit.py.
import tensorflow as _tensorflow_import_order_guard  # noqa: F401,E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.leakage_free_tabular_ext.common import load_extension_context  # noqa: E402

STAGES = {
    "1": ("metadata", "splits", "prepare", "train", "audit", "summarize", "all"),
    "2": ("groups", "perturb", "summarize", "all"),
    "3": ("panel", "probes", "probes-stochastic", "summarize", "all"),
    "4": ("probes", "functional", "summarize", "all"),
    "5": ("thresholds", "lime-convergence", "lime-ablation", "all"),
}


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        required=True,
        choices=sorted(STAGES) + ["all"],
        help="Extension study number",
    )
    parser.add_argument("--stage", default="all")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/audit_extensions.yaml")
    )
    parser.add_argument("--tasks", help="Comma-separated task IDs (subset for a pilot)")
    parser.add_argument("--models", help="Comma-separated model families")
    parser.add_argument("--seeds", help="Comma-separated fitted-model seeds")
    parser.add_argument("--methods", help="Comma-separated probes")
    parser.add_argument("--policies", help="Study 1: comma-separated split policies")
    parser.add_argument("--operators", help="Study 2: comma-separated operator IDs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("shap").setLevel(logging.WARNING)
    logger = logging.getLogger("audit_extensions")
    context = load_extension_context(ROOT, args.config)
    started = time.perf_counter()

    studies = sorted(STAGES) if args.study == "all" else [args.study]
    for study in studies:
        if args.stage != "all" and args.stage not in STAGES[study]:
            raise SystemExit(
                f"Study {study} has no stage {args.stage!r}; choose from {STAGES[study]}"
            )
        stage = args.stage
        logger.info("study %s stage %s starting", study, stage)
        if study == "5":
            _run_study5(context, stage, args, logger)
        elif study == "4":
            _run_study4(context, stage, args, logger)
        elif study == "3":
            _run_study3(context, stage, args, logger)
        elif study == "2":
            _run_study2(context, stage, args, logger)
        elif study == "1":
            _run_study1(context, stage, args, logger)
        logger.info("study %s stage %s done", study, stage)

    logger.info("extensions finished in %.1fs", time.perf_counter() - started)


def _run_study5(context, stage, args, logger) -> None:
    from src.leakage_free_tabular_ext import study5_thresholds

    if stage in {"thresholds", "all"}:
        frames = study5_thresholds.run_threshold_sensitivity(
            context, inventories=_parse_csv(args.methods), logger=logger
        )
        counts = frames["sensitivity_gate_counts"]
        logger.info(
            "threshold sweep: %d summary rows, %d gate-status flips",
            len(frames["sensitivity_summary"]),
            len(frames["sensitivity_gate_flips"]),
        )
        for _, row in counts[counts["setting_id"].eq("baseline")].iterrows():
            logger.info(
                "baseline %s: %d pass / %d fail / %d not evaluable",
                row["inventory"],
                int(row["pass"]),
                int(row["fail"]),
                int(row["not_evaluable"]),
            )
    if stage in {"lime-convergence", "lime-ablation", "all"}:
        from src.leakage_free_tabular_ext import study5_lime

        if stage in {"lime-convergence", "all"}:
            study5_lime.run_convergence(context, logger=logger, force=args.force)
        if stage in {"lime-ablation", "all"}:
            study5_lime.run_kernel_ablation(context, logger=logger, force=args.force)


def _run_study4(context, stage, args, logger) -> None:
    from src.leakage_free_tabular_ext import study4_outcome

    if stage in {"probes", "all"}:
        study4_outcome.run_probes(
            context,
            tasks=_parse_csv(args.tasks),
            models=_parse_csv(args.models),
            seeds=[int(value) for value in (_parse_csv(args.seeds) or [])] or None,
            methods=_parse_csv(args.methods),
            force=args.force,
            logger=logger,
        )
    if stage in {"functional", "all"}:
        study4_outcome.run_functional(
            context,
            tasks=_parse_csv(args.tasks),
            models=_parse_csv(args.models),
            seeds=[int(value) for value in (_parse_csv(args.seeds) or [])] or None,
            methods=_parse_csv(args.methods),
            force=args.force,
            logger=logger,
        )
    if stage in {"summarize", "all"}:
        study4_outcome.run_summaries(context, logger=logger)


def _run_study3(context, stage, args, logger) -> None:
    from src.leakage_free_tabular_ext import study3_seed_vs_architecture as study3

    if stage in {"panel", "all"}:
        study3.build_panel(context, logger=logger, force=args.force)
    if stage in {"probes", "all"}:
        study3.run_probes(
            context,
            methods=_parse_csv(args.methods),
            stochastic=False,
            force=args.force,
            logger=logger,
        )
    if stage in {"probes-stochastic", "all"}:
        study3.run_probes(
            context,
            methods=_parse_csv(args.methods),
            stochastic=True,
            force=args.force,
            logger=logger,
        )
    if stage in {"summarize", "all"}:
        study3.run_summaries(context, logger=logger)


def _run_study2(context, stage, args, logger) -> None:
    from src.leakage_free_tabular_ext import study2_dependency_perturbations as study2

    if stage in {"groups", "all"}:
        study2.build_feature_groups(context, logger=logger, force=args.force)
    if stage in {"perturb", "all"}:
        study2.run_perturbations(
            context,
            tasks=_parse_csv(args.tasks),
            models=_parse_csv(args.models),
            seeds=[int(value) for value in (_parse_csv(args.seeds) or [])] or None,
            methods=_parse_csv(args.methods),
            operators=_parse_csv(args.operators),
            force=args.force,
            logger=logger,
        )
    if stage in {"summarize", "all"}:
        study2.run_summaries(context, logger=logger)


def _run_study1(context, stage, args, logger) -> None:
    from src.leakage_free_tabular_ext import study1_deployment_splits as study1

    policies = _parse_csv(args.policies)
    if stage in {"metadata", "all"}:
        study1.build_flow_metadata(context, logger=logger, force=args.force)
    if stage in {"splits", "all"}:
        study1.build_splits(context, policies=policies, logger=logger, force=args.force)
    if stage in {"prepare", "all"}:
        study1.prepare_partitions(
            context, policies=policies, logger=logger, force=args.force
        )
    if stage in {"train", "all"}:
        study1.train_models(
            context,
            policies=policies,
            seeds=[int(value) for value in (_parse_csv(args.seeds) or [])] or None,
            logger=logger,
            force=args.force,
        )
    if stage in {"audit", "all"}:
        study1.run_audit(
            context,
            policies=policies,
            seeds=[int(value) for value in (_parse_csv(args.seeds) or [])] or None,
            logger=logger,
            force=args.force,
        )
    if stage in {"summarize", "all"}:
        study1.run_summaries(context, policies=policies, logger=logger)


if __name__ == "__main__":
    main()
