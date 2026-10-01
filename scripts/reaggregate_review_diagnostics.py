#!/usr/bin/env python3
"""Rebuild saved RQ2 sensitivity summaries from included per-source records.

This stage performs no model inference or attribution calculation. Use --write
only when intentionally updating the derived summary files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.revision_statistics import crossed_ci, stable_seed

OUT = ROOT / "results/reliability_diagnostics"
DETAIL = OUT / "rq2_sensitivity_per_flow.csv.gz"


def build_outputs() -> dict[str, pd.DataFrame]:
    detail = pd.read_csv(DETAIL)
    required = {
        "task_id", "model", "method", "operator", "functional_check",
        "seed", "flow_id", "target_class_id", "variant", "value",
    }
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Per-source record file is missing columns: {sorted(missing)}")

    rows, changes, contrasts = [], [], []
    keys = ["task_id", "model", "method", "operator", "functional_check"]
    for identity, group in detail.groupby(keys, sort=True):
        base = dict(zip(keys, identity))
        stable = group[group.variant.eq("stable")]
        for variant, frame in group.groupby("variant", sort=True):
            if len(frame) != 80:
                continue
            ci = crossed_ci(frame, "value", 5000, stable_seed("review", *identity, variant))
            rows.append(base | dict(variant=variant, **ci))
            paired = frame.merge(
                stable,
                on=["seed", "flow_id", "target_class_id"],
                suffixes=("", "_stable"),
                validate="one_to_one",
            )
            paired["difference"] = paired.value - paired.value_stable
            change = crossed_ci(
                paired,
                "difference",
                5000,
                stable_seed("review_change", *identity, variant),
            )
            changes.append(
                base
                | dict(
                    variant=variant,
                    **change,
                    max_source_change=float(paired.difference.abs().max()),
                )
            )

    contrast_keys = ["task_id", "model", "method", "functional_check", "variant"]
    for identity, group in detail.groupby(contrast_keys, sort=True):
        base = dict(zip(contrast_keys, identity))
        median = group[group.operator.eq("training_median")]
        for operator in ["conditional_gaussian_mean", "joint_donor"]:
            frame = group[group.operator.eq(operator)]
            if len(frame) != 80:
                continue
            paired = frame.merge(
                median,
                on=["seed", "flow_id", "target_class_id"],
                suffixes=("", "_median"),
                validate="one_to_one",
            )
            paired["difference"] = paired.value - paired.value_median
            ci = crossed_ci(
                paired,
                "difference",
                5000,
                stable_seed("review_contrast", *identity, operator),
            )
            contrasts.append(base | dict(operator=operator, **ci))

    bounds = pd.read_csv(OUT / "rq2_boundaries.csv")
    residuals = pd.read_csv(OUT / "ig_rq2_residuals.csv")
    if len(bounds) != 8000:
        raise ValueError(f"Expected 8,000 boundary records; found {len(bounds)}")
    if residuals.empty:
        raise ValueError("The included IG residual records are empty")

    boundary_summary = bounds.groupby(
        ["task_id", "model", "method", "fraction", "k"]
    ).agg(
        records=("flow_id", "size"),
        nonidentifiable=("nonidentifiable", "sum"),
        fraction_nonidentifiable=("nonidentifiable", "mean"),
        constant_accounts=("constant_account", "sum"),
    )
    residual_summary = residuals.groupby(["task_id", "model", "steps"]).agg(
        records=("flow_id", "size"),
        median=("absolute_residual", "median"),
        p95=("absolute_residual", lambda x: x.quantile(.95)),
        maximum=("absolute_residual", "max"),
        above_tolerance=("absolute_residual", lambda x: int((x > .01).sum())),
    )

    case_rows = []
    case_keys = ["task_id", "model", "seed", "flow_id", "target_class_id", "functional_check"]
    ig = detail[
        detail.method.eq("integrated_gradients")
        & detail.variant.isin(["stable", "steps_128", "final_resolution"])
    ]
    for operator in ["conditional_gaussian_mean", "joint_donor"]:
        frame = ig[ig.operator.eq(operator)].merge(
            ig[ig.operator.eq("training_median")],
            on=case_keys + ["variant"],
            suffixes=("", "_median"),
            validate="one_to_one",
        )
        frame["operator_contrast"] = frame.value - frame.value_median
        cases = frame.pivot(
            index=case_keys, columns="variant", values="operator_contrast"
        ).reset_index()
        cases["operator"] = operator
        cases["change_128_vs_64"] = cases.steps_128 - cases.stable
        cases["change_final_vs_64"] = cases.final_resolution - cases.stable
        case_rows.append(cases)
    cases = pd.concat(case_rows, ignore_index=True)
    residual_keys = ["task_id", "model", "seed", "flow_id", "target_class_id"]
    initial = residuals[residuals.steps.eq(64)][
        residual_keys + ["absolute_residual"]
    ].rename(columns={"absolute_residual": "residual_64"})
    final = residuals.sort_values("steps").drop_duplicates(residual_keys, keep="last")[
        residual_keys + ["steps", "absolute_residual"]
    ]
    cases = cases.merge(initial, on=residual_keys, validate="many_to_one").merge(
        final, on=residual_keys, validate="many_to_one"
    )
    cases["high_initial_residual"] = cases.residual_64 > .01

    outputs = {
        "rq2_boundary_summary.csv": boundary_summary.reset_index(),
        "rq2_sensitivity_summary.csv": pd.DataFrame(rows),
        "rq2_sensitivity_changes.csv": pd.DataFrame(changes),
        "rq2_sensitivity_contrasts.csv": pd.DataFrame(contrasts),
        "ig_rq2_residual_summary.csv": residual_summary.reset_index(),
        "ig_case_diagnostics.csv": cases,
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="replace saved derived summaries")
    parser.add_argument("--check", action="store_true", help="compare rebuilt summaries to saved outputs")
    args = parser.parse_args()
    if args.write == args.check:
        parser.error("choose exactly one of --write or --check")
    checks = pd.read_csv(OUT / "baseline_reconstruction_checks.csv")
    if len(checks) != 100 or checks.baseline_max_absolute_difference.max() > 3e-6:
        raise SystemExit("The 100-record baseline reconstruction checks did not pass")
    outputs = build_outputs()
    for name, rebuilt in outputs.items():
        path = OUT / name
        if args.write:
            rebuilt.to_csv(path, index=False)
            continue
        stored = pd.read_csv(path)
        try:
            pd.testing.assert_frame_equal(
                rebuilt.reset_index(drop=True),
                stored.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as error:
            raise SystemExit(f"Stored summary differs from source records: {name}\n{error}")
    action = "wrote" if args.write else "verified"
    print(f"{action} {len(outputs)} summaries from source-level records; no model calls made")


if __name__ == "__main__":
    main()
