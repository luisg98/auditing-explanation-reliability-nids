#!/usr/bin/env python3
"""Reanalyse frozen binary evidence and generate the six-page manuscript assets.

No training, attribution generation, network calls, or historical-output writes.
An old interval is reused only after reconstructing its identical cell estimand
and support. Newly conditional metrics and paired contrasts are bootstrapped
from source/seed records, never from previously reported interval endpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .revision_statistics import crossed_ci, equal_seed_mean, stable_seed
except ImportError:
    from revision_statistics import crossed_ci, equal_seed_mean, stable_seed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper"
ASSETS = ROOT / "latex/assets"
REV = ROOT / "results/source_records"
TABLES = ROOT / "tables/leakage_free_tabular"
EXT = ROOT / "results/leakage_free_tabular/audit_extensions"
TASKS = ["cicids2017_binary", "ton_iot_binary"]
MODELS = ["mlp", "cnn"]
METHODS = ["gradient_x_input", "integrated_gradients", "occlusion", "lime", "shap"]
METHOD_LABEL = dict(zip(METHODS, ["G×I", "IG", "Occlusion", "LIME", "SHAP"]))
OPERATORS = ["training_median", "conditional_gaussian_mean", "joint_donor"]
ARMS = ["deletion", "insertion"]
STRATA = ["true_positive", "false_positive", "true_negative", "false_negative"]
VALUE = "attribution_minus_random_aopc"
INPUTS: set[Path] = set()
SEED_ROWS: list[dict] = []
CHECKS: list[dict] = []


def read(path, **kwargs):
    INPUTS.add(path)
    return pd.read_csv(path, **kwargs)


def require(name, condition, detail=""):
    CHECKS.append(dict(check=name, passed=bool(condition), detail=detail))
    if not condition:
        raise AssertionError((name, detail))


def binary(frame):
    return frame.loc[frame.task_id.isin(TASKS)].copy()


def boolean(series):
    return series.astype(str).str.lower().isin(["true", "1"])


def support(frame):
    return dict(n_unique_sources=frame.flow_id.nunique(),
                n_seed_flow_cells=len(frame[["seed", "flow_id"]].drop_duplicates()),
                n_seeds=frame.seed.nunique(), n_rows=len(frame))


def seed_inventory(frame, value, metadata):
    for seed, group in frame.groupby("seed"):
        SEED_ROWS.append({**metadata, "seed": int(seed),
                          "estimate": equal_seed_mean(group, value), **support(group)})


def verify_existing(group, old, metric, label):
    require(label + ": estimate", np.isclose(equal_seed_mean(group, metric), old.estimate, atol=1e-10, rtol=0))
    counts = support(group)
    for key, count in counts.items():
        require(label + ": " + key, count == old[key], f"{count} vs {old[key]}")
    return {k: old[k] for k in ["estimate", "ci_low", "ci_high", "bootstrap_design"]}


def repeatability(n_resamples):
    path = REV / "statistics/repeatability_metric_fixed.csv.gz"
    columns = ["task_id", "model", "method", "seed", "flow_id", "analysis_role",
               "class_specific_inference_eligible", "spearman", "jaccard_10",
               "rank_pair_identifiable", "top_k_pair_identifiable"]
    detail = binary(read(path, usecols=columns))
    detail = detail[boolean(detail.class_specific_inference_eligible)
                    & detail.analysis_role.ne("numerical_sensitivity")]
    summary_path = TABLES / "epistemic_audit/repeatability_summary.csv"
    original = binary(read(summary_path))
    rows = []
    for identity, group in detail.groupby(["task_id", "model", "method", "analysis_role"]):
        base = dict(zip(["task_id", "model", "method", "analysis_role"], identity))
        for metric, eligible_column in [("spearman", "rank_pair_identifiable"),
                                        ("jaccard_10", "top_k_pair_identifiable")]:
            old = original.loc[original.metric.eq(metric)]
            for k, v in base.items():
                old = old[old[k].eq(v)]
            require("unique repeatability cell", len(old) == 1, str(base))
            old = old.iloc[0]
            inference = verify_existing(group, old, metric, "RQ1 " + str(identity) + metric)
            valid = boolean(group[eligible_column])
            eligible = group.loc[valid]
            coverage = equal_seed_mean(group.assign(coverage=valid.astype(float)), "coverage")
            if not valid.all():
                # LIME intervals must retain shared whole-panel run uncertainty.
                # The frozen binary LIME pairs all have full coverage. Do not
                # silently apply an observed-pairs-only bootstrap if that changes.
                require("partial coverage is not LIME", base["method"] != "lime")
                inference = crossed_ci(eligible, metric, n_resamples, stable_seed("binary", *identity, metric))
            counts = support(group)
            row = {**base, **inference, "metric": metric, "coverage": coverage,
                   "available_sources": counts["n_unique_sources"],
                   "available_seed_flow_cells": counts["n_seed_flow_cells"],
                   "available_pair_rows": counts["n_rows"], **support(eligible),
                   "status": "reaggregated" if not valid.all() else "available_verified",
                   "panel": "stochastic" if base["method"] in ["lime", "shap"] else "central",
                   "target": "fixed correct class shared by MLP and CNN",
                   "reference": "training_median" if base["method"] != "lime" else "resampled_training_background",
                   "operator": "not_applicable", "seed": "42;43;44;45;46",
                   "source": str(path.relative_to(ROOT)),
                   "interval_source": "new conditional source/seed bootstrap" if not valid.all() else str(summary_path.relative_to(ROOT)),
                   "uncertainty_scope": "source/seed + shared-run jackknife" if base["method"] == "lime" else "source/seed; observed probes only"}
            rows.append(row)
            seed_inventory(eligible, metric, {**base, "rq": "RQ1", "metric": metric})
    result = pd.DataFrame(rows)
    require("32 binary repeatability estimates", len(result) == 32)
    result.to_csv(OUT / "rq1_repeatability.csv", index=False)
    return result


def verify_pairing(frame, variants, context):
    """Demand the same seed/source/target inventory, not just equal counts."""
    keys = ["seed", "flow_id", "target_class_id"]
    inventories = []
    for identity, group in frame.groupby(variants):
        require(context + ": unique identities " + str(identity), not group.duplicated(keys).any())
        inventories.append(set(map(tuple, group[keys].to_numpy())))
    require(context + ": identical flow/seed/target sets", bool(inventories) and all(s == inventories[0] for s in inventories))
    return len(inventories[0])


def paired_contrast(group, operator, value=VALUE):
    keys = ["seed", "flow_id", "target_class_id"]
    selected = group[group.operator.eq(operator)][keys + [value]]
    reference = group[group.operator.eq("training_median")][keys + [value]]
    paired = selected.merge(reference, on=keys, suffixes=("_operator", "_median"), validate="one_to_one")
    if len(paired) != len(selected) or len(paired) != len(reference):
        raise ValueError("Unpaired sources/targets in operator contrast")
    return paired.assign(value=paired[value + "_operator"] - paired[value + "_median"])


def operators(n_resamples):
    frames = []
    for task in TASKS:
        for path in sorted((EXT / "study2_dependency_perturbations/operators" / task).glob("*/seed_*/*/operator_aopc_per_flow.csv")):
            frame = read(path)
            frame = frame[frame.operator.isin(OPERATORS) & boolean(frame.class_specific_inference_eligible)].copy()
            frame["source"] = str(path.relative_to(ROOT))
            frames.append(frame)
    detail = pd.concat(frames, ignore_index=True)
    require("100 binary operator source files", len(frames) == 100)
    for identity, group in detail.groupby(["task_id", "model", "functional_check"]):
        n = verify_pairing(group, ["method", "operator"], "RQ2 " + str(identity))
        require("RQ2 80 common source-seed-target records", n == 80)
    # The central eligibility condition implies the two models share targets too.
    for task, group in detail.groupby("task_id"):
        verify_pairing(group, ["model", "method", "operator", "functional_check"], "RQ2 cross-model " + task)
    summary_path = TABLES / "audit_extensions/study2_dependency_perturbations/operator_summary.csv"
    old = binary(read(summary_path))
    old = old[old.metric.eq(VALUE) & old.operator.isin(OPERATORS)]
    keys = ["task_id", "model", "method", "operator", "functional_check"]
    rows = []
    for identity, group in detail.groupby(keys):
        base = dict(zip(keys, identity))
        original = old
        for k, v in base.items():
            original = original[original[k].eq(v)]
        require("unique RQ2 cell", len(original) == 1)
        inference = verify_existing(group, original.iloc[0], VALUE, "RQ2 " + str(identity))
        require("50 RQ2 random controls", group.random_feature_repetitions.eq(50).all())
        rows.append({**base, **inference, **support(group), "metric": VALUE,
                     "coverage": 1., "status": "available_verified", "panel": "common_stochastic",
                     "target": "fixed correct class shared by MLP and CNN", "reference": "training_median attribution; operator-specific replacement",
                     "seed": "42;43;44;45;46", "source": ";".join(sorted(group.source.unique())),
                     "interval_source": str(summary_path.relative_to(ROOT)),
                     "uncertainty_scope": "source/seed; frozen attribution and random-control draws",
                     "random_controls": 50,
                     "attribution_aopc": equal_seed_mean(group, "attribution_effect_aopc"),
                     "random_aopc": equal_seed_mean(group, "random_effect_mean_aopc")})
        seed_inventory(group, VALUE, {**base, "rq": "RQ2", "metric": VALUE})
    contrasts = []
    for identity, group in detail.groupby(["task_id", "model", "method", "functional_check"]):
        base = dict(zip(["task_id", "model", "method", "functional_check"], identity))
        for operator in OPERATORS[1:]:
            paired = paired_contrast(group, operator)
            ci = crossed_ci(paired, "value", n_resamples, stable_seed("binary_operator", *identity, operator))
            contrasts.append({**base, "operator": operator, "reference_operator": "training_median", **ci,
                              "metric": "operator_minus_median_adjusted_aopc", "status": "reaggregated",
                              "panel": "common_stochastic", "target": "fixed correct class",
                              "coverage": 1., "seed": "42;43;44;45;46",
                              "source": ";".join(sorted(group.source.unique())),
                              "uncertainty_scope": "paired source/seed; observed draws only"})
    result = pd.DataFrame(rows)
    require("120 RQ2 effects", len(result) == 120)
    result.to_csv(OUT / "rq2_operator_effects.csv", index=False)
    contrast = pd.DataFrame(contrasts)
    require("80 paired operator contrasts", len(contrast) == 80)
    contrast.to_csv(OUT / "rq2_paired_operator_contrasts.csv", index=False)
    return result, contrast


def outcomes():
    frames = []
    for task in TASKS:
        for path in sorted((EXT / "study4_outcome_reliability/functional" / task).glob("*/seed_*/*/functional_aopc_per_flow.csv")):
            frame = read(path)
            # Low-support FN strata belong in the diagnostic, but must not be
            # promoted to inferentially eligible by filtering away this flag.
            frame = frame[frame.evaluator.eq("training_median")].copy()
            frame["source"] = str(path.relative_to(ROOT))
            frames.append(frame)
    require("60 binary outcome source files", len(frames) == 60)
    detail = pd.concat(frames, ignore_index=True)
    require("RQ3 only three evaluated methods", set(detail.method) == set(METHODS[:3]))
    for identity, group in detail.groupby(["task_id", "model"]):
        verify_pairing(group, ["method", "functional_check"], "RQ3 " + str(identity))
    summary_path = TABLES / "audit_extensions/study4_outcome_reliability/outcome_functional_summary.csv"
    old = binary(read(summary_path))
    old = old[old.stratum_kind.eq("error_type") & old.evaluator.eq("training_median") & old.metric.eq(VALUE)]
    rows = []
    keys = ["task_id", "model", "method", "functional_check", "error_type"]
    for identity, group in detail.groupby(keys):
        base = dict(zip(keys, identity))
        original = old
        for k, v in base.items():
            original = original[original["stratum" if k == "error_type" else k].eq(v)]
        require("unique RQ3 cell", len(original) == 1)
        inference = verify_existing(group, original.iloc[0], VALUE, "RQ3 " + str(identity))
        rows.append({**base, **inference, **support(group), "metric": VALUE,
                     "coverage": 1., "inference_eligible": bool(boolean(group.class_specific_inference_eligible).all()),
                     "status": "available_verified", "panel": "model_specific_outcome",
                     "target": "fixed predicted class: attack for TP/FP, benign for TN/FN",
                     "reference": "training_median", "operator": "training_median",
                     "seed": "42;43;44;45;46", "source": ";".join(sorted(group.source.unique())),
                     "interval_source": str(summary_path.relative_to(ROOT)),
                     "uncertainty_scope": "source/seed; selected stratum, not score-adjusted"})
        seed_inventory(group, VALUE, {**base, "rq": "RQ3", "metric": VALUE})
    result = pd.DataFrame(rows)
    require("96 RQ3 effects", len(result) == 96)
    result.to_csv(OUT / "rq3_outcome_effects.csv", index=False)
    identity = detail.drop_duplicates(["task_id", "model", "seed", "flow_id"])
    support_rows = []
    for keys, group in identity.groupby(["task_id", "model", "error_type"]):
        support_rows.append(dict(zip(["task_id", "model", "error_type"], keys)) | support(group))
    pd.DataFrame(support_rows).to_csv(OUT / "rq3_support.csv", index=False)
    require("9017 binary outcome model-seed records", len(identity) == 9017)
    return result


def interval(row, digits=3):
    return f"{row.estimate:.{digits}f} [{row.ci_low:.{digits}f}, {row.ci_high:.{digits}f}]"


def repeatability_table(result):
    tex = [r"\begin{tabular}{llrrlrlr}", r"\toprule",
           r"Condition & Method/test & $N$ & $S$ & $\rho$ [95\% CI] & $C_\rho$ & $J_{10}$ [95\% CI] & $C_J$ \\", r"\midrule"]
    for task in TASKS:
        for model in MODELS:
            condition = ("CICIDS" if task == TASKS[0] else "ToN-IoT") + "/" + model.upper()
            for i, method in enumerate(["lime", "shap", "integrated_gradients", "occlusion"]):
                group = result[result.task_id.eq(task) & result.model.eq(model) & result.method.eq(method)]
                rho = group[group.metric.eq("spearman")].iloc[0]
                jac = group[group.metric.eq("jaccard_10")].iloc[0]
                label = {"lime": "LIME/R", "shap": "SHAP/R", "integrated_gradients": "IG/Q", "occlusion": "Occl./D"}[method]
                digits = 4 if method == "integrated_gradients" else 3
                tex.append(f"{condition if i == 0 else ''} & {label} & {rho.available_seed_flow_cells} & {rho.available_sources} & {interval(rho, digits)} & {100*rho.coverage:.1f} & {interval(jac)} & {100*jac.coverage:.1f}" + r" \\")
            tex.append(r"\midrule" if not (task == TASKS[-1] and model == MODELS[-1]) else r"\bottomrule")
    tex.append(r"\end{tabular}")
    (ASSETS / "binary_repeatability.tex").write_text("\n".join(tex) + "\n")


def plots(operators_frame, outcomes_frame):
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    colors, markers = ["#0072B2", "#D55E00", "#009E73"], ["o", "s", "^"]
    conditions = [(t, m) for t in TASKS for m in MODELS]
    for kind, frame, labels, variants, variant_column, filename in [
        ("operators", operators_frame, METHODS, OPERATORS, "operator", "binary_operators.pdf"),
        ("outcomes", outcomes_frame, STRATA, METHODS[:3], "method", "binary_outcomes.pdf"),
    ]:
        fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.35 if kind == "operators" else 3.25), sharex=True)
        for col, (task, model) in enumerate(conditions):
            for row, arm in enumerate(ARMS):
                ax = axes[row, col]
                cell = frame[frame.task_id.eq(task) & frame.model.eq(model) & frame.functional_check.eq(arm)]
                ax.axvline(0, color=".55", linewidth=.6)
                ax.grid(axis="x", linewidth=.3, alpha=.35)
                for y, label in enumerate(labels):
                    for k, variant in enumerate(variants):
                        item = cell[cell["method" if kind == "operators" else "error_type"].eq(label)
                                    & cell[variant_column].eq(variant)].iloc[0]
                        yy = y + (k - 1) * .22
                        eligible = kind == "operators" or item.inference_eligible
                        if eligible:
                            ax.plot([item.ci_low, item.ci_high], [yy, yy], color=colors[k], lw=.9)
                        ax.plot(item.estimate, yy, markers[k], color=colors[k], ms=3,
                                markerfacecolor=colors[k] if eligible else "white")
                if kind == "operators":
                    ylabels = [METHOD_LABEL[m] for m in labels]
                else:
                    ylabels = []
                    for label, short in zip(labels, ["TP", "FP", "TN", "FN"]):
                        item = cell[cell.error_type.eq(label)].iloc[0]
                        ylabels.append(f"{short} {item.n_seed_flow_cells}/{item.n_unique_sources}")
                ax.set_yticks(range(len(labels)), ylabels)
                if kind == "operators" and col:
                    ax.set_yticklabels([])
                ax.set_ylim(len(labels) - .5, -.55)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(labelsize=7 if kind == "outcomes" else 8, length=2)
                if row == 0:
                    ax.set_title(("CICIDS" if task == TASKS[0] else "ToN-IoT") + " / " + model.upper(), fontsize=8, pad=4)
                if col == 0:
                    ax.set_ylabel(arm.capitalize(), fontsize=8)
                if row == 1:
                    ax.set_xlabel(r"$\Delta$AOPC", fontsize=8, labelpad=2)
        names = ["Training median", "Conditional Gaussian mean", "Joint donor"] if kind == "operators" else [METHOD_LABEL[m] for m in variants]
        handles = [plt.Line2D([], [], marker=markers[i], color=colors[i], lw=.9, ms=3, label=names[i]) for i in range(3)]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(.5, 0))
        fig.tight_layout(rect=(0, .055, 1, 1), pad=.4, h_pad=.8, w_pad=.65)
        fig.savefig(ASSETS / filename)
        plt.close(fig)


def context_outputs():
    for relative, destination in [
        ("data/panel_counts.csv", "panel_counts.csv"),
        ("data/predictive_by_seed.csv", "predictive_by_seed.csv"),
        ("data/split_class_counts.csv", "split_class_counts.csv"),
        ("xai/ig_completeness_summary.csv", "ig_completeness.csv"),
        ("statistics/fp_tp_matched_summary.csv", "fp_tp_matched_summary.csv"),
        ("statistics/fp_tp_matching_balance.csv", "fp_tp_matching_balance.csv"),
        ("statistics/outcome_score_distributions.csv", "outcome_score_distributions.csv"),
    ]:
        binary(read(REV / relative)).to_csv(OUT / destination, index=False)
    predicted = read(OUT / "predictive_by_seed.csv")
    rows = []
    for identity, group in predicted.groupby(["task_id", "model"]):
        for metric in ["accuracy", "f1", "fpr", "threshold"]:
            rows.append(dict(zip(["task_id", "model"], identity)) | dict(metric=metric, mean=group[metric].mean(), sd=group[metric].std(ddof=1), n_seeds=len(group)))
    pd.DataFrame(rows).to_csv(OUT / "predictive_context.csv", index=False)
    timings = read(REV / "xai/pilot_main_budget/pilot_times.csv")
    cost_rows = []
    arms = {m: m for m in METHODS[:3]} | {"lime": "lime_resampled_background", "shap": "shap_sparse10"}
    for method, arm in arms.items():
        group = timings[timings.method.eq(arm) & timings.run.gt(0)]
        milliseconds = 1000 * group.explanation_seconds
        cost_rows.append(dict(method=method, measured_arm=arm, n=len(group),
                              median_ms=milliseconds.median(), p95_ms=milliseconds.quantile(.95),
                              min_ms=milliseconds.min(), max_ms=milliseconds.max(),
                              unit="one explanation of one input",
                              batch_size=int(group.batch_size.iloc[0]),
                              background_rows=group.background_rows.iloc[0]))
    pd.DataFrame(cost_rows).to_csv(OUT / "cost_microbenchmark.csv", index=False)
    tex = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"Method & Median & Min--max & $n$ \\", r"\midrule"]
    for r in cost_rows:
        label = {"gradient_x_input": r"G$\times$I", "integrated_gradients": "IG",
                 "occlusion": "Occlusion", "lime": "LIME", "shap": "SHAP"}[r["method"]]
        tex.append(f"{label} & {r['median_ms']:,.1f} & {r['min_ms']:,.1f}--{r['max_ms']:,.1f} & {r['n']}" + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (ASSETS / "binary_cost.tex").write_text("\n".join(tex) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()
    if args.resamples < 1000:
        parser.error("Use at least 1000 resamples; manuscript release uses 5000.")
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    print("Reconstructing RQ1: conditional metrics and metric-specific coverage", flush=True)
    rq1 = repeatability(args.resamples)
    print("Verifying RQ2 pairing; reconstructing effects and paired operator contrasts", flush=True)
    rq2, contrasts = operators(args.resamples)
    print("Reconstructing RQ3 outcomes and independent support counts", flush=True)
    rq3 = outcomes()
    context_outputs()
    trace = pd.concat([rq1.assign(rq="RQ1"), rq2.assign(rq="RQ2"), contrasts.assign(rq="RQ2 contrast"), rq3.assign(rq="RQ3")], ignore_index=True)
    unavailable = pd.DataFrame([
        dict(rq="RQ1", method="gradient_x_input", metric="stochastic_variability", status="unavailable", reason="No stochastic repeatability estimand in the frozen protocol"),
        dict(rq="RQ1", method="shap", metric="run_population_uncertainty", status="unavailable", reason="Two observed runs do not estimate arbitrary-run variability"),
        *[dict(rq="RQ3", method=m, metric=VALUE, status="unavailable", reason="Method not executed on outcome panel") for m in ["lime", "shap"]],
    ])
    pd.concat([trace, unavailable], ignore_index=True).to_csv(OUT / "traceability.csv", index=False)
    pd.DataFrame(SEED_ROWS).to_csv(OUT / "seed_estimates.csv", index=False)
    repeatability_table(rq1)
    plots(rq2, rq3)
    manifest = {"scope": TASKS, "seeds": list(range(42, 47)), "new_training": False, "new_attributions": False,
                "resamples": args.resamples, "bootstrap_seed_rule": "revision_statistics.stable_seed; 20260927 + cell identity",
                "command": [sys.executable, *sys.argv], "python": platform.python_version(),
                "packages": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
                "scripts": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                            for p in [Path(__file__).resolve(), ROOT / "scripts/revision_statistics.py"]],
                "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(INPUTS)],
                "checks": CHECKS}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(CHECKS)} checks; wrote {len(trace)} estimates and 4 explicit evidence gaps.", flush=True)


if __name__ == "__main__":
    main()
