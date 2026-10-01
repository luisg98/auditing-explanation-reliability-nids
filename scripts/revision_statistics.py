#!/usr/bin/env python3
"""Rebuild estimates from frozen source-level outputs; no training or XAI execution.

Original intervals are retained only where
the point estimate is independently reconstructed. New intervals are labelled
exploratory and resample source IDs jointly with fitted seeds. No interval is
invented for a mean of reported cell estimates.
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables/leakage_free_tabular"
OUT = ROOT / "results/paper"
def boolean(series):
    return series.astype(str).str.lower().isin(["true", "1"])

EXT = ROOT / "tables/leakage_free_tabular/audit_extensions"
RESULT_EXT = ROOT / "results/leakage_free_tabular/audit_extensions"
ASSETS = ROOT / "latex/assets"
TASKS = ["cicids2017_binary", "cicids2017_multiclass", "ton_iot_binary"]
TASK_NAMES = dict(zip(TASKS, ["CICIDS binary", "CICIDS multiclass", "ToN-IoT binary"]))
METHODS = ["gradient_x_input", "integrated_gradients", "occlusion", "lime", "shap"]
SHORT = dict(zip(METHODS, ["GxI", "IG", "Occl.", "LIME", "SHAP"]))
SEED = 20260927
INPUTS: set[Path] = set()


def read(path: Path, **kwargs):
    INPUTS.add(path)
    return pd.read_csv(path, **kwargs)


def seed_means(frame, value):
    return frame.groupby(["seed", "flow_id"])[value].mean().groupby("seed").mean()


def equal_seed_mean(frame, value):
    return float(seed_means(frame, value).mean())


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, [SEED, *parts])).encode()).digest()[:4], "little")


def crossed_ci(frame, value="value", n_resamples=5000, seed=SEED):
    """Equal-seed mean with one global source draw per replicate.

    Values must already be per-source paired contrasts, or means conditional
    on the observed probe runs. This function makes NO stochastic-run claim.
    Missing cells are handled by normalising source weights inside each seed.
    """
    cells = frame.groupby(["seed", "flow_id"], as_index=False)[value].mean().dropna(subset=[value])
    if cells.empty:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_unique_sources": 0, "n_seed_flow_cells": 0, "n_seeds": 0}
    matrix = cells.pivot(index="flow_id", columns="seed", values=value)
    x = matrix.to_numpy()
    observed = np.isfinite(x)
    values = np.nan_to_num(x)
    ns, nk = x.shape
    estimate = float(np.nanmean(x, axis=0).mean())
    rng = np.random.default_rng(seed)
    draws = []
    while sum(len(v) for v in draws) < n_resamples:
        count = min(200, n_resamples - sum(len(v) for v in draws))
        weights = rng.multinomial(ns, np.repeat(1 / ns, ns), size=count)
        counts = weights @ observed
        valid = (counts > 0).all(axis=1)
        if not valid.any():
            continue
        means = (weights[valid] @ values) / counts[valid]
        selected = rng.integers(0, nk, size=(len(means), nk))
        draws.append(np.take_along_axis(means, selected, axis=1).mean(axis=1))
    draws = np.concatenate(draws)[:n_resamples]
    low, high = np.quantile(draws, [.025, .975])
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high),
            "n_unique_sources": ns, "n_seed_flow_cells": len(cells), "n_seeds": nk,
            "bootstrap_design": "crossed_source_seed_equal_seed_observed_runs_conditional", "n_resamples": n_resamples}


def seed_inventory(frame, keys, value, analysis):
    source = frame.groupby([*keys, "seed", "flow_id"], as_index=False)[value].mean()
    points = source.groupby([*keys, "seed"], as_index=False).agg(estimate=(value, "mean"), n_sources=("flow_id", "nunique"))
    points.insert(0, "analysis", analysis)
    loso = []
    for identity, group in points.groupby(keys):
        base = dict(zip(keys, identity if isinstance(identity, tuple) else [identity]))
        for seed in group.seed:
            loso.append({"analysis": analysis, **base, "left_out_seed": seed, "estimate": group.loc[group.seed.ne(seed), "estimate"].mean(), "n_seeds": len(group) - 1})
    return points, pd.DataFrame(loso)


def tex_escape(text):
    return str(text).replace("_", r"\_").replace("%", r"\%")


def interval(row, digits=3):
    return f"{row['estimate']:.{digits}f} [{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"


def repeatability(n_resamples):
    summary = read(TABLES / "repeatability_summary.csv")
    detail = read(OUT / "repeatability_metric_fixed.csv.gz")
    detail = detail[boolean(detail.class_specific_inference_eligible)]
    rows, seed_rows, loso_rows = [], [], []
    keys = ["task_id", "model", "method", "analysis_role"]
    for identity, group in detail.groupby(keys, sort=True):
        base = dict(zip(keys, identity))
        original = summary.copy()
        for key, value in base.items():
            original = original[original[key].eq(value)]
        original = original[original.metric.eq("spearman")].iloc[0]
        # The correction file has already verified every original 0/1 shortcut.
        eligible = group[boolean(group.rank_pair_identifiable)]
        corrected_screen = equal_seed_mean(group, "spearman")
        assert np.isclose(corrected_screen, original.estimate, rtol=0, atol=1e-10), (base, corrected_screen, original.estimate)
        coverage = equal_seed_mean(group.assign(covered=boolean(group.rank_pair_identifiable).astype(float)), "covered")
        if np.isclose(coverage, 1):
            inference = {key: original[key] for key in ["estimate", "ci_low", "ci_high", "n_unique_sources", "n_seed_flow_cells", "n_seeds", "bootstrap_design"]}
            inference["interval_provenance"] = "VERIFICADA: original interval; same estimand and reconstructed estimate"
        else:
            inference = crossed_ci(eligible, "spearman", n_resamples, stable_seed("repeat_conditional", *identity))
            inference["interval_provenance"] = "NOVA_EXECUÇÃO: exploratory conditional estimate from frozen pairs"
        rows.append({**base, **inference, "rank_coverage": coverage, "available_sources": group.flow_id.nunique(),
                     "available_seed_source_cells": len(group[["seed", "flow_id"]].drop_duplicates()),
                     "eligible_pair_rows": len(eligible), "available_pair_rows": len(group),
                     "screening_score": corrected_screen, "screening_ci_low": original.ci_low, "screening_ci_high": original.ci_high,
                     "run_population_limit": "two observed runs; run-population uncertainty unestimated" if base["method"] == "shap" else
                         "whole-panel run jackknife included" if base["method"] == "lime" else "numerical or deterministic contrast"})
        p, l = seed_inventory(eligible, keys, "spearman", "repeatability_conditional")
        seed_rows.append(p); loso_rows.append(l)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "repeatability_conditional.csv", index=False)
    pd.concat(seed_rows).to_csv(OUT / "repeatability_seed_points.csv", index=False)
    pd.concat(loso_rows).to_csv(OUT / "repeatability_leave_one_seed_out.csv", index=False)
    tex = [r"\begin{tabular}{llccc}", r"\toprule", r"Task & Model & LIME $\rho$ [95\% CI] & SHAP $\rho$ [95\% CI] & SHAP coverage \\", r"\midrule"]
    for task in TASKS:
        for model in ["mlp", "cnn"]:
            cell = result[result.task_id.eq(task) & result.model.eq(model)]
            lime, shap = (cell[cell.method.eq(m)].iloc[0] for m in ["lime", "shap"])
            tex.append(f"{TASK_NAMES[task]} & {model.upper()} & {interval(lime)} & {interval(shap)} & {shap.rank_coverage:.1%}".replace("%", r"\%") + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (ASSETS / "repeatability_disaggregated.tex").write_text("\n".join(tex) + "\n")
    return result


def functional():
    detail = read(TABLES / "functional_aopc_per_flow.csv")
    detail = detail[boolean(detail.in_central) & boolean(detail.class_specific_inference_eligible)]
    summary = read(TABLES / "functional_summary.csv")
    keys = ["task_id", "model", "method", "functional_check", "evaluator"]
    means = detail.groupby(keys).apply(lambda x: equal_seed_mean(x, "attribution_minus_random_aopc"), include_groups=False)
    for row in summary.itertuples():
        key = tuple(getattr(row, k) for k in keys)
        assert np.isclose(means.loc[key], row.estimate, rtol=0, atol=1e-10), key
    summary["evidence_status"] = "VERIFICADA"
    summary.to_csv(OUT / "functional_disaggregated.csv", index=False)
    points, loso = seed_inventory(detail, keys, "attribution_minus_random_aopc", "functional")
    points.to_csv(OUT / "functional_seed_points.csv", index=False)
    loso.to_csv(OUT / "functional_leave_one_seed_out.csv", index=False)
    absolute = detail.groupby(keys).agg(attribution_aopc_mean=("attribution_aopc", "mean"), random_aopc_mean=("random_aopc", "mean"),
                                       mc_se_mean=("random_aopc_monte_carlo_standard_error", "mean"),
                                       displacement_difference_mean=("attribution_minus_matched_displacement", "mean"))
    # These are explicitly record-weighted diagnostics, not replacement primary estimators.
    absolute.to_csv(OUT / "functional_absolute_record_weighted_diagnostics.csv")
    plot_functional_combined(summary, points)
    return summary


def plot_functional_combined(summary, points):
    """LNCS native-width figure; fonts remain legible at 12.2 cm text width."""
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 2, figsize=(4.803, 5.6), sharey=True)
    for row_index, task in enumerate(TASKS):
        for col, check in enumerate(["deletion", "insertion"]):
            ax = axes[row_index, col]
            ax.axvline(0, color=".7", linewidth=.6)
            ax.grid(axis="x", linewidth=.3, alpha=.3)
            for i, method in enumerate(METHODS):
                for model, offset, marker in [("mlp", -.16, "o"), ("cnn", .16, "s")]:
                    mask = (summary.task_id.eq(task) & summary.model.eq(model) & summary.method.eq(method)
                            & summary.functional_check.eq(check) & summary.evaluator.eq("training_median"))
                    cell = summary[mask].iloc[0]
                    p = points[points.task_id.eq(task) & points.model.eq(model) & points.method.eq(method)
                               & points.functional_check.eq(check) & points.evaluator.eq("training_median")]
                    color = "0.1" if model == "mlp" else "0.5"
                    ax.scatter(p.estimate, np.full(len(p), i+offset), marker="|", color=color, s=15, alpha=.45)
                    ax.plot([cell.ci_low, cell.ci_high], [i+offset]*2, color=color, linewidth=.85)
                    ax.plot(cell.estimate, i+offset, marker, color=color, markersize=3,
                            markerfacecolor="white" if model == "mlp" else color)
            ax.set_yticks(range(5), [SHORT[m] for m in METHODS])
            ax.set_ylim(4.55,-.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7, length=2)
            ax.set_title(TASK_NAMES[task] + (" / deletion" if col == 0 else " / insertion"), fontsize=7.5, pad=5)
            all_check = summary[summary.evaluator.eq("training_median") & summary.functional_check.eq(check)]
            low, high = min(-.03, all_check.ci_low.min()), max(.03, all_check.ci_high.max())
            ax.set_xlim(low-.03, high+.03)
            if row_index == 2:
                ax.set_xlabel("Attribution − random AOPC", fontsize=7.5)
    handles = [plt.Line2D([],[],marker='o',color='.1',markerfacecolor='white',linestyle='-',markersize=3,label='MLP: mean and 95% CI'),
               plt.Line2D([],[],marker='s',color='.5',linestyle='-',markersize=3,label='CNN: mean and 95% CI'),
               plt.Line2D([],[],marker='|',color='.5',linestyle='none',markersize=5,label='Five seed means')]
    fig.legend(handles=handles,loc='lower center',ncol=2,fontsize=7,frameon=False,bbox_to_anchor=(.53,0))
    fig.tight_layout(rect=(0,.07,1,1),pad=.7,h_pad=1.2,w_pad=.7)
    fig.savefig(ASSETS / "functional_response_forest.pdf")
    plt.close(fig)


def operators(n_resamples):
    summary = read(EXT / "study2_dependency_perturbations/operator_summary.csv")
    summary.to_csv(OUT / "all_nine_operators_all_metrics.csv", index=False)
    summary = summary[summary.metric.eq("attribution_minus_random_aopc")].copy()
    paths = sorted((RESULT_EXT / "study2_dependency_perturbations/operators").glob("*/*/seed_*/*/operator_aopc_per_flow.csv"))
    detail = pd.concat([read(p) for p in paths], ignore_index=True)
    detail = detail[boolean(detail.class_specific_inference_eligible)]
    keys = ["task_id", "model", "method", "operator", "functional_check"]
    means = detail.groupby(keys).apply(lambda x: equal_seed_mean(x, "attribution_minus_random_aopc"), include_groups=False)
    for row in summary.itertuples():
        key = tuple(getattr(row, k) for k in keys)
        assert np.isclose(means.loc[key], row.estimate, rtol=0, atol=1e-10), key
    summary["evidence_status"] = "VERIFICADA"
    summary.to_csv(OUT / "all_nine_operators.csv", index=False)
    p, l = seed_inventory(detail, keys, "attribution_minus_random_aopc", "operators")
    p.to_csv(OUT / "operator_seed_points.csv", index=False)
    l.to_csv(OUT / "operator_leave_one_seed_out.csv", index=False)
    # Full, declared post-hoc family: each method vs IG, each operator vs the
    # training median, separately by task/model/check. Pair before resampling.
    index = ["task_id", "model", "functional_check", "seed", "flow_id"]
    wide = detail.pivot(index=index, columns=["method", "operator"], values="attribution_minus_random_aopc")
    rows = []
    for identity, group in wide.groupby(level=["task_id", "model", "functional_check"]):
        base = dict(zip(["task_id", "model", "functional_check"], identity))
        for method in [m for m in METHODS if m != "integrated_gradients"]:
            for operator in sorted(set(summary.operator) - {"training_median"}):
                a = (group[method, operator] - group["integrated_gradients", operator]) - (group[method, "training_median"] - group["integrated_gradients", "training_median"])
                joined = a.rename("value").reset_index().dropna(subset=["value"])
                result = crossed_ci(joined, n_resamples=n_resamples, seed=stable_seed("interaction", *identity, method, operator))
                rows.append({**base, "method": method, "comparator": "integrated_gradients", "operator": operator,
                             "baseline_operator": "training_median", **result, "analysis_role": "exploratory_unadjusted_family_384_contrasts",
                             "estimand": "difference of method-specific attribution-minus-matched-random effects across operators"})
    pd.DataFrame(rows).to_csv(OUT / "paired_method_operator_interactions.csv", index=False)
    return summary


def outcomes():
    paths = sorted((RESULT_EXT / "study4_outcome_reliability/functional").glob("*/*/seed_*/*/functional_aopc_per_flow.csv"))
    detail = pd.concat([read(p) for p in paths], ignore_index=True)
    binary = detail[detail.task_id.str.endswith("_binary")]
    identity = binary[["task_id", "model", "seed", "flow_id", "error_type", "target_class_id", "confidence", "model_margin"]].drop_duplicates()
    assert not identity.duplicated(["task_id", "model", "seed", "flow_id"]).any()
    attack = identity[identity.error_type.isin(["false_positive", "true_positive"])]
    assert attack.target_class_id.eq(1).all()
    distributions, support = [], []
    for keys, group in identity.groupby(["task_id", "model", "seed", "error_type"]):
        row = dict(zip(["task_id", "model", "seed", "error_type"], keys))
        row["n_sources"] = group.flow_id.nunique()
        for column in ["confidence", "model_margin"]:
            for q, value in group[column].quantile([0, .05, .25, .5, .75, .95, 1]).items():
                row[f"{column}_q{int(q * 100):02d}"] = value
        distributions.append(row)
    for keys, group in attack.groupby(["task_id", "model", "seed"]):
        fp, tp = (group[group.error_type.eq(o)] for o in ["false_positive", "true_positive"])
        lo, hi = max(fp.confidence.min(), tp.confidence.min()), min(fp.confidence.max(), tp.confidence.max())
        common = lo <= hi
        fp_keep = fp.confidence.between(lo, hi) if common else np.zeros(len(fp), bool)
        tp_keep = tp.confidence.between(lo, hi) if common else np.zeros(len(tp), bool)
        support.append({**dict(zip(["task_id", "model", "seed"], keys)), "target": "attack_probability",
                        "n_fp": len(fp), "n_tp": len(tp), "fp_retained": int(fp_keep.sum()), "tp_retained": int(tp_keep.sum()),
                        "overlap_low": lo if common else np.nan, "overlap_high": hi if common else np.nan,
                        "fp_retention": float(np.mean(fp_keep)), "tp_retention": float(np.mean(tp_keep)),
                        "score_margin_max_residual": float(np.ptp(group.confidence - group.model_margin)),
                        "support_definition": "intersection of within-task/model/seed empirical score ranges; descriptive, not matching"})
    pd.DataFrame(distributions).to_csv(OUT / "outcome_score_distributions.csv", index=False)
    pd.DataFrame(support).to_csv(OUT / "fp_tp_common_score_support.csv", index=False)
    selected = binary[binary.error_type.isin(["false_positive", "true_positive"])]
    assert selected.target_class_id.eq(1).all()
    keys = ["task_id", "model", "seed", "method", "functional_check", "evaluator", "error_type"]
    rows = []
    for identity_key, group in selected.groupby(keys):
        row = dict(zip(keys, identity_key))
        row.update(n=len(group), mean_score=float(group.confidence.mean()), min_score=float(group.confidence.min()),
                   max_score=float(group.confidence.max()), mean_aopc=float(group.attribution_aopc.mean()),
                   mean_random_aopc=float(group.random_aopc.mean()), mean_gap=float(group.attribution_minus_random_aopc.mean()),
                   positive_probability_deletion_bound_violations=int(((group.attribution_aopc - group.confidence) > 1e-7).sum()) if row["functional_check"] == "deletion" else 0)
        for value in ["attribution_aopc", "random_aopc", "attribution_minus_random_aopc"]:
            row[f"score_spearman_{value}"] = float(spearmanr(group.confidence, group[value]).statistic) if group.confidence.nunique() > 1 and group[value].nunique() > 1 else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "fp_tp_score_effect_diagnostics.csv", index=False)
    return pd.DataFrame(support)


def corrected_rank_summaries(n_resamples):
    grouping = {
        "cross_model": ["task_id", "method"],
        "cross_probe_primary": ["task_id", "model", "panel_scope", "probe_pair"],
        "reference_sensitivity": ["task_id", "model", "method", "reference_primary", "reference_variant"],
        "parameter_sanity": ["task_id", "model", "method"],
    }
    for family, keys in grouping.items():
        path = OUT / f"{family}_metric_fixed.csv.gz"
        if not path.exists():
            continue
        detail = read(path)
        detail = detail[boolean(detail.class_specific_inference_eligible)]
        rows = []
        for identity, group in detail.groupby(keys):
            identified = group[boolean(group.rank_pair_identifiable)]
            result = crossed_ci(identified, "spearman", n_resamples, stable_seed("corrected", family, *identity))
            coverage = equal_seed_mean(group.assign(covered=boolean(group.rank_pair_identifiable).astype(float)), "covered")
            rows.append({**dict(zip(keys, identity)), **result, "rank_coverage": coverage,
                         "screening_score": equal_seed_mean(group, "spearman"), "available_sources": group.flow_id.nunique(),
                         "available_seed_source_cells": len(group[["seed", "flow_id"]].drop_duplicates()),
                         "analysis_role": "NOVA_EXECUÇÃO: metric repair, conditional on observed probe runs"})
        pd.DataFrame(rows).to_csv(OUT / f"{family}_corrected_conditional_summary.csv", index=False)
        p, l = seed_inventory(detail[boolean(detail.rank_pair_identifiable)], keys, "spearman", family)
        p.to_csv(OUT / f"{family}_corrected_seed_points.csv", index=False)
        l.to_csv(OUT / f"{family}_corrected_leave_one_seed_out.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "core", "operators", "outcomes", "corrected"], default="all")
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, function in [("core", lambda: (repeatability(args.resamples), functional())), ("operators", lambda: operators(args.resamples)),
                           ("outcomes", outcomes), ("corrected", lambda: corrected_rank_summaries(args.resamples))]:
        if args.stage in ["all", name]:
            print(f"Starting {name}", flush=True)
            function()
            print(f"Completed {name}", flush=True)
    (OUT / f"aggregation_manifest_{args.stage}.json").write_text(json.dumps({
        "status": "NOVA_EXECUÇÃO", "command": f"{sys.executable} scripts/revision_statistics.py --stage {args.stage} --resamples {args.resamples}",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "seed": SEED, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in sorted(INPUTS)]
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
