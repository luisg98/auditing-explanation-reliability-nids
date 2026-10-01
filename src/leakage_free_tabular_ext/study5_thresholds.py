"""Study 5c/5d: measurement-choice sensitivity and non-evaluable diagnostics.

5c sweeps the four measurement choices that decide what the audit reports -
the top-k cut, the account-mass floor, the rank-tie tolerance and the 95%
identification-coverage rule - one axis at a time around the baseline defaults,
re-running the frozen bootstrap and the frozen gate rules at each point.

5d explains the baseline's not-evaluable cells: how far each fell short of
coverage, which identification criterion failed, whether the shortfall came
from one side of the comparison, and which cells become evaluable under each
relaxation.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.leakage_free_tabular.epistemic_audit import (
    _apply_cross_model_gates,
    _apply_cross_probe_gates,
    _apply_repeatability_gates,
    _as_bool,
    summarize_lime_u_statistics,
    summarize_metrics,
)
from src.leakage_free_tabular_ext.common import (
    ExtensionContext,
    crossed_source_seed_bootstrap_fast,
    write_decision,
    write_extension_table,
)
from src.leakage_free_tabular_ext.detail_recompute import (
    GATED_METRICS,
    METRIC_COLUMNS,
    RepresentationSetting,
    build_inventories_at,
    reproduces_baseline,
)

INVENTORY_GROUPS = {
    "repeatability": ["task_id", "model", "method", "analysis_role"],
    "cross_model": ["task_id", "method"],
    "cross_probe": ["task_id", "model", "panel_scope", "probe_left", "probe_right"],
}
INVENTORY_RUN_COLUMN = {
    "repeatability": None,
    "cross_model": "probe_run",
    "cross_probe": "probe_run",
}
GATE_APPLIER = {
    "repeatability": _apply_repeatability_gates,
    "cross_model": _apply_cross_model_gates,
    "cross_probe": _apply_cross_probe_gates,
}


def sweep_settings(part: Mapping[str, Any]) -> list[RepresentationSetting]:
    """One-axis-at-a-time settings, with the baseline point declared once."""

    defaults = {
        "top_k": int(part["top_k_default"]),
        "minimum_account_l1": float(part["minimum_account_l1_default"]),
        "rank_tie_relative_tolerance": float(
            part["rank_tie_relative_tolerance_default"]
        ),
        "minimum_identifiable_fraction": float(
            part["minimum_identifiable_fraction_default"]
        ),
    }
    settings = [
        RepresentationSetting(setting_id="baseline", axis="baseline", **defaults)
    ]
    axes = {
        "top_k": [int(value) for value in part["top_k"]],
        "minimum_account_l1": [float(value) for value in part["minimum_account_l1"]],
        "rank_tie_relative_tolerance": [
            float(value) for value in part["rank_tie_relative_tolerance"]
        ],
        "minimum_identifiable_fraction": [
            float(value) for value in part["minimum_identifiable_fraction"]
        ],
    }
    for axis, values in axes.items():
        for value in values:
            if value == defaults[axis]:
                continue
            overrides = dict(defaults)
            overrides[axis] = value
            label = f"{axis}={value:g}" if isinstance(value, float) else f"{axis}={value}"
            settings.append(
                RepresentationSetting(setting_id=label, axis=axis, **overrides)
            )
    return settings


def _config_for(
    audit_config: Mapping[str, Any], setting: RepresentationSetting
) -> dict[str, Any]:
    """A copy of the frozen protocol with only the swept choices overridden.

    The file on disk is never touched; the frozen summarising and gating code
    is simply handed a modified mapping so that the sweep runs through exactly
    the baseline's own bootstrap and gate logic.
    """

    config = copy.deepcopy(dict(audit_config))
    representation = config["representation"]
    representation["top_k"] = int(setting.top_k)
    representation["minimum_identifiable_fraction_for_gate"] = float(
        setting.minimum_identifiable_fraction
    )
    representation["minimum_account_l1"] = float(setting.minimum_account_l1)
    representation["rank_tie_relative_tolerance"] = float(
        setting.rank_tie_relative_tolerance
    )
    return config


def summarize_inventory(
    detail: pd.DataFrame,
    inventory: str,
    config: Mapping[str, Any],
    *,
    by_class: bool = False,
) -> pd.DataFrame:
    """Frozen summarise-and-gate path for one inventory."""

    if detail.empty:
        return pd.DataFrame()
    eligible = detail[detail["class_specific_inference_eligible"].map(_as_bool)]
    if eligible.empty:
        return pd.DataFrame()
    groups = list(INVENTORY_GROUPS[inventory])
    if by_class:
        groups = groups + ["true_class_id", "true_class"]
    run_column = INVENTORY_RUN_COLUMN[inventory]
    metric_columns = list(METRIC_COLUMNS)
    if inventory == "repeatability":
        lime = eligible[eligible["method"].eq("lime")]
        deterministic = eligible[~eligible["method"].eq("lime")]
        frames = []
        if len(deterministic):
            frames.append(
                summarize_metrics(
                    deterministic,
                    group_columns=groups,
                    metric_columns=metric_columns,
                    config=config,
                    analysis_id="ext_repeatability",
                )
            )
        if len(lime):
            frames.append(
                summarize_lime_u_statistics(
                    lime,
                    group_columns=groups,
                    metric_columns=metric_columns,
                    config=config,
                )
            )
        summary = pd.concat(frames, ignore_index=True, sort=False)
    else:
        summary = summarize_metrics(
            eligible,
            group_columns=groups,
            metric_columns=metric_columns,
            config=config,
            run_column=run_column,
            analysis_id=f"ext_{inventory}",
        )
    return GATE_APPLIER[inventory](summary, config)


def _gate_counts(summary: pd.DataFrame, inventory: str) -> dict[str, int]:
    gated = summary[summary["metric"].isin(GATED_METRICS[inventory])]
    status = gated["gate_status"].astype(str)
    return {
        "gated_cells": int(len(gated)),
        "pass": int(status.eq("pass").sum()),
        "fail": int(status.eq("fail").sum()),
        "not_evaluable": int(status.str.startswith("not_evaluable").sum()),
        "not_evaluable_account_coverage": int(
            status.eq("not_evaluable_account_coverage").sum()
        ),
        "not_evaluable_seed_coverage": int(
            status.eq("not_evaluable_seed_coverage").sum()
        ),
    }


def _detail_key(setting: RepresentationSetting) -> tuple[int, float, float]:
    """Settings sharing this key produce identical per-flow measurements.

    ``minimum_identifiable_fraction`` only decides whether a cell is evaluable,
    so it neither changes a per-flow value nor the bootstrap; those settings
    reuse their group's single expensive pass.
    """

    return (
        int(setting.top_k),
        float(setting.minimum_account_l1),
        float(setting.rank_tie_relative_tolerance),
    )


def _recoverage(
    summary: pd.DataFrame,
    inventory: str,
    config: Mapping[str, Any],
    setting: RepresentationSetting,
) -> pd.DataFrame:
    """Re-decide evaluability at a different coverage threshold, then re-gate."""

    result = summary.copy()
    threshold = float(setting.minimum_identifiable_fraction)
    result["minimum_identifiable_fraction"] = threshold
    result["identification_coverage_sufficient"] = result[
        "identification_seed_inventory_complete"
    ].map(_as_bool) & pd.to_numeric(
        result["minimum_seed_identification_fraction"], errors="coerce"
    ).ge(threshold)
    return GATE_APPLIER[inventory](result, config)


def run_threshold_sensitivity(
    context: ExtensionContext,
    *,
    inventories: Sequence[str] | None = None,
    settings: Sequence[RepresentationSetting] | None = None,
    logger=None,
) -> dict[str, pd.DataFrame]:
    """Recompute, summarise and gate every inventory at every sweep setting."""

    study = context.study("study5_lime_and_thresholds")
    part = study["parts"]["threshold_and_topk_sensitivity"]
    if not bool(part.get("enabled", False)):
        raise RuntimeError("threshold_and_topk_sensitivity is disabled in the config")
    inventory_names = list(inventories or part["inventories"])
    sweep = list(settings or sweep_settings(part))
    results_dir = context.results_dir("study5_thresholds")
    tables_dir = context.tables_dir("study5_thresholds")

    grouped: dict[tuple[int, float, float], list[RepresentationSetting]] = {}
    for setting in sweep:
        grouped.setdefault(_detail_key(setting), []).append(setting)

    summary_frames: list[pd.DataFrame] = []
    count_rows: list[dict[str, Any]] = []
    flip_frames: list[pd.DataFrame] = []
    diagnostics_frames: list[pd.DataFrame] = []
    agreement_frames: list[pd.DataFrame] = []
    reproduction: pd.DataFrame | None = None
    baseline_summaries: dict[str, pd.DataFrame] = {}

    ordered_keys = sorted(grouped, key=lambda key: 0 if key == (
        int(part["top_k_default"]),
        float(part["minimum_account_l1_default"]),
        float(part["rank_tie_relative_tolerance_default"]),
    ) else 1)
    for key in ordered_keys:
        group_settings = grouped[key]
        representative = group_settings[0]
        if logger is not None:
            logger.info(
                "study5c detail pass top_k=%s l1=%.0e tie=%.0e (%d settings)",
                key[0], key[1], key[2], len(group_settings),
            )
        detail = build_inventories_at(
            context.audit_config,
            context.training_config,
            context.paths,
            representative,
            inventories=inventory_names,
        )
        base_config = _config_for(context.audit_config, representative)
        has_baseline = any(
            setting.setting_id == "baseline" for setting in group_settings
        )
        if has_baseline:
            reproduction = reproduces_baseline(
                detail,
                context.project_root / "tables/leakage_free_tabular/epistemic_audit",
            )
            reproduction.insert(0, "setting_id", "baseline")
            diagnostics_frames.append(non_evaluable_diagnostics(detail, base_config))
            agreement_frames.append(
                bootstrap_agreement_check(
                    context,
                    detail["cross_model"],
                    inventory="cross_model",
                    tolerance=float(part["bootstrap_agreement_tolerance"]),
                )
            )
        for inventory in inventory_names:
            if logger is not None:
                logger.info("study5c summarising %s", inventory)
            summary = summarize_inventory(detail[inventory], inventory, base_config)
            if summary.empty:
                continue
            for setting in group_settings:
                variant = (
                    summary
                    if setting.minimum_identifiable_fraction
                    == representative.minimum_identifiable_fraction
                    else _recoverage(
                        summary, inventory, _config_for(context.audit_config, setting), setting
                    )
                )
                variant = variant.copy()
                variant.insert(0, "inventory", inventory)
                for field, value in setting.payload().items():
                    variant[field] = value
                summary_frames.append(variant)
                count_rows.append(
                    {
                        "inventory": inventory,
                        **setting.payload(),
                        **_gate_counts(variant, inventory),
                    }
                )
                if setting.setting_id == "baseline":
                    baseline_summaries[inventory] = variant
        del detail

    for frame in summary_frames:
        inventory = str(frame["inventory"].iloc[0])
        setting_id = str(frame["setting_id"].iloc[0])
        if setting_id == "baseline" or inventory not in baseline_summaries:
            continue
        flip_frames.append(
            _gate_flips_frame(baseline_summaries[inventory], frame, inventory)
        )

    summaries = pd.concat(summary_frames, ignore_index=True, sort=False)
    counts = pd.DataFrame(count_rows)
    flips = (
        pd.concat(flip_frames, ignore_index=True, sort=False)
        if flip_frames
        else pd.DataFrame()
    )
    diagnostics = (
        pd.concat(diagnostics_frames, ignore_index=True, sort=False)
        if diagnostics_frames
        else pd.DataFrame()
    )
    agreement = (
        pd.concat(agreement_frames, ignore_index=True, sort=False)
        if agreement_frames
        else pd.DataFrame()
    )
    outputs = {
        "sensitivity_summary": tables_dir / "threshold_sensitivity_summary.csv",
        "sensitivity_gate_counts": tables_dir / "threshold_sensitivity_gate_counts.csv",
        "sensitivity_gate_flips": tables_dir / "threshold_sensitivity_gate_flips.csv",
        "non_evaluable_diagnostics": tables_dir / "non_evaluable_diagnostics.csv",
        "baseline_reproduction": tables_dir / "baseline_reproduction_check.csv",
        "bootstrap_agreement": tables_dir / "bootstrap_implementation_agreement.csv",
    }
    frames = {
        "sensitivity_summary": summaries,
        "sensitivity_gate_counts": counts,
        "sensitivity_gate_flips": flips,
        "non_evaluable_diagnostics": diagnostics,
        "baseline_reproduction": (
            reproduction if reproduction is not None else pd.DataFrame()
        ),
        "bootstrap_agreement": agreement,
    }
    for name, path in outputs.items():
        write_extension_table(frames[name], path)
    write_decision(
        results_dir / "study5_thresholds_decision.json",
        {
            "study": "study5_threshold_and_topk_sensitivity",
            "status": "complete",
            "inventories": inventory_names,
            "settings": [setting.payload() for setting in sweep],
            "detail_passes": len(ordered_keys),
            "baseline_reproduction_within_tolerance": (
                bool(reproduction["within_tolerance"].all())
                if reproduction is not None and len(reproduction)
                else None
            ),
            "bootstrap_agreement_within_tolerance": (
                bool(agreement["within_tolerance"].all())
                if len(agreement)
                else None
            ),
            "role": "exploratory_measurement_sensitivity",
            "outputs": {name: str(path) for name, path in outputs.items()},
        },
    )
    return frames


def _gate_flips_frame(
    baseline: pd.DataFrame,
    variant: pd.DataFrame,
    inventory: str,
) -> pd.DataFrame:
    """Cells whose pass/fail/not-evaluable verdict moves away from the baseline."""

    keys = INVENTORY_GROUPS[inventory] + ["metric"]
    gated = list(GATED_METRICS[inventory])
    columns = keys + ["estimate", "ci_low", "ci_high", "gate_bound", "gate_status"]
    left = baseline[baseline["metric"].isin(gated)][columns]
    right = variant[variant["metric"].isin(gated)][columns]
    merged = left.merge(right, on=keys, suffixes=("_baseline", "_variant"))
    merged.insert(0, "inventory", inventory)
    for field in (
        "setting_id",
        "axis",
        "top_k",
        "minimum_account_l1",
        "rank_tie_relative_tolerance",
        "minimum_identifiable_fraction",
    ):
        merged[field] = variant[field].iloc[0]
    merged["gate_status_changed"] = merged["gate_status_baseline"].ne(
        merged["gate_status_variant"]
    )
    return merged[merged["gate_status_changed"]].copy()


# --------------------------------------------------------------------------- #
# 5d - non-evaluable diagnostics
# --------------------------------------------------------------------------- #
_COVERAGE_BASIS = {
    "spearman": "rank_pair_identifiable",
    "jaccard_10": "top_k_pair_identifiable",
    "weighted_jaccard": "account_pair_nonzero",
    "normalized_mass_l1": "account_pair_nonzero",
}
_SIDE_FLAGS = {
    "spearman": ("left_rank_identifiable", "right_rank_identifiable"),
    "jaccard_10": ("left_top_k_identifiable", "right_top_k_identifiable"),
    "weighted_jaccard": ("left_account_nonzero", "right_account_nonzero"),
    "normalized_mass_l1": ("left_account_nonzero", "right_account_nonzero"),
}


def non_evaluable_diagnostics(
    detail: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Decompose every gated cell's identification coverage row by row."""

    threshold = float(config["representation"]["minimum_identifiable_fraction_for_gate"])
    rows: list[dict[str, Any]] = []
    for inventory, frame in detail.items():
        if frame.empty:
            continue
        eligible = frame[frame["class_specific_inference_eligible"].map(_as_bool)]
        if eligible.empty:
            continue
        groups = list(INVENTORY_GROUPS[inventory])
        for keys, group in eligible.groupby(groups, dropna=False, sort=True):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            identity = dict(zip(groups, key_values))
            seeds = pd.to_numeric(group["seed"], errors="raise").astype(int)
            for metric in GATED_METRICS[inventory]:
                basis = _COVERAGE_BASIS[metric]
                flags = group[basis].map(_as_bool)
                left_flag, right_flag = _SIDE_FLAGS[metric]
                left = group[left_flag].map(_as_bool)
                right = group[right_flag].map(_as_bool)
                per_seed = flags.groupby(seeds).mean()
                fraction = float(flags.mean())
                minimum_seed = float(per_seed.min()) if len(per_seed) else 0.0
                gaps = pd.to_numeric(
                    pd.concat(
                        [
                            group["left_top_k_boundary_gap"],
                            group["right_top_k_boundary_gap"],
                        ]
                    ),
                    errors="coerce",
                ).replace([np.inf, -np.inf], np.nan)
                rows.append(
                    {
                        "inventory": inventory,
                        **identity,
                        "metric": metric,
                        "identification_basis": basis,
                        "rows": int(len(group)),
                        "seeds": int(seeds.nunique()),
                        "identification_coverage_fraction": fraction,
                        "minimum_seed_identification_fraction": minimum_seed,
                        "minimum_identifiable_fraction": threshold,
                        "coverage_shortfall": float(max(0.0, threshold - minimum_seed)),
                        "identification_coverage_sufficient": bool(
                            minimum_seed >= threshold
                        ),
                        "left_only_unidentified_fraction": float(
                            ((~left) & right).mean()
                        ),
                        "right_only_unidentified_fraction": float(
                            (left & (~right)).mean()
                        ),
                        "both_unidentified_fraction": float(((~left) & (~right)).mean()),
                        "shortfall_is_one_sided": bool(
                            float(((~left) & (~right)).mean()) == 0.0
                            and float(((~left) | (~right)).mean()) > 0.0
                        ),
                        "null_left_account_fraction": float(
                            (~group["left_account_nonzero"].map(_as_bool)).mean()
                        ),
                        "null_right_account_fraction": float(
                            (~group["right_account_nonzero"].map(_as_bool)).mean()
                        ),
                        "agreement_on_null_accounts_fraction": float(
                            group["agreement_on_null_accounts"].map(_as_bool).mean()
                        ),
                        "rank_tied_left_fraction": float(
                            (
                                group["left_account_nonzero"].map(_as_bool)
                                & ~group["left_rank_identifiable"].map(_as_bool)
                            ).mean()
                        ),
                        "rank_tied_right_fraction": float(
                            (
                                group["right_account_nonzero"].map(_as_bool)
                                & ~group["right_rank_identifiable"].map(_as_bool)
                            ).mean()
                        ),
                        "top_k_boundary_gap_median": float(gaps.median()),
                        "top_k_boundary_gap_p05": float(gaps.quantile(0.05)),
                        "diagnostic_role": "descriptive_no_gate",
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_agreement_check(
    context: ExtensionContext,
    detail: pd.DataFrame,
    *,
    inventory: str,
    tolerance: float,
    max_cells: int = 12,
) -> pd.DataFrame:
    """Frozen versus vectorised bootstrap on the same cells, for the record."""

    groups = list(INVENTORY_GROUPS[inventory])
    eligible = detail[detail["class_specific_inference_eligible"].map(_as_bool)]
    rows: list[dict[str, Any]] = []
    uncertainty = context.audit_config["uncertainty"]
    for index, (keys, group) in enumerate(
        eligible.groupby(groups, dropna=False, sort=True)
    ):
        if index >= int(max_cells):
            break
        key_values = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(groups, key_values))
        for metric in GATED_METRICS[inventory]:
            frozen = summarize_metrics(
                group,
                group_columns=groups,
                metric_columns=[metric],
                config=context.audit_config,
                analysis_id="ext_bootstrap_agreement",
            ).iloc[0]
            fast = crossed_source_seed_bootstrap_fast(
                group,
                metric,
                n_resamples=int(uncertainty["bootstrap_resamples"]),
                confidence_level=float(uncertainty["confidence_level"]),
                seed=20260908,
            )
            rows.append(
                {
                    "inventory": inventory,
                    **identity,
                    "metric": metric,
                    "frozen_estimate": float(frozen["estimate"]),
                    "vectorised_estimate": float(fast["estimate"]),
                    "estimate_abs_diff": abs(
                        float(frozen["estimate"]) - float(fast["estimate"])
                    ),
                    "frozen_ci_low": float(frozen["ci_low"]),
                    "vectorised_ci_low": float(fast["ci_low"]),
                    "ci_low_abs_diff": abs(
                        float(frozen["ci_low"]) - float(fast["ci_low"])
                    ),
                    "frozen_ci_high": float(frozen["ci_high"]),
                    "vectorised_ci_high": float(fast["ci_high"]),
                    "ci_high_abs_diff": abs(
                        float(frozen["ci_high"]) - float(fast["ci_high"])
                    ),
                    "tolerance": float(tolerance),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["within_tolerance"] = (
            frame[["estimate_abs_diff", "ci_low_abs_diff", "ci_high_abs_diff"]].max(
                axis=1
            )
            <= float(tolerance)
        )
    return frame
