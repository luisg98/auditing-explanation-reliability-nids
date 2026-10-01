"""Study 3: is architecture disagreement larger than seed disagreement?

The baseline reports MLP-versus-CNN disagreement at the same seed, and reads it
as an architecture effect.  That reading needs a within-architecture control it
does not have: its central panel is resampled per fitted seed, so the five
per-seed panels are almost disjoint (zero shared flows on both binary tasks)
and no flow is explained under two seeds of the same architecture.

This module builds one shared panel of flows that every fitted condition
classifies correctly, explains it under all ten (architecture, seed)
conditions with the same target and the same declared reference, and then
compares three contrasts on the same flows:

* within-architecture, cross-seed (10 seed pairs per architecture);
* cross-architecture, same seed (the baseline's own estimand);
* cross-architecture, cross-seed (both sources of variation together).
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.leakage_free_tabular import training
from src.leakage_free_tabular.cache import (
    artifact_records,
    cache_hit,
    sha256_file,
    signature,
    write_cache_metadata,
)
from src.leakage_free_tabular.epistemic_audit import (
    ALLOWED_MODELS,
    ALLOWED_TASKS,
    IDENTITY_COLUMNS,
    ProbeSpec,
    _as_bool,
    _compute_probe,
    _flow_id,
    _load_frozen_model,
    _processed_probe_inputs,
    load_condition_data,
    require_valid_training_condition,
)
from src.leakage_free_tabular_ext.common import (
    ExtensionContext,
    cluster_bootstrap,
    paired_cluster_bootstrap,
    similarity_matrix_at,
    write_decision,
    write_extension_table,
)

STUDY = "study3_seed_vs_architecture"

DETERMINISTIC_SPECS = {
    "gradient_x_input": ProbeSpec("gradient_x_input", "input_zero", role="primary"),
    "integrated_gradients": ProbeSpec(
        "integrated_gradients", "training_median", 64, role="primary"
    ),
    "occlusion": ProbeSpec("occlusion", "training_median", role="primary"),
}
STOCHASTIC_SPECS = {
    "lime": ProbeSpec("lime", "local_surrogate", run_id=0, role="independent_run"),
    "shap": ProbeSpec("shap", "training_median", role="primary"),
}
ALL_SPECS = {**DETERMINISTIC_SPECS, **STOCHASTIC_SPECS}
METRICS = ("spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1")


def _panel_path(context: ExtensionContext) -> Path:
    return context.results_root / STUDY / "panel" / "cross_seed_panel.csv"


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
def build_panel(
    context: ExtensionContext,
    *,
    tasks: Sequence[str] | None = None,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Flows every fitted condition gets right, sampled attribution-blind."""

    study = context.study(STUDY)
    spec = study["panel"]
    path = _panel_path(context)
    if path.is_file() and not force:
        return pd.read_csv(path)
    class_support = pd.read_csv(context.paths.class_support)
    frames: list[pd.DataFrame] = []
    for task_id in tasks or ALLOWED_TASKS:
        predictions: dict[tuple[str, int], pd.DataFrame] = {}
        for model in ALLOWED_MODELS:
            for seed in training.FINAL_SEEDS:
                condition = require_valid_training_condition(
                    context.paths,
                    context.training_config,
                    task_id=task_id,
                    model=model,
                    seed=int(seed),
                )
                frame = pd.read_csv(condition["predictions"])
                predictions[(model, int(seed))] = frame
        base = predictions[(ALLOWED_MODELS[0], int(training.FINAL_SEEDS[0]))]
        correct = np.ones(len(base), dtype=bool)
        for frame in predictions.values():
            if len(frame) != len(base):
                raise RuntimeError("Prediction tables disagree on the test partition")
            correct &= frame["correct"].map(_as_bool).to_numpy()
        pool = base.loc[correct, [
            "task_id",
            "sample_order",
            "source_file",
            "source_row",
            "raw_fingerprint",
            "tensor_fingerprint_exact",
            "tensor_fingerprint_round5",
            "true_class_id",
            "true_class",
        ]].copy()
        pool["flow_id"] = [
            _flow_id(row) for row in pool.to_dict("records")
        ]
        rng = np.random.default_rng(int(spec["sampling_seed"]))
        cap = int(spec["maximum_per_true_class"])
        stochastic_cap = int(spec["stochastic_subset_per_true_class"])
        selected_frames: list[pd.DataFrame] = []
        for class_id, group in pool.groupby("true_class_id", sort=True):
            ordered = group.sort_values("sample_order", kind="mergesort").reset_index(
                drop=True
            )
            take = min(cap, len(ordered))
            indices = np.sort(
                rng.choice(len(ordered), size=take, replace=False)
            )
            chosen = ordered.iloc[indices].copy()
            chosen["eligible_pool_support"] = int(len(ordered))
            chosen["panel_class_support"] = int(take)
            stochastic = np.zeros(len(chosen), dtype=bool)
            stochastic_take = min(stochastic_cap, len(chosen))
            stochastic[
                np.sort(rng.choice(len(chosen), size=stochastic_take, replace=False))
            ] = True
            chosen["in_stochastic_subset"] = stochastic
            selected_frames.append(chosen)
        panel = pd.concat(selected_frames, ignore_index=True)
        support = class_support[class_support["task_id"].astype(str).eq(task_id)]
        validation_support = (
            support.groupby("true_class_id")["validation_support"].max().to_dict()
        )
        validation_eligible = (
            support.groupby("true_class_id")["validation_inference_eligible"]
            .apply(lambda values: bool(values.map(_as_bool).all()))
            .to_dict()
        )
        minimum = int(spec["minimum_class_support_for_class_specific_inference"])
        panel["validation_support"] = panel["true_class_id"].map(validation_support)
        panel["validation_inference_eligible"] = panel["true_class_id"].map(
            validation_eligible
        ).fillna(True)
        panel["panel_sample_support_sufficient"] = panel["panel_class_support"].ge(
            minimum
        )
        panel["class_specific_inference_eligible"] = panel[
            "validation_inference_eligible"
        ].map(_as_bool) & panel["panel_sample_support_sufficient"]
        panel["target_class_id"] = panel["true_class_id"]
        panel["target_class"] = panel["true_class"]
        panel["target_basis"] = "true_class_correct_for_every_fitted_condition"
        panel["panel_id"] = str(spec["panel_id"])
        panel = panel.sort_values("sample_order", kind="mergesort").reset_index(
            drop=True
        )
        frames.append(panel)
        if logger is not None:
            logger.info(
                "%s shared cross-seed panel: %d flows (%d stochastic) from a pool of %d",
                task_id,
                len(panel),
                int(panel["in_stochastic_subset"].sum()),
                int(correct.sum()),
            )
    combined = pd.concat(frames, ignore_index=True)
    write_extension_table(combined, path)
    write_extension_table(
        combined.groupby(["task_id", "true_class_id", "true_class"], as_index=False).agg(
            panel_class_support=("flow_id", "size"),
            stochastic_support=("in_stochastic_subset", "sum"),
            eligible_pool_support=("eligible_pool_support", "max"),
            validation_support=("validation_support", "max"),
            class_specific_inference_eligible=(
                "class_specific_inference_eligible",
                "all",
            ),
        ),
        context.tables_dir(STUDY) / "cross_seed_panel_support.csv",
    )
    return combined


def panel_units(
    context: ExtensionContext, *, task_id: str, stochastic_only: bool = False
) -> pd.DataFrame:
    path = _panel_path(context)
    if not path.is_file():
        raise RuntimeError("Run the study 3 'panel' stage first")
    panel = pd.read_csv(path)
    panel = panel[panel["task_id"].astype(str).eq(task_id)].copy()
    if stochastic_only:
        panel = panel[panel["in_stochastic_subset"].map(_as_bool)].copy()
    if panel.empty:
        raise RuntimeError(f"Shared cross-seed panel is empty for {task_id}")
    return panel.sort_values("sample_order", kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def _probe_directory(
    context: ExtensionContext, task_id: str, model: str, seed: int, spec: ProbeSpec
) -> Path:
    return (
        context.results_root
        / STUDY
        / "probes"
        / task_id
        / model
        / f"seed_{seed}"
        / spec.method
        / spec.spec_id
    )


def _probe_paths(directory: Path) -> dict[str, Path]:
    return {
        "array": directory / "attributions.npz",
        "index": directory / "per_flow_index.csv",
        "decision": directory / "probe_decision.json",
        "metadata": directory / "cache_metadata.json",
    }


def load_panel_probe(
    context: ExtensionContext,
    *,
    task_id: str,
    model: str,
    seed: int,
    spec: ProbeSpec,
) -> tuple[pd.DataFrame, np.ndarray]:
    paths = _probe_paths(_probe_directory(context, task_id, model, int(seed), spec))
    if not paths["metadata"].is_file():
        raise RuntimeError(f"Study 3 probe is missing: {paths['metadata'].parent}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    hit, reason = cache_hit(
        paths["metadata"],
        expected_signature=str(metadata.get("signature", "")),
        expected_outputs=[paths["array"], paths["index"], paths["decision"]],
    )
    if not hit:
        raise RuntimeError(f"Study 3 probe outputs failed their digests: {reason}")
    index = pd.read_csv(paths["index"])
    with np.load(paths["array"], allow_pickle=False) as archive:
        attributions = archive["attributions"].astype(np.float32)
    if len(attributions) != len(index) or not np.isfinite(attributions).all():
        raise RuntimeError("Study 3 probe arrays and identity table disagree")
    return index, attributions


def run_probes(
    context: ExtensionContext,
    *,
    tasks: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    methods: Sequence[str] | None = None,
    stochastic: bool = False,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Explain the shared panel under every requested fitted condition."""

    study = context.study(STUDY)
    if methods is None:
        methods = list(
            study["probes"]["stochastic" if stochastic else "deterministic"]
        )
    rows: list[dict[str, Any]] = []
    for task_id in tasks or ALLOWED_TASKS:
        data = load_condition_data(context.paths, task_id)
        units = panel_units(context, task_id=task_id, stochastic_only=stochastic)
        for model in models or ALLOWED_MODELS:
            for seed in seeds or training.FINAL_SEEDS:
                condition = require_valid_training_condition(
                    context.paths,
                    context.training_config,
                    task_id=task_id,
                    model=model,
                    seed=int(seed),
                )
                frozen_model = None
                for method in methods:
                    spec = ALL_SPECS[method]
                    directory = _probe_directory(
                        context, task_id, model, int(seed), spec
                    )
                    paths = _probe_paths(directory)
                    inputs = artifact_records(
                        [
                            context.extension_config_path,
                            context.paths.config_path,
                            context.paths.training_config_path,
                            _panel_path(context),
                            condition["model"],
                            condition["metadata"],
                            *_processed_probe_inputs(
                                context.paths.processed_root / task_id
                            ),
                        ]
                    )
                    implementation_hash = signature(
                        {"files": context.implementation_records()}
                    )
                    stage_signature = signature(
                        {
                            "extension_study": STUDY,
                            "stage": "shared_panel_probe",
                            "implementation_hash": implementation_hash,
                            "inputs": inputs,
                            "task_id": task_id,
                            "model": model,
                            "seed": int(seed),
                            "probe": spec.payload(),
                            "panel_rows": int(len(units)),
                            "stochastic_subset_only": bool(stochastic),
                        }
                    )
                    expected = [paths["array"], paths["index"], paths["decision"]]
                    hit, _ = cache_hit(
                        paths["metadata"],
                        expected_signature=stage_signature,
                        expected_outputs=expected,
                    )
                    if hit and not force:
                        continue
                    if frozen_model is None:
                        frozen_model = _load_frozen_model(condition["model"])
                    # The shared panel is deliberately seed-independent; the
                    # fitted seed is attached here so the probe index records
                    # which fitted condition produced the account (and so the
                    # frozen stochastic probes derive their run seed from it).
                    conditioned = units.assign(seed=int(seed))
                    started = time.perf_counter()
                    attributions, probabilities = _compute_probe(
                        context.audit_config, data, frozen_model, conditioned, spec
                    )
                    runtime = float(time.perf_counter() - started)
                    index = conditioned[
                        list(IDENTITY_COLUMNS)
                        + [
                            "in_stochastic_subset",
                            "class_specific_inference_eligible",
                            "target_class_id",
                            "target_class",
                            "target_basis",
                        ]
                    ].copy()
                    index["model"] = model
                    index["method"] = spec.method
                    index["spec_id"] = spec.spec_id
                    index["reference"] = spec.reference
                    index["probe_run"] = int(spec.run_id)
                    index["target_probability"] = probabilities
                    directory.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        paths["array"],
                        attributions=attributions,
                        target_probability=probabilities,
                        sample_order=index["sample_order"].to_numpy(np.int64),
                        target_class_id=index["target_class_id"].to_numpy(np.int64),
                        feature_names=np.asarray(data.feature_names, dtype="S128"),
                    )
                    index.to_csv(paths["index"], index=False)
                    decision = {
                        "extension_study": STUDY,
                        "status": "complete",
                        "task_id": task_id,
                        "model": model,
                        "seed": int(seed),
                        "probe": spec.payload(),
                        "rows": int(len(index)),
                        "panel": "shared_cross_seed_panel",
                        "target": "true_class_correct_for_every_fitted_condition",
                        "runtime_seconds": runtime,
                        "stage_signature": stage_signature,
                    }
                    paths["decision"].write_text(
                        json.dumps(decision, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    write_cache_metadata(
                        paths["metadata"],
                        stage="shared_panel_probe",
                        signature_value=stage_signature,
                        config_hash=sha256_file(context.extension_config_path),
                        implementation_hash=implementation_hash,
                        inputs=inputs,
                        outputs=expected,
                        extra=decision,
                    )
                    rows.append(
                        {
                            "task_id": task_id,
                            "model": model,
                            "seed": int(seed),
                            "method": method,
                            "rows": int(len(index)),
                            "runtime_seconds": runtime,
                        }
                    )
                    if logger is not None:
                        logger.info(
                            "shared-panel probe %s/%s/seed_%d/%s: rows=%d, %.1fs",
                            task_id, model, seed, method, len(index), runtime,
                        )
    frame = pd.DataFrame(rows)
    if len(frame):
        name = "stochastic" if stochastic else "deterministic"
        write_extension_table(
            frame, context.tables_dir(STUDY) / f"probe_runtime_{name}.csv"
        )
    return frame


# --------------------------------------------------------------------------- #
# Contrasts
# --------------------------------------------------------------------------- #
def _available_conditions(
    context: ExtensionContext, task_id: str, method: str
) -> list[tuple[str, int]]:
    spec = ALL_SPECS[method]
    available: list[tuple[str, int]] = []
    for model in ALLOWED_MODELS:
        for seed in training.FINAL_SEEDS:
            paths = _probe_paths(
                _probe_directory(context, task_id, model, int(seed), spec)
            )
            if paths["metadata"].is_file():
                available.append((model, int(seed)))
    return available


def contrast_details(
    context: ExtensionContext,
    *,
    task_id: str,
    method: str,
    top_k: int,
) -> pd.DataFrame:
    """Per-flow similarity for every pair of fitted conditions."""

    spec = ALL_SPECS[method]
    conditions = _available_conditions(context, task_id, method)
    if len(conditions) < 2:
        return pd.DataFrame()
    loaded = {
        key: load_panel_probe(
            context, task_id=task_id, model=key[0], seed=key[1], spec=spec
        )
        for key in conditions
    }
    reference_index = loaded[conditions[0]][0]
    flows = reference_index["flow_id"].astype(str).to_numpy()
    identity = reference_index[list(IDENTITY_COLUMNS) + [
        "in_stochastic_subset",
        "class_specific_inference_eligible",
        "target_class_id",
        "target_class",
    ]].reset_index(drop=True)
    ordered: dict[tuple[str, int], np.ndarray] = {}
    for key, (index, attributions) in loaded.items():
        lookup = {
            str(flow): position for position, flow in enumerate(index["flow_id"])
        }
        missing = [flow for flow in flows if flow not in lookup]
        if missing:
            raise RuntimeError(
                f"Shared-panel probes disagree on {len(missing)} flows for {key}"
            )
        ordered[key] = attributions[
            np.asarray([lookup[flow] for flow in flows], dtype=int)
        ]
    frames: list[pd.DataFrame] = []
    for left, right in itertools.combinations(conditions, 2):
        left_model, left_seed = left
        right_model, right_seed = right
        if left_model == right_model:
            contrast = "within_architecture_cross_seed"
        elif left_seed == right_seed:
            contrast = "cross_architecture_same_seed"
        else:
            contrast = "cross_architecture_cross_seed"
        metrics = similarity_matrix_at(ordered[left], ordered[right], top_k=int(top_k))
        frame = identity.copy()
        frame["method"] = method
        frame["contrast"] = contrast
        frame["left_model"] = left_model
        frame["left_seed"] = int(left_seed)
        frame["right_model"] = right_model
        frame["right_seed"] = int(right_seed)
        frame["pair_id"] = (
            f"{left_model}_{left_seed}__{right_model}_{right_seed}"
        )
        for key, value in metrics.items():
            frame[key] = value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_summaries(
    context: ExtensionContext,
    *,
    tasks: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    logger=None,
) -> dict[str, pd.DataFrame]:
    """Contrast summaries, the architecture-excess decomposition, and gates."""

    study = context.study(STUDY)
    shared = context.shared
    top_k = int(shared["representation"]["top_k"])
    resamples = int(study["uncertainty"]["bootstrap_resamples"])
    confidence = float(shared["uncertainty"]["confidence_level"])
    gates = shared["gates"]["equivalence"]
    candidate_methods = list(
        methods
        or list(study["probes"]["deterministic"]) + list(study["probes"]["stochastic"])
    )
    detail_frames: list[pd.DataFrame] = []
    for task_id in tasks or ALLOWED_TASKS:
        for method in candidate_methods:
            frame = contrast_details(
                context, task_id=task_id, method=method, top_k=top_k
            )
            if len(frame):
                detail_frames.append(frame)
    if not detail_frames:
        raise RuntimeError("Run the study 3 'probes' stage first")
    detail = pd.concat(detail_frames, ignore_index=True)
    eligible = detail[detail["class_specific_inference_eligible"].map(_as_bool)]

    summary_rows: list[dict[str, Any]] = []
    for keys, group in eligible.groupby(
        ["task_id", "method", "contrast"], dropna=False, sort=True
    ):
        identity = dict(zip(["task_id", "method", "contrast"], keys))
        for metric in METRICS:
            result = cluster_bootstrap(
                group,
                metric,
                cluster_column="flow_id",
                stratum_column="pair_id",
                n_resamples=resamples,
                confidence_level=confidence,
                seed=int(shared["uncertainty"]["bootstrap_seed"]),
            )
            direction = "upper" if metric == "normalized_mass_l1" else "lower"
            bound_key = (
                "normalized_mass_l1_ci_upper"
                if metric == "normalized_mass_l1"
                else f"{metric}_ci_lower"
            )
            bound = float(gates[bound_key])
            passed = (
                result["ci_high"] <= bound
                if direction == "upper"
                else result["ci_low"] >= bound
            )
            summary_rows.append(
                {
                    **identity,
                    "metric": metric,
                    **result,
                    "n_pairs": int(group["pair_id"].nunique()),
                    "gate_bound": bound,
                    "gate_direction": direction,
                    "gate_pass": bool(passed) if np.isfinite(result["ci_low"]) else False,
                    "gate_status": (
                        ("pass" if passed else "fail")
                        if np.isfinite(result["ci_low"])
                        else "not_evaluable"
                    ),
                    "identification_coverage_fraction": float(
                        group[
                            {
                                "spearman": "rank_pair_identifiable",
                                "jaccard_10": "top_k_pair_identifiable",
                                "weighted_jaccard": "account_pair_nonzero",
                                "normalized_mass_l1": "account_pair_nonzero",
                            }[metric]
                        ]
                        .map(_as_bool)
                        .mean()
                    ),
                    "analysis_role": "exploratory_contrast",
                }
            )
    summary = pd.DataFrame(summary_rows)

    # Architecture excess: same flows in both arms, so the difference is paired.
    excess_rows: list[dict[str, Any]] = []
    for keys, group in eligible.groupby(["task_id", "method"], dropna=False, sort=True):
        identity = dict(zip(["task_id", "method"], keys))
        for metric in METRICS:
            wide = (
                group.pivot_table(
                    index="flow_id",
                    columns="contrast",
                    values=metric,
                    aggfunc="mean",
                )
                .reset_index()
            )
            if not {
                "within_architecture_cross_seed",
                "cross_architecture_same_seed",
            }.issubset(wide.columns):
                continue
            result = paired_cluster_bootstrap(
                wide,
                "cross_architecture_same_seed",
                "within_architecture_cross_seed",
                cluster_column="flow_id",
                n_resamples=resamples,
                confidence_level=confidence,
                seed=int(shared["uncertainty"]["bootstrap_seed"]),
            )
            excess_rows.append(
                {
                    **identity,
                    "metric": metric,
                    "contrast": "cross_architecture_same_seed_minus_within_architecture",
                    "within_architecture_mean": float(
                        wide["within_architecture_cross_seed"].mean()
                    ),
                    "cross_architecture_mean": float(
                        wide["cross_architecture_same_seed"].mean()
                    ),
                    **result,
                    # How much of the disagreement two architectures show is
                    # attributable to the architecture rather than to training
                    # stochasticity: the excess over the same-architecture
                    # cross-seed baseline, as a share of the total shortfall
                    # from perfect agreement.
                    "seed_disagreement": (
                        float(1.0 - wide["within_architecture_cross_seed"].mean())
                        if metric != "normalized_mass_l1"
                        else float(wide["within_architecture_cross_seed"].mean())
                    ),
                    "total_disagreement": (
                        float(1.0 - wide["cross_architecture_same_seed"].mean())
                        if metric != "normalized_mass_l1"
                        else float(wide["cross_architecture_same_seed"].mean())
                    ),
                    "architecture_share_of_disagreement": float(
                        abs(result["estimate"])
                        / max(
                            1e-12,
                            (
                                1.0 - wide["cross_architecture_same_seed"].mean()
                                if metric != "normalized_mass_l1"
                                else wide["cross_architecture_same_seed"].mean()
                            ),
                        )
                    ),
                    "architecture_effect_established": bool(
                        result["ci_high"] < 0.0
                        if metric != "normalized_mass_l1"
                        else result["ci_low"] > 0.0
                    ),
                    "interpretation": (
                        "negative difference means the two architectures agree "
                        "less than two seeds of one architecture"
                        if metric != "normalized_mass_l1"
                        else "positive difference means the two architectures "
                        "distribute attribution mass more differently than two "
                        "seeds of one architecture"
                    ),
                    "analysis_role": "exploratory_decomposition",
                }
            )
    excess = pd.DataFrame(excess_rows)

    baseline_path = (
        context.project_root
        / "tables/leakage_free_tabular/epistemic_audit/cross_model_summary.csv"
    )
    comparison = pd.DataFrame()
    if baseline_path.is_file() and len(summary):
        baseline = pd.read_csv(baseline_path)[
            ["task_id", "method", "metric", "estimate", "ci_low", "ci_high", "gate_status"]
        ].rename(
            columns={
                "estimate": "baseline_estimate",
                "ci_low": "baseline_ci_low",
                "ci_high": "baseline_ci_high",
                "gate_status": "baseline_gate_status",
            }
        )
        same_seed = summary[summary["contrast"].eq("cross_architecture_same_seed")]
        comparison = same_seed.merge(
            baseline, on=["task_id", "method", "metric"], how="left"
        )
        comparison["shared_panel_minus_baseline_panel"] = (
            comparison["estimate"] - comparison["baseline_estimate"]
        )

    tables_dir = context.tables_dir(STUDY)
    outputs = {
        "contrast_per_flow": tables_dir / "contrast_per_flow.csv",
        "contrast_summary": tables_dir / "contrast_summary.csv",
        "architecture_excess": tables_dir / "architecture_excess.csv",
        "baseline_panel_comparison": tables_dir / "baseline_panel_comparison.csv",
    }
    frames = {
        "contrast_per_flow": detail,
        "contrast_summary": summary,
        "architecture_excess": excess,
        "baseline_panel_comparison": comparison,
    }
    for name, path in outputs.items():
        write_extension_table(frames[name], path)
    write_decision(
        context.results_dir(STUDY) / "study3_decision.json",
        {
            "study": STUDY,
            "status": "complete",
            "methods_analysed": sorted(detail["method"].unique().tolist()),
            "methods_declared": candidate_methods,
            "methods_pending": sorted(
                set(candidate_methods) - set(detail["method"].unique().tolist())
            ),
            "panel": str(study["panel"]["panel_id"]),
            "contrasts": sorted(detail["contrast"].unique().tolist()),
            "role": "exploratory_contrast",
            "outputs": {name: str(path) for name, path in outputs.items()},
        },
    )
    if logger is not None and len(summary):
        pivot = summary[summary["metric"].eq("spearman")].pivot_table(
            index=["task_id", "method"], columns="contrast", values="estimate"
        )
        for keys, row in pivot.iterrows():
            logger.info("spearman %s: %s", keys, row.round(4).to_dict())
    return frames
