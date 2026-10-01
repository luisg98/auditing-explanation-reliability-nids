"""Study 4: the deferred outcome panel - correct decisions versus FP and FN.

The baseline froze an 18,952-flow outcome-stratified panel attribution-blind,
pre-registered a correct-versus-error contrast on it, and then deferred both
the attributions and the contrast ("outcome_attributions_and_functional_compute
_outside_bounded_central_audit").  Its matching code shipped but is switched
off.  This module computes the deferred attributions for the three
pre-registered deterministic probes and runs the contrast, adding the class and
confidence stratification the baseline never declared.

Every number here is exploratory: the panel was frozen before attributions, but
this analysis is declared after the baseline's central results were read.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    _load_frozen_model,
    _processed_probe_inputs,
    _apply_cross_model_gates,
    _apply_repeatability_gates,
    correct_vs_error_detail,
    functional_checks_for_arrays,
    load_condition_data,
    require_valid_training_condition,
    summarize_metrics,
)
from src.leakage_free_tabular_ext.common import (
    ExtensionContext,
    error_type_labels,
    quantile_bins,
    similarity_matrix_at,
    write_decision,
    write_extension_table,
)

STUDY = "study4_outcome_reliability"

# The probe inventory study 4 needs: the primary account for each
# pre-registered method, the recomputation partners that make repeatability
# observable, and the reference variants for reference sensitivity.
PROBE_INVENTORY: tuple[ProbeSpec, ...] = (
    ProbeSpec("gradient_x_input", "input_zero", role="primary"),
    ProbeSpec("integrated_gradients", "training_median", 64, role="primary"),
    ProbeSpec("integrated_gradients", "training_median", 32, role="numerical_sensitivity"),
    ProbeSpec("integrated_gradients", "training_median", 128, role="numerical_sensitivity"),
    ProbeSpec("integrated_gradients", "target_class_median", 64, role="reference_sensitivity"),
    ProbeSpec("integrated_gradients", "zero_scaled", 64, role="reference_sensitivity"),
    ProbeSpec("occlusion", "training_median", role="primary"),
    ProbeSpec("occlusion", "training_median", run_id=1, role="determinism_replicate"),
    ProbeSpec("occlusion", "target_class_median", role="reference_sensitivity"),
    ProbeSpec("occlusion", "zero_scaled", role="reference_sensitivity"),
)
PRIMARY_SPEC = {
    "gradient_x_input": ProbeSpec("gradient_x_input", "input_zero", role="primary"),
    "integrated_gradients": ProbeSpec(
        "integrated_gradients", "training_median", 64, role="primary"
    ),
    "occlusion": ProbeSpec("occlusion", "training_median", role="primary"),
}
REPEATABILITY_PAIRS = (
    (
        ProbeSpec("integrated_gradients", "training_median", 64, role="primary"),
        ProbeSpec("integrated_gradients", "training_median", 128, role="numerical_sensitivity"),
        "steps_64_vs_128",
        "primary_main_to_128",
    ),
    (
        ProbeSpec("integrated_gradients", "training_median", 32, role="numerical_sensitivity"),
        ProbeSpec("integrated_gradients", "training_median", 128, role="numerical_sensitivity"),
        "steps_32_vs_128",
        "numerical_sensitivity",
    ),
    (
        ProbeSpec("occlusion", "training_median", role="primary"),
        ProbeSpec("occlusion", "training_median", run_id=1, role="determinism_replicate"),
        "fixed_rule_repeat",
        "primary_fixed_rule_repeat",
    ),
)
REFERENCE_PAIRS = (
    (
        ProbeSpec("integrated_gradients", "training_median", 64, role="primary"),
        ProbeSpec("integrated_gradients", "target_class_median", 64, role="reference_sensitivity"),
        "target_class_median",
    ),
    (
        ProbeSpec("integrated_gradients", "training_median", 64, role="primary"),
        ProbeSpec("integrated_gradients", "zero_scaled", 64, role="reference_sensitivity"),
        "zero_scaled",
    ),
    (
        ProbeSpec("occlusion", "training_median", role="primary"),
        ProbeSpec("occlusion", "target_class_median", role="reference_sensitivity"),
        "target_class_median",
    ),
    (
        ProbeSpec("occlusion", "training_median", role="primary"),
        ProbeSpec("occlusion", "zero_scaled", role="reference_sensitivity"),
        "zero_scaled",
    ),
)

FUNCTIONAL_IDENTITY_COLUMNS = list(IDENTITY_COLUMNS) + [
    "in_central",
    "in_outcome_correct",
    "in_outcome_error",
    "class_specific_inference_eligible",
    "target_class_id",
    "target_class",
    "target_basis",
    "confidence",
    "model_margin",
]
STRATA_COLUMNS = (
    "outcome_status",
    "error_type",
    "confidence_bin",
    "margin_bin",
)


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
def outcome_units(
    context: ExtensionContext,
    *,
    task_id: str,
    model: str,
    seed: int,
) -> pd.DataFrame:
    """The frozen outcome-panel rows for one condition, with strata attached."""

    units = pd.read_csv(context.paths.audit_units)
    selected = units[
        units["task_id"].astype(str).eq(task_id)
        & units["model"].astype(str).eq(model)
        & pd.to_numeric(units["seed"], errors="raise").astype(int).eq(int(seed))
    ].copy()
    outcome = selected["in_outcome_correct"].map(_as_bool) | selected[
        "in_outcome_error"
    ].map(_as_bool)
    selected = selected[outcome].copy()
    if selected.empty:
        raise RuntimeError(f"No outcome-panel rows for {task_id}/{model}/seed_{seed}")
    selected["correct"] = selected["in_outcome_correct"].map(_as_bool)
    selected["outcome_status"] = np.where(selected["correct"], "correct", "error")
    task = "binary" if task_id.endswith("_binary") else "multiclass"
    selected["task"] = task
    selected["error_type"] = error_type_labels(selected, task=task)
    strata = context.study(STUDY)["strata"]
    selected["confidence_bin"] = quantile_bins(
        selected["confidence"],
        bins=int(strata["confidence"]["bins"]),
        labels=list(strata["confidence"]["labels"]),
    )
    selected["margin_bin"] = quantile_bins(
        selected["model_margin"],
        bins=int(strata["margin"]["bins"]),
        labels=list(strata["margin"]["labels"]),
    )
    return selected.sort_values("sample_order", kind="mergesort").reset_index(drop=True)


def _probe_directory(context: ExtensionContext, task_id: str, model: str, seed: int, spec: ProbeSpec) -> Path:
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


def _probe_signature(
    context: ExtensionContext,
    *,
    task_id: str,
    model: str,
    seed: int,
    spec: ProbeSpec,
    condition: Mapping[str, Path],
) -> tuple[str, list[dict[str, Any]], str]:
    inputs = artifact_records(
        [
            context.extension_config_path,
            context.paths.config_path,
            context.paths.training_config_path,
            context.paths.audit_units,
            condition["model"],
            condition["metadata"],
            *_processed_probe_inputs(context.paths.processed_root / task_id),
        ]
    )
    implementation_hash = signature({"files": context.implementation_records()})
    stage_signature = signature(
        {
            "extension_study": STUDY,
            "stage": "outcome_probe",
            "implementation_hash": implementation_hash,
            "inputs": inputs,
            "task_id": task_id,
            "model": model,
            "seed": int(seed),
            "probe": spec.payload(),
            "panel": "frozen_outcome_panel",
            "target": "predicted_class_of_the_audited_model",
        }
    )
    return stage_signature, inputs, implementation_hash


def load_outcome_probe(
    context: ExtensionContext,
    *,
    task_id: str,
    model: str,
    seed: int,
    spec: ProbeSpec,
) -> tuple[pd.DataFrame, np.ndarray]:
    directory = _probe_directory(context, task_id, model, int(seed), spec)
    paths = _probe_paths(directory)
    if not paths["metadata"].is_file():
        raise RuntimeError(f"Outcome probe is not computed yet: {directory}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    hit, reason = cache_hit(
        paths["metadata"],
        expected_signature=str(metadata.get("signature", "")),
        expected_outputs=[paths["array"], paths["index"], paths["decision"]],
    )
    if not hit:
        raise RuntimeError(f"Outcome probe outputs failed their digests: {reason}")
    index = pd.read_csv(paths["index"])
    with np.load(paths["array"], allow_pickle=False) as archive:
        attributions = archive["attributions"].astype(np.float32)
    if len(attributions) != len(index) or not np.isfinite(attributions).all():
        raise RuntimeError("Outcome probe arrays and identity table disagree")
    return index, attributions


def run_probes(
    context: ExtensionContext,
    *,
    tasks: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    methods: Sequence[str] | None = None,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Compute the deferred attributions on the frozen outcome panel."""

    study = context.study(STUDY)
    allowed_methods = set(methods or study["probes"])
    rows: list[dict[str, Any]] = []
    for task_id in tasks or ALLOWED_TASKS:
        data = load_condition_data(context.paths, task_id)
        for model in models or ALLOWED_MODELS:
            for seed in seeds or training.FINAL_SEEDS:
                condition = require_valid_training_condition(
                    context.paths,
                    context.training_config,
                    task_id=task_id,
                    model=model,
                    seed=int(seed),
                )
                units = outcome_units(
                    context, task_id=task_id, model=model, seed=int(seed)
                )
                frozen_model = None
                for spec in PROBE_INVENTORY:
                    if spec.method not in allowed_methods:
                        continue
                    directory = _probe_directory(
                        context, task_id, model, int(seed), spec
                    )
                    paths = _probe_paths(directory)
                    stage_signature, inputs, implementation_hash = _probe_signature(
                        context,
                        task_id=task_id,
                        model=model,
                        seed=int(seed),
                        spec=spec,
                        condition=condition,
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
                    started = time.perf_counter()
                    attributions, probabilities = _compute_probe(
                        context.audit_config, data, frozen_model, units, spec
                    )
                    runtime = float(time.perf_counter() - started)
                    index = units[
                        FUNCTIONAL_IDENTITY_COLUMNS
                        + ["model", "correct", "outcome_status", "error_type",
                           "confidence_bin", "margin_bin", "task"]
                    ].copy()
                    index["method"] = spec.method
                    index["spec_id"] = spec.spec_id
                    index["reference"] = spec.reference
                    index["steps"] = spec.steps
                    index["probe_run"] = int(spec.run_id)
                    index["probe_role"] = spec.role
                    index["target_probability"] = probabilities
                    index["attribution_l1"] = np.abs(attributions).sum(axis=1)
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
                        "features": int(attributions.shape[1]),
                        "panel": "frozen_outcome_panel",
                        "target": "predicted_class_of_the_audited_model",
                        "runtime_seconds": runtime,
                        "stage_signature": stage_signature,
                    }
                    paths["decision"].write_text(
                        json.dumps(decision, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    write_cache_metadata(
                        paths["metadata"],
                        stage="outcome_probe",
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
                            "method": spec.method,
                            "spec_id": spec.spec_id,
                            "rows": int(len(index)),
                            "runtime_seconds": runtime,
                        }
                    )
                    if logger is not None:
                        logger.info(
                            "outcome probe %s/%s/seed_%d/%s/%s: rows=%d, %.1fs",
                            task_id, model, seed, spec.method, spec.spec_id,
                            len(index), runtime,
                        )
    frame = pd.DataFrame(rows)
    if len(frame):
        write_extension_table(
            frame, context.tables_dir(STUDY) / "outcome_probe_runtime.csv"
        )
    return frame


# --------------------------------------------------------------------------- #
# Functional faithfulness on the outcome panel
# --------------------------------------------------------------------------- #
def _functional_directory(
    context: ExtensionContext, task_id: str, model: str, seed: int, method: str
) -> Path:
    return (
        context.results_root
        / STUDY
        / "functional"
        / task_id
        / model
        / f"seed_{seed}"
        / method
    )


def _functional_config(context: ExtensionContext) -> dict[str, Any]:
    """Baseline functional operators, re-scoped to the outcome panel.

    The replacement rules, fractions, random-control design and Monte Carlo
    accounting are the baseline's.  Only the panel scope changes, and the
    secondary signed-intervention block is switched off: it is descriptive in
    the baseline is reduced to the single full-strength operator declared in
    the extension config.
    """

    config = copy.deepcopy(dict(context.audit_config))
    study = context.study(STUDY)
    functional = config["functional_checks"]
    functional["panel_scope"] = "frozen_outcome_panel"
    deletion = functional["deletion_and_insertion"]
    declared = study["dimensions"]["functional_faithfulness"]
    deletion["evaluators"] = list(declared["evaluators"])
    deletion["feature_fractions"] = [float(v) for v in declared["fractions"]]
    deletion["random_feature_repetitions"] = int(declared["random_feature_repetitions"])
    signed = declared["signed_intervention"]
    functional["signed_intervention"]["strengths"] = [
        float(value) for value in signed["strengths"]
    ]
    functional["signed_intervention"]["status"] = str(signed["status"])
    return config


def run_functional(
    context: ExtensionContext,
    *,
    tasks: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    methods: Sequence[str] | None = None,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Deletion/insertion AOPC against matched random controls, per outcome row."""

    study = context.study(STUDY)
    allowed = list(methods or study["probes"])
    config = _functional_config(context)
    rows: list[dict[str, Any]] = []
    for task_id in tasks or ALLOWED_TASKS:
        data = load_condition_data(context.paths, task_id)
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
                for method in allowed:
                    directory = _functional_directory(
                        context, task_id, model, int(seed), method
                    )
                    outputs = {
                        "curves": directory / "deletion_insertion_per_flow.csv",
                        "aopc": directory / "functional_aopc_per_flow.csv",
                        "signed": directory / "signed_intervention_per_flow.csv",
                        "decision": directory / "functional_decision.json",
                    }
                    metadata_path = directory / "cache_metadata.json"
                    spec = PRIMARY_SPEC[method]
                    probe_metadata = _probe_paths(
                        _probe_directory(context, task_id, model, int(seed), spec)
                    )["metadata"]
                    inputs = artifact_records(
                        [
                            context.extension_config_path,
                            context.paths.config_path,
                            context.paths.training_config_path,
                            context.paths.audit_units,
                            condition["model"],
                            condition["metadata"],
                            probe_metadata,
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
                            "stage": "outcome_functional",
                            "implementation_hash": implementation_hash,
                            "inputs": inputs,
                            "task_id": task_id,
                            "model": model,
                            "seed": int(seed),
                            "method": method,
                            "functional": config["functional_checks"],
                        }
                    )
                    expected = list(outputs.values())
                    hit, _ = cache_hit(
                        metadata_path,
                        expected_signature=stage_signature,
                        expected_outputs=expected,
                    )
                    if hit and not force:
                        continue
                    index, attributions = load_outcome_probe(
                        context, task_id=task_id, model=model, seed=int(seed), spec=spec
                    )
                    if frozen_model is None:
                        frozen_model = _load_frozen_model(condition["model"])
                    identity = index[FUNCTIONAL_IDENTITY_COLUMNS].copy()
                    orders = pd.to_numeric(
                        index["sample_order"], errors="raise"
                    ).to_numpy(int)
                    started = time.perf_counter()
                    curves, aopc, signed = functional_checks_for_arrays(
                        frozen_model,
                        np.asarray(data.X_test[orders], dtype=np.float32),
                        pd.to_numeric(index["target_class_id"], errors="raise").to_numpy(int),
                        attributions,
                        identity,
                        data,
                        config,
                        model_name=model,
                        method=method,
                        seed=int(seed),
                    )
                    runtime = float(time.perf_counter() - started)
                    strata = index[
                        ["flow_id", "correct", "outcome_status", "error_type",
                         "confidence_bin", "margin_bin", "task"]
                    ].drop_duplicates("flow_id")
                    curves = curves.merge(strata, on="flow_id", how="left")
                    aopc = aopc.merge(strata, on="flow_id", how="left")
                    signed = signed.merge(strata, on="flow_id", how="left")
                    directory.mkdir(parents=True, exist_ok=True)
                    curves.to_csv(outputs["curves"], index=False)
                    aopc.to_csv(outputs["aopc"], index=False)
                    signed.to_csv(outputs["signed"], index=False)
                    decision = {
                        "extension_study": STUDY,
                        "status": "complete",
                        "task_id": task_id,
                        "model": model,
                        "seed": int(seed),
                        "method": method,
                        "panel": "frozen_outcome_panel",
                        "flows": int(aopc["flow_id"].nunique()),
                        "signed_intervention": config["functional_checks"][
                            "signed_intervention"
                        ]["status"],
                        "runtime_seconds": runtime,
                        "stage_signature": stage_signature,
                    }
                    outputs["decision"].write_text(
                        json.dumps(decision, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    write_cache_metadata(
                        metadata_path,
                        stage="outcome_functional",
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
                            "flows": int(aopc["flow_id"].nunique()),
                            "runtime_seconds": runtime,
                        }
                    )
                    if logger is not None:
                        logger.info(
                            "outcome functional %s/%s/seed_%d/%s: flows=%d, %.1fs",
                            task_id, model, seed, method,
                            aopc["flow_id"].nunique(), runtime,
                        )
    frame = pd.DataFrame(rows)
    if len(frame):
        write_extension_table(
            frame, context.tables_dir(STUDY) / "outcome_functional_runtime.csv"
        )
    return frame


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
STRATUM_DEFINITIONS = (
    ("all", None),
    ("outcome_status", "outcome_status"),
    ("error_type", "error_type"),
    ("confidence_bin", "confidence_bin"),
    ("margin_bin", "margin_bin"),
    ("true_class", "true_class"),
    ("error_type_by_confidence", ("error_type", "confidence_bin")),
)


def _stratified_summary(
    detail: pd.DataFrame,
    *,
    base_groups: Sequence[str],
    metric_columns: Sequence[str],
    config: Mapping[str, Any],
    analysis_id: str,
    run_column: str | None = None,
    gate: str | None = None,
) -> pd.DataFrame:
    """Summarise one inventory over each declared stratification, in long form."""

    frames: list[pd.DataFrame] = []
    for stratum_kind, columns in STRATUM_DEFINITIONS:
        if columns is None:
            frame = detail.copy()
            frame["__stratum"] = "all"
        else:
            keys = [columns] if isinstance(columns, str) else list(columns)
            if any(key not in detail.columns for key in keys):
                continue
            frame = detail.copy()
            frame["__stratum"] = frame[keys].astype(str).agg(" | ".join, axis=1)
        summary = summarize_metrics(
            frame,
            group_columns=list(base_groups) + ["__stratum"],
            metric_columns=list(metric_columns),
            config=config,
            run_column=run_column,
            analysis_id=f"{analysis_id}_{stratum_kind}",
        )
        if summary.empty:
            continue
        summary = summary.rename(columns={"__stratum": "stratum"})
        summary.insert(0, "stratum_kind", stratum_kind)
        if gate == "repeatability":
            summary = _apply_repeatability_gates(summary, config)
        elif gate == "cross_model":
            summary = _apply_cross_model_gates(summary, config)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _outcome_repeatability_detail(
    context: ExtensionContext,
    *,
    top_k: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for task_id in ALLOWED_TASKS:
        for model in ALLOWED_MODELS:
            for seed in training.FINAL_SEEDS:
                for left, right, comparison, role in REPEATABILITY_PAIRS:
                    try:
                        left_index, left_attrs = load_outcome_probe(
                            context, task_id=task_id, model=model, seed=int(seed), spec=left
                        )
                        _, right_attrs = load_outcome_probe(
                            context, task_id=task_id, model=model, seed=int(seed), spec=right
                        )
                    except RuntimeError:
                        continue
                    metrics = similarity_matrix_at(left_attrs, right_attrs, top_k=top_k)
                    frame = left_index[
                        [
                            "task_id",
                            "seed",
                            "flow_id",
                            "true_class_id",
                            "true_class",
                            "class_specific_inference_eligible",
                            "correct",
                            "outcome_status",
                            "error_type",
                            "confidence_bin",
                            "margin_bin",
                            "confidence",
                            "model_margin",
                        ]
                    ].copy()
                    frame["model"] = model
                    frame["method"] = left.method
                    frame["comparison"] = comparison
                    frame["analysis_role"] = role
                    for key, value in metrics.items():
                        frame[key] = value
                    frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _outcome_cross_model_detail(
    context: ExtensionContext,
    *,
    top_k: int,
) -> pd.DataFrame:
    """Pair the architectures only where both explain the same flow and target.

    On the central panel the two architectures share the panel by construction.
    On the outcome panel they do not: each model's panel is built from its own
    predictions, so the shared rows are the flows both models placed in their
    panel with the same explained target.
    """

    frames: list[pd.DataFrame] = []
    for task_id in ALLOWED_TASKS:
        for seed in training.FINAL_SEEDS:
            for method, spec in PRIMARY_SPEC.items():
                try:
                    left_index, left_attrs = load_outcome_probe(
                        context, task_id=task_id, model=ALLOWED_MODELS[0],
                        seed=int(seed), spec=spec,
                    )
                    right_index, right_attrs = load_outcome_probe(
                        context, task_id=task_id, model=ALLOWED_MODELS[1],
                        seed=int(seed), spec=spec,
                    )
                except RuntimeError:
                    continue
                left_key = left_index["flow_id"].astype(str) + "::" + left_index[
                    "target_class_id"
                ].astype(str)
                right_key = right_index["flow_id"].astype(str) + "::" + right_index[
                    "target_class_id"
                ].astype(str)
                right_lookup = {key: position for position, key in enumerate(right_key)}
                shared = [
                    (position, right_lookup[key])
                    for position, key in enumerate(left_key)
                    if key in right_lookup
                ]
                if not shared:
                    continue
                left_rows = np.asarray([item[0] for item in shared], dtype=int)
                right_rows = np.asarray([item[1] for item in shared], dtype=int)
                metrics = similarity_matrix_at(
                    left_attrs[left_rows], right_attrs[right_rows], top_k=top_k
                )
                frame = left_index.iloc[left_rows][
                    [
                        "task_id",
                        "seed",
                        "flow_id",
                        "true_class_id",
                        "true_class",
                        "class_specific_inference_eligible",
                        "correct",
                        "outcome_status",
                        "error_type",
                        "confidence_bin",
                        "margin_bin",
                    ]
                ].reset_index(drop=True)
                frame["method"] = method
                frame["model_left"] = ALLOWED_MODELS[0]
                frame["model_right"] = ALLOWED_MODELS[1]
                frame["probe_run"] = 0
                frame["shared_outcome_rows"] = int(len(shared))
                frame["left_panel_rows"] = int(len(left_index))
                frame["right_panel_rows"] = int(len(right_index))
                for key, value in metrics.items():
                    frame[key] = value
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _outcome_reference_detail(
    context: ExtensionContext,
    *,
    top_k: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for task_id in ALLOWED_TASKS:
        for model in ALLOWED_MODELS:
            for seed in training.FINAL_SEEDS:
                for primary, variant, label in REFERENCE_PAIRS:
                    try:
                        index, primary_attrs = load_outcome_probe(
                            context, task_id=task_id, model=model,
                            seed=int(seed), spec=primary,
                        )
                        _, variant_attrs = load_outcome_probe(
                            context, task_id=task_id, model=model,
                            seed=int(seed), spec=variant,
                        )
                    except RuntimeError:
                        continue
                    metrics = similarity_matrix_at(
                        primary_attrs, variant_attrs, top_k=top_k
                    )
                    frame = index[
                        [
                            "task_id",
                            "seed",
                            "flow_id",
                            "true_class_id",
                            "true_class",
                            "class_specific_inference_eligible",
                            "correct",
                            "outcome_status",
                            "error_type",
                            "confidence_bin",
                            "margin_bin",
                        ]
                    ].copy()
                    frame["model"] = model
                    frame["method"] = primary.method
                    frame["reference_variant"] = label
                    for key, value in metrics.items():
                        frame[key] = value
                    frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _collect_functional(context: ExtensionContext) -> pd.DataFrame:
    root = context.results_root / STUDY / "functional"
    paths = sorted(root.glob("*/*/seed_*/*/functional_aopc_per_flow.csv"))
    if not paths:
        raise RuntimeError("Run the study 4 'functional' stage first")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def run_summaries(context: ExtensionContext, *, logger=None) -> dict[str, pd.DataFrame]:
    """Reliability by outcome type, class and confidence, plus the matched contrast."""

    study = context.study(STUDY)
    top_k = int(context.shared["representation"]["top_k"])
    config = _functional_config(context)
    repeatability = _outcome_repeatability_detail(context, top_k=top_k)
    cross_model = _outcome_cross_model_detail(context, top_k=top_k)
    reference = _outcome_reference_detail(context, top_k=top_k)
    functional = _collect_functional(context)

    repeatability_summary = (
        _stratified_summary(
            repeatability,
            base_groups=["task_id", "model", "method", "analysis_role"],
            metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
            config=config,
            analysis_id="outcome_repeatability",
            gate="repeatability",
        )
        if len(repeatability)
        else pd.DataFrame()
    )
    cross_model_summary = (
        _stratified_summary(
            cross_model,
            base_groups=["task_id", "method"],
            metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
            config=config,
            analysis_id="outcome_cross_model",
            run_column="probe_run",
            gate="cross_model",
        )
        if len(cross_model)
        else pd.DataFrame()
    )
    reference_summary = (
        _stratified_summary(
            reference,
            base_groups=["task_id", "model", "method", "reference_variant"],
            metric_columns=["spearman", "jaccard_10"],
            config=config,
            analysis_id="outcome_reference_sensitivity",
        )
        if len(reference)
        else pd.DataFrame()
    )
    functional_summary = _stratified_summary(
        functional,
        base_groups=["task_id", "model", "method", "functional_check", "evaluator"],
        # The matched-random gap is the estimand; the raw attribution AOPC is
        # carried so the gap can be read against the size of the effect.  The
        # random and uniform-control areas are recoverable per flow from the
        # per-flow tables and are not bootstrapped over ~30 strata as well.
        metric_columns=["attribution_minus_random_aopc", "attribution_aopc"],
        config=config,
        analysis_id="outcome_functional",
    )
    if len(functional_summary):
        functional_summary["positive_against_matched_random"] = functional_summary[
            "metric"
        ].eq("attribution_minus_random_aopc") & functional_summary["ci_low"].gt(0.0)

    # Matched contrast: the baseline's own machinery, switched on.
    matching_config = copy.deepcopy(dict(context.audit_config))
    declared = study["matched_contrast"]
    matching_config["correct_vs_error"] = {
        **matching_config["correct_vs_error"],
        "enabled": True,
        "status": "executed_by_extension_study4",
        "methods": list(study["probes"]),
        "excluded_true_classes": list(declared["excluded_true_classes"]),
        "primary_caliper_standard_deviations": float(
            declared["primary_caliper_standard_deviations"]
        ),
        "sensitivity_calipers_standard_deviations": [
            float(value) for value in declared["sensitivity_calipers_standard_deviations"]
        ],
    }
    pairs, contrasts = correct_vs_error_detail(functional, matching_config)
    if len(contrasts):
        contrast_summary = summarize_metrics(
            contrasts[contrasts["analysis_role"].eq("primary")],
            group_columns=[
                "task_id",
                "model",
                "method",
                "functional_check",
                "evaluator",
                "true_class",
            ],
            metric_columns=["correct_minus_error_aopc_gap"],
            config=config,
            analysis_id="outcome_matched_contrast",
        )
        contrast_summary["error_gap_smaller_than_correct"] = contrast_summary[
            "ci_low"
        ].gt(0.0)
        caliper_summary = summarize_metrics(
            contrasts,
            group_columns=[
                "task_id",
                "model",
                "method",
                "functional_check",
                "evaluator",
                "caliper_standard_deviations",
            ],
            metric_columns=["correct_minus_error_aopc_gap"],
            config=config,
            analysis_id="outcome_matched_contrast_caliper",
        )
    else:
        contrast_summary = pd.DataFrame()
        caliper_summary = pd.DataFrame()

    # Support table: how many rows each stratum actually carries.
    support = (
        functional[
            [
                "task_id",
                "model",
                "seed",
                "flow_id",
                "outcome_status",
                "error_type",
                "confidence_bin",
                "true_class",
            ]
        ]
        .drop_duplicates()
        .groupby(
            ["task_id", "model", "error_type", "confidence_bin"], as_index=False
        )
        .agg(flows=("flow_id", "nunique"), seeds=("seed", "nunique"))
    )

    tables_dir = context.tables_dir(STUDY)
    outputs = {
        "repeatability_per_flow": tables_dir / "outcome_repeatability_per_flow.csv",
        "repeatability_summary": tables_dir / "outcome_repeatability_summary.csv",
        "cross_model_per_flow": tables_dir / "outcome_cross_model_per_flow.csv",
        "cross_model_summary": tables_dir / "outcome_cross_model_summary.csv",
        "reference_summary": tables_dir / "outcome_reference_sensitivity_summary.csv",
        "functional_summary": tables_dir / "outcome_functional_summary.csv",
        "matched_pairs": tables_dir / "outcome_matched_pairs.csv",
        "matched_contrast_per_pair": tables_dir / "outcome_matched_contrast_per_pair.csv",
        "matched_contrast_summary": tables_dir / "outcome_matched_contrast_summary.csv",
        "matched_contrast_caliper_sensitivity": tables_dir
        / "outcome_matched_contrast_caliper_sensitivity.csv",
        "stratum_support": tables_dir / "outcome_stratum_support.csv",
    }
    frames = {
        "repeatability_per_flow": repeatability,
        "repeatability_summary": repeatability_summary,
        "cross_model_per_flow": cross_model,
        "cross_model_summary": cross_model_summary,
        "reference_summary": reference_summary,
        "functional_summary": functional_summary,
        "matched_pairs": pairs,
        "matched_contrast_per_pair": contrasts,
        "matched_contrast_summary": contrast_summary,
        "matched_contrast_caliper_sensitivity": caliper_summary,
        "stratum_support": support,
    }
    for name, path in outputs.items():
        write_extension_table(frames[name], path)
    write_decision(
        context.results_dir(STUDY) / "study4_decision.json",
        {
            "study": STUDY,
            "status": "complete",
            "panel": "frozen_outcome_panel",
            "panel_rows": int(study["panel_rows"]),
            "probes_executed": list(study["probes"]),
            "probes_pending": ["lime", "shap"],
            "probes_pending_reason": str(study["probe_note"]).strip(),
            "baseline_correct_vs_error_status": str(
                context.audit_config["correct_vs_error"]["status"]
            ),
            "matched_pairs": int(len(pairs)),
            "role": "exploratory_outcome_contrast",
            "no_causal_interpretation": True,
            "outputs": {name: str(path) for name, path in outputs.items()},
        },
    )
    if logger is not None and len(functional_summary):
        headline = functional_summary[
            functional_summary["metric"].eq("attribution_minus_random_aopc")
            & functional_summary["stratum_kind"].eq("error_type")
        ]
        for _, row in (
            headline.groupby(["stratum", "functional_check"], as_index=False)["estimate"]
            .mean()
            .iterrows()
        ):
            logger.info(
                "functional %s / %s: mean attribution-minus-random AOPC %+.4f",
                row["stratum"], row["functional_check"], float(row["estimate"]),
            )
    return frames
