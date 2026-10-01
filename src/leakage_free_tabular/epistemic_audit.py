"""Bounded, attribution-blind audit of neural decision accounts.

The module implements the frozen protocol in
``configs/leakage_free_epistemic_audit.yaml``.  Panel selection depends only on
held-out predictions and immutable flow identities.  Attribution arrays are
opened only after the panel has been written, so a feature account cannot make
its own audit cases easier.

Every computational stage is content addressed.  A cache is accepted only
when its signature, output inventory, byte sizes, and SHA-256 digests match.
The two admitted models are ordinary flat tabular networks; no sequential
geometry or high-budget coalition explainer is part of this audit.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# Must precede `import pandas`: importing pandas before tensorflow has been
# touched causes tf.data's worker-thread pool to deadlock on its first use
# (observed as an indefinite hang inside ParallelMapDatasetV2 on this
# environment) — verified by process-level bisection, not a TensorFlow config
# issue. Importing tensorflow first sidesteps it.
import tensorflow as _tensorflow_import_order_guard  # noqa: F401

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from tqdm import tqdm

from src.leakage_free_tabular.cache import (
    artifact_records,
    cache_hit,
    sha256_file,
    signature,
    write_cache_metadata,
)
from src.leakage_free_tabular import training


ALLOWED_MODELS = ("mlp", "cnn")
ALLOWED_TASKS = ("cicids2017_binary", "cicids2017_multiclass", "ton_iot_binary")
ALLOWED_PROBES = (
    "gradient_x_input",
    "integrated_gradients",
    "occlusion",
    "lime",
    "shap",
)
# Probes computed only over the smaller stochastic subset of the central
# panel (not the full central panel) because each explanation call is itself
# randomized (coalition/perturbation sampling) and expensive to repeat at
# full panel scale.
STOCHASTIC_SUBSET_PROBES = ("lime", "shap")
PRIMARY_REFERENCE = "training_median"
AUDIT_VERSION = "leakage_free_epistemic_audit_v2"
MINIMUM_ACCOUNT_L1 = 1e-12
RANK_TIE_RELATIVE_TOLERANCE = 1e-8
CROSSED_SOURCE_SEED_BOOTSTRAP = (
    "crossed_source_seed_multinomial_equal_seed_bootstrap"
)
CROSSED_SOURCE_SEED_LIME_BOOTSTRAP = (
    "crossed_source_seed_multinomial_equal_seed_shared_run_"
    "jackknife_u_statistic_bootstrap"
)
IDENTITY_COLUMNS = (
    "task_id",
    "seed",
    "sample_order",
    "source_file",
    "source_row",
    "raw_fingerprint",
    "tensor_fingerprint_exact",
    "tensor_fingerprint_round5",
    "true_class_id",
    "true_class",
    "flow_id",
)
AUDIT_SUMMARY_TABLES = (
    "repeatability_per_flow",
    "repeatability_summary",
    "repeatability_summary_by_class",
    "cross_model_per_flow",
    "cross_model_summary",
    "cross_model_summary_by_class",
    "reference_sensitivity_per_flow",
    "reference_sensitivity_summary",
    "reference_sensitivity_summary_by_class",
    "cross_probe_per_flow",
    "cross_probe_summary",
    "cross_probe_summary_by_class",
    "parameter_sanity_per_flow",
    "parameter_sanity_summary",
    "parameter_sanity_summary_by_class",
    "functional_aopc_per_flow",
    "functional_summary",
    "functional_summary_by_class",
    "functional_monte_carlo_sensitivity_summary",
    "functional_monte_carlo_sensitivity_summary_by_class",
    "functional_displacement_summary",
    "functional_displacement_summary_by_class",
    "signed_per_flow",
    "signed_coverage",
    "signed_coverage_by_class",
    "signed_summary",
    "signed_summary_by_class",
    "correct_error_matches",
    "correct_error_per_pair",
    "correct_error_summary",
    "correct_error_summary_by_class",
    "decision_inventory",
)


@dataclass(frozen=True)
class AuditPaths:
    project_root: Path
    config_path: Path
    training_config_path: Path
    processed_root: Path
    training_results_root: Path
    audit_root: Path
    tables_root: Path

    @classmethod
    def create(
        cls,
        project_root: Path,
        config_path: Path,
        training_config_path: Path,
        config: Mapping[str, Any],
        training_config: Mapping[str, Any],
    ) -> "AuditPaths":
        root = project_root.resolve()
        processed_root = root / str(config["data_guard"]["processed_root"])
        training_results_root = root / str(training_config["paths"]["results_root"])
        return cls(
            project_root=root,
            config_path=config_path.resolve(),
            training_config_path=training_config_path.resolve(),
            processed_root=processed_root,
            training_results_root=training_results_root,
            audit_root=training_results_root / "epistemic_audit",
            tables_root=root / "tables/leakage_free_tabular/epistemic_audit",
        )

    @property
    def panel_dir(self) -> Path:
        return self.audit_root / "panel"

    @property
    def central_panel(self) -> Path:
        return self.panel_dir / "central_panel.csv"

    @property
    def outcome_panel(self) -> Path:
        return self.panel_dir / "outcome_panel.csv"

    @property
    def audit_units(self) -> Path:
        return self.panel_dir / "audit_units.csv"

    @property
    def class_support(self) -> Path:
        return self.panel_dir / "class_support.csv"


@dataclass(frozen=True)
class ProbeSpec:
    method: str
    reference: str
    steps: int | None = None
    run_id: int = 0
    role: str = "primary"

    @property
    def spec_id(self) -> str:
        fields = [self.reference]
        if self.steps is not None:
            fields.append(f"steps_{self.steps}")
        if self.run_id:
            fields.append(f"run_{self.run_id:02d}")
        return "__".join(fields)

    def payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "reference": self.reference,
            "steps": self.steps,
            "run_id": self.run_id,
            "role": self.role,
        }


@dataclass(frozen=True)
class ConditionData:
    task_id: str
    task: str
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    source_file: np.ndarray
    source_row: np.ndarray
    raw_fingerprint: np.ndarray
    tensor_fingerprint_exact: np.ndarray
    tensor_fingerprint_round5: np.ndarray
    references: Mapping[str, np.ndarray]
    class_medians: np.ndarray
    feature_min: np.ndarray
    feature_max: np.ndarray


@dataclass(frozen=True)
class ProbeResult:
    index: pd.DataFrame
    attributions: np.ndarray
    target_probability: np.ndarray
    path: Path
    metadata_path: Path


def load_audit_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    validate_audit_config(config)
    return config


def validate_audit_config(config: Mapping[str, Any]) -> None:
    """Reject silent departures from the frozen confirmatory scope."""

    if str(config.get("protocol_id")) != AUDIT_VERSION:
        raise ValueError(f"protocol_id must be {AUDIT_VERSION!r}")
    amendments = config.get("amendments", []) or []
    bounded = next(
        (
            value
            for value in amendments
            if value.get("amendment_id") == "bounded_stochastic_probe_20260809"
        ),
        None,
    )
    if not isinstance(bounded, dict):
        raise ValueError("The bounded stochastic-probe amendment must be recorded")
    clarification = next(
        (
            value
            for value in amendments
            if value.get("amendment_id")
            == "decision_target_and_triangulation_clarification_20260809"
        ),
        None,
    )
    if not isinstance(clarification, dict):
        raise ValueError("The decision-target and triangulation amendment must be recorded")
    identifiability_amendment = next(
        (
            value
            for value in amendments
            if value.get("amendment_id")
            == "coordinate_identifiability_and_target_reference_20260809"
        ),
        None,
    )
    if not isinstance(identifiability_amendment, dict):
        raise ValueError("The coordinate-identifiability amendment must be recorded")
    scope = config.get("scope", {}) or {}
    if tuple(scope.get("tasks", [])) != ALLOWED_TASKS:
        raise ValueError(f"scope.tasks must be exactly {ALLOWED_TASKS}")
    if tuple(scope.get("model_families", [])) != ALLOWED_MODELS:
        raise ValueError(f"scope.model_families must be exactly {ALLOWED_MODELS}")
    if tuple(int(value) for value in scope.get("model_seeds", [])) != training.FINAL_SEEDS:
        raise ValueError("scope.model_seeds must match the five final training seeds")
    probes = config.get("measurement_probes", {}) or {}
    if tuple(probes) != ALLOWED_PROBES:
        raise ValueError(f"measurement_probes must be exactly {ALLOWED_PROBES}")
    panel = config.get("panel", {}) or {}
    if bool(panel.get("selection_reads_attributions", True)):
        raise ValueError("Panel selection must not read attributions")
    if panel.get("central_eligibility") != "correct_for_both_models_at_the_same_seed":
        raise ValueError("Unexpected central-panel eligibility rule")
    if panel.get("target") != "true_class" or not bool(panel.get("require_same_target")):
        raise ValueError("The paired central panel must target the shared true class")
    if (
        panel.get("central_target") != "true_class"
        or panel.get("outcome_target") != "predicted_class_of_the_audited_model"
    ):
        raise ValueError("Central and model-specific outcome targets must remain distinct")
    if int(panel.get("maximum_per_true_class_per_seed", 0)) <= 0:
        raise ValueError("The central per-class cap must be positive")
    stochastic_cap = int(panel.get("stochastic_subset_per_true_class_per_seed", 0))
    if not 0 < stochastic_cap <= int(panel["maximum_per_true_class_per_seed"]):
        raise ValueError("The stochastic cap must be positive and no larger than the panel cap")
    ig = probes["integrated_gradients"]
    if int(ig.get("main_steps", 0)) <= 0 or 128 not in {
        int(value) for value in ig.get("numerical_sensitivity_steps", [])
    }:
        raise ValueError("Integrated gradients requires a positive main grid and a 128-step check")
    if "target_class_median" not in set(ig.get("reference_variants", [])):
        raise ValueError("Integrated gradients must register an explained-target reference")
    occlusion = probes["occlusion"]
    if "target_class_median" not in set(occlusion.get("replacement_variants", [])):
        raise ValueError("Occlusion must register an explained-target replacement")
    lime = probes["lime"]
    if int(lime.get("num_samples", 0)) <= 0 or int(lime.get("independent_runs", 0)) < 3:
        raise ValueError(
            "The stochastic surrogate requires a positive budget and at least three "
            "runs for the jackknife U-statistic bootstrap"
        )
    representation = config.get("representation", {}) or {}
    identifiable_fraction = float(
        representation.get("minimum_identifiable_fraction_for_gate", 0.0)
    )
    if (
        representation.get("null_account_policy")
        != "fail_closed_not_perfect_agreement"
        or not np.isclose(
            float(representation.get("minimum_account_l1", float("nan"))),
            MINIMUM_ACCOUNT_L1,
            rtol=0.0,
            atol=0.0,
        )
        or tuple(
            float(value)
            for value in representation.get("account_l1_sensitivity_thresholds", [])
        )
        != (1e-10, 1e-8)
        or not np.isclose(
            float(representation.get("rank_tie_relative_tolerance", float("nan"))),
            RANK_TIE_RELATIVE_TOLERANCE,
            rtol=0.0,
            atol=0.0,
        )
        or not bool(
            representation.get("weighted_jaccard_uses_l1_normalized_absolute_mass")
        )
        or not 0.0 < identifiable_fraction <= 1.0
    ):
        raise ValueError("Feature-identification gates require fail-closed account coverage")
    if int(representation.get("top_k", -1)) != 10 or not np.isclose(
        identifiable_fraction, 0.95, rtol=0.0, atol=0.0
    ):
        raise ValueError("The v2 top-k and per-seed identification coverage are frozen")
    amended = bounded.get("amended_values", {}) or {}
    registered_amendment = (
        int(amended.get("stochastic_subset_per_true_class_per_seed", -1)),
        int(amended.get("lime_num_samples", -1)),
        int(amended.get("lime_independent_runs", -1)),
    )
    active_values = (
        stochastic_cap,
        int(lime["num_samples"]),
        int(lime["independent_runs"]),
    )
    if registered_amendment != active_values:
        raise ValueError("Active stochastic budgets differ from the recorded amendment")
    prior = bounded.get("prior_registered_values", {}) or {}
    if (
        int(prior.get("stochastic_subset_per_true_class_per_seed", -1)),
        int(prior.get("lime_num_samples", -1)),
        int(prior.get("lime_independent_runs", -1)),
    ) != (24, 2000, 20):
        raise ValueError("The amendment must retain the original stochastic budgets")
    uncertainty = config.get("uncertainty", {}) or {}
    if not bool(uncertainty.get("equal_seed_weight")):
        raise ValueError("The frozen uncertainty estimand gives every fitted seed equal weight")
    if tuple(uncertainty.get("hierarchy", [])) != ("model_seed", "source_flow"):
        raise ValueError("Unexpected bootstrap hierarchy")
    if (
        uncertainty.get("cluster_relationship") != "crossed"
        or not bool(uncertainty.get("source_resample_shared_across_seeds"))
        or uncertainty.get("lime_run_resampling")
        != "jackknife_pseudovalues_shared_across_flows_within_seed"
    ):
        raise ValueError("The bootstrap must preserve crossed sources and whole-panel runs")
    if (
        int(uncertainty.get("bootstrap_resamples", -1)) != 5000
        or not np.isclose(
            float(uncertainty.get("confidence_level", float("nan"))),
            0.95,
            rtol=0.0,
            atol=0.0,
        )
        or int(uncertainty.get("minimum_seed_coverage", -1)) != 5
    ):
        raise ValueError("The v2 bootstrap count, confidence level, and seed coverage are frozen")
    if (
        config.get("repeatability", {}).get("lime_uncertainty")
        != "crossed_source_seed_shared_run_jackknife_pseudovalue_bootstrap"
    ):
        raise ValueError("Stochastic repeatability must preserve run-level dependence")
    expected_gate_contracts = {
        "repeatability": {"spearman_ci_lower": 0.90, "jaccard_10_ci_lower": 0.80},
        "cross_model_identification": {
            "spearman_ci_lower": 0.80,
            "weighted_jaccard_ci_lower": 0.70,
            "jaccard_10_ci_lower": 0.60,
            "normalized_mass_l1_ci_upper": 0.20,
        },
        "cross_probe_triangulation": {
            "spearman_ci_lower": 0.80,
            "weighted_jaccard_ci_lower": 0.70,
            "jaccard_10_ci_lower": 0.60,
            "normalized_mass_l1_ci_upper": 0.20,
        },
    }
    observed_gate_contracts = {
        "repeatability": config.get("repeatability", {}).get("gates", {}),
        "cross_model_identification": config.get("cross_model_identification", {}).get(
            "equivalence_gates", {}
        ),
        "cross_probe_triangulation": config.get("cross_probe_triangulation", {}).get(
            "gates", {}
        ),
    }
    if observed_gate_contracts != expected_gate_contracts:
        raise ValueError("The v2 repeatability and identification gate values are frozen")
    triangulation = config.get("cross_probe_triangulation", {}) or {}
    if tuple(triangulation.get("deterministic_central_probes", [])) != ALLOWED_PROBES[:3]:
        raise ValueError("Unexpected deterministic triangulation probe set")
    if tuple(triangulation.get("all_probe_stochastic_subset", [])) != ALLOWED_PROBES:
        raise ValueError("Unexpected all-probe triangulation set")
    parameter_sanity = config.get("parameter_dependence_sanity", {}) or {}
    if (
        parameter_sanity.get("status") != "necessary_not_sufficient"
        or tuple(parameter_sanity.get("probes", []))
        != ("gradient_x_input", "integrated_gradients")
        or not bool(parameter_sanity.get("common_reinitialization_across_probes"))
        or parameter_sanity.get("primary_metric") != "normalized_mass_l1"
        or tuple((parameter_sanity.get("primary_change_gate") or {}).keys())
        != ("normalized_mass_l1_ci_lower",)
        or not np.isclose(
            float(
                (parameter_sanity.get("primary_change_gate") or {}).get(
                    "normalized_mass_l1_ci_lower", float("nan")
                )
            ),
            0.20,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ValueError("Unexpected learned-parameter sanity-check contract")
    functional = config.get("functional_checks", {}) or {}
    deletion = functional.get("deletion_and_insertion", {}) or {}
    if (
        deletion.get("primary_random_control")
        != "per_flow_per_fraction_displacement_bin_matched_subset"
        or deletion.get("random_control_uncertainty")
        != "frozen_independent_draws_per_flow_and_fraction"
        or deletion.get("gate_sensitivity")
        != "per_flow_gap_minus_1.96_monte_carlo_standard_errors"
    ):
        raise ValueError("Primary functional control must match replacement displacement")
    if tuple(deletion.get("evaluators", [])) != (
        "training_median",
        "target_class_median",
    ):
        raise ValueError("Functional references must be target-consistent")
    signed = functional.get("signed_intervention", {}) or {}
    if (
        signed.get("status") != "secondary_descriptive"
        or not bool(signed.get("no_general_faithfulness_gate"))
    ):
        raise ValueError("Signed interventions must remain secondary diagnostics")
    correct_error = config.get("correct_vs_error", {}) or {}
    if (
        bool(correct_error.get("enabled", True))
        or correct_error.get("status") != "deferred_before_attribution_generation"
        or correct_error.get("unit") != "true_attack_event_decision"
        or tuple(correct_error.get("excluded_true_classes", [])) != ("BENIGN",)
        or tuple(correct_error.get("methods", [])) != ALLOWED_PROBES[:3]
    ):
        raise ValueError("The outcome contrast must remain deferred in the bounded audit")
    if functional.get("panel_scope") != "central_stochastic_subset":
        raise ValueError("Functional checks must use the frozen central stochastic subset")
    if not bool(config.get("evidence_policy", {}).get("fail_closed_cache")):
        raise ValueError("The audit requires size-and-SHA fail-closed caches")
    _admissible_claim_types(config)


def load_context(
    project_root: Path,
    config_path: Path,
    training_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], AuditPaths]:
    root = project_root.resolve()
    audit_path = config_path if config_path.is_absolute() else root / config_path
    train_path = (
        training_config_path
        if training_config_path.is_absolute()
        else root / training_config_path
    )
    config = load_audit_config(audit_path)
    training_config = training.load_config(train_path)
    paths = AuditPaths.create(root, audit_path, train_path, config, training_config)
    if tuple(training_config["models"]) != ALLOWED_MODELS:
        raise ValueError("Training and audit model scopes differ")
    if tuple(training_config["tasks"]) != ALLOWED_TASKS:
        raise ValueError("Training and audit task scopes differ")
    return config, training_config, paths


def stable_seed(*parts: object) -> int:
    return int(signature({"parts": [str(value) for value in parts]})[:16], 16) % (2**32)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _flow_id(row: Mapping[str, Any]) -> str:
    return signature(
        {
            "task_id": str(row["task_id"]),
            "source_file": str(row["source_file"]),
            "source_row": int(row["source_row"]),
            "tensor_fingerprint_exact": str(row["tensor_fingerprint_exact"]),
        }
    )


def _safe_probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if str(column).startswith("prob_")]
    if len(columns) < 2:
        raise RuntimeError("Prediction table lacks per-class probability columns")
    return columns


def _prediction_margin(frame: pd.DataFrame) -> np.ndarray:
    probability_columns = _safe_probability_columns(frame)
    probabilities = frame[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Prediction table contains non-finite probabilities")
    if "decision_threshold" in frame.columns:
        return np.abs(
            probabilities[:, 1]
            - pd.to_numeric(frame["decision_threshold"], errors="raise").to_numpy(float)
        )
    ordered = np.sort(probabilities, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    task_id: str,
    model: str,
    seed: int,
) -> pd.DataFrame:
    required = {
        "task_id",
        "model",
        "seed",
        "sample_order",
        "source_file",
        "source_row",
        "raw_fingerprint",
        "tensor_fingerprint_exact",
        "tensor_fingerprint_round5",
        "true_class_id",
        "true_class",
        "pred_class_id",
        "pred_class",
        "confidence",
        "correct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Prediction table is missing columns: {missing}")
    clean = frame.copy()
    if set(clean["task_id"].astype(str)) != {task_id}:
        raise RuntimeError("Prediction task identity mismatch")
    if set(clean["model"].astype(str)) != {model} or model not in ALLOWED_MODELS:
        raise RuntimeError("Prediction model identity mismatch")
    if set(pd.to_numeric(clean["seed"], errors="raise").astype(int)) != {int(seed)}:
        raise RuntimeError("Prediction seed identity mismatch")
    if clean.duplicated(["source_file", "source_row"]).any():
        raise RuntimeError("Prediction table repeats a source-level identity")
    clean["correct"] = clean["correct"].astype(str).str.lower().isin({"true", "1"})
    expected_correct = (
        pd.to_numeric(clean["true_class_id"], errors="raise").astype(int)
        == pd.to_numeric(clean["pred_class_id"], errors="raise").astype(int)
    )
    if not np.array_equal(clean["correct"].to_numpy(bool), expected_correct.to_numpy(bool)):
        raise RuntimeError("Stored correctness flag disagrees with class IDs")
    clean["model_margin"] = _prediction_margin(clean)
    clean["flow_id"] = [_flow_id(row) for row in clean.to_dict("records")]
    return clean.sort_values("sample_order", kind="mergesort").reset_index(drop=True)


def _sample_group(
    frame: pd.DataFrame,
    maximum: int,
    *,
    seed_parts: Sequence[object],
) -> pd.DataFrame:
    if len(frame) <= int(maximum):
        return frame.sort_values(["source_file", "source_row"], kind="mergesort").copy()
    rng = np.random.default_rng(stable_seed(*seed_parts))
    indices = np.sort(rng.choice(len(frame), size=int(maximum), replace=False))
    return frame.iloc[indices].copy()


def build_panels_from_predictions(
    predictions: Mapping[tuple[str, str, int], pd.DataFrame],
    config: Mapping[str, Any],
    *,
    validation_inference_rules: Mapping[str, Mapping[int, Mapping[str, Any]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select central and outcome panels without opening any feature account."""

    panel_cfg = config["panel"]
    sampling_seed = int(panel_cfg["sampling_seed"])
    central_cap = int(panel_cfg["maximum_per_true_class_per_seed"])
    stochastic_cap = int(panel_cfg["stochastic_subset_per_true_class_per_seed"])
    minimum_class_support = int(panel_cfg["minimum_class_support_for_class_specific_inference"])
    outcome_cfg = panel_cfg["outcome_panel"]
    correct_cap = int(outcome_cfg["maximum_correct_per_true_class_model_seed"])
    error_cap = int(outcome_cfg["include_all_available_errors_up_to"])
    central_frames: list[pd.DataFrame] = []
    outcome_frames: list[pd.DataFrame] = []

    for task_id in ALLOWED_TASKS:
        for seed in training.FINAL_SEEDS:
            left = _validate_prediction_frame(
                predictions[(task_id, ALLOWED_MODELS[0], int(seed))],
                task_id=task_id,
                model=ALLOWED_MODELS[0],
                seed=int(seed),
            )
            right = _validate_prediction_frame(
                predictions[(task_id, ALLOWED_MODELS[1], int(seed))],
                task_id=task_id,
                model=ALLOWED_MODELS[1],
                seed=int(seed),
            )
            join_keys = ["source_file", "source_row"]
            paired = left.merge(
                right,
                on=join_keys,
                how="inner",
                validate="one_to_one",
                suffixes=(f"_{ALLOWED_MODELS[0]}", f"_{ALLOWED_MODELS[1]}"),
            )
            if len(paired) != len(left) or len(paired) != len(right):
                raise RuntimeError("The two models were not evaluated on the identical test rows")
            for column in (
                "sample_order",
                "raw_fingerprint",
                "tensor_fingerprint_exact",
                "tensor_fingerprint_round5",
                "true_class_id",
                "true_class",
                "flow_id",
            ):
                if not np.array_equal(
                    paired[f"{column}_{ALLOWED_MODELS[0]}"].to_numpy(),
                    paired[f"{column}_{ALLOWED_MODELS[1]}"].to_numpy(),
                ):
                    raise RuntimeError(f"Paired prediction identity differs in {column!r}")
            eligible = paired[
                paired[f"correct_{ALLOWED_MODELS[0]}"].astype(bool)
                & paired[f"correct_{ALLOWED_MODELS[1]}"].astype(bool)
            ].copy()
            base = pd.DataFrame(
                {
                    "task_id": task_id,
                    "seed": int(seed),
                    "sample_order": eligible[f"sample_order_{ALLOWED_MODELS[0]}"].astype(int),
                    "source_file": eligible["source_file"].astype(str),
                    "source_row": eligible["source_row"].astype(np.int64),
                    "raw_fingerprint": eligible[
                        f"raw_fingerprint_{ALLOWED_MODELS[0]}"
                    ].astype(str),
                    "tensor_fingerprint_exact": eligible[
                        f"tensor_fingerprint_exact_{ALLOWED_MODELS[0]}"
                    ].astype(str),
                    "tensor_fingerprint_round5": eligible[
                        f"tensor_fingerprint_round5_{ALLOWED_MODELS[0]}"
                    ].astype(str),
                    "true_class_id": eligible[
                        f"true_class_id_{ALLOWED_MODELS[0]}"
                    ].astype(int),
                    "true_class": eligible[f"true_class_{ALLOWED_MODELS[0]}"].astype(str),
                    "flow_id": eligible[f"flow_id_{ALLOWED_MODELS[0]}"].astype(str),
                    f"confidence_{ALLOWED_MODELS[0]}": eligible[
                        f"confidence_{ALLOWED_MODELS[0]}"
                    ].astype(float),
                    f"confidence_{ALLOWED_MODELS[1]}": eligible[
                        f"confidence_{ALLOWED_MODELS[1]}"
                    ].astype(float),
                    f"margin_{ALLOWED_MODELS[0]}": eligible[
                        f"model_margin_{ALLOWED_MODELS[0]}"
                    ].astype(float),
                    f"margin_{ALLOWED_MODELS[1]}": eligible[
                        f"model_margin_{ALLOWED_MODELS[1]}"
                    ].astype(float),
                }
            )
            base = (
                base.sort_values(["source_file", "source_row"], kind="mergesort")
                .drop_duplicates("tensor_fingerprint_exact", keep="first")
                .drop_duplicates("tensor_fingerprint_round5", keep="first")
                .reset_index(drop=True)
            )
            chosen_by_class: list[pd.DataFrame] = []
            for class_id, group in base.groupby("true_class_id", sort=True):
                rule = (
                    validation_inference_rules.get(task_id, {}).get(int(class_id), {})
                    if validation_inference_rules is not None
                    else {}
                )
                validation_eligible = bool(
                    rule.get("eligible_for_inference", True)
                )
                chosen = _sample_group(
                    group.reset_index(drop=True),
                    central_cap,
                    seed_parts=(sampling_seed, "central", task_id, seed, class_id),
                )
                chosen["in_stochastic_subset"] = False
                chosen["eligible_pool_support"] = int(len(group))
                chosen["central_class_seed_support"] = int(len(chosen))
                chosen["validation_support"] = rule.get("validation_support", np.nan)
                chosen["validation_inference_eligible"] = validation_eligible
                chosen["central_sample_support_sufficient"] = bool(
                    len(chosen) >= minimum_class_support
                )
                chosen["class_specific_inference_eligible"] = bool(
                    validation_eligible and len(chosen) >= minimum_class_support
                )
                stochastic = _sample_group(
                    chosen.reset_index(drop=True),
                    stochastic_cap,
                    seed_parts=(sampling_seed, "stochastic", task_id, seed, class_id),
                )
                chosen.loc[
                    chosen["flow_id"].isin(set(stochastic["flow_id"])),
                    "in_stochastic_subset",
                ] = True
                chosen_by_class.append(chosen)
            if chosen_by_class:
                central_frames.append(pd.concat(chosen_by_class, ignore_index=True))

            for model, frame in zip(ALLOWED_MODELS, (left, right)):
                frame = (
                    frame.sort_values(["source_file", "source_row"], kind="mergesort")
                    .drop_duplicates("tensor_fingerprint_exact", keep="first")
                    .drop_duplicates("tensor_fingerprint_round5", keep="first")
                    .reset_index(drop=True)
                )
                for class_id, class_group in frame.groupby("true_class_id", sort=True):
                    rule = (
                        validation_inference_rules.get(task_id, {}).get(int(class_id), {})
                        if validation_inference_rules is not None
                        else {}
                    )
                    validation_eligible = bool(rule.get("eligible_for_inference", True))
                    for correctness, cap, status in (
                        (True, correct_cap, "correct"),
                        (False, error_cap, "error"),
                    ):
                        candidates = class_group[class_group["correct"].eq(correctness)]
                        if candidates.empty:
                            continue
                        chosen = _sample_group(
                            candidates.reset_index(drop=True),
                            cap,
                            seed_parts=(
                                sampling_seed,
                                "outcome",
                                task_id,
                                seed,
                                model,
                                class_id,
                                status,
                            ),
                        )
                        selected = chosen[list(IDENTITY_COLUMNS)].copy()
                        selected["model"] = model
                        selected["correct"] = bool(correctness)
                        selected["outcome_status"] = status
                        selected["target_class_id"] = chosen[
                            "pred_class_id"
                        ].to_numpy(dtype=int)
                        selected["target_class"] = chosen["pred_class"].astype(str).to_numpy()
                        selected["target_basis"] = "predicted_class_of_audited_model"
                        selected["confidence"] = chosen["confidence"].to_numpy(float)
                        selected["model_margin"] = chosen["model_margin"].to_numpy(float)
                        selected["outcome_class_status_support"] = int(len(chosen))
                        selected["validation_support"] = rule.get(
                            "validation_support", np.nan
                        )
                        selected["validation_inference_eligible"] = validation_eligible
                        selected["central_sample_support_sufficient"] = bool(
                            len(chosen) >= minimum_class_support
                        )
                        selected["class_specific_inference_eligible"] = bool(
                            validation_eligible and len(chosen) >= minimum_class_support
                        )
                        outcome_frames.append(selected)

    if not central_frames:
        raise RuntimeError("No flow was correct for both models in any frozen condition")
    central = pd.concat(central_frames, ignore_index=True)
    central = central.sort_values(
        ["task_id", "seed", "true_class_id", "source_file", "source_row"],
        kind="mergesort",
    ).reset_index(drop=True)
    central.insert(0, "central_order", np.arange(len(central), dtype=int))
    if (
        central.duplicated(["task_id", "seed", "flow_id"]).any()
        or central.duplicated(["task_id", "seed", "tensor_fingerprint_exact"]).any()
        or central.duplicated(["task_id", "seed", "tensor_fingerprint_round5"]).any()
    ):
        raise RuntimeError("Central panel is not unique by source/tensor identity")

    outcome = (
        pd.concat(outcome_frames, ignore_index=True)
        if outcome_frames
        else pd.DataFrame(columns=[*IDENTITY_COLUMNS, "model", "correct"])
    )
    outcome = outcome.sort_values(
        ["task_id", "seed", "model", "true_class_id", "correct", "source_file", "source_row"],
        kind="mergesort",
    ).reset_index(drop=True)
    outcome.insert(0, "outcome_order", np.arange(len(outcome), dtype=int))

    unit_frames: list[pd.DataFrame] = []
    for model in ALLOWED_MODELS:
        unit = central[
            list(IDENTITY_COLUMNS)
            + [
                "in_stochastic_subset",
                "central_class_seed_support",
                "validation_support",
                "validation_inference_eligible",
                "central_sample_support_sufficient",
                "class_specific_inference_eligible",
            ]
        ].copy()
        unit["model"] = model
        unit["target_class_id"] = central["true_class_id"].to_numpy(dtype=int)
        unit["target_class"] = central["true_class"].astype(str).to_numpy()
        unit["target_basis"] = "true_class_correct_for_both"
        unit["in_central"] = True
        unit["in_outcome_correct"] = False
        unit["in_outcome_error"] = False
        unit["confidence"] = central[f"confidence_{model}"].to_numpy(float)
        unit["model_margin"] = central[f"margin_{model}"].to_numpy(float)
        unit_frames.append(unit)
    units = pd.concat(unit_frames, ignore_index=True)
    if len(outcome):
        outcome_units = outcome[
            list(IDENTITY_COLUMNS)
            + [
                "model",
                "target_class_id",
                "target_class",
                "target_basis",
                "confidence",
                "model_margin",
            ]
        ].copy()
        outcome_units["in_stochastic_subset"] = False
        outcome_units["central_class_seed_support"] = 0
        outcome_units["validation_support"] = outcome["validation_support"].to_numpy()
        outcome_units["validation_inference_eligible"] = outcome[
            "validation_inference_eligible"
        ].to_numpy(bool)
        outcome_units["central_sample_support_sufficient"] = outcome[
            "central_sample_support_sufficient"
        ].to_numpy(bool)
        outcome_units["class_specific_inference_eligible"] = outcome[
            "class_specific_inference_eligible"
        ].to_numpy(bool)
        outcome_units["in_central"] = False
        outcome_units["in_outcome_correct"] = outcome["correct"].to_numpy(bool)
        outcome_units["in_outcome_error"] = ~outcome["correct"].to_numpy(bool)
        units = pd.concat([units, outcome_units], ignore_index=True)
    aggregation = {
        "in_stochastic_subset": "max",
        "in_central": "max",
        "in_outcome_correct": "max",
        "in_outcome_error": "max",
        "confidence": "first",
        "model_margin": "first",
        "target_class_id": "first",
        "target_class": "first",
        "target_basis": "first",
        "central_class_seed_support": "max",
        "validation_support": "max",
        "validation_inference_eligible": "max",
        "central_sample_support_sufficient": "max",
        "class_specific_inference_eligible": "max",
    }
    units = (
        units.groupby([*IDENTITY_COLUMNS, "model"], as_index=False, dropna=False)
        .agg(aggregation)
        .sort_values(["task_id", "seed", "model", "sample_order"], kind="mergesort")
        .reset_index(drop=True)
    )
    units.insert(0, "audit_unit_order", np.arange(len(units), dtype=int))
    error_flags = units["in_outcome_error"].map(_as_bool)
    correct_flags = units["in_central"].map(_as_bool) | units[
        "in_outcome_correct"
    ].map(_as_bool)
    target_ids = pd.to_numeric(units["target_class_id"], errors="raise").astype(int)
    true_ids = pd.to_numeric(units["true_class_id"], errors="raise").astype(int)
    if (error_flags & target_ids.eq(true_ids)).any():
        raise RuntimeError("An outcome error unit does not target the erroneous prediction")
    if (correct_flags & target_ids.ne(true_ids)).any():
        raise RuntimeError("A correct/central unit does not target its correct prediction")
    return central, outcome, units


def _training_condition_paths(
    paths: AuditPaths,
    task_id: str,
    model: str,
    seed: int,
) -> dict[str, Path]:
    condition = paths.training_results_root / "final" / task_id / model / f"seed_{seed}"
    return {
        "directory": condition,
        "model": condition / "model.keras",
        "predictions": condition / "test_predictions.csv",
        "history": condition / "training_history.csv",
        "metrics": condition / "metrics.csv",
        "confusion": condition / "confusion_matrix.csv",
        "per_class": condition / "per_class_metrics.csv",
        "class_weights": condition / "class_weights.csv",
        "metadata": condition / "cache_metadata.json",
        "selection_lock": paths.training_results_root
        / "model_selection"
        / task_id
        / model
        / "selected_config.lock.json",
    }


def _expected_training_signature(
    paths: AuditPaths,
    training_config: Mapping[str, Any],
    *,
    task_id: str,
    model: str,
    seed: int,
) -> str:
    condition = _training_condition_paths(paths, task_id, model, seed)
    processed_dir = paths.processed_root / task_id
    inputs = artifact_records(
        training.final_input_paths(processed_dir) + [condition["selection_lock"]]
    )
    implementations = training.implementation_records(paths.project_root)
    return signature(
        {
            "pipeline_version": training.PIPELINE_VERSION,
            "stage": "final_predictive",
            "config_hash": sha256_file(paths.training_config_path),
            "implementation_hash": signature({"files": implementations}),
            "inputs": inputs,
            "task_id": task_id,
            "model": model,
            "seed": int(seed),
        }
    )


def require_valid_training_condition(
    paths: AuditPaths,
    training_config: Mapping[str, Any],
    *,
    task_id: str,
    model: str,
    seed: int,
) -> dict[str, Path]:
    """Verify the predictive cache against current data, code, and config bytes."""

    if task_id not in ALLOWED_TASKS or model not in ALLOWED_MODELS:
        raise ValueError("Condition lies outside the frozen audit scope")
    condition = _training_condition_paths(paths, task_id, model, seed)
    expected_outputs = [
        condition["model"],
        condition["history"],
        condition["predictions"],
        condition["metrics"],
        condition["confusion"],
        condition["per_class"],
        condition["class_weights"],
    ]
    expected_signature = _expected_training_signature(
        paths,
        training_config,
        task_id=task_id,
        model=model,
        seed=seed,
    )
    hit, reason = cache_hit(
        condition["metadata"],
        expected_signature=expected_signature,
        expected_outputs=expected_outputs,
    )
    if not hit:
        raise RuntimeError(
            f"Predictive condition failed the content-addressed guard "
            f"({task_id}/{model}/seed_{seed}): {reason}"
        )
    metadata = json.loads(condition["metadata"].read_text(encoding="utf-8"))
    if (
        metadata.get("stage") != "final_predictive"
        or metadata.get("task_id") != task_id
        or metadata.get("model") != model
        or int(metadata.get("seed", -1)) != int(seed)
        or metadata.get("threshold_fitted_on") != "validation"
        or not bool(metadata.get("test_evaluated_once_after_training"))
    ):
        raise RuntimeError("Predictive cache metadata violates the final-condition contract")
    return condition


def panel_input_paths(
    paths: AuditPaths,
    training_config: Mapping[str, Any],
) -> list[Path]:
    """Return prediction-only panel inputs; model feature accounts are absent."""

    inputs = [paths.config_path, paths.training_config_path]
    for task_id in ALLOWED_TASKS:
        for model in ALLOWED_MODELS:
            for seed in training.FINAL_SEEDS:
                condition = require_valid_training_condition(
                    paths,
                    training_config,
                    task_id=task_id,
                    model=model,
                    seed=int(seed),
                )
                inputs.extend([condition["predictions"], condition["metadata"]])
    return inputs


def _panel_cache_contract(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
) -> tuple[str, list[dict[str, Any]], str, list[Path]]:
    """Recompute the complete content-addressed contract for the frozen panel."""

    inputs = artifact_records(panel_input_paths(paths, training_config))
    implementation_hash = signature({"files": _audit_implementation_records(paths)})
    stage_signature = signature(
        {
            "audit_version": AUDIT_VERSION,
            "stage": "attribution_blind_panel",
            "config_hash": sha256_file(paths.config_path),
            "training_config_hash": sha256_file(paths.training_config_path),
            "implementation_hash": implementation_hash,
            "inputs": inputs,
        }
    )
    outputs = [
        paths.central_panel,
        paths.outcome_panel,
        paths.audit_units,
        paths.class_support,
        paths.panel_dir / "panel_decision.json",
    ]
    return stage_signature, inputs, implementation_hash, outputs


def require_valid_panel(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
) -> None:
    """Fail closed if any panel input, output, config, or implementation byte changed."""

    stage_signature, _, _, outputs = _panel_cache_contract(
        config, training_config, paths
    )
    hit, reason = cache_hit(
        paths.panel_dir / "cache_metadata.json",
        expected_signature=stage_signature,
        expected_outputs=outputs,
    )
    if not hit:
        raise RuntimeError(f"Attribution-blind panel failed its cache guard: {reason}")


def validation_inference_rules(
    paths: AuditPaths,
    training_config: Mapping[str, Any],
) -> dict[str, dict[int, dict[str, Any]]]:
    """Read class eligibility frozen from validation, never from test support."""

    rules: dict[str, dict[int, dict[str, Any]]] = {}
    for task_id in ALLOWED_TASKS:
        processed_dir = paths.processed_root / task_id
        _, class_names = training._read_label_metadata(processed_dir)
        if str(training_config["tasks"][task_id]["task"]) == "binary":
            rules[task_id] = {
                class_id: {
                    "class_id": class_id,
                    "class_name": class_name,
                    "validation_support": np.nan,
                    "eligible_for_inference": True,
                    "eligibility_partition": "binary_task",
                }
                for class_id, class_name in enumerate(class_names)
            }
            continue
        specifications: list[dict[str, Any]] = []
        for model in ALLOWED_MODELS:
            lock_path = _training_condition_paths(
                paths, task_id, model, training.FINAL_SEEDS[0]
            )["selection_lock"]
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            specification = lock.get("multiclass_inference")
            if not isinstance(specification, dict):
                raise RuntimeError("Selection lock lacks validation-derived class eligibility")
            specifications.append(specification)
        if signature(specifications[0]) != signature(specifications[1]):
            raise RuntimeError("The two model families froze different multiclass class sets")
        specification = specifications[0]
        expected_minimum = int(
            training_config["selection_policy"]["multiclass_min_validation_support"]
        )
        if (
            specification.get("eligibility_partition") != "validation"
            or int(specification.get("minimum_validation_support", -1)) != expected_minimum
        ):
            raise RuntimeError("Multiclass eligibility lock disagrees with the training protocol")
        rows = specification.get("all_classes")
        if not isinstance(rows, list) or len(rows) != len(class_names):
            raise RuntimeError("Selection lock lacks a complete class-support inventory")
        task_rules: dict[int, dict[str, Any]] = {}
        for row in rows:
            class_id = int(row["class_id"])
            if str(row["class_name"]) != class_names[class_id]:
                raise RuntimeError("Selection-lock class name differs from label mapping")
            support = int(row["validation_support"])
            eligible = bool(row["eligible_for_inference"])
            if eligible != (support >= expected_minimum):
                raise RuntimeError("Selection-lock class eligibility disagrees with support")
            task_rules[class_id] = {
                **row,
                "eligibility_partition": "validation",
            }
        rules[task_id] = task_rules
    return rules


def _audit_implementation_records(paths: AuditPaths) -> list[dict[str, Any]]:
    return artifact_records(
        [
            paths.project_root / "src/leakage_free_tabular/cache.py",
            paths.project_root / "src/leakage_free_tabular/epistemic_audit.py",
            paths.project_root / "src/leakage_free_tabular/training.py",
            paths.project_root / "src/models/mlp.py",
            paths.project_root / "src/models/cnn.py",
            paths.project_root / "src/utils/seed.py",
        ]
    )


def run_panel_selection(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
    *,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Write the immutable central/outcome panels before any probe is read."""

    stage_signature, inputs, implementation_hash, outputs = _panel_cache_contract(
        config, training_config, paths
    )
    decision_path = outputs[-1]
    metadata_path = paths.panel_dir / "cache_metadata.json"
    hit, _ = cache_hit(
        metadata_path,
        expected_signature=stage_signature,
        expected_outputs=outputs,
    )
    if hit and not force:
        return {
            "central": pd.read_csv(paths.central_panel),
            "outcome": pd.read_csv(paths.outcome_panel),
            "units": pd.read_csv(paths.audit_units),
            "class_support": pd.read_csv(paths.class_support),
        }

    predictions: dict[tuple[str, str, int], pd.DataFrame] = {}
    for task_id in ALLOWED_TASKS:
        for model in ALLOWED_MODELS:
            for seed in training.FINAL_SEEDS:
                condition = _training_condition_paths(paths, task_id, model, int(seed))
                predictions[(task_id, model, int(seed))] = pd.read_csv(
                    condition["predictions"]
                )
    inference_rules = validation_inference_rules(paths, training_config)
    central, outcome, units = build_panels_from_predictions(
        predictions,
        config,
        validation_inference_rules=inference_rules,
    )
    paths.panel_dir.mkdir(parents=True, exist_ok=True)
    central.to_csv(paths.central_panel, index=False)
    outcome.to_csv(paths.outcome_panel, index=False)
    units.to_csv(paths.audit_units, index=False)
    class_support = central[
        [
            "task_id",
            "seed",
            "true_class_id",
            "true_class",
            "eligible_pool_support",
            "central_class_seed_support",
            "validation_support",
            "validation_inference_eligible",
            "central_sample_support_sufficient",
            "class_specific_inference_eligible",
        ]
    ].drop_duplicates()
    class_support["minimum_required"] = int(
        config["panel"]["minimum_class_support_for_class_specific_inference"]
    )
    class_support.to_csv(paths.class_support, index=False)
    decision = {
        "audit_version": AUDIT_VERSION,
        "status": "panel_frozen_before_feature_accounts",
        "selection_reads_attributions": False,
        "central_eligibility": config["panel"]["central_eligibility"],
        "central_target": config["panel"]["central_target"],
        "outcome_target": config["panel"]["outcome_target"],
        "central_rows": int(len(central)),
        "stochastic_rows": int(central["in_stochastic_subset"].astype(bool).sum()),
        "outcome_rows": int(len(outcome)),
        "audit_unit_rows": int(len(units)),
        "class_seed_cells_ineligible_for_class_specific_inference": int(
            (~class_support["class_specific_inference_eligible"].astype(bool)).sum()
        ),
        "stage_signature": stage_signature,
    }
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_cache_metadata(
        metadata_path,
        stage="attribution_blind_panel",
        signature_value=stage_signature,
        config_hash=sha256_file(paths.config_path),
        implementation_hash=implementation_hash,
        inputs=inputs,
        outputs=outputs,
        extra={
            "training_config_hash": sha256_file(paths.training_config_path),
            "selection_reads_attributions": False,
        },
    )
    return {
        "central": central,
        "outcome": outcome,
        "units": units,
        "class_support": class_support,
    }


def _decode_strings(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8")
    return values.astype(str)


def load_condition_data(paths: AuditPaths, task_id: str) -> ConditionData:
    processed_dir = paths.processed_root / task_id
    training._require_guard_pass(processed_dir)
    feature_names, class_names = training._read_label_metadata(processed_dir)
    X_train = np.load(processed_dir / "X_train.npy", allow_pickle=False, mmap_mode="r")
    y_train = np.load(processed_dir / "y_train.npy", allow_pickle=False)
    test = training.load_test_data(
        processed_dir,
        n_features=len(feature_names),
        n_classes=len(class_names),
    )
    baseline = np.load(processed_dir / "baseline.npy", allow_pickle=False).astype(np.float32)
    computed_median = np.median(np.asarray(X_train, dtype=np.float32), axis=0).astype(np.float32)
    if baseline.shape != (len(feature_names),) or not np.allclose(
        baseline, computed_median, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError("Stored training baseline is not the declared training median")
    class_medians = np.vstack(
        [
            np.median(np.asarray(X_train)[np.asarray(y_train) == class_id], axis=0)
            for class_id in range(len(class_names))
        ]
    ).astype(np.float32)
    X_train_array = np.asarray(X_train, dtype=np.float32)
    references = {
        "zero_scaled": np.zeros(len(feature_names), dtype=np.float32),
        PRIMARY_REFERENCE: computed_median,
    }
    return ConditionData(
        task_id=task_id,
        task="binary" if len(class_names) == 2 else "multiclass",
        feature_names=feature_names,
        class_names=class_names,
        X_train=X_train,
        y_train=np.asarray(y_train, dtype=np.int64),
        X_test=test.X,
        y_test=np.asarray(test.y, dtype=np.int64),
        source_file=_decode_strings(np.asarray(test.source_file)),
        source_row=np.asarray(test.source_row, dtype=np.int64),
        raw_fingerprint=training._fingerprint_hex(np.asarray(test.raw_fingerprint)),
        tensor_fingerprint_exact=training._fingerprint_hex(
            np.asarray(test.tensor_fingerprint_exact)
        ),
        tensor_fingerprint_round5=training._fingerprint_hex(
            np.asarray(test.tensor_fingerprint_round5)
        ),
        references=references,
        class_medians=class_medians,
        feature_min=np.min(X_train_array, axis=0).astype(np.float32),
        feature_max=np.max(X_train_array, axis=0).astype(np.float32),
    )


def _condition_units(
    paths: AuditPaths,
    *,
    task_id: str,
    model: str,
    seed: int,
    stochastic_only: bool,
) -> pd.DataFrame:
    if not paths.audit_units.is_file():
        raise FileNotFoundError("Freeze the attribution-blind panel first")
    units = pd.read_csv(paths.audit_units)
    selected = units[
        units["task_id"].astype(str).eq(task_id)
        & units["model"].astype(str).eq(model)
        & pd.to_numeric(units["seed"], errors="raise").astype(int).eq(int(seed))
    ].copy()
    central_flags = selected["in_central"].astype(str).str.lower().isin({"true", "1"})
    selected = selected[central_flags].copy()
    if stochastic_only:
        flags = selected["in_stochastic_subset"].astype(str).str.lower().isin({"true", "1"})
        selected = selected[flags].copy()
    if selected.empty:
        raise RuntimeError(f"No audit units for {task_id}/{model}/seed_{seed}")
    return selected.sort_values("sample_order", kind="mergesort").reset_index(drop=True)


def _verify_units_against_test(units: pd.DataFrame, data: ConditionData) -> np.ndarray:
    orders = pd.to_numeric(units["sample_order"], errors="raise").to_numpy(dtype=int)
    if np.any(orders < 0) or np.any(orders >= len(data.X_test)) or len(np.unique(orders)) != len(orders):
        raise RuntimeError("Audit panel contains invalid or repeated test row offsets")
    checks = {
        "source_file": data.source_file[orders].astype(str),
        "source_row": data.source_row[orders].astype(np.int64),
        "raw_fingerprint": data.raw_fingerprint[orders].astype(str),
        "tensor_fingerprint_exact": data.tensor_fingerprint_exact[orders].astype(str),
        "tensor_fingerprint_round5": data.tensor_fingerprint_round5[orders].astype(str),
        "true_class_id": data.y_test[orders].astype(int),
    }
    for column, expected in checks.items():
        observed = units[column].to_numpy()
        if column in {"source_row", "true_class_id"}:
            observed = pd.to_numeric(observed, errors="raise").astype(expected.dtype)
        else:
            observed = observed.astype(str)
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"Audit unit identity disagrees with X_test in {column!r}")
    target_ids = pd.to_numeric(units["target_class_id"], errors="raise").to_numpy(dtype=int)
    if np.any(target_ids < 0) or np.any(target_ids >= len(data.class_names)):
        raise RuntimeError("Audit unit target lies outside the frozen label mapping")
    expected_target_names = np.asarray(
        [data.class_names[class_id] for class_id in target_ids], dtype=str
    )
    if not np.array_equal(units["target_class"].astype(str).to_numpy(), expected_target_names):
        raise RuntimeError("Audit unit target class name disagrees with its class ID")
    return orders


def reference_matrix(
    data: ConditionData,
    reference_classes: np.ndarray,
    name: str,
) -> np.ndarray:
    if name in {"true_class_median", "target_class_median"}:
        classes = np.asarray(reference_classes, dtype=int)
        if np.any(classes < 0) or np.any(classes >= len(data.class_names)):
            raise ValueError("Reference class is outside the training label mapping")
        return data.class_medians[classes].astype(np.float32, copy=True)
    try:
        vector = np.asarray(data.references[name], dtype=np.float32)
    except KeyError as exc:
        raise KeyError(f"Unknown frozen reference {name!r}") from exc
    return np.broadcast_to(vector, (len(reference_classes), len(vector))).copy()


def probe_specs(config: Mapping[str, Any], method: str) -> list[ProbeSpec]:
    if method not in ALLOWED_PROBES:
        raise ValueError(f"Probe lies outside the frozen set: {method!r}")
    probes = config["measurement_probes"]
    if method == "gradient_x_input":
        return [ProbeSpec(method, "input_zero", role="primary")]
    if method == "integrated_gradients":
        cfg = probes[method]
        main_steps = int(cfg["main_steps"])
        specs = [ProbeSpec(method, str(cfg["main_reference"]), main_steps, role="primary")]
        for steps in cfg["numerical_sensitivity_steps"]:
            candidate = ProbeSpec(
                method,
                str(cfg["main_reference"]),
                int(steps),
                role="numerical_sensitivity",
            )
            if candidate.spec_id not in {item.spec_id for item in specs}:
                specs.append(candidate)
        for reference in cfg["reference_variants"]:
            candidate = ProbeSpec(method, str(reference), main_steps, role="reference_sensitivity")
            if candidate.spec_id not in {item.spec_id for item in specs}:
                specs.append(candidate)
        return specs
    if method == "occlusion":
        cfg = probes[method]
        specs = [ProbeSpec(method, str(cfg["main_replacement"]), role="primary")]
        # The second fixed-rule calculation makes determinism observable rather
        # than assuming it from an implementation label.
        specs.append(
            ProbeSpec(
                method,
                str(cfg["main_replacement"]),
                run_id=1,
                role="determinism_replicate",
            )
        )
        for reference in cfg["replacement_variants"]:
            candidate = ProbeSpec(method, str(reference), role="reference_sensitivity")
            if candidate.spec_id not in {item.spec_id for item in specs}:
                specs.append(candidate)
        return specs
    if method == "shap":
        cfg = probes[method]
        specs = [ProbeSpec(method, str(cfg["main_background"]), role="primary")]
        # A second, differently-seeded coalition-sampling run makes SHAP's own
        # stochastic run-to-run stability observable (unlike occlusion's
        # determinism_replicate, exact reproduction is not expected here).
        specs.append(
            ProbeSpec(
                method,
                str(cfg["main_background"]),
                run_id=1,
                role="stochastic_repeatability_replicate",
            )
        )
        for reference in cfg["background_variants"]:
            candidate = ProbeSpec(method, str(reference), role="reference_sensitivity")
            if candidate.spec_id not in {item.spec_id for item in specs}:
                specs.append(candidate)
        return specs
    if method != "lime":  # pragma: no cover - validate_audit_config makes this unreachable.
        raise ValueError(f"Unsupported probe {method!r}")
    return [
        ProbeSpec(method, "local_surrogate", run_id=run_id, role="independent_run")
        for run_id in range(int(probes[method]["independent_runs"]))
    ]


def predict_probabilities(model: Any, X: np.ndarray, *, batch_size: int = 4096) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(X), int(batch_size)):
        value = model(np.asarray(X[start : start + int(batch_size)], dtype=np.float32), training=False)
        if hasattr(value, "numpy"):
            value = value.numpy()
        rows.append(np.asarray(value, dtype=float))
    if not rows:
        raise ValueError("Cannot predict an empty matrix")
    probabilities = np.concatenate(rows, axis=0)
    if probabilities.ndim == 1 or (
        probabilities.ndim == 2 and probabilities.shape[1] == 1
    ):
        positive = probabilities.reshape(-1)
        if not np.isfinite(positive).all() or np.any((positive < 0) | (positive > 1)):
            raise RuntimeError("Binary model produced invalid probabilities")
        return np.column_stack([1.0 - positive, positive])
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise RuntimeError(f"Unexpected model output geometry: {probabilities.shape}")
    totals = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or np.any(totals <= 0):
        raise RuntimeError("Multiclass model produced invalid probabilities")
    return probabilities / totals


def target_probabilities(
    model: Any,
    X: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    probabilities = predict_probabilities(model, X, batch_size=batch_size)
    target = np.asarray(targets, dtype=int).reshape(-1)
    if len(target) != len(probabilities) or np.any(target < 0) or np.any(
        target >= probabilities.shape[1]
    ):
        raise ValueError("Target geometry does not match model probabilities")
    return probabilities[np.arange(len(target)), target]


def _target_tensor(outputs: Any, targets: np.ndarray) -> Any:
    import tensorflow as tf

    if len(outputs.shape) == 1 or int(outputs.shape[-1]) == 1:
        positive = tf.reshape(outputs, [-1])
        target = tf.cast(targets, tf.float32)
        return positive * target + (1.0 - positive) * (1.0 - target)
    indices = tf.stack(
        [tf.range(tf.shape(outputs)[0]), tf.cast(targets, tf.int32)],
        axis=1,
    )
    return tf.gather_nd(outputs, indices)


def explain_gradient_x_input(
    model: Any,
    X: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    import tensorflow as tf

    rows: list[np.ndarray] = []
    for start in range(0, len(X), int(batch_size)):
        xb = np.asarray(X[start : start + int(batch_size)], dtype=np.float32)
        tb = np.asarray(targets[start : start + int(batch_size)], dtype=np.int32)
        tensor = tf.convert_to_tensor(xb, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            output = model(tensor, training=False)
            target_output = _target_tensor(output, tb)
        gradient = tape.gradient(target_output, tensor)
        if gradient is None:
            raise RuntimeError("The model target is not differentiable with respect to its input")
        rows.append(np.asarray(gradient.numpy(), dtype=np.float32) * xb)
    return np.vstack(rows).astype(np.float32)


def explain_integrated_gradients(
    model: Any,
    X: np.ndarray,
    targets: np.ndarray,
    references: np.ndarray,
    *,
    steps: int,
    batch_size: int,
) -> np.ndarray:
    """Trapezoidal path integral for the probability of each declared target."""

    import tensorflow as tf

    if int(steps) <= 0:
        raise ValueError("Integrated-gradient steps must be positive")
    X_array = np.asarray(X, dtype=np.float32)
    reference = np.asarray(references, dtype=np.float32)
    if reference.shape != X_array.shape:
        raise ValueError("Integrated-gradient reference must have one row per flow")
    all_rows: list[np.ndarray] = []
    alphas = np.linspace(0.0, 1.0, int(steps) + 1, dtype=np.float32)
    for start in range(0, len(X_array), int(batch_size)):
        xb = X_array[start : start + int(batch_size)]
        rb = reference[start : start + int(batch_size)]
        tb = np.asarray(targets[start : start + int(batch_size)], dtype=np.int32)
        previous_gradient: np.ndarray | None = None
        trapezoid_sum = np.zeros_like(xb, dtype=np.float32)
        for alpha in alphas:
            interpolated = rb + alpha * (xb - rb)
            tensor = tf.convert_to_tensor(interpolated, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(tensor)
                output = model(tensor, training=False)
                target_output = _target_tensor(output, tb)
            current = tape.gradient(target_output, tensor)
            if current is None:
                raise RuntimeError("The model target is not input-differentiable")
            current_array = np.asarray(current.numpy(), dtype=np.float32)
            if previous_gradient is not None:
                trapezoid_sum += 0.5 * (previous_gradient + current_array)
            previous_gradient = current_array
        all_rows.append((xb - rb) * trapezoid_sum / float(steps))
    return np.vstack(all_rows).astype(np.float32)


def explain_occlusion(
    model: Any,
    X: np.ndarray,
    targets: np.ndarray,
    replacements: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    X_array = np.asarray(X, dtype=np.float32)
    replacement = np.asarray(replacements, dtype=np.float32)
    if replacement.shape != X_array.shape:
        raise ValueError("Occlusion replacement must have one row per flow")
    clean = target_probabilities(model, X_array, targets, batch_size=batch_size)
    attributes = np.zeros_like(X_array, dtype=np.float32)
    for feature_index in range(X_array.shape[1]):
        perturbed = X_array.copy()
        perturbed[:, feature_index] = replacement[:, feature_index]
        attributes[:, feature_index] = clean - target_probabilities(
            model,
            perturbed,
            targets,
            batch_size=batch_size,
        )
    return attributes


def _stratified_background_indices(
    y_train: np.ndarray,
    maximum: int,
    *,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(y_train, dtype=int)
    if maximum >= len(labels):
        return np.arange(len(labels), dtype=int)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    allocation = max(1, int(math.ceil(maximum / len(classes))))
    selected: list[int] = []
    for class_id in classes:
        candidates = np.flatnonzero(labels == class_id)
        selected.extend(
            rng.choice(candidates, size=min(allocation, len(candidates)), replace=False).tolist()
        )
    selected = list(dict.fromkeys(selected))
    if len(selected) > maximum:
        selected = rng.choice(np.asarray(selected), size=maximum, replace=False).tolist()
    elif len(selected) < maximum:
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected), assume_unique=False)
        selected.extend(
            rng.choice(
                remaining,
                size=min(maximum - len(selected), len(remaining)),
                replace=False,
            ).tolist()
        )
    return np.sort(np.asarray(selected, dtype=int))


def lime_prediction_matrix(model: Any, X: np.ndarray) -> np.ndarray:
    """Classification matrix used by the local surrogate for both task types."""

    return predict_probabilities(model, np.asarray(X, dtype=np.float32))


def explain_lime(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_panel: np.ndarray,
    targets: np.ndarray,
    feature_names: Sequence[str],
    *,
    num_samples: int,
    background_max_rows: int,
    seed: int,
) -> np.ndarray:
    from lime.lime_tabular import LimeTabularExplainer

    background_indices = _stratified_background_indices(
        y_train,
        int(background_max_rows),
        seed=stable_seed(seed, "background"),
    )
    background = np.asarray(X_train[background_indices], dtype=np.float32)
    explainer = LimeTabularExplainer(
        background,
        feature_names=list(feature_names),
        discretize_continuous=False,
        mode="classification",
        random_state=int(seed),
    )
    attributes = np.zeros((len(X_panel), len(feature_names)), dtype=np.float32)

    def predict_fn(values: np.ndarray) -> np.ndarray:
        return lime_prediction_matrix(model, values)

    for row_index, row in tqdm(
        list(enumerate(np.asarray(X_panel, dtype=np.float32))),
        desc="lime rows",
        unit="row",
        leave=False,
    ):
        target = int(targets[row_index])
        explanation = explainer.explain_instance(
            row,
            predict_fn,
            labels=[target],
            num_features=len(feature_names),
            num_samples=int(num_samples),
        )
        for feature_index, weight in explanation.as_map().get(target, []):
            attributes[row_index, int(feature_index)] = float(weight)
    return attributes


def explain_kernel_shap(
    model: Any,
    X: np.ndarray,
    targets: np.ndarray,
    background: np.ndarray,
    *,
    background_size: int,
    nsamples: int,
    seed: int,
) -> np.ndarray:
    """Target-class-aware KernelSHAP.

    `background` carries one resolved reference row per panel row (the same
    convention as explain_occlusion's `replacements`). Rows sharing an
    identical resolved reference and target class are batched into one
    KernelExplainer call, whose background set is that single reference row
    repeated `background_size` times — a fixed deterministic baseline rather
    than a stochastic background sample, for consistency with this audit's
    other single-reference-point probes (occlusion, integrated gradients).
    """

    import shap

    X_array = np.asarray(X, dtype=np.float32)
    reference = np.asarray(background, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.int32)
    if reference.shape != X_array.shape:
        raise ValueError("KernelSHAP background must have one row per flow")
    if int(background_size) <= 0:
        raise ValueError("KernelSHAP background_size must be positive")
    attributes = np.zeros_like(X_array, dtype=np.float32)
    # Group by (target class, reference row) so every explainer call has a
    # single fixed background and a single fixed prediction target.
    group_key = np.concatenate(
        [target_array.reshape(-1, 1).astype(np.float32), reference], axis=1
    )
    _, group_ids = np.unique(group_key, axis=0, return_inverse=True)
    rng_state = np.random.get_state()
    try:
        np.random.seed(int(seed) % (2**32))
        for group_id in tqdm(
            np.unique(group_ids), desc="shap groups", unit="group", leave=False
        ):
            row_indices = np.flatnonzero(group_ids == group_id)
            target_class = int(target_array[row_indices[0]])
            background_row = reference[row_indices[0]]
            background_matrix = np.broadcast_to(
                background_row, (int(background_size), background_row.shape[0])
            ).astype(np.float32)

            def predict_target(values: np.ndarray) -> np.ndarray:
                probabilities = predict_probabilities(model, np.asarray(values, dtype=np.float32))
                return probabilities[:, target_class]

            explainer = shap.KernelExplainer(predict_target, background_matrix)
            values = explainer.shap_values(
                X_array[row_indices], nsamples=int(nsamples), silent=True
            )
            values_array = np.asarray(values, dtype=np.float32)
            if values_array.ndim == 3:
                values_array = values_array[:, :, 0]
            attributes[row_indices] = values_array
    finally:
        np.random.set_state(rng_state)
    return attributes


def _processed_probe_inputs(processed_dir: Path) -> list[Path]:
    names = [
        "X_train.npy",
        "y_train.npy",
        "X_test.npy",
        "y_test.npy",
        "source_file_test.npy",
        "source_row_test.npy",
        "raw_fingerprint_test.npy",
        "tensor_fingerprint_exact_test.npy",
        "tensor_fingerprint_round5_test.npy",
        "feature_names.json",
        "label_mapping.json",
        "baseline.npy",
        "split_validation.csv",
        "cache_metadata.json",
    ]
    return [processed_dir / name for name in names]


def _probe_directory(
    paths: AuditPaths,
    task_id: str,
    model: str,
    seed: int,
    spec: ProbeSpec,
) -> Path:
    return (
        paths.audit_root
        / "probes"
        / task_id
        / model
        / f"seed_{seed}"
        / spec.method
        / spec.spec_id
    )


def _probe_output_paths(directory: Path) -> dict[str, Path]:
    return {
        "array": directory / "attributions.npz",
        "index": directory / "per_flow_index.csv",
        "decision": directory / "probe_decision.json",
        "metadata": directory / "cache_metadata.json",
    }


def _probe_signature(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
    *,
    task_id: str,
    model: str,
    seed: int,
    spec: ProbeSpec,
) -> tuple[str, list[dict[str, Any]], str]:
    require_valid_panel(config, training_config, paths)
    condition = require_valid_training_condition(
        paths,
        training_config,
        task_id=task_id,
        model=model,
        seed=seed,
    )
    input_paths = [
        paths.config_path,
        paths.training_config_path,
        paths.central_panel,
        paths.class_support,
        paths.audit_units,
        paths.panel_dir / "cache_metadata.json",
        condition["model"],
        condition["predictions"],
        condition["metadata"],
        *_processed_probe_inputs(paths.processed_root / task_id),
    ]
    inputs = artifact_records(input_paths)
    implementation_hash = signature({"files": _audit_implementation_records(paths)})
    value = signature(
        {
            "audit_version": AUDIT_VERSION,
            "stage": "measurement_probe",
            "config_hash": sha256_file(paths.config_path),
            "training_config_hash": sha256_file(paths.training_config_path),
            "implementation_hash": implementation_hash,
            "inputs": inputs,
            "task_id": task_id,
            "model": model,
            "seed": int(seed),
            "probe": spec.payload(),
            "stochastic_panel_only": spec.method in STOCHASTIC_SUBSET_PROBES,
        }
    )
    return value, inputs, implementation_hash


def _load_frozen_model(path: Path) -> Any:
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


def _compute_probe(
    config: Mapping[str, Any],
    data: ConditionData,
    model: Any,
    units: pd.DataFrame,
    spec: ProbeSpec,
) -> tuple[np.ndarray, np.ndarray]:
    orders = _verify_units_against_test(units, data)
    X = np.asarray(data.X_test[orders], dtype=np.float32)
    true_classes = data.y_test[orders].astype(np.int32)
    targets = pd.to_numeric(units["target_class_id"], errors="raise").to_numpy(
        dtype=np.int32
    )
    if np.any(targets < 0) or np.any(targets >= len(data.class_names)):
        raise RuntimeError("Audit target class lies outside the frozen label mapping")
    method_cfg = config["measurement_probes"][spec.method]
    if spec.method == "gradient_x_input":
        attributes = explain_gradient_x_input(
            model,
            X,
            targets,
            batch_size=int(method_cfg["batch_size"]),
        )
    elif spec.method == "integrated_gradients":
        reference_classes = targets if spec.reference == "target_class_median" else true_classes
        references = reference_matrix(data, reference_classes, spec.reference)
        attributes = explain_integrated_gradients(
            model,
            X,
            targets,
            references,
            steps=int(spec.steps or method_cfg["main_steps"]),
            batch_size=int(method_cfg["batch_size"]),
        )
    elif spec.method == "occlusion":
        reference_classes = targets if spec.reference == "target_class_median" else true_classes
        replacements = reference_matrix(data, reference_classes, spec.reference)
        attributes = explain_occlusion(
            model,
            X,
            targets,
            replacements,
            batch_size=int(method_cfg["batch_size"]),
        )
    elif spec.method == "lime":
        run_seed = stable_seed(
            config["panel"]["sampling_seed"],
            data.task_id,
            int(units["seed"].iloc[0]),
            spec.run_id,
        )
        attributes = explain_lime(
            model,
            data.X_train,
            data.y_train,
            X,
            targets,
            data.feature_names,
            num_samples=int(method_cfg["num_samples"]),
            background_max_rows=int(method_cfg["training_background_max_rows"]),
            seed=run_seed,
        )
    elif spec.method == "shap":
        reference_classes = targets if spec.reference == "target_class_median" else true_classes
        background_rows = reference_matrix(data, reference_classes, spec.reference)
        run_seed = stable_seed(
            config["panel"]["sampling_seed"],
            data.task_id,
            int(units["seed"].iloc[0]),
            spec.run_id,
        )
        attributes = explain_kernel_shap(
            model,
            X,
            targets,
            background_rows,
            background_size=int(method_cfg["background_size"]),
            nsamples=int(method_cfg["nsamples"]),
            seed=run_seed,
        )
    else:  # pragma: no cover - validate_audit_config makes this unreachable.
        raise ValueError(f"Unsupported probe {spec.method!r}")
    if attributes.shape != X.shape or not np.isfinite(attributes).all():
        raise RuntimeError(
            f"Probe returned invalid attribution geometry or values: {attributes.shape}"
        )
    probabilities = target_probabilities(model, X, targets)
    return attributes.astype(np.float32), probabilities.astype(np.float64)


def run_probe_condition(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
    *,
    task_id: str,
    model: str,
    seed: int,
    spec: ProbeSpec,
    force: bool = False,
) -> ProbeResult:
    if task_id not in ALLOWED_TASKS or model not in ALLOWED_MODELS:
        raise ValueError("Probe condition lies outside the frozen scope")
    if int(seed) not in training.FINAL_SEEDS or spec.method not in ALLOWED_PROBES:
        raise ValueError("Probe seed or method lies outside the frozen scope")
    allowed_specs = {candidate.spec_id for candidate in probe_specs(config, spec.method)}
    if spec.spec_id not in allowed_specs:
        raise ValueError("Probe specification is not declared in the frozen protocol")
    directory = _probe_directory(paths, task_id, model, int(seed), spec)
    outputs = _probe_output_paths(directory)
    stage_signature, inputs, implementation_hash = _probe_signature(
        config,
        training_config,
        paths,
        task_id=task_id,
        model=model,
        seed=int(seed),
        spec=spec,
    )
    expected = [outputs["array"], outputs["index"], outputs["decision"]]
    hit, _ = cache_hit(
        outputs["metadata"],
        expected_signature=stage_signature,
        expected_outputs=expected,
    )
    if hit and not force:
        return load_probe_result(
            config,
            training_config,
            paths,
            task_id=task_id,
            model=model,
            seed=int(seed),
            spec=spec,
        )

    units = _condition_units(
        paths,
        task_id=task_id,
        model=model,
        seed=int(seed),
        stochastic_only=spec.method in STOCHASTIC_SUBSET_PROBES,
    )
    data = load_condition_data(paths, task_id)
    condition = _training_condition_paths(paths, task_id, model, int(seed))
    frozen_model = _load_frozen_model(condition["model"])
    started = time.perf_counter()
    attributes, probabilities = _compute_probe(config, data, frozen_model, units, spec)
    runtime = float(time.perf_counter() - started)
    index = units[list(IDENTITY_COLUMNS) + [
        "model",
        "in_central",
        "in_outcome_correct",
        "in_outcome_error",
        "in_stochastic_subset",
        "central_class_seed_support",
        "validation_support",
        "validation_inference_eligible",
        "central_sample_support_sufficient",
        "class_specific_inference_eligible",
        "target_class_id",
        "target_class",
        "target_basis",
        "confidence",
        "model_margin",
    ]].copy()
    index["method"] = spec.method
    index["spec_id"] = spec.spec_id
    index["reference"] = spec.reference
    index["steps"] = spec.steps
    index["probe_run"] = int(spec.run_id)
    index["target_probability"] = probabilities
    index["attribution_l1"] = np.abs(attributes).sum(axis=1)
    index["zero_mass_account"] = index["attribution_l1"].lt(MINIMUM_ACCOUNT_L1)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outputs["array"],
        attributions=attributes,
        target_probability=probabilities,
        sample_order=index["sample_order"].to_numpy(dtype=np.int64),
        target_class_id=index["target_class_id"].to_numpy(dtype=np.int64),
        feature_names=np.asarray(data.feature_names, dtype="S128"),
    )
    index.to_csv(outputs["index"], index=False)
    decision = {
        "audit_version": AUDIT_VERSION,
        "status": "complete",
        "task_id": task_id,
        "model": model,
        "seed": int(seed),
        "probe": spec.payload(),
        "rows": int(len(index)),
        "features": int(attributes.shape[1]),
        "target": "declared_target_class",
        "stochastic_panel_only": spec.method in STOCHASTIC_SUBSET_PROBES,
        "runtime_seconds": runtime,
        "stage_signature": stage_signature,
    }
    outputs["decision"].write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_cache_metadata(
        outputs["metadata"],
        stage="measurement_probe",
        signature_value=stage_signature,
        config_hash=sha256_file(paths.config_path),
        implementation_hash=implementation_hash,
        inputs=inputs,
        outputs=expected,
        extra=decision,
    )
    return ProbeResult(
        index=index,
        attributions=attributes,
        target_probability=probabilities,
        path=outputs["array"],
        metadata_path=outputs["metadata"],
    )


def load_probe_result(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
    *,
    task_id: str,
    model: str,
    seed: int,
    spec: ProbeSpec,
) -> ProbeResult:
    directory = _probe_directory(paths, task_id, model, int(seed), spec)
    outputs = _probe_output_paths(directory)
    stage_signature, _, _ = _probe_signature(
        config,
        training_config,
        paths,
        task_id=task_id,
        model=model,
        seed=int(seed),
        spec=spec,
    )
    expected = [outputs["array"], outputs["index"], outputs["decision"]]
    hit, reason = cache_hit(
        outputs["metadata"],
        expected_signature=stage_signature,
        expected_outputs=expected,
    )
    if not hit:
        raise RuntimeError(
            f"Probe cache failed closed for {task_id}/{model}/seed_{seed}/"
            f"{spec.method}/{spec.spec_id}: {reason}"
        )
    index = pd.read_csv(outputs["index"])
    with np.load(outputs["array"], allow_pickle=False) as archive:
        attributes = archive["attributions"].astype(np.float32)
        probabilities = archive["target_probability"].astype(float)
        sample_order = archive["sample_order"].astype(np.int64)
    if (
        attributes.ndim != 2
        or len(attributes) != len(index)
        or len(probabilities) != len(index)
        or not np.array_equal(
            sample_order,
            pd.to_numeric(index["sample_order"], errors="raise").to_numpy(np.int64),
        )
        or not np.isfinite(attributes).all()
        or not np.isfinite(probabilities).all()
    ):
        raise RuntimeError("Probe arrays and per-flow identity table disagree")
    return ProbeResult(
        index=index,
        attributions=attributes,
        target_probability=probabilities,
        path=outputs["array"],
        metadata_path=outputs["metadata"],
    )


def estimate_lime_workload(
    units: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    observed_seconds_per_explanation: float | None = None,
) -> pd.DataFrame:
    """Count the confirmatory workload without changing its registered budget."""

    flags = units["in_stochastic_subset"].astype(str).str.lower().isin({"true", "1"})
    selected = units[flags].copy()
    runs = int(config["measurement_probes"]["lime"]["independent_runs"])
    samples = int(config["measurement_probes"]["lime"]["num_samples"])
    group_columns = ["task_id", "model", "seed", "true_class_id", "true_class"]
    table = (
        selected.groupby(group_columns, dropna=False)
        .size()
        .rename("flows")
        .reset_index()
    )
    table["independent_runs"] = runs
    table["samples_per_explanation"] = samples
    table["explanation_calls"] = table["flows"] * runs
    table["model_evaluations_requested"] = table["explanation_calls"] * samples
    if observed_seconds_per_explanation is not None:
        table["projected_seconds"] = (
            table["explanation_calls"] * float(observed_seconds_per_explanation)
        )
    total = {
        "task_id": "ALL",
        "model": "ALL",
        "seed": -1,
        "true_class_id": -1,
        "true_class": "ALL",
        "flows": int(table["flows"].sum()),
        "independent_runs": runs,
        "samples_per_explanation": samples,
        "explanation_calls": int(table["explanation_calls"].sum()),
        "model_evaluations_requested": int(table["model_evaluations_requested"].sum()),
    }
    if observed_seconds_per_explanation is not None:
        total["projected_seconds"] = float(
            table["projected_seconds"].sum()  # type: ignore[assignment]
        )
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


def estimate_functional_workload(
    units: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Count frozen-model row evaluations for the bounded functional audit."""

    flags = units["in_stochastic_subset"].astype(str).str.lower().isin({"true", "1"})
    selected = units[flags].copy()
    deletion = config["functional_checks"]["deletion_and_insertion"]
    signed = config["functional_checks"]["signed_intervention"]
    evaluator_count = len(deletion["evaluators"])
    fraction_count = len(deletion["feature_fractions"])
    deletion_repetitions = int(deletion["random_feature_repetitions"])
    signed_repetitions = int(signed["random_feature_repetitions"])
    signed_condition_count = 2 * len(signed["strengths"])
    # One clean prediction; one reference prediction per evaluator; two ranked
    # interventions plus four random-control interventions per fraction; one
    # ranked and one random prediction per signed repetition/condition.
    row_evaluations_per_flow = (
        1
        + evaluator_count
        * (1 + fraction_count * (2 + 4 * deletion_repetitions))
        + signed_condition_count * (1 + signed_repetitions)
    )
    group_columns = ["task_id", "model", "seed", "true_class_id", "true_class"]
    base = (
        selected.groupby(group_columns, dropna=False)
        .size()
        .rename("flows")
        .reset_index()
    )
    frames: list[pd.DataFrame] = []
    for method in ALLOWED_PROBES:
        current = base.copy()
        current["method"] = method
        frames.append(current)
    table = pd.concat(frames, ignore_index=True)
    table["evaluators"] = evaluator_count
    table["fractions"] = fraction_count
    table["deletion_random_repetitions"] = deletion_repetitions
    table["signed_conditions"] = signed_condition_count
    table["signed_random_repetitions"] = signed_repetitions
    table["row_evaluations_per_flow"] = row_evaluations_per_flow
    table["requested_model_row_evaluations"] = (
        table["flows"] * row_evaluations_per_flow
    )
    total = {
        "task_id": "ALL",
        "model": "ALL",
        "seed": -1,
        "true_class_id": -1,
        "true_class": "ALL",
        "flows": int(table["flows"].sum()),
        "method": "ALL",
        "evaluators": evaluator_count,
        "fractions": fraction_count,
        "deletion_random_repetitions": deletion_repetitions,
        "signed_conditions": signed_condition_count,
        "signed_random_repetitions": signed_repetitions,
        "row_evaluations_per_flow": row_evaluations_per_flow,
        "requested_model_row_evaluations": int(
            table["requested_model_row_evaluations"].sum()
        ),
    }
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


def run_lime_cost_benchmark(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
    *,
    task_id: str,
    model: str,
    seed: int,
    flows_per_class: int = 1,
    runs: int = 1,
) -> pd.DataFrame:
    """Run an explicitly non-confirmatory timing pilot in a separate namespace."""

    if flows_per_class <= 0 or runs <= 0:
        raise ValueError("Benchmark flow and run counts must be positive")
    units = _condition_units(
        paths,
        task_id=task_id,
        model=model,
        seed=int(seed),
        stochastic_only=True,
    )
    sampled = (
        units.sort_values(["true_class_id", "sample_order"], kind="mergesort")
        .groupby("true_class_id", group_keys=False)
        .head(int(flows_per_class))
        .reset_index(drop=True)
    )
    data = load_condition_data(paths, task_id)
    condition = require_valid_training_condition(
        paths,
        training_config,
        task_id=task_id,
        model=model,
        seed=int(seed),
    )
    frozen_model = _load_frozen_model(condition["model"])
    rows: list[dict[str, Any]] = []
    for run_id in range(int(runs)):
        spec = ProbeSpec("lime", "local_surrogate", run_id=run_id, role="cost_benchmark")
        started = time.perf_counter()
        attributes, _ = _compute_probe(config, data, frozen_model, sampled, spec)
        elapsed = float(time.perf_counter() - started)
        rows.append(
            {
                "status": "non_confirmatory_cost_benchmark",
                "task_id": task_id,
                "model": model,
                "seed": int(seed),
                "benchmark_run": run_id,
                "flows": int(len(sampled)),
                "num_samples": int(config["measurement_probes"]["lime"]["num_samples"]),
                "runtime_seconds": elapsed,
                "seconds_per_explanation": elapsed / max(len(sampled), 1),
                "finite_output": bool(np.isfinite(attributes).all()),
            }
        )
    result = pd.DataFrame(rows)
    output_dir = paths.audit_root / "nonconfirmatory_cost_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{task_id}__{model}__seed_{seed}.csv"
    result.to_csv(output, index=False)
    units_all = pd.read_csv(paths.audit_units)
    estimate = estimate_lime_workload(
        units_all,
        config,
        observed_seconds_per_explanation=float(result["seconds_per_explanation"].mean()),
    )
    estimate.to_csv(output_dir / "projected_registered_workload.csv", index=False)
    return result


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    if int(valid.sum()) < 2:
        return float("nan")
    left_valid = left_array[valid]
    right_valid = right_array[valid]
    if np.allclose(left_valid, right_valid, rtol=0.0, atol=1e-12):
        return 1.0
    if np.allclose(left_valid, left_valid[0]) or np.allclose(
        right_valid, right_valid[0]
    ):
        return 0.0
    return float(spearmanr(left_valid, right_valid).statistic)


def _topk(values: np.ndarray, k: int) -> np.ndarray:
    array = np.abs(np.asarray(values, dtype=float))
    effective = min(max(int(k), 1), len(array))
    # Feature index is the deterministic tie-break.
    return np.lexsort((np.arange(len(array)), -array))[:effective]


def _account_diagnostics(values: np.ndarray, top_k: int) -> dict[str, Any]:
    """Describe whether an attribution vector identifies a ranked account.

    A deterministic index tie-break is useful for reproducible files, but it
    must not turn a null or flat vector into apparently identified features.
    The tolerances below are scale relative after a small absolute floor.
    """

    absolute = np.abs(np.asarray(values, dtype=float).reshape(-1))
    total = float(absolute.sum())
    maximum = float(absolute.max(initial=0.0))
    tolerance = max(MINIMUM_ACCOUNT_L1, maximum * RANK_TIE_RELATIVE_TOLERANCE)
    nonzero = bool(total >= MINIMUM_ACCOUNT_L1)
    rank_identifiable = bool(
        nonzero and float(np.ptp(absolute)) > tolerance
    )
    effective = min(max(int(top_k), 1), len(absolute))
    ordered = np.sort(absolute)[::-1]
    if not nonzero:
        top_k_identifiable = False
        boundary_gap = 0.0
    elif effective >= len(ordered):
        top_k_identifiable = True
        boundary_gap = float("inf")
    else:
        boundary_gap = float(ordered[effective - 1] - ordered[effective])
        top_k_identifiable = bool(boundary_gap > tolerance)
    return {
        "mass": total,
        "maximum": maximum,
        "nonzero": nonzero,
        "rank_identifiable": rank_identifiable,
        "top_k_identifiable": top_k_identifiable,
        "top_k_boundary_gap": boundary_gap,
    }


def attribution_similarity(
    left: np.ndarray,
    right: np.ndarray,
    *,
    top_k: int,
) -> dict[str, float]:
    """Compare absolute feature rankings/mass, as declared in the protocol."""

    left_abs = np.abs(np.asarray(left, dtype=float).reshape(-1))
    right_abs = np.abs(np.asarray(right, dtype=float).reshape(-1))
    if left_abs.shape != right_abs.shape or not (
        np.isfinite(left_abs).all() and np.isfinite(right_abs).all()
    ):
        raise ValueError("Attribution vectors must have equal finite geometry")
    left_diagnostics = _account_diagnostics(left_abs, top_k)
    right_diagnostics = _account_diagnostics(right_abs, top_k)
    pair_nonzero = bool(
        left_diagnostics["nonzero"] and right_diagnostics["nonzero"]
    )
    pair_rank_identifiable = bool(
        left_diagnostics["rank_identifiable"]
        and right_diagnostics["rank_identifiable"]
    )
    pair_top_k_identifiable = bool(
        left_diagnostics["top_k_identifiable"]
        and right_diagnostics["top_k_identifiable"]
    )
    left_top = set(_topk(left_abs, top_k).tolist())
    right_top = set(_topk(right_abs, top_k).tolist())
    union = left_top | right_top
    left_total = float(left_abs.sum())
    right_total = float(right_abs.sum())
    if not pair_nonzero:
        # Fail closed: absence of an attributable account is recorded
        # separately and cannot masquerade as perfect feature agreement.
        mass_l1 = 1.0
    else:
        left_mass = left_abs / left_total
        right_mass = right_abs / right_total
        mass_l1 = float(0.5 * np.abs(left_mass - right_mass).sum())
    weighted_denominator = (
        float(np.maximum(left_mass, right_mass).sum()) if pair_nonzero else 0.0
    )
    return {
        "spearman": (
            _safe_spearman(left_abs, right_abs) if pair_rank_identifiable else 0.0
        ),
        "jaccard_10": (
            float(len(left_top & right_top) / len(union))
            if union and pair_top_k_identifiable
            else 0.0
        ),
        "weighted_jaccard": (
            float(np.minimum(left_mass, right_mass).sum() / weighted_denominator)
            if pair_nonzero and weighted_denominator >= MINIMUM_ACCOUNT_L1
            else 0.0
        ),
        "normalized_mass_l1": mass_l1,
        "left_account_mass": float(left_diagnostics["mass"]),
        "right_account_mass": float(right_diagnostics["mass"]),
        "left_account_nonzero": bool(left_diagnostics["nonzero"]),
        "right_account_nonzero": bool(right_diagnostics["nonzero"]),
        "account_pair_nonzero": pair_nonzero,
        "left_rank_identifiable": bool(left_diagnostics["rank_identifiable"]),
        "right_rank_identifiable": bool(right_diagnostics["rank_identifiable"]),
        "rank_pair_identifiable": pair_rank_identifiable,
        "left_top_k_identifiable": bool(left_diagnostics["top_k_identifiable"]),
        "right_top_k_identifiable": bool(right_diagnostics["top_k_identifiable"]),
        "top_k_pair_identifiable": pair_top_k_identifiable,
        "agreement_on_null_accounts": bool(
            not left_diagnostics["nonzero"] and not right_diagnostics["nonzero"]
        ),
    }


def crossed_source_seed_equal_seed_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    seed_column: str = "seed",
    flow_column: str = "flow_id",
    run_column: str | None = None,
) -> dict[str, Any]:
    """Crossed seed x source bootstrap with equal fitted-seed weight.

    The same source flow can enter the frozen panel under several fitted seeds.
    One global multinomial source draw is therefore shared across every seed in
    a replicate.  Fitted seeds are resampled independently.  If probe runs are
    present, one run draw is shared by all flows for each sampled seed
    occurrence, preserving common-random-number dependence.
    """

    columns = [seed_column, flow_column, value_column]
    if run_column is not None:
        columns.append(run_column)
    clean = frame[columns].copy()
    clean[value_column] = pd.to_numeric(clean[value_column], errors="coerce")
    clean = clean.dropna(subset=[value_column])
    groups: dict[int, dict[str, dict[Any, float]]] = {}
    for seed_value, seed_frame in clean.groupby(seed_column, sort=True):
        flows: dict[str, dict[Any, float]] = {}
        for flow_id, flow_frame in seed_frame.groupby(flow_column, sort=True):
            if run_column is None:
                values = {None: float(flow_frame[value_column].mean())}
            else:
                by_run = (
                    flow_frame.groupby(run_column, dropna=False)[value_column]
                    .mean()
                )
                values = {run_id: float(value) for run_id, value in by_run.items()}
            if values:
                flows[str(flow_id)] = values
        if flows:
            groups[int(seed_value)] = flows
    if not groups:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_std": float("nan"),
            "p_value_two_sided_zero": float("nan"),
            "n_rows": 0,
            "n_flows": 0,
            "n_seed_flow_cells": 0,
            "n_unique_sources": 0,
            "n_reused_sources": 0,
            "maximum_seed_reuse": 0,
            "n_seeds": 0,
            "bootstrap_design": CROSSED_SOURCE_SEED_BOOTSTRAP,
        }
    seed_values = np.asarray(sorted(groups), dtype=int)
    source_values = np.asarray(
        sorted({flow for flows in groups.values() for flow in flows}), dtype=object
    )
    source_position = {str(value): index for index, value in enumerate(source_values)}
    source_seed_reuse = {
        str(source): sum(str(source) in groups[int(seed_value)] for seed_value in seed_values)
        for source in source_values
    }
    run_ids_by_seed: dict[int, np.ndarray] = {}
    run_positions_by_seed: dict[int, dict[Any, int]] = {}
    source_positions_by_seed: dict[int, np.ndarray] = {}
    value_matrix_by_seed: dict[int, np.ndarray] = {}
    for seed_value, flows in groups.items():
        seed_key = int(seed_value)
        source_positions_by_seed[seed_key] = np.asarray(
            [source_position[str(flow_id)] for flow_id in flows],
            dtype=int,
        )
        if run_column is not None:
            inventories = {tuple(sorted(values)) for values in flows.values()}
            if len(inventories) != 1:
                raise RuntimeError("Flows within a seed do not share one probe-run inventory")
            run_ids = np.asarray(next(iter(inventories)), dtype=object)
            run_ids_by_seed[seed_key] = run_ids
            run_positions_by_seed[seed_key] = {
                run_id: position for position, run_id in enumerate(run_ids)
            }
            value_matrix_by_seed[seed_key] = np.asarray(
                [
                    [run_values[run_id] for run_id in run_ids]
                    for run_values in flows.values()
                ],
                dtype=float,
            )
        else:
            value_matrix_by_seed[seed_key] = np.asarray(
                [
                    [float(next(iter(run_values.values())))]
                    for run_values in flows.values()
                ],
                dtype=float,
            )

    estimate = float(
        np.mean(
            [float(value_matrix_by_seed[int(value)].mean()) for value in seed_values]
        )
    )
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_resamples), dtype=float)
    source_probabilities = np.full(
        len(source_values), 1.0 / len(source_values), dtype=float
    )
    for draw_index in range(int(n_resamples)):
        selected_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
        # A single source draw is reused in every selected seed occurrence.
        # Redraw only in the rare case where it misses every source observed by
        # one of those seed panels.
        for _ in range(10_000):
            source_counts = rng.multinomial(
                len(source_values),
                source_probabilities,
            )
            if all(
                np.any(
                    source_counts[source_positions_by_seed[int(selected_seed)]] > 0
                )
                for selected_seed in selected_seeds
            ):
                break
        else:
            raise RuntimeError("Could not draw source clusters represented in every seed")
        seed_draws: list[float] = []
        for selected_seed in selected_seeds:
            seed_key = int(selected_seed)
            value_matrix = value_matrix_by_seed[seed_key]
            if run_column is not None:
                run_ids = run_ids_by_seed[seed_key]
                sampled_runs = rng.choice(run_ids, size=len(run_ids), replace=True)
                sampled_run_positions = np.fromiter(
                    (
                        run_positions_by_seed[seed_key][run_id]
                        for run_id in sampled_runs
                    ),
                    dtype=int,
                    count=len(sampled_runs),
                )
                flow_values = value_matrix[:, sampled_run_positions].mean(axis=1)
            else:
                flow_values = value_matrix[:, 0]
            source_weights = source_counts[source_positions_by_seed[seed_key]]
            total_weight = int(source_weights.sum())
            if total_weight <= 0:
                raise RuntimeError("Crossed source draw produced an empty sampled seed")
            seed_draws.append(float(source_weights @ flow_values) / total_weight)
        draws[draw_index] = float(np.mean(seed_draws))
    tail = (1.0 - float(confidence_level)) / 2.0
    ci_low, ci_high = np.quantile(draws, [tail, 1.0 - tail])
    p_lower = float(np.mean(draws <= 0.0))
    p_upper = float(np.mean(draws >= 0.0))
    return {
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_std": float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0,
        "p_value_two_sided_zero": float(min(1.0, 2.0 * min(p_lower, p_upper))),
        "n_rows": int(len(clean)),
        "n_flows": int(len(source_values)),
        "n_seed_flow_cells": int(sum(len(value) for value in groups.values())),
        "n_unique_sources": int(len(source_values)),
        "n_reused_sources": int(
            sum(reuse_count > 1 for reuse_count in source_seed_reuse.values())
        ),
        "maximum_seed_reuse": int(max(source_seed_reuse.values())),
        "n_seeds": int(len(groups)),
        "bootstrap_design": CROSSED_SOURCE_SEED_BOOTSTRAP,
        "run_resampling": (
            "one_shared_run_multiset_per_sampled_seed_occurrence"
            if run_column is not None
            else "not_applicable"
        ),
    }


# Compatibility for callers written before the crossed-cluster correction.
# New summaries use the scientifically accurate name above.
hierarchical_equal_seed_bootstrap = crossed_source_seed_equal_seed_bootstrap


def crossed_source_seed_lime_u_statistic_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Crossed source/seed bootstrap with a shared run draw inside each seed."""

    required = {"seed", "flow_id", "left_run", "right_run", value_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stochastic repeatability detail lacks {missing}")
    groups: dict[int, dict[str, tuple[np.ndarray, dict[tuple[int, int], float]]]] = {}
    for seed_value, seed_frame in frame.groupby("seed", sort=True):
        flows: dict[str, tuple[np.ndarray, dict[tuple[int, int], float]]] = {}
        for flow_id, flow_frame in seed_frame.groupby("flow_id", sort=True):
            pair_values: dict[tuple[int, int], float] = {}
            runs: set[int] = set()
            for _, row in flow_frame.iterrows():
                left = int(row["left_run"])
                right = int(row["right_run"])
                value = float(row[value_column])
                if not np.isfinite(value):
                    continue
                runs.update((left, right))
                pair_values[tuple(sorted((left, right)))] = value
            run_array = np.asarray(sorted(runs), dtype=int)
            expected_pairs = len(run_array) * (len(run_array) - 1) // 2
            if len(run_array) < 2 or len(pair_values) != expected_pairs:
                raise RuntimeError("Stochastic run-pair inventory is incomplete")
            flows[str(flow_id)] = (run_array, pair_values)
        if flows:
            groups[int(seed_value)] = flows
    if not groups:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_std": float("nan"),
            "p_value_two_sided_zero": float("nan"),
            "n_rows": 0,
            "n_flows": 0,
            "n_seed_flow_cells": 0,
            "n_unique_sources": 0,
            "n_reused_sources": 0,
            "maximum_seed_reuse": 0,
            "n_seeds": 0,
            "n_probe_runs": 0,
            "bootstrap_design": CROSSED_SOURCE_SEED_LIME_BOOTSTRAP,
        }

    def flow_projection(
        run_values: np.ndarray,
        pair_values: Mapping[tuple[int, int], float],
    ) -> np.ndarray:
        # Leave-one-run projections used to form jackknife pseudovalues below.
        # This preserves whole-panel run dependence without introducing the
        # optimistic self-pair diagonal of a naive bootstrap of run labels.
        return np.asarray(
            [
                np.mean(
                    [
                        pair_values[tuple(sorted((int(run), int(other))))]
                        for other in run_values
                        if int(other) != int(run)
                    ]
                )
                for run in run_values
            ],
            dtype=float,
        )

    seed_values = np.asarray(sorted(groups), dtype=int)
    source_values = np.asarray(
        sorted({flow for flows in groups.values() for flow in flows}), dtype=object
    )
    source_position = {str(value): index for index, value in enumerate(source_values)}
    source_seed_reuse = {
        str(source): sum(str(source) in groups[int(seed_value)] for seed_value in seed_values)
        for source in source_values
    }
    run_ids_by_seed: dict[int, np.ndarray] = {}
    source_positions_by_seed: dict[int, np.ndarray] = {}
    projection_matrix_by_seed: dict[int, np.ndarray] = {}
    for seed_value, flows in groups.items():
        inventories = {tuple(runs.tolist()) for runs, _ in flows.values()}
        if len(inventories) != 1:
            raise RuntimeError("LIME flows within a seed do not share one run inventory")
        run_ids = np.asarray(next(iter(inventories)), dtype=int)
        run_ids_by_seed[int(seed_value)] = run_ids
        source_positions_by_seed[int(seed_value)] = np.asarray(
            [source_position[str(flow_id)] for flow_id in flows],
            dtype=int,
        )
        # Computing every flow x run projection is the expensive part of the
        # order-two U-statistic.  It depends only on the observed run pairs, so
        # cache it once and let each bootstrap draw only apply source weights.
        projection_matrix_by_seed[int(seed_value)] = np.vstack(
            [flow_projection(runs, pairs) for runs, pairs in flows.values()]
        )
    run_counts = {len(run_ids) for run_ids in run_ids_by_seed.values()}
    if len(run_counts) != 1:
        raise RuntimeError("Stochastic flows do not share one independent-run count")
    run_count = int(next(iter(run_counts)))
    if run_count <= 2:
        raise RuntimeError("Jackknife U-statistic bootstrap needs at least three runs")
    estimate = float(
        np.mean(
            [
                float(projection_matrix_by_seed[int(value)].mean())
                for value in seed_values
            ]
        )
    )
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_resamples), dtype=float)
    source_probabilities = np.full(
        len(source_values), 1.0 / len(source_values), dtype=float
    )
    for draw_index in range(int(n_resamples)):
        sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
        for _ in range(10_000):
            source_counts = rng.multinomial(
                len(source_values),
                source_probabilities,
            )
            if all(
                np.any(
                    source_counts[source_positions_by_seed[int(sampled_seed)]] > 0
                )
                for sampled_seed in sampled_seeds
            ):
                break
        else:
            raise RuntimeError("Could not draw LIME source clusters represented in every seed")
        seed_draws: list[float] = []
        for sampled_seed in sampled_seeds:
            run_ids = run_ids_by_seed[int(sampled_seed)]
            sampled_runs = rng.choice(run_ids, size=len(run_ids), replace=True)
            source_weights = source_counts[
                source_positions_by_seed[int(sampled_seed)]
            ]
            total_weight = int(source_weights.sum())
            if total_weight <= 0:
                raise RuntimeError("Crossed LIME draw produced an empty sampled seed")
            projection = (
                source_weights @ projection_matrix_by_seed[int(sampled_seed)]
            ) / total_weight
            u_statistic = float(projection.mean())
            # Order-two U-statistic jackknife pseudovalues.  Their mean is U,
            # while their spread has the correct first-order factor of two.
            pseudovalues = (
                2.0 * (run_count - 1) * projection - run_count * u_statistic
            ) / (run_count - 2)
            sampled_run_positions = np.searchsorted(run_ids, sampled_runs)
            seed_draws.append(
                float(np.mean(pseudovalues[sampled_run_positions]))
            )
        draws[draw_index] = float(np.mean(seed_draws))
    tail = (1.0 - float(confidence_level)) / 2.0
    ci_low, ci_high = np.quantile(draws, [tail, 1.0 - tail])
    p_lower = float(np.mean(draws <= 0.0))
    p_upper = float(np.mean(draws >= 0.0))
    return {
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_std": float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0,
        "p_value_two_sided_zero": float(min(1.0, 2.0 * min(p_lower, p_upper))),
        "n_rows": int(len(frame)),
        "n_flows": int(len(source_values)),
        "n_seed_flow_cells": int(sum(len(value) for value in groups.values())),
        "n_unique_sources": int(len(source_values)),
        "n_reused_sources": int(
            sum(reuse_count > 1 for reuse_count in source_seed_reuse.values())
        ),
        "maximum_seed_reuse": int(max(source_seed_reuse.values())),
        "n_seeds": int(len(groups)),
        "n_probe_runs": run_count,
        "bootstrap_design": CROSSED_SOURCE_SEED_LIME_BOOTSTRAP,
        "run_resampling": "shared_run_pseudovalue_draw_per_sampled_seed_occurrence",
    }


# Compatibility for callers written before the crossed-cluster correction.
lime_u_statistic_equal_seed_bootstrap = (
    crossed_source_seed_lime_u_statistic_bootstrap
)


def _identification_coverage(
    frame: pd.DataFrame,
    metric: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the account coverage a similarity metric needs to identify features."""

    diagnostic_by_metric = {
        "spearman": "rank_pair_identifiable",
        "jaccard_10": "top_k_pair_identifiable",
        "weighted_jaccard": "account_pair_nonzero",
        "normalized_mass_l1": "account_pair_nonzero",
    }
    diagnostic = diagnostic_by_metric.get(str(metric))
    if diagnostic is None or diagnostic not in frame.columns:
        return {
            "identification_coverage_basis": "not_applicable",
            "identification_coverage_fraction": 1.0,
            "minimum_identifiable_fraction": float("nan"),
            "identification_coverage_sufficient": True,
            "identification_seed_inventory_complete": True,
            "minimum_seed_identification_fraction": 1.0,
            "null_account_pair_fraction": float("nan"),
        }
    flags = frame[diagnostic].map(_as_bool)
    threshold = float(
        config["representation"]["minimum_identifiable_fraction_for_gate"]
    )
    null_fraction = (
        float(frame["agreement_on_null_accounts"].map(_as_bool).mean())
        if "agreement_on_null_accounts" in frame.columns and len(frame)
        else 0.0
    )
    fraction = float(flags.mean()) if len(flags) else 0.0
    if "seed" not in frame.columns:
        raise ValueError("Identification coverage requires fitted-seed identities")
    seed_values = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    per_seed = flags.groupby(seed_values).mean()
    expected_seeds = set(int(value) for value in training.FINAL_SEEDS)
    observed_seeds = set(int(value) for value in per_seed.index)
    seed_inventory_complete = observed_seeds == expected_seeds
    minimum_seed_fraction = float(per_seed.min()) if len(per_seed) else 0.0
    sensitivity: dict[str, float] = {}
    if {"left_account_mass", "right_account_mass"}.issubset(frame.columns):
        minimum_pair_mass = np.minimum(
            pd.to_numeric(frame["left_account_mass"], errors="raise").to_numpy(float),
            pd.to_numeric(frame["right_account_mass"], errors="raise").to_numpy(float),
        )
        for sensitivity_threshold in config["representation"][
            "account_l1_sensitivity_thresholds"
        ]:
            label = f"account_pair_nonzero_fraction_l1_ge_{float(sensitivity_threshold):.0e}"
            sensitivity[label] = float(
                np.mean(minimum_pair_mass >= float(sensitivity_threshold))
            )
    return {
        "identification_coverage_basis": diagnostic,
        "identification_coverage_fraction": fraction,
        "minimum_identifiable_fraction": threshold,
        "identification_seed_inventory_complete": seed_inventory_complete,
        "minimum_seed_identification_fraction": minimum_seed_fraction,
        "identification_coverage_sufficient": bool(
            seed_inventory_complete and minimum_seed_fraction >= threshold
        ),
        "null_account_pair_fraction": null_fraction,
        **sensitivity,
    }


def summarize_lime_u_statistics(
    detail: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    uncertainty = config["uncertainty"]
    rows: list[dict[str, Any]] = []
    for keys, group in detail.groupby(list(group_columns), dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, key_values))
        for metric in metric_columns:
            result = crossed_source_seed_lime_u_statistic_bootstrap(
                group,
                metric,
                n_resamples=int(uncertainty["bootstrap_resamples"]),
                confidence_level=float(uncertainty["confidence_level"]),
                seed=stable_seed(
                    uncertainty["bootstrap_seed"], "lime_u_statistic", *key_values, metric
                ),
            )
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    **result,
                    **_identification_coverage(group, metric, config),
                    "minimum_seed_coverage": int(uncertainty["minimum_seed_coverage"]),
                    "coverage_sufficient": result["n_seeds"]
                    >= int(uncertainty["minimum_seed_coverage"]),
                    "run_uncertainty": "shared_whole_panel_run_jackknife_pseudovalues",
                }
            )
    return pd.DataFrame(rows)


def summarize_metrics(
    detail: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    config: Mapping[str, Any],
    run_column: str | None = None,
    analysis_id: str,
) -> pd.DataFrame:
    uncertainty = config["uncertainty"]
    rows: list[dict[str, Any]] = []
    for keys, group in detail.groupby(list(group_columns), dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, key_values))
        for metric in metric_columns:
            result = crossed_source_seed_equal_seed_bootstrap(
                group,
                metric,
                n_resamples=int(uncertainty["bootstrap_resamples"]),
                confidence_level=float(uncertainty["confidence_level"]),
                seed=stable_seed(uncertainty["bootstrap_seed"], analysis_id, *key_values, metric),
                run_column=run_column,
            )
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    **result,
                    **_identification_coverage(group, metric, config),
                    "minimum_seed_coverage": int(uncertainty["minimum_seed_coverage"]),
                    "coverage_sufficient": result["n_seeds"]
                    >= int(uncertainty["minimum_seed_coverage"]),
                }
            )
    return pd.DataFrame(rows)


def _align_probe_results(results: Sequence[ProbeResult]) -> tuple[pd.DataFrame, list[np.ndarray]]:
    if not results:
        raise ValueError("At least one probe result is required")
    base = results[0].index.reset_index(drop=True)
    flow_ids = base["flow_id"].astype(str).to_numpy()
    arrays = [results[0].attributions]
    for result in results[1:]:
        current = result.index.reset_index(drop=True)
        if not np.array_equal(flow_ids, current["flow_id"].astype(str).to_numpy()):
            raise RuntimeError("Probe results do not share the identical ordered flow panel")
        arrays.append(result.attributions)
    return base, arrays


def repeatability_detail_for_condition(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
    *,
    task_id: str,
    model: str,
    seed: int,
    method: str,
) -> pd.DataFrame:
    top_k = int(config["representation"]["top_k"])
    specs = probe_specs(config, method)
    result_by_spec = {
        spec.spec_id: load_probe_result(
            config,
            training_config,
            paths,
            task_id=task_id,
            model=model,
            seed=int(seed),
            spec=spec,
        )
        for spec in specs
    }
    comparisons: list[tuple[ProbeSpec, ProbeSpec, str]] = []
    if method == "lime":
        for left, right in itertools.combinations(specs, 2):
            comparisons.append((left, right, f"run_{left.run_id:02d}_vs_{right.run_id:02d}"))
    elif method == "integrated_gradients":
        main = next(spec for spec in specs if spec.role == "primary")
        reference = next(
            spec
            for spec in specs
            if spec.reference == main.reference and int(spec.steps or -1) == 128
        )
        for candidate in specs:
            if (
                candidate.reference == main.reference
                and candidate.steps != reference.steps
                and candidate.role in {"primary", "numerical_sensitivity"}
            ):
                comparisons.append(
                    (
                        candidate,
                        reference,
                        f"steps_{candidate.steps}_vs_{reference.steps}",
                    )
                )
    elif method == "occlusion":
        primary = next(spec for spec in specs if spec.role == "primary")
        replicate = next(spec for spec in specs if spec.role == "determinism_replicate")
        comparisons.append((primary, replicate, "fixed_rule_repeat"))
    elif method == "shap":
        primary = next(spec for spec in specs if spec.role == "primary")
        replicate = next(
            spec for spec in specs if spec.role == "stochastic_repeatability_replicate"
        )
        comparisons.append((primary, replicate, "stochastic_repeat"))
    else:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for left_spec, right_spec, comparison in comparisons:
        index, arrays = _align_probe_results(
            [result_by_spec[left_spec.spec_id], result_by_spec[right_spec.spec_id]]
        )
        central_rows = index["in_central"].map(_as_bool).to_numpy()
        for row_index in np.flatnonzero(central_rows):
            identity = index.iloc[int(row_index)]
            rows.append(
                {
                    **{column: identity[column] for column in IDENTITY_COLUMNS},
                    "model": model,
                    "method": method,
                    "comparison": comparison,
                    "left_run": int(left_spec.run_id),
                    "right_run": int(right_spec.run_id),
                    "probe_pair": f"{left_spec.spec_id}::{right_spec.spec_id}",
                    "class_specific_inference_eligible": _as_bool(
                        identity["class_specific_inference_eligible"]
                    ),
                    **attribution_similarity(
                        arrays[0][row_index],
                        arrays[1][row_index],
                        top_k=top_k,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _primary_attributions_for_condition(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
    *,
    task_id: str,
    model: str,
    seed: int,
    method: str,
) -> tuple[pd.DataFrame, np.ndarray, list[Path]]:
    specs = probe_specs(config, method)
    if method != "lime":
        spec = next(item for item in specs if item.role == "primary")
        result = load_probe_result(
            config,
            training_config,
            paths,
            task_id=task_id,
            model=model,
            seed=int(seed),
            spec=spec,
        )
        return result.index, result.attributions, [result.metadata_path]
    results = [
        load_probe_result(
            config,
            training_config,
            paths,
            task_id=task_id,
            model=model,
            seed=int(seed),
            spec=spec,
        )
        for spec in specs
    ]
    index, arrays = _align_probe_results(results)
    return index, np.mean(np.stack(arrays, axis=0), axis=0), [
        result.metadata_path for result in results
    ]


def cross_model_detail_for_condition(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
    *,
    task_id: str,
    seed: int,
    method: str,
) -> pd.DataFrame:
    comparisons: list[tuple[int, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]] = []
    if method in STOCHASTIC_SUBSET_PROBES:
        repeated_run_specs = [
            spec
            for spec in probe_specs(config, method)
            if spec.role in {"primary", "independent_run", "stochastic_repeatability_replicate"}
        ]
        for spec in repeated_run_specs:
            left = load_probe_result(
                config,
                training_config,
                paths,
                task_id=task_id,
                model=ALLOWED_MODELS[0],
                seed=int(seed),
                spec=spec,
            )
            right = load_probe_result(
                config,
                training_config,
                paths,
                task_id=task_id,
                model=ALLOWED_MODELS[1],
                seed=int(seed),
                spec=spec,
            )
            comparisons.append(
                (spec.run_id, left.index, left.attributions, right.index, right.attributions)
            )
    else:
        left_index, left_attrs, _ = _primary_attributions_for_condition(
            config,
            training_config,
            paths,
            task_id=task_id,
            model=ALLOWED_MODELS[0],
            seed=int(seed),
            method=method,
        )
        right_index, right_attrs, _ = _primary_attributions_for_condition(
            config,
            training_config,
            paths,
            task_id=task_id,
            model=ALLOWED_MODELS[1],
            seed=int(seed),
            method=method,
        )
        comparisons.append((0, left_index, left_attrs, right_index, right_attrs))
    central = pd.read_csv(paths.central_panel)
    central = central[
        central["task_id"].astype(str).eq(task_id)
        & pd.to_numeric(central["seed"], errors="raise").astype(int).eq(int(seed))
    ].copy()
    if method in STOCHASTIC_SUBSET_PROBES:
        flags = central["in_stochastic_subset"].astype(str).str.lower().isin({"true", "1"})
        central = central[flags].copy()
    top_k = int(config["representation"]["top_k"])
    rows: list[dict[str, Any]] = []
    for probe_run, left_index, left_attrs, right_index, right_attrs in comparisons:
        left_lookup = {
            str(flow): index for index, flow in enumerate(left_index["flow_id"])
        }
        right_lookup = {
            str(flow): index for index, flow in enumerate(right_index["flow_id"])
        }
        missing = [
            flow
            for flow in central["flow_id"].astype(str)
            if flow not in left_lookup or flow not in right_lookup
        ]
        if missing:
            raise RuntimeError(
                f"Primary paired feature accounts are missing {len(missing)} flows"
            )
        for _, identity in central.iterrows():
            flow = str(identity["flow_id"])
            rows.append(
                {
                    **{column: identity[column] for column in IDENTITY_COLUMNS},
                    "method": method,
                    "model_left": ALLOWED_MODELS[0],
                    "model_right": ALLOWED_MODELS[1],
                    "probe_run": int(probe_run),
                    "class_specific_inference_eligible": _as_bool(
                        identity["class_specific_inference_eligible"]
                    ),
                    **attribution_similarity(
                        left_attrs[left_lookup[flow]],
                        right_attrs[right_lookup[flow]],
                        top_k=top_k,
                    ),
                }
            )
    return pd.DataFrame(rows)


def reference_sensitivity_detail_for_condition(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
    *,
    task_id: str,
    model: str,
    seed: int,
    method: str,
) -> pd.DataFrame:
    declared = tuple(config["reference_sensitivity"]["methods"])
    if method not in declared:
        raise ValueError(f"Reference sensitivity is not registered for {method!r}")
    specs = probe_specs(config, method)
    primary = next(spec for spec in specs if spec.role == "primary")
    primary_result = load_probe_result(
        config,
        training_config,
        paths,
        task_id=task_id,
        model=model,
        seed=int(seed),
        spec=primary,
    )
    top_k = int(config["representation"]["top_k"])
    rows: list[dict[str, Any]] = []
    for variant in [spec for spec in specs if spec.role == "reference_sensitivity"]:
        variant_result = load_probe_result(
            config,
            training_config,
            paths,
            task_id=task_id,
            model=model,
            seed=int(seed),
            spec=variant,
        )
        index, arrays = _align_probe_results([primary_result, variant_result])
        central_rows = index["in_central"].map(_as_bool).to_numpy()
        for row_index in np.flatnonzero(central_rows):
            identity = index.iloc[int(row_index)]
            rows.append(
                {
                    **{column: identity[column] for column in IDENTITY_COLUMNS},
                    "model": model,
                    "method": method,
                    "reference_primary": primary.reference,
                    "reference_variant": variant.reference,
                    "comparison": f"{variant.reference}_vs_{primary.reference}",
                    "class_specific_inference_eligible": _as_bool(
                        identity["class_specific_inference_eligible"]
                    ),
                    **attribution_similarity(
                        arrays[1][row_index],
                        arrays[0][row_index],
                        top_k=top_k,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _representative_stochastic_spec(config: Mapping[str, Any], method: str) -> ProbeSpec:
    """Pick one spec to stand in for a stochastic method in a pairwise comparison.

    SHAP registers a "primary" role; LIME does not — every independent run
    is equally arbitrary, so its first run (run_id 0) is used instead.
    """

    specs = probe_specs(config, method)
    primary = next((spec for spec in specs if spec.role == "primary"), None)
    if primary is not None:
        return primary
    return next(spec for spec in specs if spec.run_id == 0)


def cross_probe_detail_for_condition(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
    *,
    task_id: str,
    model: str,
    seed: int,
) -> pd.DataFrame:
    """Pair probes without choosing or ranking a preferred measurement."""

    triangulation = config["cross_probe_triangulation"]
    deterministic_methods = tuple(triangulation["deterministic_central_probes"])
    all_methods = tuple(triangulation["all_probe_stochastic_subset"])
    stochastic_methods = tuple(method for method in all_methods if method in STOCHASTIC_SUBSET_PROBES)
    deterministic: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for method in deterministic_methods:
        index, attributes, _ = _primary_attributions_for_condition(
            config,
            training_config,
            paths,
            task_id=task_id,
            model=model,
            seed=int(seed),
            method=method,
        )
        deterministic[method] = (index, attributes)
    stochastic_results_by_method: dict[str, list[ProbeResult]] = {
        method: [
            load_probe_result(
                config,
                training_config,
                paths,
                task_id=task_id,
                model=model,
                seed=int(seed),
                spec=spec,
            )
            for spec in probe_specs(config, method)
        ]
        for method in stochastic_methods
    }
    central = pd.read_csv(paths.central_panel)
    central = central[
        central["task_id"].astype(str).eq(task_id)
        & pd.to_numeric(central["seed"], errors="raise").astype(int).eq(int(seed))
    ].copy()
    top_k = int(config["representation"]["top_k"])
    rows: list[dict[str, Any]] = []

    def append_comparison(
        panel: pd.DataFrame,
        panel_scope: str,
        left_method: str,
        right_method: str,
        left_index: pd.DataFrame,
        left_attrs: np.ndarray,
        right_index: pd.DataFrame,
        right_attrs: np.ndarray,
        probe_run: int,
    ) -> None:
        left_lookup = {
            str(flow): position for position, flow in enumerate(left_index["flow_id"])
        }
        right_lookup = {
            str(flow): position for position, flow in enumerate(right_index["flow_id"])
        }
        for _, identity in panel.iterrows():
            flow_id = str(identity["flow_id"])
            if flow_id not in left_lookup or flow_id not in right_lookup:
                raise RuntimeError("Cross-probe comparison lacks a registered panel flow")
            rows.append(
                {
                    **{column: identity[column] for column in IDENTITY_COLUMNS},
                    "model": model,
                    "panel_scope": panel_scope,
                    "probe_left": left_method,
                    "probe_right": right_method,
                    "probe_pair": f"{left_method}::{right_method}",
                    "probe_run": int(probe_run),
                    "class_specific_inference_eligible": _as_bool(
                        identity["class_specific_inference_eligible"]
                    ),
                    **attribution_similarity(
                        left_attrs[left_lookup[flow_id]],
                        right_attrs[right_lookup[flow_id]],
                        top_k=top_k,
                    ),
                }
            )

    for left_method, right_method in itertools.combinations(deterministic_methods, 2):
        left_index, left_attrs = deterministic[left_method]
        right_index, right_attrs = deterministic[right_method]
        append_comparison(
            central,
            "deterministic_central",
            left_method,
            right_method,
            left_index,
            left_attrs,
            right_index,
            right_attrs,
            0,
        )

    stochastic_flags = central["in_stochastic_subset"].map(_as_bool)
    stochastic_panel = central[stochastic_flags].copy()
    for left_method, right_method in itertools.combinations(all_methods, 2):
        left_stochastic = left_method in stochastic_methods
        right_stochastic = right_method in stochastic_methods
        if not left_stochastic and not right_stochastic:
            left_index, left_attrs = deterministic[left_method]
            right_index, right_attrs = deterministic[right_method]
            append_comparison(
                stochastic_panel,
                "all_probe_stochastic_subset",
                left_method,
                right_method,
                left_index,
                left_attrs,
                right_index,
                right_attrs,
                0,
            )
        elif left_stochastic and right_stochastic:
            # Two stochastic methods: pair only one representative run each
            # (not every run-pair combination) to keep the pairwise cost
            # bounded — per-method run-to-run stability is already covered
            # by repeatability_detail_for_condition. LIME's specs have no
            # "primary" role (every independent run is equally arbitrary),
            # so its first run (run_id 0) stands in for it; SHAP does
            # register a "primary" role and uses that.
            left_primary = _representative_stochastic_spec(config, left_method)
            right_primary = _representative_stochastic_spec(config, right_method)
            left_result = stochastic_results_by_method[left_method][
                probe_specs(config, left_method).index(left_primary)
            ]
            right_result = stochastic_results_by_method[right_method][
                probe_specs(config, right_method).index(right_primary)
            ]
            append_comparison(
                stochastic_panel,
                "all_probe_stochastic_subset",
                left_method,
                right_method,
                left_result.index,
                left_result.attributions,
                right_result.index,
                right_result.attributions,
                0,
            )
        else:
            stochastic_method = left_method if left_stochastic else right_method
            deterministic_method = right_method if left_stochastic else left_method
            deterministic_index, deterministic_attrs = deterministic[deterministic_method]
            for spec, stochastic_result in zip(
                probe_specs(config, stochastic_method),
                stochastic_results_by_method[stochastic_method],
            ):
                if left_stochastic:
                    comparison = (
                        stochastic_result.index,
                        stochastic_result.attributions,
                        deterministic_index,
                        deterministic_attrs,
                    )
                else:
                    comparison = (
                        deterministic_index,
                        deterministic_attrs,
                        stochastic_result.index,
                        stochastic_result.attributions,
                    )
                append_comparison(
                    stochastic_panel,
                    "all_probe_stochastic_subset",
                    left_method,
                    right_method,
                    *comparison,
                    int(spec.run_id),
                )
    return pd.DataFrame(rows)


def _parameter_sanity_directory(
    paths: AuditPaths,
    task_id: str,
    model: str,
    seed: int,
    method: str,
) -> Path:
    return (
        paths.audit_root
        / "parameter_dependence_sanity"
        / task_id
        / model
        / f"seed_{seed}"
        / method
    )


def run_parameter_sanity_condition(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
    *,
    task_id: str,
    model: str,
    seed: int,
    method: str,
    force: bool = False,
    compute_if_missing: bool = True,
) -> pd.DataFrame:
    """Check whether a feature account changes after destroying learned weights."""

    sanity_cfg = config["parameter_dependence_sanity"]
    if method not in tuple(sanity_cfg["probes"]):
        raise ValueError(f"Parameter sanity is not registered for {method!r}")
    spec = next(value for value in probe_specs(config, method) if value.role == "primary")
    trained = load_probe_result(
        config,
        training_config,
        paths,
        task_id=task_id,
        model=model,
        seed=int(seed),
        spec=spec,
    )
    subset_flags = trained.index["in_central"].map(_as_bool) & trained.index[
        "in_stochastic_subset"
    ].map(_as_bool)
    units = trained.index[subset_flags].reset_index(drop=True)
    trained_attributes = trained.attributions[subset_flags.to_numpy()]
    if units.empty:
        raise RuntimeError("Parameter sanity subset is empty")
    condition = require_valid_training_condition(
        paths,
        training_config,
        task_id=task_id,
        model=model,
        seed=int(seed),
    )
    directory = _parameter_sanity_directory(paths, task_id, model, int(seed), method)
    outputs = {
        "array": directory / "reinitialized_attributions.npz",
        "detail": directory / "trained_vs_reinitialized_per_flow.csv",
        "decision": directory / "sanity_decision.json",
        "metadata": directory / "cache_metadata.json",
    }
    input_paths = [
        paths.config_path,
        paths.training_config_path,
        paths.audit_units,
        condition["model"],
        condition["selection_lock"],
        condition["metadata"],
        trained.metadata_path,
        *_processed_probe_inputs(paths.processed_root / task_id),
    ]
    inputs = artifact_records(input_paths)
    implementation_hash = signature({"files": _audit_implementation_records(paths)})
    # Both probes for a fitted condition must see the same random parameter
    # state.  Including ``method`` here would silently mix probe sensitivity
    # with a different reinitialised network in each paired sanity check.
    reinitialization_seed = stable_seed(
        sanity_cfg["reinitialization_seed"], task_id, model, int(seed)
    )
    stage_signature = signature(
        {
            "audit_version": AUDIT_VERSION,
            "stage": "parameter_dependence_sanity",
            "config_hash": sha256_file(paths.config_path),
            "training_config_hash": sha256_file(paths.training_config_path),
            "implementation_hash": implementation_hash,
            "inputs": inputs,
            "task_id": task_id,
            "model": model,
            "seed": int(seed),
            "method": method,
            "reinitialization_seed": reinitialization_seed,
        }
    )
    expected = [outputs["array"], outputs["detail"], outputs["decision"]]
    hit, _ = cache_hit(
        outputs["metadata"],
        expected_signature=stage_signature,
        expected_outputs=expected,
    )
    if hit and not force:
        return pd.read_csv(outputs["detail"])
    if not compute_if_missing:
        raise RuntimeError(
            f"Parameter-sanity cache is absent or stale for "
            f"{task_id}/{model}/seed_{seed}/{method}"
        )
    from src.utils.seed import set_global_seed

    set_global_seed(reinitialization_seed)
    lock = json.loads(condition["selection_lock"].read_text(encoding="utf-8"))
    data = load_condition_data(paths, task_id)
    reinitialized_model = training._MODEL_BUILDERS[model](
        len(data.feature_names),
        len(data.class_names),
        training._model_params(dict(lock["selected_candidate"])),
    )
    started = time.perf_counter()
    reinitialized_attributes, _ = _compute_probe(
        config,
        data,
        reinitialized_model,
        units,
        spec,
    )
    rows: list[dict[str, Any]] = []
    top_k = int(config["representation"]["top_k"])
    for row_index, identity in units.iterrows():
        rows.append(
            {
                **{column: identity[column] for column in IDENTITY_COLUMNS},
                "model": model,
                "method": method,
                "comparison": "trained_vs_same_architecture_reinitialized",
                "reinitialization_seed": reinitialization_seed,
                "class_specific_inference_eligible": _as_bool(
                    identity["class_specific_inference_eligible"]
                ),
                **attribution_similarity(
                    trained_attributes[row_index],
                    reinitialized_attributes[row_index],
                    top_k=top_k,
                ),
            }
        )
    detail = pd.DataFrame(rows)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outputs["array"],
        attributions=reinitialized_attributes.astype(np.float32),
        flow_id=np.asarray(units["flow_id"].astype(str), dtype="S64"),
    )
    detail.to_csv(outputs["detail"], index=False)
    decision = {
        "audit_version": AUDIT_VERSION,
        "status": "complete",
        "analysis_role": "necessary_not_sufficient",
        "task_id": task_id,
        "model": model,
        "seed": int(seed),
        "method": method,
        "comparison": sanity_cfg["comparison"],
        "reinitialization_seed": reinitialization_seed,
        "runtime_seconds": float(time.perf_counter() - started),
        "stage_signature": stage_signature,
    }
    outputs["decision"].write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_cache_metadata(
        outputs["metadata"],
        stage="parameter_dependence_sanity",
        signature_value=stage_signature,
        config_hash=sha256_file(paths.config_path),
        implementation_hash=implementation_hash,
        inputs=inputs,
        outputs=expected,
        extra=decision,
    )
    return detail


def _selected_feature_matrix(attributes: np.ndarray, k: int) -> np.ndarray:
    absolute = np.abs(np.asarray(attributes, dtype=float))
    order = np.argsort(-absolute, axis=1, kind="stable")
    return order[:, : min(max(int(k), 1), absolute.shape[1])]


def _replace_selected(
    X: np.ndarray,
    replacements: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    perturbed = np.asarray(X, dtype=np.float32).copy()
    rows = np.arange(len(perturbed))[:, None]
    perturbed[rows, selected] = np.asarray(replacements, dtype=np.float32)[rows, selected]
    return perturbed


def _insert_selected(
    X: np.ndarray,
    replacements: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    inserted = np.asarray(replacements, dtype=np.float32).copy()
    rows = np.arange(len(inserted))[:, None]
    inserted[rows, selected] = np.asarray(X, dtype=np.float32)[rows, selected]
    return inserted


def _random_feature_matrix(
    rows: int,
    features: int,
    k: int,
    *,
    seed: int,
    flow_ids: Sequence[str] | None = None,
) -> np.ndarray:
    if flow_ids is None:
        rng = np.random.default_rng(int(seed))
        random_keys = rng.random((int(rows), int(features)))
    else:
        if len(flow_ids) != int(rows):
            raise ValueError("Random-control flow IDs must align with input rows")
        random_keys = np.vstack(
            [
                np.random.default_rng(stable_seed(seed, flow_id)).random(int(features))
                for flow_id in flow_ids
            ]
        )
    return np.argsort(random_keys, axis=1, kind="stable")[:, : min(k, features)]


def _displacement_bin_matched_feature_matrix(
    displacements: np.ndarray,
    selected: np.ndarray,
    *,
    bins: int,
    seed: int,
    flow_ids: Sequence[str],
) -> np.ndarray:
    """Sample equal coordinate counts from each per-flow displacement-rank bin."""

    values = np.asarray(displacements, dtype=float)
    chosen = np.asarray(selected, dtype=int)
    if values.ndim != 2 or chosen.ndim != 2 or len(values) != len(chosen):
        raise ValueError("Displacements and selected coordinates must align by flow")
    if int(bins) <= 0 or len(flow_ids) != len(values):
        raise ValueError("Displacement bins and flow identities must be valid")
    result = np.empty_like(chosen)
    feature_count = values.shape[1]
    for row_index, flow_id in enumerate(flow_ids):
        # Stable ranks avoid data-dependent bin edges with large tied masses.
        order = np.argsort(values[row_index], kind="stable")
        bin_id = np.empty(feature_count, dtype=int)
        bin_id[order] = np.minimum(
            int(bins) - 1,
            np.floor(np.arange(feature_count) * int(bins) / feature_count).astype(int),
        )
        selected_bins = bin_id[chosen[row_index]]
        rng = np.random.default_rng(stable_seed(seed, flow_id))
        output: list[int] = []
        for current_bin in range(int(bins)):
            count = int(np.sum(selected_bins == current_bin))
            if count == 0:
                continue
            candidates = np.flatnonzero(bin_id == current_bin)
            if count > len(candidates):
                raise RuntimeError("A displacement bin cannot supply its matched count")
            output.extend(
                rng.choice(candidates, size=count, replace=False).astype(int).tolist()
            )
        if len(output) != chosen.shape[1]:
            raise RuntimeError("Displacement-matched control changed the feature count")
        result[row_index] = np.asarray(output, dtype=int)
    return result


def functional_checks_for_arrays(
    model: Any,
    X: np.ndarray,
    targets: np.ndarray,
    attributes: np.ndarray,
    identity: pd.DataFrame,
    data: ConditionData,
    config: Mapping[str, Any],
    *,
    model_name: str,
    method: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run declared coordinate operators and matched random controls per flow."""

    X_array = np.asarray(X, dtype=np.float32)
    target = np.asarray(targets, dtype=int)
    attrs = np.asarray(attributes, dtype=np.float32)
    if X_array.shape != attrs.shape or len(identity) != len(X_array):
        raise ValueError("Functional-check arrays and identities must have equal geometry")
    functional_cfg = config["functional_checks"]
    deletion_cfg = functional_cfg["deletion_and_insertion"]
    fractions = [float(value) for value in deletion_cfg["feature_fractions"]]
    repetitions = int(deletion_cfg["random_feature_repetitions"])
    clean_probability = target_probabilities(model, X_array, target)
    true_classes = pd.to_numeric(identity["true_class_id"], errors="raise").to_numpy(
        dtype=int
    )
    curve_frames: list[pd.DataFrame] = []
    for evaluator in deletion_cfg["evaluators"]:
        reference_classes = target if str(evaluator) == "target_class_median" else true_classes
        replacements = reference_matrix(data, reference_classes, str(evaluator))
        reference_probability = target_probabilities(model, replacements, target)
        for fraction in fractions:
            k = min(X_array.shape[1], max(1, int(math.ceil(fraction * X_array.shape[1]))))
            selected = _selected_feature_matrix(attrs, k)
            deleted = _replace_selected(X_array, replacements, selected)
            inserted = _insert_selected(X_array, replacements, selected)
            deletion_effect = clean_probability - target_probabilities(model, deleted, target)
            insertion_effect = target_probabilities(model, inserted, target) - reference_probability
            displacement = np.abs(X_array - replacements)
            row_indices = np.arange(len(X_array))[:, None]
            attribution_displacement = displacement[row_indices, selected].mean(axis=1)
            matched_deletion = np.empty((repetitions, len(X_array)), dtype=float)
            matched_insertion = np.empty((repetitions, len(X_array)), dtype=float)
            uniform_deletion = np.empty((repetitions, len(X_array)), dtype=float)
            uniform_insertion = np.empty((repetitions, len(X_array)), dtype=float)
            matched_displacement = np.empty((repetitions, len(X_array)), dtype=float)
            uniform_displacement = np.empty((repetitions, len(X_array)), dtype=float)
            flow_ids = identity["flow_id"].astype(str).tolist()
            for repetition in range(repetitions):
                control_seed = stable_seed(
                    config["panel"]["sampling_seed"],
                    "random_functional",
                    data.task_id,
                    model_name,
                    seed,
                    evaluator,
                    fraction,
                    repetition,
                )
                matched_selected = _displacement_bin_matched_feature_matrix(
                    displacement,
                    selected,
                    bins=int(deletion_cfg["displacement_bins"]),
                    seed=control_seed,
                    flow_ids=flow_ids,
                )
                uniform_selected = _random_feature_matrix(
                    len(X_array),
                    X_array.shape[1],
                    k,
                    seed=stable_seed(control_seed, "uniform_sensitivity"),
                    flow_ids=flow_ids,
                )
                matched_deleted = _replace_selected(X_array, replacements, matched_selected)
                matched_inserted = _insert_selected(X_array, replacements, matched_selected)
                uniform_deleted = _replace_selected(X_array, replacements, uniform_selected)
                uniform_inserted = _insert_selected(X_array, replacements, uniform_selected)
                matched_deletion[repetition] = clean_probability - target_probabilities(
                    model, matched_deleted, target
                )
                matched_insertion[repetition] = target_probabilities(
                    model, matched_inserted, target
                ) - reference_probability
                uniform_deletion[repetition] = clean_probability - target_probabilities(
                    model, uniform_deleted, target
                )
                uniform_insertion[repetition] = target_probabilities(
                    model, uniform_inserted, target
                ) - reference_probability
                matched_displacement[repetition] = displacement[
                    row_indices, matched_selected
                ].mean(axis=1)
                uniform_displacement[repetition] = displacement[
                    row_indices, uniform_selected
                ].mean(axis=1)
            for check, attribution_effect, matched_effect, uniform_effect in (
                ("deletion", deletion_effect, matched_deletion, uniform_deletion),
                ("insertion", insertion_effect, matched_insertion, uniform_insertion),
            ):
                frame = identity[list(IDENTITY_COLUMNS) + [
                    "in_central",
                    "in_outcome_correct",
                    "in_outcome_error",
                    "class_specific_inference_eligible",
                    "target_class_id",
                    "target_class",
                    "target_basis",
                    "confidence",
                    "model_margin",
                ]].copy()
                frame["model"] = model_name
                frame["method"] = method
                frame["functional_check"] = check
                frame["evaluator"] = str(evaluator)
                frame["feature_fraction"] = fraction
                frame["features_changed"] = k
                frame["attribution_effect"] = attribution_effect
                frame["random_control_primary"] = (
                    "per_flow_per_fraction_displacement_bin_matched_subset"
                )
                frame["random_effect_mean"] = matched_effect.mean(axis=0)
                frame["random_effect_std"] = matched_effect.std(
                    axis=0, ddof=1 if repetitions > 1 else 0
                )
                frame["random_effect_standard_error"] = (
                    frame["random_effect_std"] / math.sqrt(repetitions)
                )
                frame["uniform_random_effect_mean"] = uniform_effect.mean(axis=0)
                frame["uniform_random_effect_std"] = uniform_effect.std(
                    axis=0, ddof=1 if repetitions > 1 else 0
                )
                frame["uniform_random_effect_standard_error"] = (
                    frame["uniform_random_effect_std"] / math.sqrt(repetitions)
                )
                frame["attribution_minus_random"] = (
                    frame["attribution_effect"] - frame["random_effect_mean"]
                )
                frame["attribution_minus_uniform_random"] = (
                    frame["attribution_effect"] - frame["uniform_random_effect_mean"]
                )
                frame["attribution_displacement_mean"] = attribution_displacement
                frame["matched_random_displacement_mean"] = matched_displacement.mean(axis=0)
                frame["uniform_random_displacement_mean"] = uniform_displacement.mean(axis=0)
                frame["attribution_minus_matched_displacement"] = (
                    frame["attribution_displacement_mean"]
                    - frame["matched_random_displacement_mean"]
                )
                curve_frames.append(frame)
    curves = pd.concat(curve_frames, ignore_index=True)
    aopc_columns = [
        *IDENTITY_COLUMNS,
        "model",
        "method",
        "functional_check",
        "evaluator",
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
    expected_fractions = np.asarray(sorted(set(fractions)), dtype=float)

    def normalized_curve_area(group: pd.DataFrame, value_column: str) -> float:
        ordered = group.sort_values("feature_fraction", kind="mergesort")
        observed = ordered["feature_fraction"].to_numpy(dtype=float)
        if not np.array_equal(observed, expected_fractions):
            raise RuntimeError("Functional curve does not contain every registered fraction")
        x = np.concatenate([[0.0], observed])
        y = np.concatenate([[0.0], ordered[value_column].to_numpy(dtype=float)])
        return float(np.trapezoid(y, x) / expected_fractions[-1])

    def normalized_curve_standard_error(
        group: pd.DataFrame, standard_error_column: str
    ) -> float:
        """Propagate independent per-fraction Monte Carlo means through AOPC."""

        ordered = group.sort_values("feature_fraction", kind="mergesort")
        observed = ordered["feature_fraction"].to_numpy(dtype=float)
        if not np.array_equal(observed, expected_fractions):
            raise RuntimeError("Functional curve does not contain every registered fraction")
        weights = np.empty(len(observed), dtype=float)
        for index, fraction in enumerate(observed):
            previous = 0.0 if index == 0 else observed[index - 1]
            left_weight = 0.5 * (fraction - previous)
            right_weight = (
                0.5 * (observed[index + 1] - fraction)
                if index + 1 < len(observed)
                else 0.0
            )
            weights[index] = (left_weight + right_weight) / observed[-1]
        standard_errors = ordered[standard_error_column].to_numpy(dtype=float)
        return float(np.sqrt(np.sum(np.square(weights * standard_errors))))

    aopc_rows: list[dict[str, Any]] = []
    for group_keys, group in curves.groupby(aopc_columns, dropna=False, sort=False):
        row = dict(zip(aopc_columns, group_keys))
        random_aopc_mc_se = normalized_curve_standard_error(
            group, "random_effect_standard_error"
        )
        row.update(
            {
                "attribution_aopc": normalized_curve_area(group, "attribution_effect"),
                "random_aopc": normalized_curve_area(group, "random_effect_mean"),
                "attribution_minus_random_aopc": normalized_curve_area(
                    group, "attribution_minus_random"
                ),
                "random_aopc_monte_carlo_standard_error": random_aopc_mc_se,
                "attribution_minus_random_aopc_mc_lower95": (
                    normalized_curve_area(group, "attribution_minus_random")
                    - 1.96 * random_aopc_mc_se
                ),
                "uniform_random_aopc": normalized_curve_area(
                    group, "uniform_random_effect_mean"
                ),
                "attribution_minus_uniform_random_aopc": normalized_curve_area(
                    group, "attribution_minus_uniform_random"
                ),
                "attribution_displacement_mean": float(
                    group["attribution_displacement_mean"].mean()
                ),
                "matched_random_displacement_mean": float(
                    group["matched_random_displacement_mean"].mean()
                ),
                "attribution_minus_matched_displacement": float(
                    group["attribution_minus_matched_displacement"].mean()
                ),
                "fractions_evaluated": int(group["feature_fraction"].nunique()),
                "aopc_definition": "trapezoid_from_zero_normalized_by_max_fraction",
            }
        )
        aopc_rows.append(row)
    aopc = pd.DataFrame(aopc_rows)

    signed_cfg = functional_cfg["signed_intervention"]
    top_each_direction = int(signed_cfg["top_features_each_direction"])
    signed_repetitions = int(signed_cfg["random_feature_repetitions"])
    signed_frames: list[pd.DataFrame] = []
    operator = str(signed_cfg["method_aware_operators"][method])
    contribution_operator = method != "lime"
    if method == "gradient_x_input":
        contribution_reference = reference_matrix(data, true_classes, "zero_scaled")
    elif method in {"integrated_gradients", "occlusion", "shap"}:
        contribution_reference = reference_matrix(data, true_classes, PRIMARY_REFERENCE)
    else:
        contribution_reference = np.empty_like(X_array)
    for direction in ("positive", "negative"):
        eligible = attrs > 0 if direction == "positive" else attrs < 0
        signed_magnitude = np.where(eligible, np.abs(attrs), -np.inf)
        ordered = np.argsort(-signed_magnitude, axis=1, kind="stable")
        selected_count = np.minimum(eligible.sum(axis=1), top_each_direction).astype(int)
        selected_lists = [
            ordered[row, : selected_count[row]] for row in range(len(X_array))
        ]
        for strength in [float(value) for value in signed_cfg["strengths"]]:
            intervened = X_array.copy()
            changed_counts = np.zeros(len(X_array), dtype=int)
            for row_index, selected_features in enumerate(selected_lists):
                if not len(selected_features):
                    continue
                old = intervened[row_index, selected_features].copy()
                if contribution_operator:
                    destination = contribution_reference[row_index, selected_features]
                elif direction == "positive":
                    destination = data.feature_max[selected_features]
                else:
                    destination = data.feature_min[selected_features]
                intervened[row_index, selected_features] = old + strength * (destination - old)
                changed_counts[row_index] = int(
                    np.sum(np.abs(intervened[row_index, selected_features] - old) > 1e-12)
                )
            within_range = (
                (intervened >= data.feature_min[None, :] - 1e-6)
                & (intervened <= data.feature_max[None, :] + 1e-6)
            ).all(axis=1)
            changed_probability = target_probabilities(model, intervened, target)
            if contribution_operator and direction == "positive":
                aligned_effect = clean_probability - changed_probability
            elif contribution_operator and direction == "negative":
                aligned_effect = changed_probability - clean_probability
            else:
                # A local-surrogate coefficient describes the direction in
                # which the explained target increases.  Positive features are
                # moved up and negative features down, so both directions use
                # changed-minus-clean probability as the aligned response.
                aligned_effect = changed_probability - clean_probability
            random_effects = np.empty((signed_repetitions, len(X_array)), dtype=float)
            for repetition in range(signed_repetitions):
                random_intervened = X_array.copy()
                for row_index, count in enumerate(selected_count):
                    if count <= 0:
                        continue
                    rng = np.random.default_rng(
                        stable_seed(
                            config["panel"]["sampling_seed"],
                            "signed_random",
                            data.task_id,
                            model_name,
                            seed,
                            direction,
                            strength,
                            repetition,
                            identity.iloc[row_index]["flow_id"],
                        )
                    )
                    chosen = rng.choice(X_array.shape[1], size=int(count), replace=False)
                    old = random_intervened[row_index, chosen].copy()
                    if contribution_operator:
                        destination = contribution_reference[row_index, chosen]
                    elif direction == "positive":
                        destination = data.feature_max[chosen]
                    else:
                        destination = data.feature_min[chosen]
                    random_intervened[row_index, chosen] = old + strength * (
                        destination - old
                    )
                random_probability = target_probabilities(model, random_intervened, target)
                if contribution_operator and direction == "positive":
                    random_effects[repetition] = clean_probability - random_probability
                elif contribution_operator and direction == "negative":
                    random_effects[repetition] = random_probability - clean_probability
                else:
                    random_effects[repetition] = random_probability - clean_probability
            frame = identity[list(IDENTITY_COLUMNS) + [
                "in_central",
                "in_outcome_correct",
                "in_outcome_error",
                "class_specific_inference_eligible",
                "target_class_id",
                "target_class",
                "target_basis",
                "confidence",
                "model_margin",
            ]].copy()
            frame["model"] = model_name
            frame["method"] = method
            frame["analysis_role"] = "secondary_descriptive"
            frame["operator"] = operator
            frame["direction"] = direction
            frame["strength"] = strength
            frame["features_selected"] = selected_count
            frame["features_changed"] = changed_counts
            frame["signed_intervention_eligible"] = changed_counts > 0
            frame["within_training_min_max"] = within_range
            frame["aligned_probability_effect"] = aligned_effect
            frame["random_aligned_effect_mean"] = random_effects.mean(axis=0)
            frame["aligned_effect_minus_random"] = (
                frame["aligned_probability_effect"] - frame["random_aligned_effect_mean"]
            )
            frame["operator_consistent"] = np.where(
                changed_counts > 0,
                frame["aligned_probability_effect"].ge(0.0),
                np.nan,
            )
            signed_frames.append(frame)
    signed = pd.concat(signed_frames, ignore_index=True)
    return curves, aopc, signed


def _functional_directory(
    paths: AuditPaths,
    task_id: str,
    model: str,
    seed: int,
    method: str,
) -> Path:
    return paths.audit_root / "functional" / task_id / model / f"seed_{seed}" / method


def run_functional_condition(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
    *,
    task_id: str,
    model: str,
    seed: int,
    method: str,
    force: bool = False,
    compute_if_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    identity, attributes, probe_metadata = _primary_attributions_for_condition(
        config,
        training_config,
        paths,
        task_id=task_id,
        model=model,
        seed=int(seed),
        method=method,
    )
    if config["functional_checks"].get("panel_scope") == "central_stochastic_subset":
        functional_flags = identity["in_stochastic_subset"].map(_as_bool).to_numpy()
        identity = identity.loc[functional_flags].reset_index(drop=True)
        attributes = np.asarray(attributes)[functional_flags]
    condition = require_valid_training_condition(
        paths,
        training_config,
        task_id=task_id,
        model=model,
        seed=int(seed),
    )
    data = load_condition_data(paths, task_id)
    orders = _verify_units_against_test(identity, data)
    input_paths = [
        paths.config_path,
        paths.training_config_path,
        paths.audit_units,
        condition["model"],
        condition["metadata"],
        *probe_metadata,
        *_processed_probe_inputs(paths.processed_root / task_id),
    ]
    inputs = artifact_records(input_paths)
    implementation_hash = signature({"files": _audit_implementation_records(paths)})
    stage_signature = signature(
        {
            "audit_version": AUDIT_VERSION,
            "stage": "functional_checks",
            "config_hash": sha256_file(paths.config_path),
            "training_config_hash": sha256_file(paths.training_config_path),
            "implementation_hash": implementation_hash,
            "inputs": inputs,
            "task_id": task_id,
            "model": model,
            "seed": int(seed),
            "method": method,
            "target": "declared_target_class",
        }
    )
    directory = _functional_directory(paths, task_id, model, int(seed), method)
    output_paths = {
        "curves": directory / "deletion_insertion_per_flow.csv",
        "aopc": directory / "functional_aopc_per_flow.csv",
        "signed": directory / "signed_intervention_per_flow.csv",
        "decision": directory / "functional_decision.json",
        "metadata": directory / "cache_metadata.json",
    }
    expected = [
        output_paths["curves"],
        output_paths["aopc"],
        output_paths["signed"],
        output_paths["decision"],
    ]
    hit, _ = cache_hit(
        output_paths["metadata"],
        expected_signature=stage_signature,
        expected_outputs=expected,
    )
    if hit and not force:
        return {
            "curves": pd.read_csv(output_paths["curves"]),
            "aopc": pd.read_csv(output_paths["aopc"]),
            "signed": pd.read_csv(output_paths["signed"]),
        }
    if not compute_if_missing:
        raise RuntimeError(
            f"Functional cache is absent or stale for "
            f"{task_id}/{model}/seed_{seed}/{method}"
        )
    frozen_model = _load_frozen_model(condition["model"])
    started = time.perf_counter()
    curves, aopc, signed = functional_checks_for_arrays(
        frozen_model,
        np.asarray(data.X_test[orders], dtype=np.float32),
        pd.to_numeric(identity["target_class_id"], errors="raise").to_numpy(dtype=int),
        attributes,
        identity,
        data,
        config,
        model_name=model,
        method=method,
        seed=int(seed),
    )
    runtime = float(time.perf_counter() - started)
    directory.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output_paths["curves"], index=False)
    aopc.to_csv(output_paths["aopc"], index=False)
    signed.to_csv(output_paths["signed"], index=False)
    decision = {
        "audit_version": AUDIT_VERSION,
        "status": "complete",
        "task_id": task_id,
        "model": model,
        "seed": int(seed),
        "method": method,
        "target": "declared_target_class_probability",
        "operators": {
            "deletion_insertion": "coordinate replacement with displacement-matched primary random control",
            "signed": config["functional_checks"]["signed_intervention"][
                "method_aware_operators"
            ][method],
        },
        "signed_analysis_role": "secondary_descriptive_no_general_faithfulness_gate",
        "causal_interpretation": False,
        "runtime_seconds": runtime,
        "stage_signature": stage_signature,
    }
    output_paths["decision"].write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_cache_metadata(
        output_paths["metadata"],
        stage="functional_checks",
        signature_value=stage_signature,
        config_hash=sha256_file(paths.config_path),
        implementation_hash=implementation_hash,
        inputs=inputs,
        outputs=expected,
        extra=decision,
    )
    return {"curves": curves, "aopc": aopc, "signed": signed}


def _maximum_cardinality_minimum_distance_pairs(
    distances: np.ndarray,
    *,
    caliper: float,
) -> list[tuple[int, int, float]]:
    """Lexicographically maximize feasible pairs, then minimize total distance."""

    matrix = np.asarray(distances, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return []
    n_correct, n_error = matrix.shape
    size = n_correct + n_error
    unmatched_penalty = 1.0e6
    forbidden = 1.0e12
    cost = np.zeros((size, size), dtype=float)
    cost[:n_correct, :n_error] = np.where(matrix <= float(caliper), matrix, forbidden)
    cost[:n_correct, n_error:] = unmatched_penalty
    cost[n_correct:, :n_error] = unmatched_penalty
    cost[n_correct:, n_error:] = 0.0
    row_indices, column_indices = linear_sum_assignment(cost)
    pairs: list[tuple[int, int, float]] = []
    for row, column in zip(row_indices, column_indices):
        if row < n_correct and column < n_error and matrix[row, column] <= float(caliper):
            pairs.append((int(row), int(column), float(matrix[row, column])))
    return sorted(pairs, key=lambda value: (value[0], value[1]))


def match_correct_error_group(
    frame: pd.DataFrame,
    *,
    caliper: float,
) -> pd.DataFrame:
    required = {"flow_id", "correct", "confidence", "model_margin"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Outcome matching table lacks {missing}")
    unique = frame.drop_duplicates("flow_id").copy()
    unique["correct"] = unique["correct"].astype(str).str.lower().isin({"true", "1"})
    correct = unique[unique["correct"]].reset_index(drop=True)
    error = unique[~unique["correct"]].reset_index(drop=True)
    if correct.empty or error.empty:
        return pd.DataFrame()
    values = unique[["confidence", "model_margin"]].to_numpy(dtype=float)
    means = np.mean(values, axis=0)
    standard_deviations = np.std(values, axis=0, ddof=1)
    standard_deviations = np.where(standard_deviations > 1e-12, standard_deviations, 1.0)
    correct_z = (correct[["confidence", "model_margin"]].to_numpy(float) - means) / standard_deviations
    error_z = (error[["confidence", "model_margin"]].to_numpy(float) - means) / standard_deviations
    differences = correct_z[:, None, :] - error_z[None, :, :]
    distances = np.sqrt(np.mean(np.square(differences), axis=2))
    pairs = _maximum_cardinality_minimum_distance_pairs(distances, caliper=float(caliper))
    rows: list[dict[str, Any]] = []
    for pair_index, (correct_index, error_index, distance) in enumerate(pairs):
        rows.append(
            {
                "pair_index": pair_index,
                "correct_flow_id": str(correct.loc[correct_index, "flow_id"]),
                "error_flow_id": str(error.loc[error_index, "flow_id"]),
                "standardized_distance": distance,
                "caliper_standard_deviations": float(caliper),
                "correct_confidence": float(correct.loc[correct_index, "confidence"]),
                "error_confidence": float(error.loc[error_index, "confidence"]),
                "correct_margin": float(correct.loc[correct_index, "model_margin"]),
                "error_margin": float(error.loc[error_index, "model_margin"]),
            }
        )
    return pd.DataFrame(rows)


def correct_vs_error_detail(
    aopc: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match outcomes without replacement, then form descriptive paired contrasts."""

    if not bool(config.get("correct_vs_error", {}).get("enabled", False)):
        return pd.DataFrame(), pd.DataFrame()

    outcome_flags = (
        aopc["in_outcome_correct"].astype(str).str.lower().isin({"true", "1"})
        | aopc["in_outcome_error"].astype(str).str.lower().isin({"true", "1"})
    )
    outcome = aopc[outcome_flags].copy()
    outcome["correct"] = outcome["in_outcome_correct"].astype(str).str.lower().isin(
        {"true", "1"}
    )
    matching_cfg = config["correct_vs_error"]
    excluded_true_classes = {
        str(value) for value in matching_cfg.get("excluded_true_classes", [])
    }
    if excluded_true_classes:
        outcome = outcome[
            ~outcome["true_class"].astype(str).isin(excluded_true_classes)
        ].copy()
    allowed_methods = {
        str(value) for value in matching_cfg.get("methods", ALLOWED_PROBES)
    }
    outcome = outcome[outcome["method"].astype(str).isin(allowed_methods)].copy()
    calipers = [
        float(matching_cfg["primary_caliper_standard_deviations"]),
        *[float(value) for value in matching_cfg["sensitivity_calipers_standard_deviations"]],
    ]
    group_columns = ["task_id", "seed", "model", "true_class_id", "true_class"]
    pair_frames: list[pd.DataFrame] = []
    contrast_frames: list[pd.DataFrame] = []
    for group_keys, group in outcome.groupby(group_columns, dropna=False, sort=True):
        identity = dict(zip(group_columns, group_keys))
        inferentially_eligible = bool(
            group["class_specific_inference_eligible"].map(_as_bool).all()
        )
        matching_base = group[
            ["flow_id", "correct", "confidence", "model_margin"]
        ].drop_duplicates("flow_id")
        for caliper in sorted(set(calipers)):
            pairs = match_correct_error_group(matching_base, caliper=caliper)
            if pairs.empty:
                continue
            for column, value in identity.items():
                pairs[column] = value
            pairs["analysis_role"] = (
                "primary"
                if math.isclose(
                    caliper,
                    float(matching_cfg["primary_caliper_standard_deviations"]),
                )
                else "caliper_sensitivity"
            )
            pairs["class_specific_inference_eligible"] = inferentially_eligible
            pair_frames.append(pairs)
            measurement_columns = ["method", "functional_check", "evaluator"]
            for measurement_keys, measurement in group.groupby(
                measurement_columns, dropna=False, sort=True
            ):
                values = measurement.set_index("flow_id")[
                    "attribution_minus_random_aopc"
                ].to_dict()
                rows: list[dict[str, Any]] = []
                for _, pair in pairs.iterrows():
                    correct_flow = str(pair["correct_flow_id"])
                    error_flow = str(pair["error_flow_id"])
                    if correct_flow not in values or error_flow not in values:
                        continue
                    rows.append(
                        {
                            **identity,
                            **dict(zip(measurement_columns, measurement_keys)),
                            "pair_index": int(pair["pair_index"]),
                            "flow_id": f"{correct_flow}::{error_flow}",
                            "correct_flow_id": correct_flow,
                            "error_flow_id": error_flow,
                            "standardized_distance": float(pair["standardized_distance"]),
                            "caliper_standard_deviations": caliper,
                            "analysis_role": pair["analysis_role"],
                            "class_specific_inference_eligible": inferentially_eligible,
                            "correct_aopc_gap": float(values[correct_flow]),
                            "error_aopc_gap": float(values[error_flow]),
                            "correct_minus_error_aopc_gap": float(
                                values[correct_flow] - values[error_flow]
                            ),
                        }
                    )
                if rows:
                    contrast_frames.append(pd.DataFrame(rows))
    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    contrasts = (
        pd.concat(contrast_frames, ignore_index=True) if contrast_frames else pd.DataFrame()
    )
    return pairs, contrasts


def _apply_repeatability_gates(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    result = summary.copy()
    gates = config["repeatability"]["gates"]
    thresholds = {
        "spearman": ("lower", float(gates["spearman_ci_lower"])),
        "jaccard_10": ("lower", float(gates["jaccard_10_ci_lower"])),
    }
    result["gate_bound"] = np.nan
    result["gate_direction"] = "not_predeclared"
    result["gate_pass"] = False
    result["gate_status"] = "not_predeclared"
    for metric, (direction, threshold) in thresholds.items():
        metric_mask = result["metric"].eq(metric)
        result.loc[metric_mask, "gate_bound"] = threshold
        result.loc[metric_mask, "gate_direction"] = direction
        result.loc[
            metric_mask & ~result["coverage_sufficient"].astype(bool), "gate_status"
        ] = "not_evaluable_seed_coverage"
        result.loc[
            metric_mask
            & result["coverage_sufficient"].astype(bool)
            & ~result["identification_coverage_sufficient"].astype(bool),
            "gate_status",
        ] = "not_evaluable_account_coverage"
        mask = (
            metric_mask
            & result["coverage_sufficient"].astype(bool)
            & result["identification_coverage_sufficient"].astype(bool)
        )
        result.loc[mask, "gate_pass"] = result.loc[mask, "ci_low"].ge(threshold)
        result.loc[mask, "gate_status"] = np.where(
            result.loc[mask, "gate_pass"], "pass", "fail"
        )
    return result


def _apply_cross_model_gates(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    result = summary.copy()
    gates = config["cross_model_identification"]["equivalence_gates"]
    definitions = {
        "spearman": ("lower", float(gates["spearman_ci_lower"])),
        "weighted_jaccard": ("lower", float(gates["weighted_jaccard_ci_lower"])),
        "jaccard_10": ("lower", float(gates["jaccard_10_ci_lower"])),
        "normalized_mass_l1": ("upper", float(gates["normalized_mass_l1_ci_upper"])),
    }
    result["gate_bound"] = np.nan
    result["gate_direction"] = "not_predeclared"
    result["gate_pass"] = False
    result["gate_status"] = "not_predeclared"
    for metric, (direction, threshold) in definitions.items():
        metric_mask = result["metric"].eq(metric)
        result.loc[metric_mask, "gate_bound"] = threshold
        result.loc[metric_mask, "gate_direction"] = direction
        result.loc[
            metric_mask & ~result["coverage_sufficient"].astype(bool), "gate_status"
        ] = "not_evaluable_seed_coverage"
        result.loc[
            metric_mask
            & result["coverage_sufficient"].astype(bool)
            & ~result["identification_coverage_sufficient"].astype(bool),
            "gate_status",
        ] = "not_evaluable_account_coverage"
        mask = (
            metric_mask
            & result["coverage_sufficient"].astype(bool)
            & result["identification_coverage_sufficient"].astype(bool)
        )
        if direction == "lower":
            result.loc[mask, "gate_pass"] = result.loc[mask, "ci_low"].ge(threshold)
        else:
            result.loc[mask, "gate_pass"] = result.loc[mask, "ci_high"].le(threshold)
        result.loc[mask, "gate_status"] = np.where(
            result.loc[mask, "gate_pass"], "pass", "fail"
        )
    return result


def _apply_cross_probe_gates(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    result = summary.copy()
    gates = config["cross_probe_triangulation"]["gates"]
    definitions = {
        "spearman": ("lower", float(gates["spearman_ci_lower"])),
        "weighted_jaccard": ("lower", float(gates["weighted_jaccard_ci_lower"])),
        "jaccard_10": ("lower", float(gates["jaccard_10_ci_lower"])),
        "normalized_mass_l1": ("upper", float(gates["normalized_mass_l1_ci_upper"])),
    }
    result["gate_bound"] = np.nan
    result["gate_direction"] = "not_predeclared"
    result["gate_pass"] = False
    result["gate_status"] = "not_predeclared"
    for metric, (direction, threshold) in definitions.items():
        metric_mask = result["metric"].eq(metric)
        result.loc[metric_mask, "gate_bound"] = threshold
        result.loc[metric_mask, "gate_direction"] = direction
        result.loc[
            metric_mask & ~result["coverage_sufficient"].astype(bool), "gate_status"
        ] = "not_evaluable_seed_coverage"
        result.loc[
            metric_mask
            & result["coverage_sufficient"].astype(bool)
            & ~result["identification_coverage_sufficient"].astype(bool),
            "gate_status",
        ] = "not_evaluable_account_coverage"
        mask = (
            metric_mask
            & result["coverage_sufficient"].astype(bool)
            & result["identification_coverage_sufficient"].astype(bool)
        )
        result.loc[mask, "gate_pass"] = (
            result.loc[mask, "ci_low"].ge(threshold)
            if direction == "lower"
            else result.loc[mask, "ci_high"].le(threshold)
        )
        result.loc[mask, "gate_status"] = np.where(
            result.loc[mask, "gate_pass"], "pass", "fail"
        )
    return result


def _apply_parameter_sanity_gates(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    result = summary.copy()
    sanity = config["parameter_dependence_sanity"]
    gates = sanity["primary_change_gate"]
    definitions = {
        str(sanity["primary_metric"]): (
            "lower",
            float(gates["normalized_mass_l1_ci_lower"]),
        )
    }
    result["change_gate_bound"] = np.nan
    result["change_gate_direction"] = "not_predeclared"
    result["change_gate_pass"] = False
    result["change_gate_status"] = "not_predeclared"
    for metric, (direction, threshold) in definitions.items():
        metric_mask = result["metric"].eq(metric)
        result.loc[metric_mask, "change_gate_bound"] = threshold
        result.loc[metric_mask, "change_gate_direction"] = direction
        result.loc[
            metric_mask & ~result["coverage_sufficient"].astype(bool),
            "change_gate_status",
        ] = "not_evaluable_seed_coverage"
        result.loc[
            metric_mask
            & result["coverage_sufficient"].astype(bool)
            & ~result["identification_coverage_sufficient"].astype(bool),
            "change_gate_status",
        ] = "not_evaluable_account_coverage"
        mask = (
            metric_mask
            & result["coverage_sufficient"].astype(bool)
            & result["identification_coverage_sufficient"].astype(bool)
        )
        result.loc[mask, "change_gate_pass"] = (
            result.loc[mask, "ci_high"].le(threshold)
            if direction == "upper"
            else result.loc[mask, "ci_low"].ge(threshold)
        )
        result.loc[mask, "change_gate_status"] = np.where(
            result.loc[mask, "change_gate_pass"], "pass", "fail"
        )
    group_columns = ["task_id", "model", "method"]
    result["parameter_dependence_established"] = result.groupby(group_columns)[
        "change_gate_pass"
    ].transform("any")
    return result


def _class_group_columns(group_columns: Sequence[str]) -> list[str]:
    """Insert the frozen true-class identity after the task identifier."""

    columns = list(group_columns)
    if "task_id" not in columns:
        raise ValueError("Class-specific summaries require task_id")
    insertion = columns.index("task_id") + 1
    return [
        *columns[:insertion],
        "true_class_id",
        "true_class",
        *columns[insertion:],
    ]


def _expected_decision_inventory(
    config: Mapping[str, Any],
    class_support: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Enumerate every aggregate and class-specific decision cell a priori."""

    similarity_metrics = (
        "spearman",
        "jaccard_10",
        "weighted_jaccard",
        "normalized_mass_l1",
    )
    repeatability_conditions = (
        ("integrated_gradients", "primary_main_to_128"),
        ("occlusion", "primary_fixed_rule_repeat"),
        ("shap", "primary_stochastic_repeat"),
        ("lime", "primary_mean_pairwise"),
    )
    deterministic = tuple(
        str(value)
        for value in config["cross_probe_triangulation"]["deterministic_central_probes"]
    )
    all_probes = tuple(
        str(value)
        for value in config["cross_probe_triangulation"]["all_probe_stochastic_subset"]
    )
    cross_probe_conditions = [
        ("deterministic_central", left, right)
        for left, right in itertools.combinations(deterministic, 2)
    ]
    cross_probe_conditions.extend(
        ("all_probe_stochastic_subset", left, right)
        for left, right in itertools.combinations(all_probes, 2)
    )
    functional = config["functional_checks"]["deletion_and_insertion"]

    aggregate: dict[str, dict[str, Any]] = {
        "repeatability_aggregate": {
            "key_columns": ("task_id", "model", "method", "analysis_role", "metric"),
            "cells": {
                (task_id, model, method, role, metric)
                for task_id in ALLOWED_TASKS
                for model in ALLOWED_MODELS
                for method, role in repeatability_conditions
                for metric in ("spearman", "jaccard_10")
            },
        },
        "cross_model_aggregate": {
            "key_columns": ("task_id", "method", "metric"),
            "cells": {
                (task_id, method, metric)
                for task_id in ALLOWED_TASKS
                for method in ALLOWED_PROBES
                for metric in similarity_metrics
            },
        },
        "cross_probe_aggregate": {
            "key_columns": (
                "task_id",
                "model",
                "panel_scope",
                "probe_left",
                "probe_right",
                "metric",
            ),
            "cells": {
                (task_id, model, panel_scope, left, right, metric)
                for task_id in ALLOWED_TASKS
                for model in ALLOWED_MODELS
                for panel_scope, left, right in cross_probe_conditions
                for metric in similarity_metrics
            },
        },
        "parameter_sanity_aggregate": {
            "key_columns": ("task_id", "model", "method", "metric"),
            "cells": {
                (task_id, model, str(method), metric)
                for task_id in ALLOWED_TASKS
                for model in ALLOWED_MODELS
                for method in config["parameter_dependence_sanity"]["probes"]
                for metric in similarity_metrics
            },
        },
        "functional_response_aggregate": {
            "key_columns": (
                "task_id",
                "model",
                "method",
                "functional_check",
                "evaluator",
                "metric",
            ),
            "cells": {
                (task_id, model, method, check, str(evaluator), "attribution_minus_random_aopc")
                for task_id in ALLOWED_TASKS
                for model in ALLOWED_MODELS
                for method in ALLOWED_PROBES
                for check in ("deletion", "insertion")
                for evaluator in functional["evaluators"]
            },
        },
        "functional_monte_carlo_aggregate": {
            "key_columns": (
                "task_id",
                "model",
                "method",
                "functional_check",
                "evaluator",
                "metric",
            ),
            "cells": {
                (
                    task_id,
                    model,
                    method,
                    check,
                    str(evaluator),
                    "attribution_minus_random_aopc_mc_lower95",
                )
                for task_id in ALLOWED_TASKS
                for model in ALLOWED_MODELS
                for method in ALLOWED_PROBES
                for check in ("deletion", "insertion")
                for evaluator in functional["evaluators"]
            },
        },
    }

    required_support_columns = {
        "task_id",
        "seed",
        "true_class_id",
        "true_class",
        "validation_inference_eligible",
        "class_specific_inference_eligible",
    }
    missing_support = sorted(required_support_columns - set(class_support.columns))
    if missing_support:
        raise RuntimeError(f"Class-support inventory lacks {missing_support}")
    support_keys = ["task_id", "seed", "true_class_id", "true_class"]
    if class_support.duplicated(support_keys).any():
        raise RuntimeError("Class-support inventory contains duplicate class/seed cells")
    observed_tasks = set(class_support["task_id"].astype(str))
    if observed_tasks != set(ALLOWED_TASKS):
        raise RuntimeError("Class-support inventory does not contain exactly the audit tasks")
    expected_seeds = set(int(value) for value in training.FINAL_SEEDS)
    for task_id, task_support in class_support.groupby("task_id", sort=True):
        observed_seeds = set(
            pd.to_numeric(task_support["seed"], errors="raise").astype(int)
        )
        if observed_seeds != expected_seeds:
            raise RuntimeError(
                f"Class-support seed inventory is incomplete for {task_id}"
            )
    class_identity = class_support[
        ["task_id", "true_class_id", "true_class"]
    ].drop_duplicates()
    if class_identity.duplicated(["task_id", "true_class_id"]).any() or (
        class_identity.duplicated(["task_id", "true_class"]).any()
    ):
        raise RuntimeError("Class-support IDs and names are not one-to-one within a task")
    validation_flags = class_support.assign(
        _validation_flag=class_support["validation_inference_eligible"].map(_as_bool)
    ).groupby(["task_id", "true_class_id", "true_class"], dropna=False)[
        "_validation_flag"
    ].nunique()
    if validation_flags.gt(1).any():
        raise RuntimeError("Validation class eligibility is not frozen across seeds")
    eligible = class_support[
        class_support["validation_inference_eligible"].map(_as_bool)
        & class_support["class_specific_inference_eligible"].map(_as_bool)
    ][["task_id", "true_class_id", "true_class"]].drop_duplicates()
    eligible_classes = {
        str(task_id): tuple(
            (int(row.true_class_id), str(row.true_class))
            for row in task_frame.sort_values("true_class_id", kind="mergesort").itertuples()
        )
        for task_id, task_frame in eligible.groupby("task_id", sort=True)
    }
    missing_tasks = sorted(set(ALLOWED_TASKS) - set(eligible_classes))
    if missing_tasks:
        raise RuntimeError(f"No inferential class is registered for tasks {missing_tasks}")

    inventory = dict(aggregate)
    for name, specification in aggregate.items():
        key_columns = tuple(specification["key_columns"])
        task_position = key_columns.index("task_id")
        class_key_columns = tuple(_class_group_columns(key_columns))
        class_cells: set[tuple[Any, ...]] = set()
        for cell in specification["cells"]:
            task_id = str(cell[task_position])
            for class_id, class_name in eligible_classes[task_id]:
                class_cells.add(
                    (
                        *cell[: task_position + 1],
                        int(class_id),
                        str(class_name),
                        *cell[task_position + 1 :],
                    )
                )
        inventory[name.replace("_aggregate", "_by_class")] = {
            "key_columns": class_key_columns,
            "cells": class_cells,
        }
    empty_inventories = sorted(
        name for name, specification in inventory.items() if not specification["cells"]
    )
    if empty_inventories:
        raise RuntimeError(f"Decision inventories are empty for {empty_inventories}")
    return inventory


def _inventory_status(
    analysis_id: str,
    frame: pd.DataFrame,
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare one observed summary with its exact predeclared cell inventory."""

    key_columns = tuple(str(value) for value in specification["key_columns"])
    expected = set(specification["cells"])
    if not expected:
        raise RuntimeError(f"Decision inventory for {analysis_id} is empty")
    required = {*key_columns, "coverage_sufficient"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        return {
            "analysis_id": analysis_id,
            "expected_cells": int(len(expected)),
            "observed_rows": int(len(frame)),
            "observed_unique_cells": 0,
            "missing_cells": int(len(expected)),
            "unexpected_cells": 0,
            "duplicate_cells": 0,
            "structure_complete": False,
            "seed_coverage_complete": False,
            "identification_coverage_complete": False,
            "decision_coverage_complete": False,
            "missing_columns_json": json.dumps(missing_columns),
            "missing_examples_json": "[]",
            "unexpected_examples_json": "[]",
            "duplicate_examples_json": "[]",
        }

    observed_rows = [
        tuple(row[column] for column in key_columns)
        for row in frame[list(key_columns)].to_dict("records")
    ]
    counts = Counter(observed_rows)
    observed = set(counts)
    missing = expected - observed
    unexpected = observed - expected
    duplicates = {cell for cell, count in counts.items() if count != 1}
    structure_complete = not missing and not unexpected and not duplicates
    expected_mask = pd.Series(
        [cell in expected for cell in observed_rows], index=frame.index, dtype=bool
    )
    expected_rows = frame.loc[expected_mask]
    seed_coverage = bool(
        structure_complete
        and len(expected_rows) == len(expected)
        and expected_rows["coverage_sufficient"].map(_as_bool).all()
    )
    identification_coverage = bool(
        seed_coverage
        and (
            "identification_coverage_sufficient" not in expected_rows.columns
            or expected_rows["identification_coverage_sufficient"].map(_as_bool).all()
        )
    )

    def examples(values: set[tuple[Any, ...]]) -> str:
        return json.dumps(
            [list(value) for value in sorted(values, key=lambda item: tuple(map(str, item)))[:10]],
            default=str,
        )

    return {
        "analysis_id": analysis_id,
        "expected_cells": int(len(expected)),
        "observed_rows": int(len(frame)),
        "observed_unique_cells": int(len(observed)),
        "missing_cells": int(len(missing)),
        "unexpected_cells": int(len(unexpected)),
        "duplicate_cells": int(len(duplicates)),
        "structure_complete": bool(structure_complete),
        "seed_coverage_complete": seed_coverage,
        "identification_coverage_complete": identification_coverage,
        "decision_coverage_complete": identification_coverage,
        "missing_columns_json": "[]",
        "missing_examples_json": examples(missing),
        "unexpected_examples_json": examples(unexpected),
        "duplicate_examples_json": examples(duplicates),
    }


def _inventory_all_rows_pass(
    frame: pd.DataFrame,
    specification: Mapping[str, Any],
    status: Mapping[str, Any],
    pass_column: str,
) -> bool:
    """Return false on missing cells, insufficient coverage, or any failed gate."""

    if not bool(status["decision_coverage_complete"]) or pass_column not in frame.columns:
        return False
    key_columns = tuple(specification["key_columns"])
    expected = set(specification["cells"])
    keys = [
        tuple(row[column] for column in key_columns)
        for row in frame[list(key_columns)].to_dict("records")
    ]
    selected = frame.loc[[key in expected for key in keys], pass_column]
    return bool(len(selected) == len(expected) and selected.map(_as_bool).all())


def _inventory_all_conditions_pass(
    frame: pd.DataFrame,
    specification: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    condition_columns: Sequence[str],
    pass_column: str,
) -> bool:
    """Fail closed for a condition-level decision repeated across metric rows."""

    if not bool(status["decision_coverage_complete"]) or pass_column not in frame.columns:
        return False
    key_columns = tuple(specification["key_columns"])
    expected = set(specification["cells"])
    keys = [
        tuple(row[column] for column in key_columns)
        for row in frame[list(key_columns)].to_dict("records")
    ]
    selected = frame.loc[[key in expected for key in keys]]
    condition = selected.groupby(list(condition_columns), dropna=False)[
        pass_column
    ].agg(lambda values: bool(values.map(_as_bool).all()))
    expected_condition_count = len(
        {
            tuple(cell[key_columns.index(column)] for column in condition_columns)
            for cell in expected
        }
    )
    return bool(
        len(condition) == expected_condition_count and condition.map(_as_bool).all()
    )


def _admissible_claim_types(config: Mapping[str, Any]) -> list[str]:
    """Return the protocol whitelist without implying that any gate passed."""

    policy = config["evidence_policy"]
    if "accepted_claims" in policy:
        raise ValueError(
            "evidence_policy.accepted_claims is ambiguous; use admissible_claim_types"
        )
    values = policy.get("admissible_claim_types")
    if not isinstance(values, list) or not values:
        raise ValueError("evidence_policy.admissible_claim_types must be a non-empty list")
    normalized = [str(value).strip() for value in values]
    if any(not value for value in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("admissible_claim_types must be unique non-empty strings")
    return normalized


def _merge_functional_monte_carlo_sensitivity(
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Attach the conservative Monte Carlo result with an exact one-to-one key set."""

    key_columns = list(group_columns)
    primary_required = {*key_columns, "metric", "ci_low", "coverage_sufficient"}
    sensitivity_required = {
        *key_columns,
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "coverage_sufficient",
    }
    missing_primary = sorted(primary_required - set(primary.columns))
    missing_sensitivity = sorted(sensitivity_required - set(sensitivity.columns))
    if missing_primary or missing_sensitivity:
        raise RuntimeError(
            "Functional summary lacks Monte Carlo merge columns: "
            f"primary={missing_primary}, sensitivity={missing_sensitivity}"
        )
    if set(primary["metric"].astype(str)) != {"attribution_minus_random_aopc"}:
        raise RuntimeError("Unexpected primary functional metric inventory")
    if set(sensitivity["metric"].astype(str)) != {
        "attribution_minus_random_aopc_mc_lower95"
    }:
        raise RuntimeError("Unexpected functional Monte Carlo metric inventory")
    if primary.duplicated(key_columns).any() or sensitivity.duplicated(key_columns).any():
        raise RuntimeError("Functional Monte Carlo summaries contain duplicate keys")
    key_inventory = primary[key_columns].merge(
        sensitivity[key_columns],
        on=key_columns,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not key_inventory["_merge"].eq("both").all():
        raise RuntimeError(
            "Primary and Monte Carlo functional summaries have different key inventories"
        )
    result = primary.merge(
        sensitivity[
            key_columns + ["estimate", "ci_low", "ci_high", "coverage_sufficient"]
        ].rename(
            columns={
                "estimate": "mc_conservative_estimate",
                "ci_low": "mc_conservative_ci_low",
                "ci_high": "mc_conservative_ci_high",
                "coverage_sufficient": "mc_conservative_coverage_sufficient",
            }
        ),
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    result["positive_response_vs_displacement_matched_random"] = (
        result["coverage_sufficient"].map(_as_bool)
        & result["ci_low"].gt(0.0)
        & result["mc_conservative_coverage_sufficient"].map(_as_bool)
        & result["mc_conservative_ci_low"].gt(0.0)
    )
    return result


def run_audit_summaries(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
) -> dict[str, pd.DataFrame]:
    """Aggregate only complete caches; this stage never generates explanations."""

    require_valid_panel(config, training_config, paths)

    summary_input_paths: list[Path] = [
        paths.config_path,
        paths.training_config_path,
        paths.panel_dir / "cache_metadata.json",
        paths.central_panel,
        paths.class_support,
    ]
    class_support = pd.read_csv(paths.class_support)
    repeat_frames: list[pd.DataFrame] = []
    cross_frames: list[pd.DataFrame] = []
    reference_frames: list[pd.DataFrame] = []
    triangulation_frames: list[pd.DataFrame] = []
    parameter_sanity_frames: list[pd.DataFrame] = []
    aopc_frames: list[pd.DataFrame] = []
    signed_frames: list[pd.DataFrame] = []
    for task_id in ALLOWED_TASKS:
        for seed in training.FINAL_SEEDS:
            for model in ALLOWED_MODELS:
                triangulation_frames.append(
                    cross_probe_detail_for_condition(
                        config,
                        training_config,
                        paths,
                        task_id=task_id,
                        model=model,
                        seed=int(seed),
                    )
                )
                for sanity_method in config["parameter_dependence_sanity"]["probes"]:
                    parameter_sanity_frames.append(
                        run_parameter_sanity_condition(
                            config,
                            training_config,
                            paths,
                            task_id=task_id,
                            model=model,
                            seed=int(seed),
                            method=str(sanity_method),
                            compute_if_missing=False,
                        )
                    )
                    summary_input_paths.append(
                        _parameter_sanity_directory(
                            paths,
                            task_id,
                            model,
                            int(seed),
                            str(sanity_method),
                        )
                        / "cache_metadata.json"
                    )
            for method in ALLOWED_PROBES:
                cross_frames.append(
                    cross_model_detail_for_condition(
                        config,
                        training_config,
                        paths,
                        task_id=task_id,
                        seed=int(seed),
                        method=method,
                    )
                )
                for model in ALLOWED_MODELS:
                    summary_input_paths.extend(
                        _probe_output_paths(
                            _probe_directory(paths, task_id, model, int(seed), spec)
                        )["metadata"]
                        for spec in probe_specs(config, method)
                    )
                    repeat = repeatability_detail_for_condition(
                        config,
                        training_config,
                        paths,
                        task_id=task_id,
                        model=model,
                        seed=int(seed),
                        method=method,
                    )
                    if len(repeat):
                        repeat_frames.append(repeat)
                    if method in set(config["reference_sensitivity"]["methods"]):
                        reference_frames.append(
                            reference_sensitivity_detail_for_condition(
                                config,
                                training_config,
                                paths,
                                task_id=task_id,
                                model=model,
                                seed=int(seed),
                                method=method,
                            )
                        )
                    functional = run_functional_condition(
                        config,
                        training_config,
                        paths,
                        task_id=task_id,
                        model=model,
                        seed=int(seed),
                        method=method,
                        compute_if_missing=False,
                    )
                    aopc_frames.append(functional["aopc"])
                    signed_frames.append(functional["signed"])
                    summary_input_paths.append(
                        _functional_directory(paths, task_id, model, int(seed), method)
                        / "cache_metadata.json"
                    )
    repeatability = pd.concat(repeat_frames, ignore_index=True)
    repeatability["analysis_role"] = np.where(
        repeatability["method"].eq("integrated_gradients"),
        np.where(
            repeatability["comparison"].eq("steps_64_vs_128"),
            "primary_main_to_128",
            "numerical_sensitivity",
        ),
        np.where(
            repeatability["method"].eq("lime"),
            "primary_mean_pairwise",
            np.where(
                repeatability["method"].eq("shap"),
                "primary_stochastic_repeat",
                "primary_fixed_rule_repeat",
            ),
        ),
    )
    repeat_eligible = repeatability["class_specific_inference_eligible"].map(_as_bool)
    repeat_inference = repeatability[repeat_eligible]
    lime_repeat = repeat_inference[repeat_inference["method"].eq("lime")]
    deterministic_repeat = repeat_inference[~repeat_inference["method"].eq("lime")]
    deterministic_repeatability_summary = summarize_metrics(
        deterministic_repeat,
        group_columns=["task_id", "model", "method", "analysis_role"],
        metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
        config=config,
        analysis_id="repeatability",
    )
    lime_repeatability_summary = summarize_lime_u_statistics(
        lime_repeat,
        group_columns=["task_id", "model", "method", "analysis_role"],
        metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
        config=config,
    )
    repeatability_summary = pd.concat(
        [deterministic_repeatability_summary, lime_repeatability_summary],
        ignore_index=True,
        sort=False,
    )
    repeatability_summary = _apply_repeatability_gates(repeatability_summary, config)
    repeatability_class_groups = _class_group_columns(
        ["task_id", "model", "method", "analysis_role"]
    )
    deterministic_repeatability_summary_by_class = summarize_metrics(
        deterministic_repeat,
        group_columns=repeatability_class_groups,
        metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
        config=config,
        analysis_id="repeatability_by_class",
    )
    lime_repeatability_summary_by_class = summarize_lime_u_statistics(
        lime_repeat,
        group_columns=repeatability_class_groups,
        metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
        config=config,
    )
    repeatability_summary_by_class = pd.concat(
        [
            deterministic_repeatability_summary_by_class,
            lime_repeatability_summary_by_class,
        ],
        ignore_index=True,
        sort=False,
    )
    repeatability_summary_by_class = _apply_repeatability_gates(
        repeatability_summary_by_class, config
    )

    cross_model = pd.concat(cross_frames, ignore_index=True)
    cross_eligible = cross_model["class_specific_inference_eligible"].map(_as_bool)
    cross_model_summary = summarize_metrics(
        cross_model[cross_eligible],
        group_columns=["task_id", "method"],
        metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
        config=config,
        run_column="probe_run",
        analysis_id="cross_model_identification",
    )
    cross_model_summary = _apply_cross_model_gates(cross_model_summary, config)
    cross_model_summary_by_class = summarize_metrics(
        cross_model[cross_eligible],
        group_columns=_class_group_columns(["task_id", "method"]),
        metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
        config=config,
        run_column="probe_run",
        analysis_id="cross_model_identification_by_class",
    )
    cross_model_summary_by_class = _apply_cross_model_gates(
        cross_model_summary_by_class, config
    )

    reference_sensitivity = pd.concat(reference_frames, ignore_index=True)
    reference_eligible = reference_sensitivity[
        "class_specific_inference_eligible"
    ].map(_as_bool)
    reference_sensitivity_summary = summarize_metrics(
        reference_sensitivity[reference_eligible],
        group_columns=[
            "task_id",
            "model",
            "method",
            "reference_primary",
            "reference_variant",
        ],
        metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
        config=config,
        analysis_id="reference_sensitivity_descriptive",
    )
    reference_sensitivity_summary_by_class = summarize_metrics(
        reference_sensitivity[reference_eligible],
        group_columns=_class_group_columns(
            [
                "task_id",
                "model",
                "method",
                "reference_primary",
                "reference_variant",
            ]
        ),
        metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
        config=config,
        analysis_id="reference_sensitivity_descriptive_by_class",
    )

    cross_probe = pd.concat(triangulation_frames, ignore_index=True)
    cross_probe_eligible = cross_probe["class_specific_inference_eligible"].map(_as_bool)
    cross_probe_summary = summarize_metrics(
        cross_probe[cross_probe_eligible],
        group_columns=[
            "task_id",
            "model",
            "panel_scope",
            "probe_left",
            "probe_right",
        ],
        metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
        config=config,
        run_column="probe_run",
        analysis_id="cross_probe_triangulation",
    )
    cross_probe_summary = _apply_cross_probe_gates(cross_probe_summary, config)
    cross_probe_summary_by_class = summarize_metrics(
        cross_probe[cross_probe_eligible],
        group_columns=_class_group_columns(
            [
                "task_id",
                "model",
                "panel_scope",
                "probe_left",
                "probe_right",
            ]
        ),
        metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
        config=config,
        run_column="probe_run",
        analysis_id="cross_probe_triangulation_by_class",
    )
    cross_probe_summary_by_class = _apply_cross_probe_gates(
        cross_probe_summary_by_class, config
    )

    parameter_sanity = pd.concat(parameter_sanity_frames, ignore_index=True)
    parameter_eligible = parameter_sanity["class_specific_inference_eligible"].map(
        _as_bool
    )
    parameter_sanity_summary = summarize_metrics(
        parameter_sanity[parameter_eligible],
        group_columns=["task_id", "model", "method"],
        metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
        config=config,
        analysis_id="parameter_dependence_sanity",
    )
    parameter_sanity_summary = _apply_parameter_sanity_gates(
        parameter_sanity_summary, config
    )
    parameter_sanity_summary_by_class = summarize_metrics(
        parameter_sanity[parameter_eligible],
        group_columns=_class_group_columns(["task_id", "model", "method"]),
        metric_columns=["spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1"],
        config=config,
        analysis_id="parameter_dependence_sanity_by_class",
    )
    parameter_sanity_summary_by_class = _apply_parameter_sanity_gates(
        parameter_sanity_summary_by_class, config
    )
    parameter_sanity_summary_by_class["parameter_dependence_established"] = (
        parameter_sanity_summary_by_class.groupby(
            ["task_id", "true_class_id", "true_class", "model", "method"],
            dropna=False,
        )["change_gate_pass"].transform("any")
    )

    aopc = pd.concat(aopc_frames, ignore_index=True)
    central_flags = aopc["in_central"].map(_as_bool)
    inferential_flags = aopc["class_specific_inference_eligible"].map(_as_bool)
    functional_groups = [
        "task_id",
        "model",
        "method",
        "functional_check",
        "evaluator",
    ]
    functional_summary = summarize_metrics(
        aopc[central_flags & inferential_flags],
        group_columns=functional_groups,
        metric_columns=["attribution_minus_random_aopc"],
        config=config,
        analysis_id="functional_aopc",
    )
    functional_monte_carlo_sensitivity_summary = summarize_metrics(
        aopc[central_flags & inferential_flags],
        group_columns=functional_groups,
        metric_columns=["attribution_minus_random_aopc_mc_lower95"],
        config=config,
        analysis_id="functional_aopc_monte_carlo_sensitivity",
    )
    functional_summary = _merge_functional_monte_carlo_sensitivity(
        functional_summary,
        functional_monte_carlo_sensitivity_summary,
        group_columns=functional_groups,
    )
    functional_class_groups = _class_group_columns(functional_groups)
    functional_summary_by_class = summarize_metrics(
        aopc[central_flags & inferential_flags],
        group_columns=functional_class_groups,
        metric_columns=["attribution_minus_random_aopc"],
        config=config,
        analysis_id="functional_aopc_by_class",
    )
    functional_monte_carlo_sensitivity_summary_by_class = summarize_metrics(
        aopc[central_flags & inferential_flags],
        group_columns=functional_class_groups,
        metric_columns=["attribution_minus_random_aopc_mc_lower95"],
        config=config,
        analysis_id="functional_aopc_monte_carlo_sensitivity_by_class",
    )
    functional_summary_by_class = _merge_functional_monte_carlo_sensitivity(
        functional_summary_by_class,
        functional_monte_carlo_sensitivity_summary_by_class,
        group_columns=functional_class_groups,
    )
    functional_displacement_summary = summarize_metrics(
        aopc[central_flags & inferential_flags],
        group_columns=["task_id", "model", "method", "functional_check", "evaluator"],
        metric_columns=["attribution_minus_matched_displacement"],
        config=config,
        analysis_id="functional_random_control_displacement_balance",
    )
    functional_displacement_summary_by_class = summarize_metrics(
        aopc[central_flags & inferential_flags],
        group_columns=_class_group_columns(
            ["task_id", "model", "method", "functional_check", "evaluator"]
        ),
        metric_columns=["attribution_minus_matched_displacement"],
        config=config,
        analysis_id="functional_random_control_displacement_balance_by_class",
    )

    signed = pd.concat(signed_frames, ignore_index=True)
    signed_change_eligible = signed["signed_intervention_eligible"].map(_as_bool)
    signed["operator_consistent_value"] = np.where(
        signed_change_eligible,
        signed["operator_consistent"].map(_as_bool).astype(float),
        np.nan,
    )
    signed_flags = signed["in_central"].map(_as_bool)
    signed_eligible = signed["class_specific_inference_eligible"].map(_as_bool)
    signed_coverage = (
        signed[signed_flags & signed_eligible]
        .groupby(
            ["task_id", "model", "method", "direction", "strength"],
            as_index=False,
            dropna=False,
        )
        .agg(
            eligible_changed_rows=("signed_intervention_eligible", lambda x: int(x.map(_as_bool).sum())),
            total_rows=("signed_intervention_eligible", "size"),
        )
    )
    signed_coverage["eligible_changed_fraction"] = (
        signed_coverage["eligible_changed_rows"] / signed_coverage["total_rows"]
    )
    signed_coverage_by_class = (
        signed[signed_flags & signed_eligible]
        .groupby(
            [
                "task_id",
                "true_class_id",
                "true_class",
                "model",
                "method",
                "direction",
                "strength",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            eligible_changed_rows=(
                "signed_intervention_eligible",
                lambda x: int(x.map(_as_bool).sum()),
            ),
            total_rows=("signed_intervention_eligible", "size"),
        )
    )
    signed_coverage_by_class["eligible_changed_fraction"] = (
        signed_coverage_by_class["eligible_changed_rows"]
        / signed_coverage_by_class["total_rows"]
    )
    signed_summary = summarize_metrics(
        signed[signed_flags & signed_eligible & signed_change_eligible],
        group_columns=["task_id", "model", "method", "direction", "strength"],
        metric_columns=[
            "aligned_effect_minus_random",
            "aligned_probability_effect",
            "operator_consistent_value",
        ],
        config=config,
        analysis_id="signed_intervention",
    )
    signed_summary_by_class = summarize_metrics(
        signed[signed_flags & signed_eligible & signed_change_eligible],
        group_columns=_class_group_columns(
            ["task_id", "model", "method", "direction", "strength"]
        ),
        metric_columns=[
            "aligned_effect_minus_random",
            "aligned_probability_effect",
            "operator_consistent_value",
        ],
        config=config,
        analysis_id="signed_intervention_by_class",
    )

    matches, correct_error = correct_vs_error_detail(aopc, config)
    if len(correct_error):
        correct_error_eligible = correct_error[
            "class_specific_inference_eligible"
        ].map(_as_bool)
        correct_error_summary = summarize_metrics(
            correct_error[correct_error_eligible],
            group_columns=[
                "task_id",
                "model",
                "method",
                "functional_check",
                "evaluator",
                "caliper_standard_deviations",
                "analysis_role",
            ],
            metric_columns=["correct_minus_error_aopc_gap"],
            config=config,
            analysis_id="correct_vs_error_descriptive",
        )
        correct_error_summary_by_class = summarize_metrics(
            correct_error[correct_error_eligible],
            group_columns=_class_group_columns(
                [
                    "task_id",
                    "model",
                    "method",
                    "functional_check",
                    "evaluator",
                    "caliper_standard_deviations",
                    "analysis_role",
                ]
            ),
            metric_columns=["correct_minus_error_aopc_gap"],
            config=config,
            analysis_id="correct_vs_error_descriptive_by_class",
        )
    else:
        correct_error_summary = pd.DataFrame()
        correct_error_summary_by_class = pd.DataFrame()

    inventory_specifications = _expected_decision_inventory(config, class_support)
    primary_repeatability = repeatability_summary[
        repeatability_summary["analysis_role"].isin(
            [
                "primary_main_to_128",
                "primary_fixed_rule_repeat",
                "primary_stochastic_repeat",
                "primary_mean_pairwise",
            ]
        )
        & repeatability_summary["metric"].isin(["spearman", "jaccard_10"])
    ].copy()
    primary_repeatability_by_class = repeatability_summary_by_class[
        repeatability_summary_by_class["analysis_role"].isin(
            [
                "primary_main_to_128",
                "primary_fixed_rule_repeat",
                "primary_stochastic_repeat",
                "primary_mean_pairwise",
            ]
        )
        & repeatability_summary_by_class["metric"].isin(["spearman", "jaccard_10"])
    ].copy()
    inventory_frames = {
        "repeatability_aggregate": primary_repeatability,
        "repeatability_by_class": primary_repeatability_by_class,
        "cross_model_aggregate": cross_model_summary,
        "cross_model_by_class": cross_model_summary_by_class,
        "cross_probe_aggregate": cross_probe_summary,
        "cross_probe_by_class": cross_probe_summary_by_class,
        "parameter_sanity_aggregate": parameter_sanity_summary,
        "parameter_sanity_by_class": parameter_sanity_summary_by_class,
        "functional_response_aggregate": functional_summary,
        "functional_response_by_class": functional_summary_by_class,
        "functional_monte_carlo_aggregate": (
            functional_monte_carlo_sensitivity_summary
        ),
        "functional_monte_carlo_by_class": (
            functional_monte_carlo_sensitivity_summary_by_class
        ),
    }
    inventory_statuses = {
        name: _inventory_status(name, inventory_frames[name], specification)
        for name, specification in inventory_specifications.items()
    }
    decision_inventory = pd.DataFrame(list(inventory_statuses.values())).sort_values(
        "analysis_id", kind="mergesort"
    )
    summary_inventory_complete = bool(
        decision_inventory["structure_complete"].map(_as_bool).all()
    )

    repeatability_aggregate_pass = _inventory_all_rows_pass(
        primary_repeatability,
        inventory_specifications["repeatability_aggregate"],
        inventory_statuses["repeatability_aggregate"],
        "gate_pass",
    )
    repeatability_class_pass = _inventory_all_rows_pass(
        primary_repeatability_by_class,
        inventory_specifications["repeatability_by_class"],
        inventory_statuses["repeatability_by_class"],
        "gate_pass",
    )
    cross_model_aggregate_pass = _inventory_all_rows_pass(
        cross_model_summary,
        inventory_specifications["cross_model_aggregate"],
        inventory_statuses["cross_model_aggregate"],
        "gate_pass",
    )
    cross_model_class_pass = _inventory_all_rows_pass(
        cross_model_summary_by_class,
        inventory_specifications["cross_model_by_class"],
        inventory_statuses["cross_model_by_class"],
        "gate_pass",
    )
    cross_probe_aggregate_pass = _inventory_all_rows_pass(
        cross_probe_summary,
        inventory_specifications["cross_probe_aggregate"],
        inventory_statuses["cross_probe_aggregate"],
        "gate_pass",
    )
    cross_probe_class_pass = _inventory_all_rows_pass(
        cross_probe_summary_by_class,
        inventory_specifications["cross_probe_by_class"],
        inventory_statuses["cross_probe_by_class"],
        "gate_pass",
    )
    parameter_aggregate_pass = _inventory_all_conditions_pass(
        parameter_sanity_summary,
        inventory_specifications["parameter_sanity_aggregate"],
        inventory_statuses["parameter_sanity_aggregate"],
        condition_columns=["task_id", "model", "method"],
        pass_column="parameter_dependence_established",
    )
    parameter_class_pass = _inventory_all_conditions_pass(
        parameter_sanity_summary_by_class,
        inventory_specifications["parameter_sanity_by_class"],
        inventory_statuses["parameter_sanity_by_class"],
        condition_columns=[
            "task_id",
            "true_class_id",
            "true_class",
            "model",
            "method",
        ],
        pass_column="parameter_dependence_established",
    )
    functional_aggregate_pass = _inventory_all_rows_pass(
        functional_summary,
        inventory_specifications["functional_response_aggregate"],
        inventory_statuses["functional_response_aggregate"],
        "positive_response_vs_displacement_matched_random",
    ) and bool(
        inventory_statuses["functional_monte_carlo_aggregate"][
            "decision_coverage_complete"
        ]
    )
    functional_class_pass = _inventory_all_rows_pass(
        functional_summary_by_class,
        inventory_specifications["functional_response_by_class"],
        inventory_statuses["functional_response_by_class"],
        "positive_response_vs_displacement_matched_random",
    ) and bool(
        inventory_statuses["functional_monte_carlo_by_class"][
            "decision_coverage_complete"
        ]
    )

    paths.tables_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "repeatability_per_flow": repeatability,
        "repeatability_summary": repeatability_summary,
        "repeatability_summary_by_class": repeatability_summary_by_class,
        "cross_model_per_flow": cross_model,
        "cross_model_summary": cross_model_summary,
        "cross_model_summary_by_class": cross_model_summary_by_class,
        "reference_sensitivity_per_flow": reference_sensitivity,
        "reference_sensitivity_summary": reference_sensitivity_summary,
        "reference_sensitivity_summary_by_class": reference_sensitivity_summary_by_class,
        "cross_probe_per_flow": cross_probe,
        "cross_probe_summary": cross_probe_summary,
        "cross_probe_summary_by_class": cross_probe_summary_by_class,
        "parameter_sanity_per_flow": parameter_sanity,
        "parameter_sanity_summary": parameter_sanity_summary,
        "parameter_sanity_summary_by_class": parameter_sanity_summary_by_class,
        "functional_aopc_per_flow": aopc,
        "functional_summary": functional_summary,
        "functional_summary_by_class": functional_summary_by_class,
        "functional_monte_carlo_sensitivity_summary": functional_monte_carlo_sensitivity_summary,
        "functional_monte_carlo_sensitivity_summary_by_class": functional_monte_carlo_sensitivity_summary_by_class,
        "functional_displacement_summary": functional_displacement_summary,
        "functional_displacement_summary_by_class": functional_displacement_summary_by_class,
        "signed_per_flow": signed,
        "signed_coverage": signed_coverage,
        "signed_coverage_by_class": signed_coverage_by_class,
        "signed_summary": signed_summary,
        "signed_summary_by_class": signed_summary_by_class,
        "correct_error_matches": matches,
        "correct_error_per_pair": correct_error,
        "correct_error_summary": correct_error_summary,
        "correct_error_summary_by_class": correct_error_summary_by_class,
        "decision_inventory": decision_inventory,
    }
    if tuple(tables) != AUDIT_SUMMARY_TABLES:
        missing_tables = sorted(set(AUDIT_SUMMARY_TABLES) - set(tables))
        unexpected_tables = sorted(set(tables) - set(AUDIT_SUMMARY_TABLES))
        raise RuntimeError(
            "Audit table inventory differs from its frozen contract: "
            f"missing={missing_tables}, unexpected={unexpected_tables}"
        )
    for name, table in tables.items():
        table.to_csv(paths.tables_root / f"{name}.csv", index=False)
    all_registered_gates_pass = bool(
        repeatability_aggregate_pass
        and repeatability_class_pass
        and cross_model_aggregate_pass
        and cross_model_class_pass
        and cross_probe_aggregate_pass
        and cross_probe_class_pass
        and parameter_aggregate_pass
        and parameter_class_pass
        and functional_aggregate_pass
        and functional_class_pass
    )
    decision = {
        "audit_version": AUDIT_VERSION,
        "status": "complete" if summary_inventory_complete else "incomplete_summary_inventory",
        "inference": CROSSED_SOURCE_SEED_BOOTSTRAP,
        "bootstrap_design": {
            "cluster_relationship": "crossed",
            "source_resampling": (
                "one_global_multinomial_source_draw_shared_across_all_sampled_"
                "seed_occurrences"
            ),
            "fitted_seed_resampling": "equal_weight_with_replacement",
            "lime_run_resampling": (
                "one_shared_whole_panel_run_jackknife_pseudovalue_draw_per_"
                "sampled_seed_occurrence"
            ),
        },
        "confidence_level": float(config["uncertainty"]["confidence_level"]),
        "bootstrap_resamples": int(config["uncertainty"]["bootstrap_resamples"]),
        "table_inventory": list(AUDIT_SUMMARY_TABLES),
        "table_inventory_complete": True,
        "summary_inventory_complete": summary_inventory_complete,
        "inventory_conditions": inventory_statuses,
        "repeatability_all_aggregate_primary_gates_pass": repeatability_aggregate_pass,
        "repeatability_all_class_specific_primary_gates_pass": repeatability_class_pass,
        "repeatability_all_primary_gates_pass": bool(
            repeatability_aggregate_pass and repeatability_class_pass
        ),
        "cross_model_all_aggregate_equivalence_gates_pass": cross_model_aggregate_pass,
        "cross_model_all_class_specific_equivalence_gates_pass": cross_model_class_pass,
        "cross_model_all_equivalence_gates_pass": bool(
            cross_model_aggregate_pass and cross_model_class_pass
        ),
        "cross_probe_all_aggregate_identification_gates_pass": cross_probe_aggregate_pass,
        "cross_probe_all_class_specific_identification_gates_pass": cross_probe_class_pass,
        "cross_probe_all_identification_gates_pass": bool(
            cross_probe_aggregate_pass and cross_probe_class_pass
        ),
        "parameter_dependence_established_all_aggregate_conditions": parameter_aggregate_pass,
        "parameter_dependence_established_all_class_specific_conditions": parameter_class_pass,
        "parameter_dependence_established_all_conditions": bool(
            parameter_aggregate_pass and parameter_class_pass
        ),
        "functional_response_positive_all_aggregate_conditions": functional_aggregate_pass,
        "functional_response_positive_all_class_specific_conditions": functional_class_pass,
        "functional_response_positive_all_conditions": bool(
            functional_aggregate_pass and functional_class_pass
        ),
        "all_registered_evidential_gates_pass": all_registered_gates_pass,
        "correct_vs_error_interpretation": "descriptive_not_causal",
        "admissible_claim_types": _admissible_claim_types(config),
        "admissible_claim_types_are_protocol_whitelist_not_observed_passes": True,
        "prohibited_claims": list(config["evidence_policy"]["prohibited_claims"]),
    }
    decision_path = paths.tables_root / "audit_decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not summary_inventory_complete:
        raise RuntimeError(
            "Audit summary inventory is incomplete; see decision_inventory.csv and "
            "audit_decision.json"
        )
    inputs = artifact_records(summary_input_paths)
    implementation_hash = signature({"files": _audit_implementation_records(paths)})
    stage_signature = signature(
        {
            "audit_version": AUDIT_VERSION,
            "stage": "audit_summaries",
            "config_hash": sha256_file(paths.config_path),
            "training_config_hash": sha256_file(paths.training_config_path),
            "implementation_hash": implementation_hash,
            "inputs": inputs,
        }
    )
    decision["stage_signature"] = stage_signature
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_paths = [paths.tables_root / f"{name}.csv" for name in tables]
    output_paths.append(decision_path)
    write_cache_metadata(
        paths.tables_root / "cache_metadata.json",
        stage="audit_summaries",
        signature_value=stage_signature,
        config_hash=sha256_file(paths.config_path),
        implementation_hash=implementation_hash,
        inputs=inputs,
        outputs=output_paths,
        extra={
            "training_config_hash": sha256_file(paths.training_config_path),
            "rare_classes_retained_descriptively": True,
            "inferential_aggregates_exclude_ineligible_classes": True,
        },
    )
    return tables
