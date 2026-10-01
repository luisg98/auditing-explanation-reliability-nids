"""Cached, read-only diagnostics for leakage-free predictive outputs.

The stage in this module never loads a model and never retrains one.  It first
validates the complete preparation and final-training cache inventory, then
reads the immutable test-prediction tables.  Diagnostics are written only
after every required predictive condition has passed its content-addressed
cache check.

Raw CICIDS2017 labels are not stored in the processed tensors.  When the raw
input bytes recorded by preparation are still available and unchanged, they
can be recovered without ambiguity through ``(source_file, source_row)``.  A
missing or changed raw input disables only the raw-label tables and is recorded
as an explicit lineage blocker; contradictions in recovered labels fail the
whole stage.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from src.leakage_free_tabular.cache import (
    artifact_record,
    artifact_records,
    cache_hit,
    sha256_file,
    signature,
    write_cache_metadata,
)


PIPELINE_VERSION = "leakage_free_tabular_v2"
DIAGNOSTICS_VERSION = 1
ALLOWED_MODELS = ("mlp", "cnn")
FINAL_SEEDS = (42, 43, 44, 45, 46)
IDENTITY_COLUMNS = (
    "sample_order",
    "source_file",
    "source_row",
    "raw_fingerprint",
    "true_class_id",
    "true_class",
)
PREDICTION_COLUMNS = (
    "task_id",
    "dataset",
    "task",
    "model",
    "seed",
    *IDENTITY_COLUMNS,
    "pred_class_id",
    "pred_class",
    "correct",
)
PREPARATION_INVARIANTS = {
    "one_fixed_split_shared_by_all_model_seeds": True,
    "learned_preprocessing_fit_partition": "train_only",
    "exact_duplicate_feature_groups_fit_partition": "train_only",
    "model_coordinates_are_train_unique": True,
    "raw_cross_split_fingerprint_overlap": 0,
    "final_float32_exact_cross_split_overlap": 0,
    "final_float32_round5_cross_split_overlap": 0,
    "all_classes_in_each_final_split": True,
    "representation_filter_used_labels": False,
    "session_independence_claimed": False,
}


@dataclass(frozen=True)
class CacheInventory:
    """Validated paths defining one complete predictive cache snapshot."""

    config: dict[str, Any]
    task_ids: tuple[str, ...]
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    prediction_paths: dict[tuple[str, str, int], Path]
    condition_metadata_paths: tuple[Path, ...]
    preparation_metadata_paths: tuple[Path, ...]
    snapshot_records: tuple[dict[str, Any], ...]
    preparation_decision: dict[str, Any]


class RawLabelLineageUnavailable(RuntimeError):
    """The predictive cache is valid, but its optional raw-label source is not."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Required JSON artifact is unreadable: {path}: {exc!r}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML artifact is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a YAML mapping in {path}")
    return value


def _resolve_record_path(record: Mapping[str, Any], project_root: Path) -> Path:
    value = Path(str(record.get("path", "")))
    return value if value.is_absolute() else project_root / value


def validate_artifact_records(
    records: Iterable[Mapping[str, Any]],
    *,
    project_root: Path,
    required_paths: Iterable[Path] | None = None,
) -> None:
    """Validate a metadata inventory by byte size and SHA-256.

    Preparation records are project-relative, whereas the newer training cache
    records are absolute.  Both representations are accepted, but duplicate
    declarations and missing required paths are rejected.
    """

    observed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("Artifact inventory contains a non-object record")
        path = _resolve_record_path(record, project_root).resolve()
        key = path.as_posix()
        if key in observed:
            raise RuntimeError(f"Artifact inventory repeats a path: {path}")
        observed[key] = record
        if not path.is_file():
            raise FileNotFoundError(f"Cached artifact is missing: {path}")
        expected_size = int(record.get("size_bytes", -1))
        if int(path.stat().st_size) != expected_size:
            raise RuntimeError(f"Cached artifact size differs from metadata: {path}")
        expected_hash = str(record.get("sha256", ""))
        if not expected_hash or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Cached artifact SHA-256 differs from metadata: {path}")
    if not observed:
        raise RuntimeError("Artifact inventory is empty")
    if required_paths is not None:
        required = {path.resolve().as_posix() for path in required_paths}
        missing = sorted(required - set(observed))
        if missing:
            raise RuntimeError(f"Artifact inventory omits required paths: {missing}")


def validate_task_metadata_records(
    decision: Mapping[str, Any],
    *,
    project_root: Path,
    expected_paths: Iterable[Path],
) -> None:
    """Bind the preparation decision to the exact per-task metadata bytes."""

    records = decision.get("task_metadata")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Preparation decision has no task-metadata inventory")
    expected = {path.resolve().as_posix() for path in expected_paths}
    declared = {
        _resolve_record_path(record, project_root).resolve().as_posix()
        for record in records
        if isinstance(record, Mapping)
    }
    if len(records) != len(declared) or declared != expected:
        raise RuntimeError(
            "Preparation decision task-metadata inventory differs from the expected tasks"
        )
    validate_artifact_records(
        records,
        project_root=project_root,
        required_paths=[Path(path) for path in sorted(expected)],
    )


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _preparation_config_signature(config: Mapping[str, Any]) -> str:
    sections = (
        "pipeline_version",
        "dataset",
        "tasks",
        "split",
        "preprocessing",
        "representation_guard",
        "outputs",
    )
    missing = [key for key in sections if key not in config]
    if missing:
        raise RuntimeError(f"Preparation config lacks required sections: {missing}")
    return _stable_hash({key: config[key] for key in sections})


def _relative_artifact_record(path: Path, project_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        reported = resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        reported = resolved.as_posix()
    return {
        "path": reported,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _preparation_implementation_hash(project_root: Path) -> str:
    paths = [
        project_root / "src/leakage_free_tabular/prepare.py",
        project_root / "scripts/prepare_leakage_free_tabular.py",
    ]
    records = [_relative_artifact_record(path, project_root) for path in paths]
    return _stable_hash(records)


def _training_implementation_records(project_root: Path) -> list[dict[str, Any]]:
    return artifact_records(
        [
            project_root / "src/leakage_free_tabular/cache.py",
            project_root / "src/leakage_free_tabular/training.py",
            project_root / "src/models/mlp.py",
            project_root / "src/models/cnn.py",
            project_root / "src/utils/seed.py",
        ]
    )


def _condition_input_paths(processed_dir: Path, lock_path: Path) -> list[Path]:
    paths = [
        processed_dir / "X_train.npy",
        processed_dir / "y_train.npy",
        processed_dir / "X_val.npy",
        processed_dir / "y_val.npy",
        processed_dir / "feature_names.json",
        processed_dir / "feature_groups.json",
        processed_dir / "label_mapping.json",
        processed_dir / "baseline.npy",
        processed_dir / "preprocessor.joblib",
        processed_dir / "preprocessor_metadata.json",
        processed_dir / "split_validation.csv",
        processed_dir / "cache_metadata.json",
    ]
    for split in ("train", "val"):
        for stem in (
            "source_file",
            "source_row",
            "raw_fingerprint",
            "tensor_fingerprint_exact",
            "tensor_fingerprint_round5",
        ):
            paths.append(processed_dir / f"{stem}_{split}.npy")
    for name in (
        "X_test.npy",
        "y_test.npy",
        "source_file_test.npy",
        "source_row_test.npy",
        "raw_fingerprint_test.npy",
        "tensor_fingerprint_exact_test.npy",
        "tensor_fingerprint_round5_test.npy",
    ):
        paths.append(processed_dir / name)
    paths.append(lock_path)
    return paths


def _condition_output_paths(out_dir: Path) -> list[Path]:
    return [
        out_dir / "model.keras",
        out_dir / "training_history.csv",
        out_dir / "test_predictions.csv",
        out_dir / "metrics.csv",
        out_dir / "confusion_matrix.csv",
        out_dir / "per_class_metrics.csv",
        out_dir / "class_weights.csv",
    ]


def _require_content_addressed_paths(
    required_paths: Iterable[Path],
    records: Iterable[Mapping[str, Any]],
) -> None:
    """Confirm each required file's bytes are certified by some record.

    Unlike validate_artifact_records's required_paths check, this matches by
    (size, sha256) rather than by the record's own declared path string — a
    prepare-stage output copied verbatim into a second location (e.g. a
    dataset prepared in its own isolated run and copied into a processed
    root shared with other datasets) is still the exact certified bytes even
    though no record names that second location.
    """

    certified = {(int(r.get("size_bytes", -1)), str(r.get("sha256", ""))) for r in records}
    missing = []
    for path in required_paths:
        if not path.is_file():
            missing.append(str(path))
            continue
        if (path.stat().st_size, sha256_file(path)) not in certified:
            missing.append(str(path))
    if missing:
        raise RuntimeError(f"Required artifacts are not certified by this metadata: {missing}")


def _validate_one_preparation_group(
    *,
    project_root: Path,
    preparation_config_path: Path,
    processed_root: Path,
    group_task_ids: tuple[str, ...],
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Validate one prepare-stage decision against the tasks it actually produced."""

    config = _read_yaml(preparation_config_path)
    if str(config.get("pipeline_version")) != PIPELINE_VERSION:
        raise RuntimeError("Preparation config is not the v2 leakage-free pipeline")
    expected_config_signature = _preparation_config_signature(config)
    expected_implementation_hash = _preparation_implementation_hash(project_root)
    decision_results_root = project_root / str(config["outputs"]["results_root"])
    decision_path = decision_results_root / "decision.json"
    decision = _read_json(decision_path)
    if (
        decision.get("pipeline_version") != PIPELINE_VERSION
        or decision.get("stage") != "leakage_free_data_preparation"
        or decision.get("status") != "complete"
    ):
        raise RuntimeError("Preparation decision does not certify a complete v2 stage")
    if decision.get("config_signature") != expected_config_signature:
        raise RuntimeError("Preparation cache was produced from a different data config")
    if decision.get("implementation_hash") != expected_implementation_hash:
        raise RuntimeError("Preparation cache was produced by a different implementation")
    validate_artifact_records(decision.get("outputs", []), project_root=project_root)

    # Preparation metadata may have been copied from a config-declared
    # results_root that differs from the shared processed_root every stage
    # downstream of preparation actually reads from (e.g. a dataset prepared
    # in an isolated staging run and then copied into the shared processed
    # root alongside other datasets' tasks). validate_task_metadata_records
    # binds the decision to the exact paths it declares, so it is checked
    # against the decision's own results_root; every other read uses the
    # shared processed_root, which validate_artifact_records below confirms
    # is byte-identical to what the decision certifies.
    declared_metadata_paths = [
        decision_results_root / "processed" / task_id / "cache_metadata.json"
        for task_id in group_task_ids
    ]
    validate_task_metadata_records(
        decision,
        project_root=project_root,
        expected_paths=declared_metadata_paths,
    )
    metadata_paths = [
        processed_root / task_id / "cache_metadata.json" for task_id in group_task_ids
    ]
    task_signatures: set[str] = set()
    for task_id, metadata_path in zip(group_task_ids, metadata_paths):
        processed_dir = processed_root / task_id
        metadata = _read_json(metadata_path)
        if (
            metadata.get("pipeline_version") != PIPELINE_VERSION
            or metadata.get("stage") != "leakage_free_data_preparation"
            or metadata.get("status") != "complete"
            or metadata.get("task_id") != task_id
        ):
            raise RuntimeError(f"Invalid preparation metadata for task {task_id}")
        if metadata.get("config_signature") != expected_config_signature:
            raise RuntimeError(f"Preparation config signature differs for task {task_id}")
        if metadata.get("implementation_hash") != expected_implementation_hash:
            raise RuntimeError(f"Preparation implementation differs for task {task_id}")
        failed = {
            key: metadata.get("invariants", {}).get(key)
            for key, expected in PREPARATION_INVARIANTS.items()
            if metadata.get("invariants", {}).get(key) != expected
        }
        if failed:
            raise RuntimeError(f"Preparation invariants failed for {task_id}: {failed}")
        required = [
            processed_dir / "source_file_test.npy",
            processed_dir / "source_row_test.npy",
            processed_dir / "y_test.npy",
            processed_dir / "raw_fingerprint_test.npy",
            processed_dir / "label_mapping.json",
            processed_dir / "split_validation.csv",
        ]
        outputs = metadata.get("outputs", [])
        validate_artifact_records(outputs, project_root=project_root)
        _require_content_addressed_paths(required, outputs)
        task_signatures.add(str(metadata.get("signature", "")))
    if task_signatures != {str(decision.get("signature", ""))}:
        raise RuntimeError("Preparation task signatures differ from the stage decision")
    return decision, tuple(metadata_paths)


def _validate_preparation_cache(
    *,
    project_root: Path,
    preparation_config_path: Path,
    results_root: Path,
    task_ids: tuple[str, ...],
    task_preparation_overrides: Mapping[str, Path] | None = None,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Validate every task's preparation provenance, grouped by originating config.

    Every task_id is normally produced by the same `preparation_config_path`
    (one prepare_all() run covering all of that dataset's tasks). A task
    listed in `task_preparation_overrides` was instead prepared by its own
    config/run (a different dataset entirely) and is validated against that
    config's own decision independently, rather than assuming one shared
    preparation decision covers every task.
    """

    overrides = dict(task_preparation_overrides or {})
    processed_root = results_root / "processed"
    groups: dict[Path, list[str]] = {}
    for task_id in task_ids:
        groups.setdefault(overrides.get(task_id, preparation_config_path), []).append(task_id)

    primary_decision: dict[str, Any] | None = None
    all_metadata_paths: list[Path] = []
    for config_path, group_task_ids in sorted(groups.items(), key=lambda item: str(item[0])):
        decision, metadata_paths = _validate_one_preparation_group(
            project_root=project_root,
            preparation_config_path=config_path,
            processed_root=processed_root,
            group_task_ids=tuple(sorted(group_task_ids)),
        )
        all_metadata_paths.extend(metadata_paths)
        if config_path == preparation_config_path:
            primary_decision = decision
    if primary_decision is None:
        raise RuntimeError("No task uses the primary preparation config")
    return primary_decision, tuple(all_metadata_paths)


def validate_cache_inventory(
    *,
    project_root: Path,
    training_config_path: Path,
    preparation_config_path: Path,
    task_preparation_overrides: Mapping[str, Path] | None = None,
) -> CacheInventory:
    """Validate every frozen condition before any prediction CSV is opened."""

    config = _read_yaml(training_config_path)
    if config.get("pipeline_id") != "leakage_free_tabular_training_v2":
        raise RuntimeError("Training config is not the frozen v2 pipeline")
    task_ids = tuple(str(value) for value in config.get("tasks", {}))
    models = tuple(str(value) for value in config.get("models", []))
    seeds = tuple(int(value) for value in config.get("final_seeds", []))
    if not task_ids:
        raise RuntimeError("Training config declares no tasks")
    if models != ALLOWED_MODELS or seeds != FINAL_SEEDS:
        raise RuntimeError(
            f"Frozen scope differs: models={models}, seeds={seeds}; "
            f"expected {ALLOWED_MODELS}, {FINAL_SEEDS}"
        )
    results_root = project_root / str(config["paths"]["results_root"])
    processed_root = project_root / str(config["paths"]["processed_root"])
    decision_path = results_root / "training_decision.json"
    decision = _read_json(decision_path)
    expected_conditions = len(task_ids) * len(models) * len(seeds)
    if (
        decision.get("pipeline_version") != PIPELINE_VERSION
        or decision.get("status") != "complete"
        or not bool(decision.get("full_frozen_scope"))
        or int(decision.get("final_condition_count", -1)) != expected_conditions
        or int(decision.get("expected_final_condition_count", -1)) != expected_conditions
        or tuple(decision.get("tasks", [])) != task_ids
        or tuple(decision.get("models", [])) != models
        or tuple(int(value) for value in decision.get("final_seeds", [])) != seeds
    ):
        raise RuntimeError("Training decision does not certify the complete frozen v2 inventory")
    config_hash = sha256_file(training_config_path)
    if decision.get("config_sha256") != config_hash:
        raise RuntimeError("Training decision config hash differs from the current config")
    implementations = _training_implementation_records(project_root)
    implementation_hash = signature({"files": implementations})
    if decision.get("implementation_hash") != implementation_hash:
        raise RuntimeError("Training decision implementation hash differs from current code")

    preparation_decision, preparation_metadata_paths = _validate_preparation_cache(
        project_root=project_root,
        preparation_config_path=preparation_config_path,
        results_root=results_root,
        task_ids=task_ids,
        task_preparation_overrides=task_preparation_overrides,
    )
    prediction_paths: dict[tuple[str, str, int], Path] = {}
    metadata_paths: list[Path] = []
    snapshot_records: list[dict[str, Any]] = []
    # Test/train tensors are identical across the five final seeds.  Hash them
    # once per task/model selection lock rather than re-reading the same large
    # arrays for every seed.
    condition_input_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task_id, model, seed in itertools.product(task_ids, models, seeds):
        processed_dir = processed_root / task_id
        lock_path = results_root / "model_selection" / task_id / model / "selected_config.lock.json"
        out_dir = results_root / "final" / task_id / model / f"seed_{seed}"
        metadata_path = out_dir / "cache_metadata.json"
        expected_outputs = _condition_output_paths(out_dir)
        input_key = (task_id, model)
        if input_key not in condition_input_records:
            condition_input_records[input_key] = artifact_records(
                _condition_input_paths(processed_dir, lock_path)
            )
        input_records = condition_input_records[input_key]
        expected_signature = signature(
            {
                "pipeline_version": PIPELINE_VERSION,
                "stage": "final_predictive",
                "config_hash": config_hash,
                "implementation_hash": implementation_hash,
                "inputs": input_records,
                "task_id": task_id,
                "model": model,
                "seed": int(seed),
            }
        )
        hit, reason = cache_hit(
            metadata_path,
            expected_signature=expected_signature,
            expected_outputs=expected_outputs,
        )
        if not hit:
            raise RuntimeError(
                f"Final predictive cache is invalid for {task_id}/{model}/seed_{seed}: {reason}"
            )
        metadata = _read_json(metadata_path)
        if (
            metadata.get("pipeline_version") != PIPELINE_VERSION
            or metadata.get("stage") != "final_predictive"
            or metadata.get("task_id") != task_id
            or metadata.get("model") != model
            or int(metadata.get("seed", -1)) != seed
            or metadata.get("threshold_fitted_on") != "validation"
            or not bool(metadata.get("test_evaluated_once_after_training"))
        ):
            raise RuntimeError(f"Final predictive metadata contract failed for {task_id}/{model}/{seed}")
        prediction_path = out_dir / "test_predictions.csv"
        prediction_paths[(task_id, model, seed)] = prediction_path
        metadata_paths.append(metadata_path)
        snapshot_records.extend([artifact_record(metadata_path), artifact_record(prediction_path)])
    return CacheInventory(
        config=config,
        task_ids=task_ids,
        models=models,
        seeds=seeds,
        prediction_paths=prediction_paths,
        condition_metadata_paths=tuple(metadata_paths),
        preparation_metadata_paths=preparation_metadata_paths,
        snapshot_records=tuple(snapshot_records),
        preparation_decision=preparation_decision,
    )


def _coerce_bool(values: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.lower()
    valid = normalized.isin({"true", "false", "1", "0"})
    if not bool(valid.all()):
        raise RuntimeError(f"Column {name!r} contains non-Boolean values")
    return normalized.isin({"true", "1"})


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    task_id: str | None = None,
    model: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Validate and canonicalize one immutable condition prediction table."""

    missing = sorted(set(PREDICTION_COLUMNS) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Prediction table lacks required columns: {missing}")
    if frame.empty:
        raise RuntimeError("Prediction table is empty")
    out = frame.copy()
    out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)
    for column in ("sample_order", "source_row", "true_class_id", "pred_class_id"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype(np.int64)
    out["correct"] = _coerce_bool(out["correct"], name="correct")
    expected = {"task_id": task_id, "model": model, "seed": seed}
    for column, value in expected.items():
        if value is None:
            continue
        observed = set(out[column].tolist())
        if observed != {value}:
            raise RuntimeError(f"Prediction table {column} differs: expected {value!r}, got {observed}")
    if out["sample_order"].duplicated().any():
        raise RuntimeError("Prediction table repeats sample_order")
    if out.duplicated(["source_file", "source_row"]).any():
        raise RuntimeError("Prediction table repeats a source-file/source-row identity")
    expected_correct = out["true_class_id"].to_numpy() == out["pred_class_id"].to_numpy()
    if not np.array_equal(out["correct"].to_numpy(dtype=bool), expected_correct):
        raise RuntimeError("Prediction table correctness column contradicts class IDs")
    if out.loc[:, list(IDENTITY_COLUMNS)].isna().any().any():
        raise RuntimeError("Prediction table contains a missing sample identity")
    return out.sort_values("sample_order", kind="stable").reset_index(drop=True)


def load_prediction_snapshot(inventory: CacheInventory) -> pd.DataFrame:
    """Read predictions only after validation, then verify snapshot bytes again."""

    frames: list[pd.DataFrame] = []
    task_identity: dict[str, pd.DataFrame] = {}
    for (task_id, model, seed), path in inventory.prediction_paths.items():
        frame = validate_prediction_frame(
            pd.read_csv(path),
            task_id=task_id,
            model=model,
            seed=seed,
        )
        identity = frame.loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True)
        if task_id not in task_identity:
            task_identity[task_id] = identity
        elif not identity.equals(task_identity[task_id]):
            raise RuntimeError(f"Test identities differ across predictive conditions for {task_id}")
        frames.append(frame)
    for record in inventory.snapshot_records:
        path = Path(str(record["path"]))
        current = artifact_record(path)
        if (
            current["size_bytes"] != int(record["size_bytes"])
            or current["sha256"] != str(record["sha256"])
        ):
            raise RuntimeError(f"Predictive cache changed while diagnostics were reading it: {path}")
    return pd.concat(frames, ignore_index=True)


def _class_metrics(frame: pd.DataFrame, class_ids: np.ndarray) -> dict[str, float]:
    y_true = frame["true_class_id"].to_numpy(dtype=int)
    y_pred = frame["pred_class_id"].to_numpy(dtype=int)
    recalls: list[float] = []
    for class_id in class_ids:
        mask = y_true == int(class_id)
        if mask.any():
            recalls.append(float(np.mean(y_pred[mask] == int(class_id))))
    attack = frame["true_class"].astype(str).eq("ATTACK").to_numpy()
    benign = frame["true_class"].astype(str).eq("BENIGN").to_numpy()
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy_present_classes": float(np.mean(recalls)) if recalls else np.nan,
        "macro_f1_all_task_classes": float(
            f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=class_ids, average="weighted", zero_division=0)
        ),
        "attack_recall": (
            float(np.mean(y_pred[attack] == y_true[attack])) if attack.any() else np.nan
        ),
        "benign_false_positive_rate": (
            float(np.mean(y_pred[benign] != y_true[benign])) if benign.any() else np.nan
        ),
    }


def per_source_file_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute capture/source-file metrics for every task/model/seed."""

    rows: list[dict[str, Any]] = []
    group_columns = ["task_id", "dataset", "task", "model", "seed"]
    for condition_key, condition in predictions.groupby(group_columns, sort=True, observed=True):
        class_ids = np.sort(condition["true_class_id"].unique().astype(int))
        for source_file, group in condition.groupby("source_file", sort=True, observed=True):
            values = _class_metrics(group, class_ids)
            rows.append(
                {
                    **dict(zip(group_columns, condition_key)),
                    "source_file": str(source_file),
                    "test_support": int(len(group)),
                    "correct_count": int(group["correct"].sum()),
                    "class_count_present": int(group["true_class_id"].nunique()),
                    "attack_support": int(group["true_class"].astype(str).eq("ATTACK").sum()),
                    "benign_support": int(group["true_class"].astype(str).eq("BENIGN").sum()),
                    **values,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["task_id", "model", "seed", "source_file"], kind="stable"
    ).reset_index(drop=True)


def source_aggregate_estimands(
    predictions: pd.DataFrame,
    source_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Contrast row-pooled, source-unweighted, and source-weighted estimands."""

    rows: list[dict[str, Any]] = []
    keys = ["task_id", "dataset", "task", "model", "seed"]
    metric_specs = {
        "accuracy": "test_support",
        "balanced_accuracy_present_classes": "test_support",
        "macro_f1_all_task_classes": "test_support",
        "weighted_f1": "test_support",
        "attack_recall": "attack_support",
        "benign_false_positive_rate": "benign_support",
    }
    for condition_key, condition in predictions.groupby(keys, sort=True, observed=True):
        selector = np.ones(len(source_metrics), dtype=bool)
        for column, value in zip(keys, condition_key):
            selector &= source_metrics[column].to_numpy() == value
        sources = source_metrics.loc[selector].copy()
        class_ids = np.sort(condition["true_class_id"].unique().astype(int))
        pooled = _class_metrics(condition, class_ids)
        for metric, weight_column in metric_specs.items():
            eligible = sources[metric].notna() & sources[weight_column].gt(0)
            values = sources.loc[eligible, metric].to_numpy(dtype=float)
            weights = sources.loc[eligible, weight_column].to_numpy(dtype=float)
            common = {
                **dict(zip(keys, condition_key)),
                "metric": metric,
                "support_rows": int(weights.sum()),
                "eligible_source_count": int(eligible.sum()),
            }
            rows.append(
                {
                    **common,
                    "estimand": "row_pooled",
                    "value": float(pooled[metric]),
                    "weight_basis": "individual test rows in the declared population",
                }
            )
            rows.append(
                {
                    **common,
                    "estimand": "source_unweighted",
                    "value": float(np.mean(values)) if len(values) else np.nan,
                    "weight_basis": "each eligible source file receives equal weight",
                }
            )
            rows.append(
                {
                    **common,
                    "estimand": "source_support_weighted",
                    "value": (
                        float(np.average(values, weights=weights))
                        if len(values) and float(weights.sum()) > 0
                        else np.nan
                    ),
                    "weight_basis": f"source files weighted by {weight_column}",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["task_id", "model", "seed", "metric", "estimand"], kind="stable"
    ).reset_index(drop=True)


def identity_weighting_sensitivity(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare row weighting with equal weight per distinct raw identity.

    Preparation keeps within-split multiplicities of identical raw feature
    vectors.  Row-pooled performance therefore gives a fingerprint weight
    proportional to its multiplicity.  This sensitivity collapses each
    ``raw_fingerprint`` to one equally weighted identity.  A fingerprint may
    be collapsed only when its true and predicted classes are unambiguous in
    the condition.
    """

    rows: list[dict[str, Any]] = []
    keys = ["task_id", "dataset", "task", "model", "seed"]
    metrics = (
        "accuracy",
        "balanced_accuracy_present_classes",
        "macro_f1_all_task_classes",
        "weighted_f1",
        "attack_recall",
        "benign_false_positive_rate",
    )
    for condition_key, condition in predictions.groupby(keys, sort=True, observed=True):
        ambiguity = condition.groupby("raw_fingerprint", sort=False, observed=True).agg(
            true_class_id_count=("true_class_id", "nunique"),
            true_class_count=("true_class", "nunique"),
            pred_class_id_count=("pred_class_id", "nunique"),
            pred_class_count=("pred_class", "nunique"),
        )
        inconsistent = ambiguity[(ambiguity > 1).any(axis=1)]
        if not inconsistent.empty:
            examples = inconsistent.index.astype(str).tolist()[:5]
            scope = "/".join(str(value) for value in condition_key)
            raise RuntimeError(
                "A raw fingerprint has inconsistent true/predicted classes in "
                f"condition {scope}; examples={examples}"
            )
        multiplicity = condition.groupby("raw_fingerprint", sort=False, observed=True).size()
        identities = condition.drop_duplicates("raw_fingerprint", keep="first").copy()
        class_ids = np.sort(condition["true_class_id"].unique().astype(int))
        row_metrics = _class_metrics(condition, class_ids)
        identity_metrics = _class_metrics(identities, class_ids)
        common = {
            **dict(zip(keys, condition_key)),
            "row_support": int(len(condition)),
            "raw_identity_support": int(len(identities)),
            "duplicate_row_excess": int(len(condition) - len(identities)),
            "raw_identities_with_multiplicity_gt_1": int((multiplicity > 1).sum()),
            "maximum_raw_identity_multiplicity": int(multiplicity.max()),
        }
        for metric in metrics:
            row_value = float(row_metrics[metric])
            identity_value = float(identity_metrics[metric])
            if metric == "attack_recall":
                row_metric_support = int(condition["true_class"].astype(str).eq("ATTACK").sum())
                identity_metric_support = int(
                    identities["true_class"].astype(str).eq("ATTACK").sum()
                )
            elif metric == "benign_false_positive_rate":
                row_metric_support = int(condition["true_class"].astype(str).eq("BENIGN").sum())
                identity_metric_support = int(
                    identities["true_class"].astype(str).eq("BENIGN").sum()
                )
            else:
                row_metric_support = int(len(condition))
                identity_metric_support = int(len(identities))
            rows.extend(
                [
                    {
                        **common,
                        "metric": metric,
                        "estimand": "row_pooled",
                        "value": row_value,
                        "difference_vs_row_pooled": 0.0,
                        "effective_unit_count": row_metric_support,
                        "weighting_definition": "equal weight per retained test row",
                    },
                    {
                        **common,
                        "metric": metric,
                        "estimand": "raw_identity_unweighted",
                        "value": identity_value,
                        "difference_vs_row_pooled": identity_value - row_value,
                        "effective_unit_count": identity_metric_support,
                        "weighting_definition": (
                            "equal weight per distinct raw_fingerprint after requiring "
                            "class-consistent duplicates"
                        ),
                    },
                ]
            )
    return pd.DataFrame(rows).sort_values(
        ["task_id", "model", "seed", "metric", "estimand"], kind="stable"
    ).reset_index(drop=True)


def _agreement_counts(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    if not left.loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True).equals(
        right.loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True)
    ):
        raise RuntimeError("Cannot pair predictions with different sample identities")
    left_pred = left["pred_class_id"].to_numpy(dtype=int)
    right_pred = right["pred_class_id"].to_numpy(dtype=int)
    left_correct = left["correct"].to_numpy(dtype=bool)
    right_correct = right["correct"].to_numpy(dtype=bool)
    support = len(left)

    def count_rate(mask: np.ndarray, name: str) -> dict[str, Any]:
        count = int(np.sum(mask))
        return {f"{name}_count": count, f"{name}_rate": float(count / support)}

    return {
        "test_support": int(support),
        **count_rate(left_pred == right_pred, "prediction_agreement"),
        **count_rate(left_pred != right_pred, "prediction_disagreement"),
        **count_rate(left_correct & right_correct, "correct_both"),
        **count_rate(~left_correct & ~right_correct, "wrong_both"),
        **count_rate(left_correct & ~right_correct, "left_only_correct"),
        **count_rate(~left_correct & right_correct, "right_only_correct"),
    }


def architecture_prediction_agreement(
    predictions: pd.DataFrame,
    *,
    left_model: str = "mlp",
    right_model: str = "cnn",
) -> pd.DataFrame:
    """Pair MLP and CNN decisions within task and final seed."""

    rows: list[dict[str, Any]] = []
    for (task_id, dataset, task, seed), condition in predictions.groupby(
        ["task_id", "dataset", "task", "seed"], sort=True, observed=True
    ):
        observed = set(condition["model"].astype(str))
        if observed != {left_model, right_model}:
            raise RuntimeError(f"Architecture pairing scope differs for {task_id}/seed_{seed}: {observed}")
        left = condition[condition["model"] == left_model].sort_values("sample_order").reset_index(drop=True)
        right = condition[condition["model"] == right_model].sort_values("sample_order").reset_index(drop=True)
        counts = _agreement_counts(left, right)
        rows.append(
            {
                "task_id": task_id,
                "dataset": dataset,
                "task": task,
                "seed": int(seed),
                "left_model": left_model,
                "right_model": right_model,
                **counts,
            }
        )
    return pd.DataFrame(rows).sort_values(["task_id", "seed"], kind="stable").reset_index(drop=True)


def cross_seed_prediction_agreement(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return seed-pair agreement and an all-seed summary per model/task."""

    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (task_id, dataset, task, model), condition in predictions.groupby(
        ["task_id", "dataset", "task", "model"], sort=True, observed=True
    ):
        seeds = sorted(int(value) for value in condition["seed"].unique())
        if len(seeds) < 2:
            raise RuntimeError(f"Cross-seed agreement requires at least two seeds: {task_id}/{model}")
        by_seed = {
            seed: condition[condition["seed"] == seed].sort_values("sample_order").reset_index(drop=True)
            for seed in seeds
        }
        pair_values: list[float] = []
        for seed_a, seed_b in itertools.combinations(seeds, 2):
            counts = _agreement_counts(by_seed[seed_a], by_seed[seed_b])
            pair_values.append(float(counts["prediction_agreement_rate"]))
            pair_rows.append(
                {
                    "task_id": task_id,
                    "dataset": dataset,
                    "task": task,
                    "model": model,
                    "seed_a": int(seed_a),
                    "seed_b": int(seed_b),
                    **counts,
                }
            )
        reference = by_seed[seeds[0]]
        predictions_matrix = np.column_stack(
            [by_seed[seed]["pred_class_id"].to_numpy(dtype=int) for seed in seeds]
        )
        correctness_matrix = np.column_stack(
            [by_seed[seed]["correct"].to_numpy(dtype=bool) for seed in seeds]
        )
        unanimous = np.all(predictions_matrix == predictions_matrix[:, [0]], axis=1)
        summary_rows.append(
            {
                "task_id": task_id,
                "dataset": dataset,
                "task": task,
                "model": model,
                "seed_count": int(len(seeds)),
                "seed_pair_count": int(len(pair_values)),
                "test_support": int(len(reference)),
                "mean_pairwise_prediction_agreement": float(np.mean(pair_values)),
                "minimum_pairwise_prediction_agreement": float(np.min(pair_values)),
                "maximum_pairwise_prediction_agreement": float(np.max(pair_values)),
                "all_seed_unanimous_prediction_count": int(unanimous.sum()),
                "all_seed_unanimous_prediction_rate": float(np.mean(unanimous)),
                "all_seed_correct_count": int(np.all(correctness_matrix, axis=1).sum()),
                "all_seed_correct_rate": float(np.mean(np.all(correctness_matrix, axis=1))),
                "any_seed_correct_count": int(np.any(correctness_matrix, axis=1).sum()),
                "any_seed_correct_rate": float(np.mean(np.any(correctness_matrix, axis=1))),
                "pair_summary_weighting": "unweighted arithmetic mean across seed pairs",
                "sample_summary_weighting": "each test row receives equal weight",
            }
        )
    pairwise = pd.DataFrame(pair_rows).sort_values(
        ["task_id", "model", "seed_a", "seed_b"], kind="stable"
    ).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["task_id", "model"], kind="stable"
    ).reset_index(drop=True)
    return pairwise, summary


def _normalize_label(value: object) -> str:
    text = str(value).strip().replace("\ufffd", " ")
    return " ".join(text.split())


def _attack_family(value: object, mapping: Mapping[str, Any]) -> str:
    clean = _normalize_label(value)
    if clean in mapping:
        return str(mapping[clean])
    low = clean.lower()
    if "benign" in low:
        return "BENIGN"
    if "ddos" in low:
        return "DDoS"
    if low.startswith("dos") or "slowloris" in low or "slowhttptest" in low:
        return "DoS"
    if "portscan" in low:
        return "PortScan"
    if "patator" in low or "brute force" in low or "ssh" in low or "ftp" in low:
        return "BruteForce"
    if "web attack" in low or "sql injection" in low or "xss" in low:
        return "WebAttack"
    if "infiltration" in low:
        return "Infiltration"
    if "heartbleed" in low:
        return "Heartbleed"
    if "bot" in low:
        return "Bot"
    return clean.replace(" ", "_")


def _raw_input_records_by_name(
    preparation_decision: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Mapping[str, Any]]:
    records = preparation_decision.get("raw_inputs", [])
    if not isinstance(records, list) or not records:
        raise RawLabelLineageUnavailable("preparation decision has no raw-input inventory")
    by_name: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise RawLabelLineageUnavailable("preparation raw-input inventory is malformed")
        path = _resolve_record_path(record, project_root)
        if path.name in by_name:
            raise RawLabelLineageUnavailable(
                f"raw-input inventory has a non-unique source basename: {path.name}"
            )
        by_name[path.name] = record
    return by_name


def recover_raw_label_lineage(
    binary_identity: pd.DataFrame,
    *,
    project_root: Path,
    preparation_config: Mapping[str, Any],
    preparation_decision: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[Path]]:
    """Recover normalized raw labels from byte-validated CICIDS inputs."""

    dataset = preparation_config.get("dataset", {})
    label_column = str(dataset.get("label_column", ""))
    if not label_column:
        raise RawLabelLineageUnavailable("preparation config has no dataset.label_column")
    records_by_name = _raw_input_records_by_name(preparation_decision, project_root=project_root)
    parts: list[pd.DataFrame] = []
    used_paths: list[Path] = []
    for source_file, rows in binary_identity.groupby("source_file", sort=True, observed=True):
        name = str(source_file)
        record = records_by_name.get(name)
        if record is None:
            raise RawLabelLineageUnavailable(
                f"source file {name!r} is absent from the preparation raw-input inventory"
            )
        path = _resolve_record_path(record, project_root).resolve()
        if not path.is_file():
            raise RawLabelLineageUnavailable(f"recorded raw source is unavailable: {path}")
        if int(path.stat().st_size) != int(record.get("size_bytes", -1)):
            raise RawLabelLineageUnavailable(f"recorded raw source size has changed: {path}")
        if sha256_file(path) != str(record.get("sha256", "")):
            raise RawLabelLineageUnavailable(f"recorded raw source SHA-256 has changed: {path}")
        try:
            raw = pd.read_parquet(path, columns=[label_column])
        except Exception as exc:  # optional lineage may lack its parquet engine/source
            raise RawLabelLineageUnavailable(
                f"could not read raw labels from {path}: {type(exc).__name__}: {exc}"
            ) from exc
        requested = rows["source_row"].to_numpy(dtype=np.int64)
        if np.any(requested < 0) or (len(requested) and int(requested.max()) >= len(raw)):
            raise RuntimeError(f"Processed source-row provenance is outside raw source bounds: {path}")
        part = rows.loc[:, list(IDENTITY_COLUMNS)].copy()
        part["raw_label"] = [
            _normalize_label(value) for value in raw.iloc[requested][label_column].tolist()
        ]
        parts.append(part)
        used_paths.append(path)
    lineage = pd.concat(parts, ignore_index=True).sort_values("sample_order").reset_index(drop=True)
    benign = {
        _normalize_label(value).lower() for value in dataset.get("benign_labels", ["BENIGN"])
    }
    recovered_binary = np.where(
        lineage["raw_label"].astype(str).str.lower().isin(benign), "BENIGN", "ATTACK"
    )
    if not np.array_equal(recovered_binary, lineage["true_class"].astype(str).to_numpy()):
        mismatch = int(np.sum(recovered_binary != lineage["true_class"].astype(str).to_numpy()))
        raise RuntimeError(
            f"Recovered raw labels contradict {mismatch} processed binary test labels"
        )
    family_map = dataset.get("attack_family_map", {}) or {}
    lineage["attack_family"] = [
        _attack_family(value, family_map) for value in lineage["raw_label"]
    ]
    return lineage, used_paths


RAW_RECALL_COLUMNS = [
    "task_id",
    "dataset",
    "task",
    "model",
    "seed",
    "subgroup_type",
    "subgroup",
    "attack_support",
    "detected_attack_count",
    "missed_attack_count",
    "attack_recall",
    "estimand",
]


def raw_attack_recall_tables(
    predictions: pd.DataFrame,
    lineage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute row-weighted binary attack recall by raw label and family."""

    # Scoped to task_id, not the generic "binary" task kind: `lineage` only
    # covers cicids2017_binary identities (see build_diagnostic_tables), and
    # ton_iot_binary also reports task == "binary" but has no raw CICIDS2017
    # attack-label lineage to join against.
    binary = predictions[predictions["task_id"].astype(str) == "cicids2017_binary"].copy()
    if binary.empty:
        raise RuntimeError("No binary predictive conditions are available")
    join_columns = list(IDENTITY_COLUMNS)
    binary = binary.merge(
        lineage[join_columns + ["raw_label", "attack_family"]],
        on=join_columns,
        how="left",
        validate="many_to_one",
    )
    if binary[["raw_label", "attack_family"]].isna().any().any():
        raise RuntimeError("Raw-label lineage does not cover every binary test prediction")
    attacks = binary[binary["true_class"].astype(str) == "ATTACK"].copy()
    if attacks.empty:
        raise RuntimeError("Binary test snapshot contains no attack rows")
    keys = ["task_id", "dataset", "task", "model", "seed"]

    def summarize(column: str, subgroup_type: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for group_key, group in attacks.groupby(keys + [column], sort=True, observed=True):
            detected = group["pred_class"].astype(str).eq("ATTACK")
            rows.append(
                {
                    **dict(zip(keys, group_key[:-1])),
                    "subgroup_type": subgroup_type,
                    "subgroup": str(group_key[-1]),
                    "attack_support": int(len(group)),
                    "detected_attack_count": int(detected.sum()),
                    "missed_attack_count": int((~detected).sum()),
                    "attack_recall": float(detected.mean()),
                    "estimand": "row-weighted recall within this true-attack subgroup",
                }
            )
        return pd.DataFrame(rows, columns=RAW_RECALL_COLUMNS).sort_values(
            ["task_id", "model", "seed", "subgroup"], kind="stable"
        ).reset_index(drop=True)

    return summarize("raw_label", "raw_cicids_label"), summarize(
        "attack_family", "configured_attack_family"
    )


def estimand_definitions() -> pd.DataFrame:
    """Machine-readable definitions for every weighting convention."""

    return pd.DataFrame(
        [
            {
                "estimand_id": "row_pooled",
                "unit": "test row",
                "weighting": "equal weight per test row",
                "interpretation": "Performance for a random retained test flow; large source files contribute more.",
            },
            {
                "estimand_id": "source_unweighted",
                "unit": "source file",
                "weighting": "equal weight per eligible source file",
                "interpretation": "Mean source/capture performance, irrespective of its retained test size.",
            },
            {
                "estimand_id": "source_support_weighted",
                "unit": "source file",
                "weighting": "source metric weighted by its eligible row support",
                "interpretation": "A decomposition of row-weighted performance for decomposable metrics; not generally equal to the pooled nonlinear metric.",
            },
            {
                "estimand_id": "raw_identity_unweighted",
                "unit": "distinct raw feature fingerprint",
                "weighting": "equal weight per distinct raw_fingerprint",
                "interpretation": "Sensitivity to within-test duplicate multiplicity; repeated identical raw feature vectors contribute once after class-consistency validation.",
            },
            {
                "estimand_id": "paired_architecture_row",
                "unit": "paired test row",
                "weighting": "equal weight per row, paired within task and seed",
                "interpretation": "Agreement and joint correctness of MLP and tabular ResNet on identical flows.",
            },
            {
                "estimand_id": "cross_seed_pair",
                "unit": "paired test row within a seed pair",
                "weighting": "equal weight per row; summary gives equal weight to every seed pair",
                "interpretation": "Sensitivity of predicted class to the final training seed.",
            },
            {
                "estimand_id": "raw_attack_subgroup_recall",
                "unit": "true attack test row",
                "weighting": "equal weight per attack row within label/family subgroup",
                "interpretation": "Probability that a retained attack flow in the subgroup is detected as ATTACK.",
            },
        ]
    )


def _raw_lineage_status(
    *,
    status: str,
    blocker: str,
    requested_sources: int,
    validated_sources: int,
    test_rows: int,
    matched_rows: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "status": status,
                "blocker": blocker,
                "lineage_key": "source_file + zero-based source_row",
                "requested_source_file_count": int(requested_sources),
                "validated_source_file_count": int(validated_sources),
                "binary_test_row_count": int(test_rows),
                "matched_raw_label_count": int(matched_rows),
                "raw_input_bytes_validated": bool(status == "recoverable"),
            }
        ]
    )


def build_diagnostic_tables(
    predictions: pd.DataFrame,
    *,
    project_root: Path,
    preparation_config: Mapping[str, Any],
    preparation_decision: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], list[Path], dict[str, Any]]:
    """Build all diagnostics in memory; raw-label failure remains explicit."""

    source = per_source_file_metrics(predictions)
    aggregates = source_aggregate_estimands(predictions, source)
    identity_sensitivity = identity_weighting_sensitivity(predictions)
    architecture = architecture_prediction_agreement(predictions)
    seed_pairs, seed_summary = cross_seed_prediction_agreement(predictions)
    # Raw-label lineage recovery is CICIDS2017-specific (it reads
    # preparation_config's CICIDS2017 dataset/attack_family_map and the
    # CICIDS2017 prepare stage's raw-input inventory). Filtering on
    # task_id rather than the generic "binary"/"multiclass" task kind keeps
    # ton_iot_binary's rows (which also carry task == "binary") out of this
    # identity table — otherwise their disjoint sample_order numbering would
    # collide under drop_duplicates("sample_order") and their unresolvable
    # source_file would fail the whole lineage recovery for both tasks.
    binary_conditions = predictions[predictions["task_id"].astype(str) == "cicids2017_binary"]
    if binary_conditions.empty:
        raise RuntimeError("Frozen diagnostics require the binary CICIDS2017 task")
    first = binary_conditions.sort_values(["model", "seed", "sample_order"]).drop_duplicates(
        "sample_order", keep="first"
    )
    identity = first.loc[:, list(IDENTITY_COLUMNS)].sort_values("sample_order").reset_index(drop=True)
    requested_sources = int(identity["source_file"].nunique())
    raw_paths: list[Path] = []
    try:
        lineage, raw_paths = recover_raw_label_lineage(
            identity,
            project_root=project_root,
            preparation_config=preparation_config,
            preparation_decision=preparation_decision,
        )
        raw_label, raw_family = raw_attack_recall_tables(predictions, lineage)
        lineage_status = _raw_lineage_status(
            status="recoverable",
            blocker="",
            requested_sources=requested_sources,
            validated_sources=len(raw_paths),
            test_rows=len(identity),
            matched_rows=len(lineage),
        )
        raw_observation = {
            "status": "recoverable",
            "source_records": artifact_records(raw_paths),
        }
    except RawLabelLineageUnavailable as exc:
        raw_label = pd.DataFrame(columns=RAW_RECALL_COLUMNS)
        raw_family = pd.DataFrame(columns=RAW_RECALL_COLUMNS)
        lineage_status = _raw_lineage_status(
            status="unavailable",
            blocker=str(exc),
            requested_sources=requested_sources,
            validated_sources=len(raw_paths),
            test_rows=len(identity),
            matched_rows=0,
        )
        raw_observation = {"status": "unavailable", "blocker": str(exc)}
    tables = {
        "per_source_file_metrics.csv": source,
        "source_aggregate_estimands.csv": aggregates,
        "identity_weighting_sensitivity.csv": identity_sensitivity,
        "architecture_prediction_agreement.csv": architecture,
        "cross_seed_pairwise_agreement.csv": seed_pairs,
        "cross_seed_agreement_summary.csv": seed_summary,
        "raw_attack_recall_by_label.csv": raw_label,
        "raw_attack_recall_by_family.csv": raw_family,
        "raw_label_lineage_status.csv": lineage_status,
        "estimand_definitions.csv": estimand_definitions(),
    }
    return tables, raw_paths, raw_observation


def _diagnostics_implementation_records(project_root: Path) -> list[dict[str, Any]]:
    paths = [
        project_root / "src/leakage_free_tabular/cache.py",
        project_root / "src/leakage_free_tabular/predictive_diagnostics.py",
        project_root / "scripts/run_leakage_free_predictive_diagnostics.py",
    ]
    return artifact_records(paths)


def run_cached_predictive_diagnostics(
    *,
    project_root: Path,
    training_config_path: Path,
    preparation_config_path: Path,
    task_preparation_overrides: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    """Validate caches, derive diagnostics, and create an immutable cache."""

    project_root = project_root.resolve()
    training_config_path = training_config_path.resolve()
    preparation_config_path = preparation_config_path.resolve()
    if task_preparation_overrides:
        task_preparation_overrides = {
            str(task_id): Path(path).resolve()
            for task_id, path in task_preparation_overrides.items()
        }
    inventory = validate_cache_inventory(
        project_root=project_root,
        training_config_path=training_config_path,
        preparation_config_path=preparation_config_path,
        task_preparation_overrides=task_preparation_overrides,
    )
    predictions = load_prediction_snapshot(inventory)
    preparation_config = _read_yaml(preparation_config_path)
    tables, raw_paths, raw_observation = build_diagnostic_tables(
        predictions,
        project_root=project_root,
        preparation_config=preparation_config,
        preparation_decision=inventory.preparation_decision,
    )
    results_root = project_root / str(inventory.config["paths"]["results_root"])
    tables_root = project_root / str(inventory.config["paths"]["tables_root"])
    output_dir = tables_root / "diagnostics"
    result_dir = results_root / "predictive_diagnostics"
    metadata_path = result_dir / "cache_metadata.json"
    decision_path = result_dir / "decision.json"
    output_paths = [output_dir / name for name in tables]
    output_paths.append(decision_path)
    implementations = _diagnostics_implementation_records(project_root)
    implementation_hash = signature({"files": implementations})
    input_paths = [
        training_config_path,
        preparation_config_path,
        results_root / "training_decision.json",
        results_root / "decision.json",
        *inventory.preparation_metadata_paths,
        *inventory.condition_metadata_paths,
        *inventory.prediction_paths.values(),
        *raw_paths,
    ]
    inputs = artifact_records(input_paths)
    stage_signature = signature(
        {
            "pipeline_version": PIPELINE_VERSION,
            "diagnostics_version": DIAGNOSTICS_VERSION,
            "stage": "cached_predictive_diagnostics",
            "implementation_hash": implementation_hash,
            "inputs": inputs,
            "raw_label_lineage_observation": raw_observation,
            "tasks": inventory.task_ids,
            "models": inventory.models,
            "seeds": inventory.seeds,
        }
    )
    hit, _ = cache_hit(
        metadata_path,
        expected_signature=stage_signature,
        expected_outputs=output_paths,
    )
    if hit:
        return {name: output_dir / name for name in tables} | {"decision": decision_path}
    if output_dir.exists() or result_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing or invalid predictive-diagnostics namespace: "
            f"{output_dir}, {result_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)
    decision = {
        "pipeline_version": PIPELINE_VERSION,
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "stage": "cached_predictive_diagnostics",
        "status": "complete",
        "read_only_predictive_cache_consumer": True,
        "models_loaded": False,
        "retraining_performed": False,
        "complete_condition_count": len(inventory.prediction_paths),
        "expected_condition_count": (
            len(inventory.task_ids) * len(inventory.models) * len(inventory.seeds)
        ),
        "tasks": list(inventory.task_ids),
        "models": list(inventory.models),
        "seeds": list(inventory.seeds),
        "raw_label_lineage": raw_observation,
        "stage_signature": stage_signature,
        "estimand_note": (
            "Row-pooled and source-unweighted results are distinct declared estimands; "
            "see estimand_definitions.csv."
        ),
        "tables": {name: (output_dir / name).resolve().as_posix() for name in tables},
    }
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_cache_metadata(
        metadata_path,
        stage="cached_predictive_diagnostics",
        signature_value=stage_signature,
        config_hash=sha256_file(training_config_path),
        implementation_hash=implementation_hash,
        inputs=inputs,
        outputs=output_paths,
        extra={
            "diagnostics_version": DIAGNOSTICS_VERSION,
            "raw_label_lineage": raw_observation,
            "complete_condition_count": len(inventory.prediction_paths),
        },
    )
    return {name: output_dir / name for name in tables} | {"decision": decision_path}
