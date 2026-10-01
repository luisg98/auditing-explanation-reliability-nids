"""Validation-only model selection and untouched-test evaluation.

This module deliberately has separate data types and loaders for selection and
test evaluation.  Hyperparameter selection can only receive train/validation
arrays; test arrays and source identities are loaded after a final model has
been trained and its binary operating threshold has been fixed on validation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Must precede `import pandas`: importing pandas before tensorflow has been
# touched causes tf.data's worker-thread pool to deadlock on its first use
# (observed as an indefinite hang inside ParallelMapDatasetV2 on this
# environment) — verified by process-level bisection, not a TensorFlow config
# issue. Importing tensorflow first sidesteps it.
import tensorflow as _tensorflow_import_order_guard  # noqa: F401

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from src.leakage_free_tabular.cache import (
    artifact_record,
    artifact_records,
    cache_hit,
    sha256_file,
    signature,
    write_cache_metadata,
)
from src.models.cnn import build_cnn_flat_input
from src.models.mlp import build_mlp
from src.utils.seed import set_global_seed


ALLOWED_MODELS = ("mlp", "cnn")
FINAL_SEEDS = (42, 43, 44, 45, 46)
PIPELINE_VERSION = "leakage_free_tabular_v2"
_MODEL_BUILDERS = {
    "mlp": build_mlp,
    "cnn": build_cnn_flat_input,
}
_MODEL_CONTROL_KEYS = {"candidate_id", "batch_size", "class_weight"}


@dataclass(frozen=True)
class SelectionData:
    """Only the data that validation-only model selection may observe."""

    task_id: str
    dataset: str
    task: str
    processed_dir: Path
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class TestData:
    """Untouched test arrays and their source-level identities."""

    X: np.ndarray
    y: np.ndarray
    source_file: np.ndarray
    source_row: np.ndarray
    raw_fingerprint: np.ndarray
    tensor_fingerprint_exact: np.ndarray
    tensor_fingerprint_round5: np.ndarray


@dataclass
class TrainedCondition:
    model: Any
    history: pd.DataFrame
    runtime_seconds: float
    validation_probabilities: np.ndarray
    threshold: float | None
    threshold_metrics: dict[str, float]
    class_weights: list[dict[str, Any]]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    models = tuple(str(value) for value in config.get("models", []))
    if models != ALLOWED_MODELS:
        raise ValueError(f"models must be exactly {ALLOWED_MODELS}, got {models}")
    final_seeds = tuple(int(value) for value in config.get("final_seeds", []))
    if final_seeds != FINAL_SEEDS:
        raise ValueError(f"final_seeds must be exactly {FINAL_SEEDS}")
    selection_seed = int(config.get("selection_seed", -1))
    if selection_seed in final_seeds:
        raise ValueError("selection_seed must be distinct from every final seed")
    policy = config.get("selection_policy", {}) or {}
    if bool(policy.get("reads_test_split", True)):
        raise ValueError("selection_policy.reads_test_split must be false")
    if policy.get("binary_primary_metric") != "validation_pr_auc":
        raise ValueError("Binary selection must use validation_pr_auc")
    if policy.get("multiclass_primary_metric") != "validation_supported_macro_f1":
        raise ValueError("Multiclass selection must use validation_supported_macro_f1")
    if policy.get("multiclass_eligibility_partition") != "validation":
        raise ValueError("Multiclass class eligibility must be defined on validation only")
    if int(policy.get("multiclass_min_validation_support", 0)) < 1:
        raise ValueError("multiclass_min_validation_support must be positive")
    class_weight_max = float(config.get("training", {}).get("class_weight_max", 0.0))
    if not np.isfinite(class_weight_max) or class_weight_max <= 0.0:
        raise ValueError("training.class_weight_max must be a finite positive value")
    if not bool(config.get("split_policy", {}).get("fixed_across_model_seeds")):
        raise ValueError("The processed split must be fixed across model seeds")
    tasks = config.get("tasks", {}) or {}
    grids = config.get("grids", {}) or {}
    for task_id in tasks:
        for model_name in models:
            candidates = grids.get(task_id, {}).get(model_name, [])
            if not 1 <= len(candidates) <= 4:
                raise ValueError(
                    f"Expected 1--4 frozen candidates for {task_id}/{model_name}"
                )
            ids = [str(candidate.get("candidate_id", "")) for candidate in candidates]
            if any(not value for value in ids) or len(ids) != len(set(ids)):
                raise ValueError(f"Candidate IDs must be non-empty and unique: {task_id}/{model_name}")
            if not all(bool(candidate.get("class_weight", False)) for candidate in candidates):
                raise ValueError(
                    f"Every frozen candidate must use the capped class-weight policy: {task_id}/{model_name}"
                )


def _task_config(config: dict[str, Any], task_id: str) -> dict[str, Any]:
    try:
        task = dict(config["tasks"][task_id])
    except KeyError as exc:
        raise KeyError(f"Unknown task_id {task_id!r}") from exc
    if task.get("task") not in {"binary", "multiclass"}:
        raise ValueError(f"Unsupported task kind for {task_id}: {task.get('task')!r}")
    return task


def _read_label_metadata(processed_dir: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    feature_names = tuple(
        str(value)
        for value in json.loads((processed_dir / "feature_names.json").read_text(encoding="utf-8"))
    )
    feature_groups = json.loads(
        (processed_dir / "feature_groups.json").read_text(encoding="utf-8")
    )
    if not isinstance(feature_groups, list) or len(feature_groups) != len(feature_names):
        raise RuntimeError("Feature-group inventory differs from the model coordinates")
    for feature_index, (feature_name, group) in enumerate(zip(feature_names, feature_groups)):
        if (
            not isinstance(group, dict)
            or int(group.get("feature_index", -1)) != feature_index
            or str(group.get("representative", "")) != feature_name
            or feature_name not in [str(value) for value in group.get("members", [])]
        ):
            raise RuntimeError("Feature-group inventory violates the representative contract")
    mapping = json.loads((processed_dir / "label_mapping.json").read_text(encoding="utf-8"))
    class_names = tuple(
        str(name) for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))
    )
    return feature_names, class_names


def _require_guard_pass(processed_dir: Path) -> None:
    """Fail closed on the preparation stage's representation guard."""

    table_path = processed_dir / "split_validation.csv"
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing split guard: {table_path}")
    table = pd.read_csv(table_path)
    if table.empty:
        raise RuntimeError(f"Empty split guard: {table_path}")
    if {"stage", "passed"}.issubset(table.columns):
        required_stages = {"raw_pre_filter", "tensor_post_filter", "raw_split_balance"}
        observed_stages = set(table["stage"].astype(str))
        missing_stages = required_stages - observed_stages
        if missing_stages:
            raise RuntimeError(f"Processed split guard lacks required stages: {sorted(missing_stages)}")
        required_rows = table[table["stage"].isin(required_stages)]
        passed = required_rows["passed"].astype(str).str.lower().isin(
            {"true", "1", "yes", "pass", "passed"}
        )
        if not bool(passed.all()):
            failed = required_rows.loc[~passed, ["stage", "representation", "split_a", "split_b"]]
            raise RuntimeError(f"Processed split guard contains failed required rows:\n{failed}")
    failure_columns = [
        column
        for column in table.columns
        if (
            column.endswith("_passed")
            or column.startswith("zero_")
            or column.startswith("all_classes_")
        )
    ]
    for column in failure_columns:
        values = table[column]
        if values.dtype == object:
            passed = values.astype(str).str.lower().isin({"true", "1", "yes", "pass", "passed"})
        else:
            passed = values.astype(bool)
        if not bool(passed.all()):
            raise RuntimeError(f"Processed split guard failed in column {column!r}")
    metadata_path = processed_dir / "cache_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing preparation cache metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("signature") or not metadata.get("implementation_hash"):
        raise RuntimeError("Preparation metadata lacks signature or implementation_hash")
    invariants = metadata.get("invariants", {})
    required_invariants = {
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
    failed_invariants = {
        key: invariants.get(key)
        for key, expected in required_invariants.items()
        if invariants.get(key) != expected
    }
    if failed_invariants:
        raise RuntimeError(f"Preparation metadata invariant failure: {failed_invariants}")


def load_selection_data(
    processed_root: Path,
    task_id: str,
    task_cfg: dict[str, Any],
) -> SelectionData:
    """Load train/validation only.  Test files are not named or opened here."""

    processed_dir = processed_root / task_id
    _require_guard_pass(processed_dir)
    feature_names, class_names = _read_label_metadata(processed_dir)
    X_train = np.load(processed_dir / "X_train.npy", allow_pickle=False, mmap_mode="r")
    y_train = np.load(processed_dir / "y_train.npy", allow_pickle=False)
    X_val = np.load(processed_dir / "X_val.npy", allow_pickle=False, mmap_mode="r")
    y_val = np.load(processed_dir / "y_val.npy", allow_pickle=False)
    _validate_selection_arrays(X_train, y_train, X_val, y_val, feature_names, class_names)
    return SelectionData(
        task_id=task_id,
        dataset=str(task_cfg["dataset"]),
        task=str(task_cfg["task"]),
        processed_dir=processed_dir,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        class_names=class_names,
    )


def load_test_data(processed_dir: Path, *, n_features: int, n_classes: int) -> TestData:
    data = TestData(
        X=np.load(processed_dir / "X_test.npy", allow_pickle=False, mmap_mode="r"),
        y=np.load(processed_dir / "y_test.npy", allow_pickle=False),
        source_file=np.load(processed_dir / "source_file_test.npy", allow_pickle=False),
        source_row=np.load(processed_dir / "source_row_test.npy", allow_pickle=False),
        raw_fingerprint=np.load(processed_dir / "raw_fingerprint_test.npy", allow_pickle=False),
        tensor_fingerprint_exact=np.load(
            processed_dir / "tensor_fingerprint_exact_test.npy", allow_pickle=False
        ),
        tensor_fingerprint_round5=np.load(
            processed_dir / "tensor_fingerprint_round5_test.npy", allow_pickle=False
        ),
    )
    lengths = {
        len(data.X),
        len(data.y),
        len(data.source_file),
        len(data.source_row),
        len(data.raw_fingerprint),
        len(data.tensor_fingerprint_exact),
        len(data.tensor_fingerprint_round5),
    }
    if len(lengths) != 1:
        raise RuntimeError(f"Test arrays have inconsistent row counts: {sorted(lengths)}")
    if data.X.ndim != 2 or data.X.shape[1] != n_features:
        raise RuntimeError("Test feature geometry differs from train/validation")
    if not np.isfinite(data.X).all():
        raise RuntimeError("Test model tensor contains non-finite values")
    if set(np.unique(data.y).tolist()) != set(range(n_classes)):
        raise RuntimeError("Test split does not contain exactly the declared classes")
    return data


def _validate_selection_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: tuple[str, ...],
    class_names: tuple[str, ...],
) -> None:
    if X_train.ndim != 2 or X_val.ndim != 2:
        raise RuntimeError("Expected two-dimensional tabular model tensors")
    if X_train.shape[1] != X_val.shape[1] or X_train.shape[1] != len(feature_names):
        raise RuntimeError("Feature geometry or feature-name contract mismatch")
    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise RuntimeError("Feature/label row-count mismatch")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise RuntimeError("Train/validation model tensors contain non-finite values")
    expected = set(range(len(class_names)))
    if set(np.unique(y_train).tolist()) != expected or set(np.unique(y_val).tolist()) != expected:
        raise RuntimeError("Train and validation must each contain every declared class")


def selection_input_paths(processed_dir: Path) -> list[Path]:
    """Exact preparation outputs permitted to influence model selection."""

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
        paths.extend(
            [
                processed_dir / f"source_file_{split}.npy",
                processed_dir / f"source_row_{split}.npy",
                processed_dir / f"raw_fingerprint_{split}.npy",
                processed_dir / f"tensor_fingerprint_exact_{split}.npy",
                processed_dir / f"tensor_fingerprint_round5_{split}.npy",
            ]
        )
    return paths


def final_input_paths(processed_dir: Path) -> list[Path]:
    return selection_input_paths(processed_dir) + [
        processed_dir / "X_test.npy",
        processed_dir / "y_test.npy",
        processed_dir / "source_file_test.npy",
        processed_dir / "source_row_test.npy",
        processed_dir / "raw_fingerprint_test.npy",
        processed_dir / "tensor_fingerprint_exact_test.npy",
        processed_dir / "tensor_fingerprint_round5_test.npy",
    ]


def implementation_records(project_root: Path) -> list[dict[str, Any]]:
    return artifact_records(
        [
            project_root / "src/leakage_free_tabular/cache.py",
            project_root / "src/leakage_free_tabular/training.py",
            project_root / "src/models/mlp.py",
            project_root / "src/models/cnn.py",
            project_root / "src/utils/seed.py",
        ]
    )


def _model_params(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if key not in _MODEL_CONTROL_KEYS}


def class_weight_specification(
    y: np.ndarray,
    n_classes: int,
    *,
    maximum_weight: float,
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    """Return capped balanced weights and an auditable raw/effective table."""

    classes = np.arange(n_classes, dtype=int)
    values = compute_class_weight("balanced", classes=classes, y=np.asarray(y, dtype=int))
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes)
    weights = {
        int(class_id): float(min(float(raw_weight), float(maximum_weight)))
        for class_id, raw_weight in zip(classes, values)
    }
    rows = [
        {
            "class_id": int(class_id),
            "training_support": int(counts[class_id]),
            "raw_balanced_weight": float(raw_weight),
            "effective_capped_weight": weights[int(class_id)],
            "maximum_weight": float(maximum_weight),
            "cap_applied": bool(float(raw_weight) > float(maximum_weight)),
        }
        for class_id, raw_weight in zip(classes, values)
    ]
    return weights, rows


def _predict_probabilities(model: Any, X: np.ndarray, *, batch_size: int = 4096) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(X), int(batch_size)):
        result = model(np.asarray(X[start : start + batch_size], dtype=np.float32), training=False)
        if hasattr(result, "numpy"):
            result = result.numpy()
        rows.append(np.asarray(result, dtype=float))
    probabilities = np.concatenate(rows, axis=0)
    if probabilities.ndim == 2 and probabilities.shape[1] == 1:
        return probabilities.reshape(-1)
    if probabilities.ndim != 2:
        raise RuntimeError(f"Unexpected prediction geometry: {probabilities.shape}")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or np.any(row_sums <= 0):
        raise RuntimeError("Model produced invalid probabilities")
    return probabilities / row_sums


def _fit(
    data: SelectionData,
    model_name: str,
    candidate: dict[str, Any],
    training_cfg: dict[str, Any],
    *,
    seed: int,
) -> TrainedCondition:
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"Model is outside the frozen audit set: {model_name!r}")
    import tensorflow as tf

    tf.keras.backend.clear_session()
    set_global_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Some older TensorFlow builds do not expose this switch.  The audited
        # environment records its exact implementation and runs on CPU.
        pass
    builder = _MODEL_BUILDERS[model_name]
    model = builder(
        int(data.X_train.shape[1]),
        len(data.class_names),
        _model_params(candidate),
    )
    monitor = str(training_cfg.get("early_stopping_monitor", "val_loss"))
    mode = str(training_cfg.get("early_stopping_mode", "min"))
    callbacks: list[Any] = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=int(training_cfg.get("early_stopping_patience", 5)),
            min_delta=float(training_cfg.get("early_stopping_min_delta", 0.0005)),
            restore_best_weights=True,
        )
    ]
    lr_cfg = training_cfg.get("reduce_lr_on_plateau", {}) or {}
    if bool(lr_cfg.get("enabled", False)):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                mode=mode,
                factor=float(lr_cfg.get("factor", 0.5)),
                patience=int(lr_cfg.get("patience", 3)),
                min_delta=float(lr_cfg.get("min_delta", 0.0005)),
                min_lr=float(lr_cfg.get("min_lr", 1.0e-6)),
                verbose=0,
            )
        )
    batch_size = int(candidate.get("batch_size", 256))
    X_train = np.asarray(data.X_train, dtype=np.float32)
    X_val = np.asarray(data.X_val, dtype=np.float32)
    y_train = np.asarray(data.y_train, dtype=np.int64)
    y_val = np.asarray(data.y_val, dtype=np.int64)
    class_weight_rows: list[dict[str, Any]] = []
    if bool(candidate.get("class_weight", False)):
        weight_map, class_weight_rows = class_weight_specification(
            y_train,
            len(data.class_names),
            maximum_weight=float(training_cfg["class_weight_max"]),
        )
        sample_weight = np.asarray(
            [weight_map[int(label)] for label in y_train], dtype=np.float32
        )
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weight))
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(
        min(len(y_train), 65536), seed=int(seed), reshuffle_each_iteration=True
    )
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )
    options = tf.data.Options()
    options.experimental_deterministic = True
    train_ds = train_ds.with_options(options)
    val_ds = val_ds.with_options(options)
    started = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(training_cfg.get("epochs", 40)),
        callbacks=callbacks,
        verbose=int(training_cfg.get("verbose", 0)),
    )
    runtime = float(time.perf_counter() - started)
    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    history_frame["training_time_seconds"] = runtime
    numeric = history_frame.select_dtypes(include=[np.number])
    if numeric.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("Training history contains non-finite values")
    validation_probabilities = _predict_probabilities(model, data.X_val)
    threshold: float | None = None
    threshold_metrics: dict[str, float] = {}
    if data.task == "binary":
        threshold, threshold_metrics = select_binary_threshold_at_fpr(
            data.y_val,
            validation_probabilities.reshape(-1),
            max_fpr=0.05,
        )
    return TrainedCondition(
        model=model,
        history=history_frame,
        runtime_seconds=runtime,
        validation_probabilities=validation_probabilities,
        threshold=threshold,
        threshold_metrics=threshold_metrics,
        class_weights=class_weight_rows,
    )


def select_binary_threshold_at_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    max_fpr: float,
) -> tuple[float, dict[str, float]]:
    """Choose maximum-recall validation threshold under an empirical FPR cap."""

    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Invalid validation labels or scores")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Binary threshold calibration requires both classes")
    order = np.argsort(-p, kind="mergesort")
    sorted_scores = p[order]
    sorted_y = y[order]
    change = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    endpoints = np.flatnonzero(change)
    tp = np.cumsum(sorted_y == 1)[endpoints].astype(float)
    fp = np.cumsum(sorted_y == 0)[endpoints].astype(float)
    thresholds = sorted_scores[endpoints]
    positives = float(np.sum(y == 1))
    negatives = float(np.sum(y == 0))
    recall = tp / positives
    fpr = fp / negatives
    precision = tp / np.maximum(tp + fp, 1.0)
    f2 = 5.0 * precision * recall / np.maximum(4.0 * precision + recall, np.finfo(float).eps)
    valid = np.flatnonzero(fpr <= float(max_fpr) + 1e-15)
    if not len(valid):
        threshold = float(np.nextafter(np.max(p), np.inf))
        return threshold, {
            "validation_fpr": 0.0,
            "validation_fnr": 1.0,
            "validation_precision": 0.0,
            "validation_recall": 0.0,
            "validation_f2": 0.0,
            "max_fpr": float(max_fpr),
        }
    # np.lexsort uses the last key as primary: recall desc, F2 desc,
    # threshold desc.  A higher threshold is the deterministic final tie-break.
    ranked = valid[np.lexsort((-thresholds[valid], -f2[valid], -recall[valid]))]
    index = int(ranked[0])
    return float(thresholds[index]), {
        "validation_fpr": float(fpr[index]),
        "validation_fnr": float(1.0 - recall[index]),
        "validation_precision": float(precision[index]),
        "validation_recall": float(recall[index]),
        "validation_f2": float(f2[index]),
        "max_fpr": float(max_fpr),
    }


def multiclass_validation_eligibility(
    y_validation: np.ndarray,
    class_names: tuple[str, ...],
    *,
    minimum_support: int,
) -> dict[str, Any]:
    """Freeze inferential class eligibility from validation labels alone."""

    labels = np.asarray(y_validation, dtype=int).reshape(-1)
    counts = np.bincount(labels, minlength=len(class_names))
    class_rows = [
        {
            "class_id": int(class_id),
            "class_name": str(class_name),
            "validation_support": int(counts[class_id]),
            "eligible_for_inference": bool(counts[class_id] >= int(minimum_support)),
        }
        for class_id, class_name in enumerate(class_names)
    ]
    eligible = [row for row in class_rows if row["eligible_for_inference"]]
    if not eligible:
        raise RuntimeError("No multiclass labels meet the frozen validation-support rule")
    return {
        "eligibility_partition": "validation",
        "minimum_validation_support": int(minimum_support),
        "eligible_classes": eligible,
        "all_classes": class_rows,
    }


def _supported_multiclass_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    eligible_class_ids: tuple[int, ...],
    n_classes: int,
) -> dict[str, float]:
    """Metrics over class IDs frozen elsewhere, without support-based reselection."""

    if not eligible_class_ids:
        raise ValueError("supported multiclass metrics require frozen eligible class IDs")
    labels = list(eligible_class_ids)
    eligible_rows = np.isin(y_true, labels)
    if not bool(eligible_rows.any()):
        raise RuntimeError("No rows have a true label in the frozen supported class set")
    return {
        "supported_macro_f1": float(
            f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "supported_balanced_accuracy": float(
            recall_score(y_true, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "supported_weighted_f1": float(
            f1_score(y_true, predictions, labels=labels, average="weighted", zero_division=0)
        ),
        "supported_log_loss": float(
            log_loss(
                np.asarray(y_true)[eligible_rows],
                np.asarray(probabilities)[eligible_rows],
                labels=np.arange(n_classes),
            )
        ),
    }


def _validation_metrics(
    data: SelectionData,
    probabilities: np.ndarray,
    threshold: float | None,
    *,
    eligible_class_ids: tuple[int, ...] = (),
) -> dict[str, float]:
    if data.task == "binary":
        scores = np.asarray(probabilities, dtype=float).reshape(-1)
        selected_threshold = float(threshold if threshold is not None else 0.5)
        pred = (scores >= selected_threshold).astype(int)
        return {
            "validation_pr_auc": float(average_precision_score(data.y_val, scores)),
            "validation_roc_auc": float(roc_auc_score(data.y_val, scores)),
            "validation_macro_f1": float(f1_score(data.y_val, pred, average="macro")),
            "validation_balanced_accuracy": float(balanced_accuracy_score(data.y_val, pred)),
            "validation_weighted_f1": float(f1_score(data.y_val, pred, average="weighted")),
            "validation_log_loss": float(log_loss(data.y_val, np.column_stack([1.0 - scores, scores]), labels=[0, 1])),
        }
    probs = np.asarray(probabilities, dtype=float)
    pred = probs.argmax(axis=1)
    supported = _supported_multiclass_metrics(
        data.y_val,
        pred,
        probs,
        eligible_class_ids=eligible_class_ids,
        n_classes=len(data.class_names),
    )
    return {
        "validation_macro_f1": float(f1_score(data.y_val, pred, average="macro", zero_division=0)),
        "validation_balanced_accuracy": float(balanced_accuracy_score(data.y_val, pred)),
        "validation_weighted_f1": float(f1_score(data.y_val, pred, average="weighted", zero_division=0)),
        "validation_log_loss": float(log_loss(data.y_val, probs, labels=np.arange(len(data.class_names)))),
        **{f"validation_{key}": value for key, value in supported.items()},
    }


def select_best_candidate(table: pd.DataFrame, task: str) -> dict[str, Any]:
    """Apply the frozen ranking; the table contains validation metrics only."""

    if table.empty:
        raise ValueError("Cannot select from an empty candidate table")
    if any(str(column).startswith("test_") for column in table.columns):
        raise ValueError("Candidate table must not contain test-derived columns")
    if task == "binary":
        columns = ["validation_pr_auc", "validation_roc_auc", "validation_log_loss", "candidate_id"]
        ascending = [False, False, True, True]
    else:
        columns = [
            "validation_supported_macro_f1",
            "validation_supported_balanced_accuracy",
            "validation_supported_weighted_f1",
            "validation_supported_log_loss",
            "candidate_id",
        ]
        ascending = [False, False, False, True, True]
    row = table.sort_values(columns, ascending=ascending, kind="mergesort").iloc[0]
    return {str(key): _clean_json(value) for key, value in row.to_dict().items()}


def _clean_json(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _merge_training_config(config: dict[str, Any], phase: str) -> dict[str, Any]:
    training = dict(config.get("training", {}).get(phase, {}) or {})
    training["class_weight_max"] = float(config["training"]["class_weight_max"])
    training["reduce_lr_on_plateau"] = dict(
        config.get("training", {}).get("reduce_lr_on_plateau", {}) or {}
    )
    return training


def run_model_selection(
    config: dict[str, Any],
    *,
    project_root: Path,
    config_path: Path,
    task_id: str,
    model_name: str,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train the frozen candidate grid without loading the test split."""

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported clean-pipeline model: {model_name!r}")
    task_cfg = _task_config(config, task_id)
    processed_root = project_root / str(config["paths"]["processed_root"])
    processed_dir = processed_root / task_id
    inputs = artifact_records(selection_input_paths(processed_dir))
    implementations = implementation_records(project_root)
    config_hash = sha256_file(config_path)
    implementation_hash = signature({"files": implementations})
    selection_seed = int(config["selection_seed"])
    stage_signature = signature(
        {
            "pipeline_version": PIPELINE_VERSION,
            "stage": "model_selection",
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
            "inputs": inputs,
            "task_id": task_id,
            "model": model_name,
            "selection_seed": selection_seed,
        }
    )
    out_dir = (
        project_root
        / str(config["paths"]["results_root"])
        / "model_selection"
        / task_id
        / model_name
    )
    candidates = list(config["grids"][task_id][model_name])
    expected_outputs = [out_dir / "candidate_results.csv", out_dir / "selected_config.lock.json"]
    for candidate in candidates:
        trial_dir = out_dir / "candidates" / str(candidate["candidate_id"])
        expected_outputs.extend(
            [
                trial_dir / "model.keras",
                trial_dir / "training_history.csv",
                trial_dir / "class_weights.csv",
            ]
        )
    metadata_path = out_dir / "cache_metadata.json"
    hit, _ = cache_hit(
        metadata_path,
        expected_signature=stage_signature,
        expected_outputs=expected_outputs,
    )
    table_path = out_dir / "candidate_results.csv"
    lock_path = out_dir / "selected_config.lock.json"
    if hit and not force:
        return pd.read_csv(table_path), json.loads(lock_path.read_text(encoding="utf-8"))
    data = load_selection_data(processed_root, task_id, task_cfg)
    multiclass_eligibility: dict[str, Any] | None = None
    eligible_class_ids: tuple[int, ...] = ()
    if data.task == "multiclass":
        multiclass_eligibility = multiclass_validation_eligibility(
            data.y_val,
            data.class_names,
            minimum_support=int(
                config["selection_policy"]["multiclass_min_validation_support"]
            ),
        )
        eligible_class_ids = tuple(
            int(row["class_id"])
            for row in multiclass_eligibility["eligible_classes"]
        )
    rows: list[dict[str, Any]] = []
    output_paths: list[Path] = []
    for trial_index, raw_candidate in enumerate(candidates):
        candidate = dict(raw_candidate)
        trained = _fit(
            data,
            model_name,
            candidate,
            _merge_training_config(config, "selection"),
            seed=selection_seed,
        )
        metrics = _validation_metrics(
            data,
            trained.validation_probabilities,
            trained.threshold,
            eligible_class_ids=eligible_class_ids,
        )
        trial_dir = out_dir / "candidates" / str(candidate["candidate_id"])
        trial_dir.mkdir(parents=True, exist_ok=True)
        model_path = trial_dir / "model.keras"
        history_path = trial_dir / "training_history.csv"
        class_weights_path = trial_dir / "class_weights.csv"
        trained.model.save(model_path)
        trained.history.to_csv(history_path, index=False)
        pd.DataFrame(trained.class_weights).to_csv(class_weights_path, index=False)
        output_paths.extend([model_path, history_path, class_weights_path])
        rows.append(
            {
                "task_id": task_id,
                "dataset": data.dataset,
                "task": data.task,
                "model": model_name,
                "selection_seed": selection_seed,
                "trial_index": trial_index,
                "candidate_id": candidate["candidate_id"],
                "candidate_json": json.dumps(candidate, sort_keys=True, separators=(",", ":")),
                "epochs_ran": int(len(trained.history)),
                "training_time_seconds": trained.runtime_seconds,
                "selected_threshold": trained.threshold,
                "class_weight_max": float(config["training"]["class_weight_max"]),
                "maximum_raw_balanced_weight": float(
                    max(row["raw_balanced_weight"] for row in trained.class_weights)
                ),
                "maximum_effective_capped_weight": float(
                    max(row["effective_capped_weight"] for row in trained.class_weights)
                ),
                "class_weights_json": json.dumps(
                    trained.class_weights, sort_keys=True, separators=(",", ":")
                ),
                "eligible_class_ids_json": (
                    json.dumps(list(eligible_class_ids))
                    if data.task == "multiclass"
                    else None
                ),
                "eligible_class_names_json": (
                    json.dumps(
                        [data.class_names[class_id] for class_id in eligible_class_ids]
                    )
                    if data.task == "multiclass"
                    else None
                ),
                **metrics,
                **{f"threshold_{key}": value for key, value in trained.threshold_metrics.items()},
            }
        )
        del trained
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
        except Exception:
            pass
    table = pd.DataFrame(rows)
    selected = select_best_candidate(table, data.task)
    selected_candidate = next(
        dict(candidate)
        for candidate in candidates
        if str(candidate["candidate_id"]) == str(selected["candidate_id"])
    )
    lock = {
        "pipeline_version": PIPELINE_VERSION,
        "task_id": task_id,
        "dataset": data.dataset,
        "task": data.task,
        "model": model_name,
        "selection_seed": selection_seed,
        "selection_split": "validation_only",
        "selected_candidate": selected_candidate,
        "selection_metrics": selected,
        "multiclass_inference": multiclass_eligibility,
        "config_hash": config_hash,
        "implementation_hash": implementation_hash,
        "processed_input_signature": signature({"inputs": inputs}),
        "stage_signature": stage_signature,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths.extend([table_path, lock_path])
    write_cache_metadata(
        metadata_path,
        stage="model_selection",
        signature_value=stage_signature,
        config_hash=config_hash,
        implementation_hash=implementation_hash,
        inputs=inputs,
        outputs=output_paths,
        extra={
            "task_id": task_id,
            "model": model_name,
            "selection_seed": selection_seed,
            "test_split_loaded": False,
            "class_weight_max": float(config["training"]["class_weight_max"]),
        },
    )
    return table, lock


def _binary_calibration_error(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    bins: int,
) -> float:
    labels = np.asarray(y_true, dtype=float).reshape(-1)
    probability = np.asarray(scores, dtype=float).reshape(-1)
    if len(labels) != len(probability) or bins <= 0:
        raise ValueError("Binary calibration inputs or bin count are invalid")
    bin_ids = np.minimum((np.clip(probability, 0.0, 1.0) * bins).astype(int), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += float(mask.mean()) * abs(
                float(probability[mask].mean()) - float(labels[mask].mean())
            )
    return error


def _multiclass_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int,
) -> float:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or len(probs) != len(y_true) or bins <= 0:
        raise ValueError("Multiclass calibration inputs or bin count are invalid")
    predictions = probs.argmax(axis=1)
    confidence = probs[np.arange(len(probs)), predictions]
    correctness = predictions == np.asarray(y_true, dtype=int)
    bin_ids = np.minimum((np.clip(confidence, 0.0, 1.0) * bins).astype(int), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += float(mask.mean()) * abs(
                float(confidence[mask].mean()) - float(correctness[mask].mean())
            )
    return error


def _test_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    task: str,
    threshold: float | None,
    n_classes: int,
    eligible_class_ids: tuple[int, ...] = (),
    calibration_bins: int = 15,
) -> tuple[dict[str, float], np.ndarray]:
    if task == "binary":
        scores = np.asarray(probabilities, dtype=float).reshape(-1)
        selected_threshold = float(threshold if threshold is not None else 0.5)
        pred = (scores >= selected_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        metrics = {
            "accuracy": accuracy_score(y_true, pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "f2": fbeta_score(y_true, pred, beta=2.0, zero_division=0),
            "fpr": fp / max(fp + tn, 1),
            "fnr": fn / max(fn + tp, 1),
            "pr_auc": average_precision_score(y_true, scores),
            "roc_auc": roc_auc_score(y_true, scores),
            "log_loss": log_loss(y_true, np.column_stack([1.0 - scores, scores]), labels=[0, 1]),
            "brier_score": float(np.mean(np.square(scores - np.asarray(y_true, dtype=float)))),
            "ece": _binary_calibration_error(
                y_true, scores, bins=int(calibration_bins)
            ),
        }
    else:
        probs = np.asarray(probabilities, dtype=float)
        pred = probs.argmax(axis=1)
        full_macro_f1 = float(
            f1_score(
                y_true,
                pred,
                labels=np.arange(n_classes),
                average="macro",
                zero_division=0,
            )
        )
        supported = _supported_multiclass_metrics(
            y_true,
            pred,
            probs,
            eligible_class_ids=eligible_class_ids,
            n_classes=n_classes,
        )
        metrics = {
            "accuracy": accuracy_score(y_true, pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, pred),
            "macro_f1": full_macro_f1,
            "macro_f1_all_classes_descriptive": full_macro_f1,
            "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(y_true, pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
            "log_loss": log_loss(y_true, probs, labels=np.arange(n_classes)),
            "brier_score": float(
                np.mean(
                    np.sum(
                        np.square(
                            probs
                            - np.eye(n_classes, dtype=float)[
                                np.asarray(y_true, dtype=int)
                            ]
                        ),
                        axis=1,
                    )
                )
            ),
            "ece": _multiclass_calibration_error(
                y_true, probs, bins=int(calibration_bins)
            ),
            **supported,
        }
    return {key: float(value) for key, value in metrics.items()}, pred


def _per_class_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    class_names: tuple[str, ...],
    *,
    eligible_class_ids: tuple[int, ...],
    task_id: str,
    dataset: str,
    task: str,
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=labels,
        zero_division=0,
    )
    eligible = set(int(value) for value in eligible_class_ids)
    return pd.DataFrame(
        [
            {
                "task_id": task_id,
                "dataset": dataset,
                "task": task,
                "model": model_name,
                "seed": int(seed),
                "class_id": int(class_id),
                "class_name": class_name,
                "test_support": int(support[class_id]),
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "eligible_for_inference": bool(class_id in eligible),
                "eligibility_source": (
                    "validation_support" if task == "multiclass" else "binary_task"
                ),
            }
            for class_id, class_name in enumerate(class_names)
        ]
    )


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def predicted_class_confidence(
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Return the probability of the class selected by the decision rule."""

    probs = np.asarray(probabilities, dtype=float)
    pred = np.asarray(predictions, dtype=int).reshape(-1)
    if probs.ndim == 1:
        if len(probs) != len(pred) or np.any((pred < 0) | (pred > 1)):
            raise ValueError("Binary probabilities and predictions do not align")
        return np.where(pred == 1, probs, 1.0 - probs)
    if probs.ndim == 2:
        if len(probs) != len(pred) or np.any((pred < 0) | (pred >= probs.shape[1])):
            raise ValueError("Multiclass probabilities and predictions do not align")
        return probs[np.arange(len(pred)), pred]
    raise ValueError("Probabilities must be one- or two-dimensional")


def _fingerprint_hex(values: np.ndarray) -> np.ndarray:
    """Encode raw SHA-256 arrays without decoding arbitrary digest bytes.

    Preparation stores digests as ``S32`` to avoid quadrupling their size.
    Viewing each fixed-width element as 32 uint8 values preserves trailing NUL
    bytes, which both ``astype(str)`` and scalar ``bytes`` conversion can lose.
    Synthetic/tests may use already-readable Unicode fingerprints.
    """

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("Fingerprint arrays must be one-dimensional")
    if array.dtype.kind == "U":
        return array.astype(str)
    if array.dtype.kind == "S" and array.dtype.itemsize == 32:
        octets = np.ascontiguousarray(array).view(np.uint8).reshape(len(array), 32)
        return np.asarray([row.tobytes().hex() for row in octets], dtype="<U64")
    if array.dtype.kind == "S":
        return np.char.decode(array, "ascii").astype(str)
    raise TypeError(f"Unsupported fingerprint dtype: {array.dtype}")


def _prediction_table(
    data: SelectionData,
    test: TestData,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    *,
    model_name: str,
    seed: int,
    threshold: float | None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "task_id": data.task_id,
            "dataset": data.dataset,
            "task": data.task,
            "model": model_name,
            "seed": int(seed),
            "sample_order": np.arange(len(test.y)),
            "source_file": test.source_file.astype(str),
            "source_row": test.source_row.astype(np.int64),
            "raw_fingerprint": _fingerprint_hex(test.raw_fingerprint),
            "tensor_fingerprint_exact": _fingerprint_hex(test.tensor_fingerprint_exact),
            "tensor_fingerprint_round5": _fingerprint_hex(test.tensor_fingerprint_round5),
            "true_class_id": test.y.astype(int),
            "true_class": [data.class_names[int(value)] for value in test.y],
            "pred_class_id": predictions.astype(int),
            "pred_class": [data.class_names[int(value)] for value in predictions],
            # Probability assigned to the class selected by the declared
            # decision rule.  For binary models the validation threshold need
            # not be 0.5, so max(p, 1-p) can refer to the *other* class.
            "confidence": predicted_class_confidence(probabilities, predictions),
            "correct": predictions == test.y,
        }
    )
    if data.task == "binary":
        scores = probabilities.reshape(-1)
        frame["decision_threshold"] = float(threshold if threshold is not None else 0.5)
        frame[f"prob_{_safe_name(data.class_names[0])}"] = 1.0 - scores
        frame[f"prob_{_safe_name(data.class_names[1])}"] = scores
    else:
        for class_index, class_name in enumerate(data.class_names):
            frame[f"prob_{_safe_name(class_name)}"] = probabilities[:, class_index]
    return frame


def _frozen_eligible_class_ids(
    lock: dict[str, Any],
    data: SelectionData,
    config: dict[str, Any],
) -> tuple[int, ...]:
    """Read multiclass eligibility from the validation-derived selection lock."""

    if data.task != "multiclass":
        return tuple(range(len(data.class_names)))
    spec = lock.get("multiclass_inference")
    if not isinstance(spec, dict):
        raise RuntimeError("Selection lock lacks multiclass validation eligibility")
    minimum = int(config["selection_policy"]["multiclass_min_validation_support"])
    if spec.get("eligibility_partition") != "validation":
        raise RuntimeError("Selection lock eligibility was not defined on validation")
    if int(spec.get("minimum_validation_support", -1)) != minimum:
        raise RuntimeError("Selection lock validation-support rule differs from config")
    eligible_rows = spec.get("eligible_classes")
    if not isinstance(eligible_rows, list) or not eligible_rows:
        raise RuntimeError("Selection lock has no eligible multiclass labels")
    counts = np.bincount(np.asarray(data.y_val, dtype=int), minlength=len(data.class_names))
    eligible_ids: list[int] = []
    for row in eligible_rows:
        class_id = int(row["class_id"])
        if class_id < 0 or class_id >= len(data.class_names):
            raise RuntimeError(f"Selection lock has invalid class ID {class_id}")
        if str(row["class_name"]) != data.class_names[class_id]:
            raise RuntimeError(f"Selection lock class name differs for ID {class_id}")
        if int(row["validation_support"]) != int(counts[class_id]):
            raise RuntimeError(f"Selection lock validation support differs for ID {class_id}")
        if int(counts[class_id]) < minimum:
            raise RuntimeError(f"Selection lock includes below-threshold class ID {class_id}")
        eligible_ids.append(class_id)
    if len(eligible_ids) != len(set(eligible_ids)):
        raise RuntimeError("Selection lock repeats an eligible class ID")
    expected_ids = tuple(int(value) for value in np.flatnonzero(counts >= minimum))
    if tuple(eligible_ids) != expected_ids:
        raise RuntimeError("Selection lock does not exactly match validation-derived eligibility")
    return tuple(eligible_ids)


def run_final_condition(
    config: dict[str, Any],
    *,
    project_root: Path,
    config_path: Path,
    task_id: str,
    model_name: str,
    seed: int,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Train on the locked config, freeze threshold on validation, then open test."""

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported clean-pipeline model: {model_name!r}")
    if int(seed) not in FINAL_SEEDS:
        raise ValueError(f"Final seed must be one of {FINAL_SEEDS}")
    task_cfg = _task_config(config, task_id)
    processed_root = project_root / str(config["paths"]["processed_root"])
    processed_dir = processed_root / task_id
    selection_dir = (
        project_root
        / str(config["paths"]["results_root"])
        / "model_selection"
        / task_id
        / model_name
    )
    lock_path = selection_dir / "selected_config.lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Run validation-only selection first: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if int(lock.get("selection_seed", -1)) in FINAL_SEEDS:
        raise RuntimeError("Invalid selection lock: selection seed overlaps final seeds")
    inputs = artifact_records(final_input_paths(processed_dir) + [lock_path])
    implementations = implementation_records(project_root)
    config_hash = sha256_file(config_path)
    implementation_hash = signature({"files": implementations})
    stage_signature = signature(
        {
            "pipeline_version": PIPELINE_VERSION,
            "stage": "final_predictive",
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
            "inputs": inputs,
            "task_id": task_id,
            "model": model_name,
            "seed": int(seed),
        }
    )
    out_dir = (
        project_root
        / str(config["paths"]["results_root"])
        / "final"
        / task_id
        / model_name
        / f"seed_{seed}"
    )
    metadata_path = out_dir / "cache_metadata.json"
    metrics_path = out_dir / "metrics.csv"
    predictions_path = out_dir / "test_predictions.csv"
    history_path = out_dir / "training_history.csv"
    confusion_path = out_dir / "confusion_matrix.csv"
    per_class_path = out_dir / "per_class_metrics.csv"
    class_weights_path = out_dir / "class_weights.csv"
    model_path = out_dir / "model.keras"
    expected_outputs = [
        model_path,
        history_path,
        predictions_path,
        metrics_path,
        confusion_path,
        per_class_path,
        class_weights_path,
    ]
    hit, _ = cache_hit(
        metadata_path,
        expected_signature=stage_signature,
        expected_outputs=expected_outputs,
    )
    if hit and not force:
        return {
            "metrics": pd.read_csv(metrics_path),
            "predictions": pd.read_csv(predictions_path),
            "history": pd.read_csv(history_path),
            "confusion": pd.read_csv(confusion_path, index_col=0),
            "per_class": pd.read_csv(per_class_path),
            "class_weights": pd.read_csv(class_weights_path),
        }

    # This loader has no test arrays.  Training and threshold calibration finish
    # before load_test_data is called below.
    selection_data = load_selection_data(processed_root, task_id, task_cfg)
    eligible_class_ids = _frozen_eligible_class_ids(lock, selection_data, config)
    candidate = dict(lock["selected_candidate"])
    trained = _fit(
        selection_data,
        model_name,
        candidate,
        _merge_training_config(config, "final"),
        seed=int(seed),
    )
    if selection_data.task == "binary":
        threshold, threshold_info = select_binary_threshold_at_fpr(
            selection_data.y_val,
            trained.validation_probabilities.reshape(-1),
            max_fpr=float(config["selection_policy"]["binary_max_fpr"]),
        )
    else:
        threshold, threshold_info = None, {}

    # First access to the held-out feature/label arrays in this condition.
    test = load_test_data(
        processed_dir,
        n_features=len(selection_data.feature_names),
        n_classes=len(selection_data.class_names),
    )
    test_probabilities = _predict_probabilities(trained.model, test.X)
    metrics, predictions = _test_metrics(
        test.y,
        test_probabilities,
        task=selection_data.task,
        threshold=threshold,
        n_classes=len(selection_data.class_names),
        eligible_class_ids=eligible_class_ids,
        calibration_bins=int(config["calibration"]["bins"]),
    )
    predictions_frame = _prediction_table(
        selection_data,
        test,
        test_probabilities,
        predictions,
        model_name=model_name,
        seed=int(seed),
        threshold=threshold,
    )
    labels = np.arange(len(selection_data.class_names))
    confusion = pd.DataFrame(
        confusion_matrix(test.y, predictions, labels=labels),
        index=[f"true_{name}" for name in selection_data.class_names],
        columns=[f"pred_{name}" for name in selection_data.class_names],
    )
    per_class = _per_class_metrics(
        test.y,
        predictions,
        selection_data.class_names,
        eligible_class_ids=eligible_class_ids,
        task_id=task_id,
        dataset=selection_data.dataset,
        task=selection_data.task,
        model_name=model_name,
        seed=int(seed),
    )
    metrics_frame = pd.DataFrame(
        [
            {
                "task_id": task_id,
                "dataset": selection_data.dataset,
                "task": selection_data.task,
                "model": model_name,
                "seed": int(seed),
                "selected_candidate_id": candidate["candidate_id"],
                "epochs_ran": int(len(trained.history)),
                "training_time_seconds": trained.runtime_seconds,
                "class_weight_max": float(config["training"]["class_weight_max"]),
                "maximum_raw_balanced_weight": float(
                    max(row["raw_balanced_weight"] for row in trained.class_weights)
                ),
                "maximum_effective_capped_weight": float(
                    max(row["effective_capped_weight"] for row in trained.class_weights)
                ),
                "validation_selected_threshold": threshold,
                "eligible_class_ids_json": json.dumps(list(eligible_class_ids)),
                "eligible_class_names_json": json.dumps(
                    [selection_data.class_names[class_id] for class_id in eligible_class_ids]
                ),
                "class_eligibility_partition": (
                    "validation" if selection_data.task == "multiclass" else "not_applicable"
                ),
                **{f"threshold_{key}": value for key, value in threshold_info.items()},
                **metrics,
            }
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    trained.model.save(model_path)
    trained.history.to_csv(history_path, index=False)
    predictions_frame.to_csv(predictions_path, index=False)
    metrics_frame.to_csv(metrics_path, index=False)
    confusion.to_csv(confusion_path)
    per_class.to_csv(per_class_path, index=False)
    class_weights = pd.DataFrame(trained.class_weights)
    class_weights.insert(0, "seed", int(seed))
    class_weights.insert(0, "model", model_name)
    class_weights.insert(0, "task_id", task_id)
    class_weights.to_csv(class_weights_path, index=False)
    outputs = [
        model_path,
        history_path,
        predictions_path,
        metrics_path,
        confusion_path,
        per_class_path,
        class_weights_path,
    ]
    write_cache_metadata(
        metadata_path,
        stage="final_predictive",
        signature_value=stage_signature,
        config_hash=config_hash,
        implementation_hash=implementation_hash,
        inputs=inputs,
        outputs=outputs,
        extra={
            "task_id": task_id,
            "model": model_name,
            "seed": int(seed),
            "selection_lock": artifact_record(lock_path),
            "threshold_fitted_on": "validation",
            "test_evaluated_once_after_training": True,
            "eligible_class_ids_frozen_before_test": list(eligible_class_ids),
            "class_weight_max": float(config["training"]["class_weight_max"]),
            "class_weights": trained.class_weights,
        },
    )
    return {
        "metrics": metrics_frame,
        "predictions": predictions_frame,
        "history": trained.history,
        "confusion": confusion,
        "per_class": per_class,
        "class_weights": class_weights,
    }


def write_summary_tables(
    config: dict[str, Any],
    *,
    project_root: Path,
    selection_tables: Iterable[pd.DataFrame],
    final_metric_tables: Iterable[pd.DataFrame],
) -> dict[str, Path]:
    tables_root = project_root / str(config["paths"]["tables_root"])
    tables_root.mkdir(parents=True, exist_ok=True)
    selection = pd.concat(list(selection_tables), ignore_index=True)
    metrics = pd.concat(list(final_metric_tables), ignore_index=True)
    selection_path = tables_root / "model_selection.csv"
    metrics_path = tables_root / "predictive_metrics_by_seed.csv"
    aggregate_path = tables_root / "predictive_metrics_summary.csv"
    selection.to_csv(selection_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    id_columns = {
        "task_id",
        "dataset",
        "task",
        "model",
        "seed",
        "selected_candidate_id",
    }
    metric_columns = [
        column
        for column in metrics.columns
        if column not in id_columns and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    aggregate = (
        metrics.groupby(["task_id", "dataset", "task", "model"], dropna=False)[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if part != "")
        if isinstance(column, tuple)
        else str(column)
        for column in aggregate.columns
    ]
    aggregate.to_csv(aggregate_path, index=False)
    return {
        "selection": selection_path,
        "metrics": metrics_path,
        "aggregate": aggregate_path,
    }
