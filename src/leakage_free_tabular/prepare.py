"""Leakage-resistant CICIDS2017 tabular data preparation.

This module deliberately has no model or XAI dependency.  It creates one fixed
split per task and fits every learned preprocessing operation on the training
partition only.  Exact feature-vector groups cannot cross raw partitions.  A
second, label-blind guard removes hold-out rows that collide with an earlier
partition after the *actual* float32 model representation is produced, both
exactly and after rounding to five decimal places.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "leakage_free_tabular.yaml"
STAGE_VERSION = 2
SPLITS = ("train", "validation", "test")
CONFIG_SECTIONS = (
    "pipeline_version",
    "dataset",
    "tasks",
    "split",
    "preprocessing",
    "representation_guard",
    "outputs",
)


@dataclass
class TaskFrame:
    """Capped, finite rows for one task before splitting."""

    task_id: str
    task: str
    class_names: list[str]
    feature_names: list[str]
    X: np.ndarray
    y: np.ndarray
    source_file_id: np.ndarray
    source_row: np.ndarray
    source_files: list[str]
    raw_fingerprint: np.ndarray


@dataclass
class PreparedTask:
    """Final arrays and audit payload for one task."""

    task_id: str
    task: str
    class_names: list[str]
    input_feature_names: list[str]
    feature_names: list[str]
    feature_groups: list[dict[str, Any]]
    pipeline: Pipeline
    splits: dict[str, dict[str, np.ndarray]]
    dropped_rows: pd.DataFrame
    validation: pd.DataFrame
    distribution: pd.DataFrame
    representation_summary: pd.DataFrame
    preprocessing_summary: dict[str, Any]


class ExactDuplicateFeatureRemover(TransformerMixin, BaseEstimator):
    """Keep one coordinate from each train-identical feature group.

    The support is learned only from the training partition.  This is necessary
    for coordinate-level attribution: two identical model inputs are not
    separately identifiable, and leaving both in the tensor lets attribution
    mass split between aliases for purely representational reasons.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ExactDuplicateFeatureRemover":
        del y
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError("ExactDuplicateFeatureRemover expects a 2-D array")
        feature_count = int(values.shape[1])
        support = np.ones(feature_count, dtype=bool)
        duplicate_of = np.arange(feature_count, dtype=np.int64)
        if self.enabled:
            digest_buckets: dict[str, list[int]] = {}
            for feature_index in range(feature_count):
                column = np.ascontiguousarray(values[:, feature_index])
                digest = hashlib.sha256(column.view(np.uint8)).hexdigest()
                representative: int | None = None
                for candidate in digest_buckets.get(digest, []):
                    if np.array_equal(values[:, candidate], values[:, feature_index]):
                        representative = int(candidate)
                        break
                if representative is None:
                    digest_buckets.setdefault(digest, []).append(feature_index)
                else:
                    support[feature_index] = False
                    duplicate_of[feature_index] = representative
        self.n_features_in_ = feature_count
        self.support_ = support
        self.duplicate_of_ = duplicate_of
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Feature geometry differs from the fitted duplicate guard")
        return values[:, self.support_]

    def get_support(self) -> np.ndarray:
        if not hasattr(self, "support_"):
            raise RuntimeError("ExactDuplicateFeatureRemover is not fitted")
        return self.support_.copy()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_record(path: Path, *, reported_path: Path | None = None) -> dict[str, Any]:
    target = reported_path or path
    try:
        relative = target.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        relative = target.as_posix()
    return {
        "path": relative,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def records_valid(records: Iterable[dict[str, Any]]) -> bool:
    for record in records:
        path = ROOT / str(record.get("path", ""))
        if not path.is_file():
            return False
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            return False
        if sha256_file(path) != str(record.get("sha256", "")):
            return False
    return True


def read_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    missing = [key for key in CONFIG_SECTIONS if key not in data]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")
    return data


def relevant_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return only data-stage config, so later training settings do not poison this cache."""

    return {key: config[key] for key in CONFIG_SECTIONS}


def source_paths(config: dict[str, Any]) -> list[Path]:
    dataset = config["dataset"]
    source_root = ROOT / str(dataset["source_path"])
    paths = sorted(source_root.glob(str(dataset.get("glob_pattern", "*.parquet"))))
    if not paths:
        raise FileNotFoundError(f"No source parquet files found below {source_root}")
    return paths


def implementation_records() -> list[dict[str, Any]]:
    paths = [Path(__file__).resolve()]
    wrapper = ROOT / "scripts" / "prepare_leakage_free_tabular.py"
    if wrapper.is_file():
        paths.append(wrapper)
    return [file_record(path) for path in paths]


def stage_payload(config: dict[str, Any]) -> dict[str, Any]:
    raw = [file_record(path) for path in source_paths(config)]
    implementation = implementation_records()
    payload = {
        "stage_version": STAGE_VERSION,
        "config": relevant_config(config),
        "config_signature": stable_hash(relevant_config(config)),
        "raw_inputs": raw,
        "implementation": implementation,
        "implementation_hash": stable_hash(implementation),
    }
    return payload


def normalize_label(value: object) -> str:
    text = str(value).strip().replace("\ufffd", " ")
    return " ".join(text.split())


def attack_family(value: object, mapping: dict[str, str]) -> str:
    clean = normalize_label(value)
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


def canonicalize_zeros(values: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(values)
    out[out == 0] = 0
    return out


def row_fingerprints(values: np.ndarray, *, dtype: str) -> np.ndarray:
    """SHA-256 each canonical numeric row; return fixed-width raw digests."""

    array = np.asarray(values, dtype=np.dtype(dtype))
    if array.ndim != 2:
        raise ValueError("row_fingerprints expects a two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("Cannot fingerprint non-finite values")
    array = canonicalize_zeros(array)
    output = np.empty(array.shape[0], dtype="S32")
    for index, row in enumerate(array):
        output[index] = hashlib.sha256(row.tobytes(order="C")).digest()
    return output


def tensor_fingerprints(values: np.ndarray, decimals: int) -> tuple[np.ndarray, np.ndarray]:
    actual = canonicalize_zeros(np.asarray(values, dtype="<f4"))
    rounded = np.round(actual, decimals=int(decimals)).astype("<f4", copy=False)
    rounded = canonicalize_zeros(rounded)
    return row_fingerprints(actual, dtype="<f4"), row_fingerprints(rounded, dtype="<f4")


def fingerprint_hex_strings(values: np.ndarray) -> np.ndarray:
    """Hex-encode fixed-width S32 digests without losing trailing NUL bytes."""

    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind != "S" or array.dtype.itemsize != 32:
        raise TypeError(f"Expected one-dimensional S32 digests, got {array.shape} {array.dtype}")
    octets = np.ascontiguousarray(array).view(np.uint8).reshape(len(array), 32)
    return np.asarray([row.tobytes().hex() for row in octets], dtype="<U64")


def _seed_for(seed: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _priority_for_rows(seed: int, task_id: str, source_name: str, row_count: int) -> np.ndarray:
    rng = np.random.default_rng(_seed_for(seed, task_id, source_name, "class-cap"))
    return rng.integers(0, np.iinfo(np.uint64).max, size=row_count, dtype=np.uint64)


def _trim_reservoir(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame
    priority = frame["__priority"].to_numpy(dtype=np.uint64)
    source_id = frame["__source_file_id"].to_numpy(dtype=np.int64)
    source_row = frame["__source_row"].to_numpy(dtype=np.int64)
    order = np.lexsort((source_row, source_id, priority))[:cap]
    return frame.iloc[np.sort(order)].reset_index(drop=True)


def _encode_task_labels(
    raw_labels: pd.Series,
    task_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
) -> np.ndarray:
    class_names = [str(name) for name in task_cfg["class_names"]]
    mapping = {name: idx for idx, name in enumerate(class_names)}
    if str(task_cfg["task"]) == "binary":
        benign = {normalize_label(value).lower() for value in dataset_cfg.get("benign_labels", ["BENIGN"])}
        labels = np.where(raw_labels.map(normalize_label).str.lower().isin(benign), "BENIGN", "ATTACK")
    else:
        family_map = {normalize_label(key): str(value) for key, value in dataset_cfg.get("attack_family_map", {}).items()}
        labels = raw_labels.map(lambda value: attack_family(value, family_map)).to_numpy(dtype=object)
    unknown = sorted(set(labels) - set(mapping))
    if unknown:
        raise ValueError(f"Task {task_cfg['task_id']} produced unconfigured classes: {unknown}")
    return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64)


def load_finite_capped_tasks(
    config: dict[str, Any],
) -> tuple[dict[str, TaskFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read once, apply fixed cleaning, and maintain deterministic class reservoirs."""

    dataset_cfg = config["dataset"]
    label_column = str(dataset_cfg.get("label_column", "Label"))
    drop_columns = {str(column) for column in dataset_cfg.get("drop_columns", [])}
    clip = float(dataset_cfg["fixed_abs_clip"])
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("dataset.fixed_abs_clip must be a positive finite constant")
    seed = int(config["split"]["seed"])
    tasks = config["tasks"]
    paths = source_paths(config)
    reservoirs: dict[str, dict[int, pd.DataFrame]] = {
        str(task["task_id"]): {} for task in tasks
    }
    valid_counts: dict[tuple[str, int], int] = {}
    inventory_rows: list[dict[str, Any]] = []
    nonfinite_rows: list[pd.DataFrame] = []
    feature_names: list[str] | None = None

    for source_id, path in enumerate(paths):
        frame = pd.read_parquet(path)
        frame.columns = frame.columns.astype(str).str.strip()
        if label_column not in frame.columns:
            raise ValueError(f"Label column {label_column!r} missing from {path}")
        columns = [
            column
            for column in frame.columns
            if column != label_column and column not in drop_columns
        ]
        if feature_names is None:
            feature_names = columns
        elif columns != feature_names:
            raise ValueError(f"Numeric feature schema/order differs in {path.name}")

        # The mask is fixed: coercion, preserve non-finite as missing, fixed
        # clipping of finite values, then remove every row with any non-finite.
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce").astype(np.float64)
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.clip(lower=-clip, upper=clip)
        finite_mask = np.isfinite(numeric.to_numpy(dtype=np.float64, copy=False)).all(axis=1)
        source_rows = np.arange(len(frame), dtype=np.int64)
        invalid = np.flatnonzero(~finite_mask)
        if len(invalid):
            nonfinite_rows.append(
                pd.DataFrame(
                    {
                        "source_file": path.name,
                        "source_row": source_rows[invalid],
                        "raw_label": frame[label_column].iloc[invalid].map(normalize_label).to_numpy(),
                        "reason": "nonfinite_after_fixed_coercion_and_clipping",
                    }
                )
            )
        numeric = numeric.loc[finite_mask].reset_index(drop=True)
        raw_labels = frame.loc[finite_mask, label_column].reset_index(drop=True)
        retained_source_rows = source_rows[finite_mask]
        inventory_rows.append(
            {
                "source_file": path.name,
                "raw_rows": int(len(frame)),
                "finite_rows": int(len(numeric)),
                "dropped_nonfinite_rows": int((~finite_mask).sum()),
                "feature_count": int(len(columns)),
            }
        )

        for task_cfg in tasks:
            task_id = str(task_cfg["task_id"])
            class_names = [str(name) for name in task_cfg["class_names"]]
            labels = _encode_task_labels(raw_labels, task_cfg, dataset_cfg)
            priorities = _priority_for_rows(seed, task_id, path.name, len(numeric))
            cap = int(task_cfg["max_rows_per_class"])
            for class_id in range(len(class_names)):
                selected = np.flatnonzero(labels == class_id)
                valid_counts[(task_id, class_id)] = valid_counts.get((task_id, class_id), 0) + int(len(selected))
                if not len(selected):
                    continue
                if len(selected) > cap:
                    local = np.argpartition(priorities[selected], cap - 1)[:cap]
                    selected = selected[local]
                chunk = numeric.iloc[selected].copy()
                chunk["__source_file_id"] = np.int16(source_id)
                chunk["__source_row"] = retained_source_rows[selected]
                chunk["__class_id"] = np.int16(class_id)
                chunk["__priority"] = priorities[selected]
                current = reservoirs[task_id].get(class_id)
                merged = chunk if current is None else pd.concat([current, chunk], ignore_index=True)
                reservoirs[task_id][class_id] = _trim_reservoir(merged, cap)

    if feature_names is None:
        raise RuntimeError("No feature schema was loaded")
    source_files = [path.name for path in paths]
    task_frames: dict[str, TaskFrame] = {}
    cap_rows: list[dict[str, Any]] = []
    for task_cfg in tasks:
        task_id = str(task_cfg["task_id"])
        class_names = [str(name) for name in task_cfg["class_names"]]
        missing = [class_names[i] for i in range(len(class_names)) if i not in reservoirs[task_id]]
        if missing:
            raise ValueError(f"Task {task_id} has no finite rows for classes: {missing}")
        combined = pd.concat(
            [reservoirs[task_id][class_id] for class_id in range(len(class_names))],
            ignore_index=True,
        )
        combined = combined.sort_values(["__source_file_id", "__source_row", "__class_id"]).reset_index(drop=True)
        X = combined[feature_names].to_numpy(dtype="<f8", copy=True)
        fingerprint = row_fingerprints(X, dtype="<f8")
        for class_id, class_name in enumerate(class_names):
            cap_rows.append(
                {
                    "task_id": task_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "finite_rows_before_cap": int(valid_counts.get((task_id, class_id), 0)),
                    "rows_after_cap": int(np.sum(combined["__class_id"].to_numpy() == class_id)),
                    "configured_cap": int(task_cfg["max_rows_per_class"]),
                }
            )
        task_frames[task_id] = TaskFrame(
            task_id=task_id,
            task=str(task_cfg["task"]),
            class_names=class_names,
            feature_names=feature_names,
            X=X,
            y=combined["__class_id"].to_numpy(dtype=np.int64),
            source_file_id=combined["__source_file_id"].to_numpy(dtype=np.int16),
            source_row=combined["__source_row"].to_numpy(dtype=np.int64),
            source_files=source_files,
            raw_fingerprint=fingerprint,
        )
    nonfinite = (
        pd.concat(nonfinite_rows, ignore_index=True)
        if nonfinite_rows
        else pd.DataFrame(columns=["source_file", "source_row", "raw_label", "reason"])
    )
    return (
        task_frames,
        pd.DataFrame(inventory_rows),
        pd.DataFrame(cap_rows),
        nonfinite,
    )


def _pairwise_overlap(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.intersect1d(left, right, assume_unique=False).size)


def group_safe_split(
    fingerprints: np.ndarray,
    labels: np.ndarray,
    *,
    train_size: float,
    validation_size: float,
    test_size: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Balance unique, label-consistent fingerprint groups by row count."""

    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("train/validation/test sizes must sum to one")
    groups = pd.DataFrame({"fingerprint": fingerprints, "label": labels})
    conflict = groups.groupby("fingerprint", sort=False)["label"].nunique().gt(1)
    conflict_values = conflict[conflict].index.to_numpy(dtype="S32")
    conflict_mask = np.isin(fingerprints, conflict_values) if len(conflict_values) else np.zeros(len(labels), dtype=bool)
    eligible = ~conflict_mask
    assignments = np.full(len(labels), "", dtype="U10")
    proportions = np.asarray([train_size, validation_size, test_size], dtype=float)
    for class_id in sorted(np.unique(labels[eligible]).tolist()):
        class_groups, group_sizes = np.unique(
            fingerprints[eligible & (labels == class_id)], return_counts=True
        )
        if len(class_groups) < 4:
            raise ValueError(
                f"Class {class_id} has only {len(class_groups)} unique raw groups; "
                "at least four are required for three-way splitting"
            )
        # Largest groups are allocated first against row-count targets.  A
        # seeded tie order makes the result fixed without letting source order
        # decide.  The empty-split constraint guarantees class coverage.
        rng = np.random.default_rng(_seed_for(seed, class_id, "balanced-groups"))
        tie_order = rng.permutation(len(class_groups))
        order = np.lexsort((tie_order, -group_sizes))
        target_rows = proportions * float(group_sizes.sum())
        allocated_rows = np.zeros(3, dtype=float)
        allocated_groups: list[list[bytes]] = [[], [], []]
        for position, group_index in enumerate(order):
            empty = [index for index, values in enumerate(allocated_groups) if not values]
            remaining_after = len(order) - position - 1
            candidates = empty if len(empty) > remaining_after else [0, 1, 2]
            candidate_order = rng.permutation(candidates)
            best_split = min(
                candidate_order,
                key=lambda candidate: float(
                    np.square(
                        allocated_rows
                        + np.eye(3)[candidate] * float(group_sizes[group_index])
                        - target_rows
                    ).sum()
                ),
            )
            allocated_groups[int(best_split)].append(class_groups[group_index])
            allocated_rows[int(best_split)] += float(group_sizes[group_index])
        for split_index, split in enumerate(SPLITS):
            assignments[np.isin(fingerprints, allocated_groups[split_index])] = split
    split_indices = {split: np.flatnonzero(assignments == split) for split in SPLITS}
    if sum(len(index) for index in split_indices.values()) + int(conflict_mask.sum()) != len(labels):
        raise AssertionError("Every non-conflicting row must receive exactly one split")
    for split in SPLITS:
        if set(np.unique(labels[split_indices[split]])) != set(np.unique(labels[eligible])):
            raise ValueError(f"Raw split {split} does not contain every class")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if _pairwise_overlap(fingerprints[split_indices[left]], fingerprints[split_indices[right]]) != 0:
            raise AssertionError(f"Raw fingerprint leakage between {left} and {right}")
    return split_indices, conflict_mask


def build_preprocessor(config: dict[str, Any]) -> Pipeline:
    preprocessing = config["preprocessing"]
    if str(preprocessing.get("scaler", "standard")).lower() != "standard":
        raise ValueError("Only the train-fitted StandardScaler is supported")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=str(preprocessing.get("imputer_strategy", "median")))),
            ("selector", VarianceThreshold(threshold=float(preprocessing.get("variance_threshold", 0.0)))),
            (
                "deduplicator",
                ExactDuplicateFeatureRemover(
                    enabled=bool(preprocessing.get("drop_exact_duplicate_features", True))
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )


def fit_transform_train_only(
    X: np.ndarray,
    split_indices: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[Pipeline, dict[str, np.ndarray]]:
    """Fit the complete learned pipeline once, exclusively on training rows."""

    pipeline = build_preprocessor(config)
    pipeline.fit(X[split_indices["train"]])
    transformed: dict[str, np.ndarray] = {}
    for split in SPLITS:
        values = pipeline.transform(X[split_indices[split]]).astype("<f4", copy=False)
        values = canonicalize_zeros(values)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite tensor after train-only transform in {split}")
        transformed[split] = values
    return pipeline, transformed


def representation_unique_holdouts(
    transformed: dict[str, np.ndarray],
    *,
    decimals: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    """Create label-blind train > validation > test representation masks."""

    fingerprints: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        exact, rounded = tensor_fingerprints(transformed[split], decimals)
        fingerprints[split] = {"exact": exact, "rounded": rounded}

    masks = {split: np.ones(len(transformed[split]), dtype=bool) for split in SPLITS}
    rows: list[dict[str, Any]] = []
    train_exact = fingerprints["train"]["exact"]
    train_rounded = fingerprints["train"]["rounded"]
    val_exact_conflict = np.isin(fingerprints["validation"]["exact"], train_exact)
    val_round_conflict = np.isin(fingerprints["validation"]["rounded"], train_rounded)
    masks["validation"] = ~(val_exact_conflict | val_round_conflict)

    earlier_exact = np.concatenate(
        [train_exact, fingerprints["validation"]["exact"][masks["validation"]]]
    )
    earlier_rounded = np.concatenate(
        [train_rounded, fingerprints["validation"]["rounded"][masks["validation"]]]
    )
    test_exact_conflict = np.isin(fingerprints["test"]["exact"], earlier_exact)
    test_round_conflict = np.isin(fingerprints["test"]["rounded"], earlier_rounded)
    masks["test"] = ~(test_exact_conflict | test_round_conflict)

    for split, exact_conflict, round_conflict in (
        ("validation", val_exact_conflict, val_round_conflict),
        ("test", test_exact_conflict, test_round_conflict),
    ):
        rows.append(
            {
                "split": split,
                "rows_before": int(len(masks[split])),
                "rows_dropped_exact": int(exact_conflict.sum()),
                "rows_dropped_rounded5": int(round_conflict.sum()),
                "rows_dropped_union": int((~masks[split]).sum()),
                "rows_after": int(masks[split].sum()),
                "filter_used_labels": False,
            }
        )
    rows.insert(
        0,
        {
            "split": "train",
            "rows_before": int(len(masks["train"])),
            "rows_dropped_exact": 0,
            "rows_dropped_rounded5": 0,
            "rows_dropped_union": 0,
            "rows_after": int(len(masks["train"])),
            "filter_used_labels": False,
        },
    )
    return masks, fingerprints, pd.DataFrame(rows)


def _validation_rows(
    raw_fingerprints: dict[str, np.ndarray],
    tensor_fingerprints_by_split: dict[str, dict[str, np.ndarray]],
    masks: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    class_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for stage, kind in (("raw_pre_filter", "raw"), ("tensor_pre_filter", "exact"), ("tensor_pre_filter", "rounded5")):
        for left, right in pairs:
            if kind == "raw":
                left_fp, right_fp = raw_fingerprints[left], raw_fingerprints[right]
            elif kind == "exact":
                left_fp = tensor_fingerprints_by_split[left]["exact"]
                right_fp = tensor_fingerprints_by_split[right]["exact"]
            else:
                left_fp = tensor_fingerprints_by_split[left]["rounded"]
                right_fp = tensor_fingerprints_by_split[right]["rounded"]
            overlap = _pairwise_overlap(left_fp, right_fp)
            rows.append(
                {
                    "stage": stage,
                    "representation": kind,
                    "split_a": left,
                    "split_b": right,
                    "overlap_unique_fingerprints": overlap,
                    "passed": bool(overlap == 0) if kind == "raw" else "diagnostic",
                }
            )
    for kind in ("exact", "rounded"):
        for left, right in pairs:
            overlap = _pairwise_overlap(
                tensor_fingerprints_by_split[left][kind][masks[left]],
                tensor_fingerprints_by_split[right][kind][masks[right]],
            )
            rows.append(
                {
                    "stage": "tensor_post_filter",
                    "representation": "rounded5" if kind == "rounded" else "exact",
                    "split_a": left,
                    "split_b": right,
                    "overlap_unique_fingerprints": overlap,
                    "passed": bool(overlap == 0),
                }
            )
    for split in SPLITS:
        present = set(np.unique(labels[split][masks[split]]).tolist())
        rows.append(
            {
                "stage": "tensor_post_filter",
                "representation": "class_coverage",
                "split_a": split,
                "split_b": "",
                "overlap_unique_fingerprints": "",
                "passed": bool(present == set(range(class_count))),
            }
        )
    return pd.DataFrame(rows)


def _split_balance_rows(
    labels_by_split: dict[str, np.ndarray],
    class_names: list[str],
    split_config: dict[str, Any],
) -> pd.DataFrame:
    proportions = {
        "train": float(split_config["train_size"]),
        "validation": float(split_config["validation_size"]),
        "test": float(split_config["test_size"]),
    }
    configured_tolerance = float(split_config["max_absolute_split_fraction_deviation"])
    rows: list[dict[str, Any]] = []
    total = sum(len(values) for values in labels_by_split.values())
    for split in SPLITS:
        observed = len(labels_by_split[split]) / max(total, 1)
        deviation = abs(observed - proportions[split])
        rows.append(
            {
                "stage": "raw_split_balance",
                "representation": "overall_row_fraction",
                "split_a": split,
                "split_b": "",
                "overlap_unique_fingerprints": "",
                "class_id": "",
                "class_name": "ALL",
                "expected_fraction": proportions[split],
                "observed_fraction": observed,
                "absolute_deviation": deviation,
                "tolerance": configured_tolerance,
                "passed": bool(deviation <= configured_tolerance),
            }
        )
    for class_id, class_name in enumerate(class_names):
        class_total = sum(int(np.sum(values == class_id)) for values in labels_by_split.values())
        # With rare classes, a single indivisible row/group can exceed the
        # global tolerance.  The class gate therefore includes the unavoidable
        # one-row discretisation bound and records it explicitly.
        tolerance = max(configured_tolerance, 1.0 / max(class_total, 1))
        for split in SPLITS:
            observed = int(np.sum(labels_by_split[split] == class_id)) / max(class_total, 1)
            deviation = abs(observed - proportions[split])
            rows.append(
                {
                    "stage": "raw_split_balance",
                    "representation": "class_row_fraction",
                    "split_a": split,
                    "split_b": "",
                    "overlap_unique_fingerprints": "",
                    "class_id": class_id,
                    "class_name": class_name,
                    "expected_fraction": proportions[split],
                    "observed_fraction": observed,
                    "absolute_deviation": deviation,
                    "tolerance": tolerance,
                    "passed": bool(deviation <= tolerance),
                }
            )
    return pd.DataFrame(rows)


def prepare_task(frame: TaskFrame, config: dict[str, Any]) -> PreparedTask:
    split_cfg = config["split"]
    split_indices, conflict_mask = group_safe_split(
        frame.raw_fingerprint,
        frame.y,
        train_size=float(split_cfg["train_size"]),
        validation_size=float(split_cfg["validation_size"]),
        test_size=float(split_cfg["test_size"]),
        seed=int(split_cfg["seed"]),
    )
    if conflict_mask.any() and not bool(split_cfg.get("drop_label_conflicting_fingerprint_groups", False)):
        raise ValueError("Label-conflicting exact feature groups exist and dropping is disabled")
    pipeline, transformed = fit_transform_train_only(frame.X, split_indices, config)
    decimals = int(config["representation_guard"]["rounded_float32_decimals"])
    masks, tensor_fp, representation_summary = representation_unique_holdouts(
        transformed,
        decimals=decimals,
    )

    raw_by_split = {split: frame.raw_fingerprint[index] for split, index in split_indices.items()}
    labels_by_split = {split: frame.y[index] for split, index in split_indices.items()}
    validation = _validation_rows(
        raw_by_split,
        tensor_fp,
        masks,
        labels_by_split,
        len(frame.class_names),
    )
    validation = pd.concat(
        [validation, _split_balance_rows(labels_by_split, frame.class_names, split_cfg)],
        ignore_index=True,
        sort=False,
    )
    required = validation[
        (validation["stage"] == "tensor_post_filter")
        | (validation["stage"] == "raw_pre_filter")
        | (validation["stage"] == "raw_split_balance")
    ]
    if not required["passed"].map(bool).all():
        raise AssertionError(f"Leakage/coverage invariant failed for {frame.task_id}")

    selector = pipeline.named_steps["selector"]
    deduplicator = pipeline.named_steps["deduplicator"]
    variance_features = np.asarray(frame.feature_names, dtype=object)[selector.get_support()]
    selected_features = variance_features[deduplicator.get_support()].tolist()
    representative_positions = np.flatnonzero(deduplicator.get_support())
    output_index = {
        int(source_position): int(index)
        for index, source_position in enumerate(representative_positions)
    }
    feature_groups: list[dict[str, Any]] = []
    for source_position in representative_positions:
        members = np.flatnonzero(deduplicator.duplicate_of_ == int(source_position))
        feature_groups.append(
            {
                "feature_index": output_index[int(source_position)],
                "representative": str(variance_features[int(source_position)]),
                "members": [str(variance_features[int(member)]) for member in members],
                "train_identical_member_count": int(len(members)),
            }
        )
    final: dict[str, dict[str, np.ndarray]] = {}
    distribution_rows: list[dict[str, Any]] = []
    dropped_parts: list[pd.DataFrame] = []
    if conflict_mask.any():
        conflict_index = np.flatnonzero(conflict_mask)
        dropped_parts.append(
            pd.DataFrame(
                {
                    "source_file": [frame.source_files[int(i)] for i in frame.source_file_id[conflict_index]],
                    "source_row": frame.source_row[conflict_index],
                    "class_id": frame.y[conflict_index],
                    "original_split": "unassigned",
                    "reason": "label_conflicting_raw_fingerprint_group",
                    "raw_fingerprint_sha256": fingerprint_hex_strings(
                        frame.raw_fingerprint[conflict_index]
                    ),
                    "tensor_fingerprint_exact_sha256": "",
                    "tensor_fingerprint_round5_sha256": "",
                }
            )
        )
    for split in SPLITS:
        source_index = split_indices[split]
        keep = masks[split]
        removed = np.flatnonzero(~keep)
        if len(removed):
            exact_reference = tensor_fp["train"]["exact"] if split == "validation" else np.concatenate(
                [tensor_fp["train"]["exact"], tensor_fp["validation"]["exact"][masks["validation"]]]
            )
            rounded_reference = tensor_fp["train"]["rounded"] if split == "validation" else np.concatenate(
                [tensor_fp["train"]["rounded"], tensor_fp["validation"]["rounded"][masks["validation"]]]
            )
            exact_conflict = np.isin(tensor_fp[split]["exact"][removed], exact_reference)
            rounded_conflict = np.isin(tensor_fp[split]["rounded"][removed], rounded_reference)
            reasons = np.where(
                exact_conflict & rounded_conflict,
                "tensor_exact_and_round5_collision",
                np.where(exact_conflict, "tensor_exact_collision", "tensor_round5_collision"),
            )
            absolute = source_index[removed]
            dropped_parts.append(
                pd.DataFrame(
                    {
                        "source_file": [frame.source_files[int(i)] for i in frame.source_file_id[absolute]],
                        "source_row": frame.source_row[absolute],
                        "class_id": frame.y[absolute],
                        "original_split": split,
                        "reason": reasons,
                        "raw_fingerprint_sha256": fingerprint_hex_strings(
                            frame.raw_fingerprint[absolute]
                        ),
                        "tensor_fingerprint_exact_sha256": fingerprint_hex_strings(
                            tensor_fp[split]["exact"][removed]
                        ),
                        "tensor_fingerprint_round5_sha256": fingerprint_hex_strings(
                            tensor_fp[split]["rounded"][removed]
                        ),
                    }
                )
            )
        retained = source_index[keep]
        final[split] = {
            "X": transformed[split][keep],
            "y": frame.y[retained].astype(np.int64, copy=False),
            "source_file": np.asarray(
                [frame.source_files[int(i)] for i in frame.source_file_id[retained]], dtype="S96"
            ),
            "source_row": frame.source_row[retained].astype(np.int64, copy=False),
            "raw_fingerprint": frame.raw_fingerprint[retained],
            "tensor_fingerprint_exact": tensor_fp[split]["exact"][keep],
            "tensor_fingerprint_round5": tensor_fp[split]["rounded"][keep],
        }
        pre_labels = frame.y[source_index]
        post_labels = frame.y[retained]
        for class_id, class_name in enumerate(frame.class_names):
            distribution_rows.append(
                {
                    "task_id": frame.task_id,
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "rows_before_representation_filter": int(np.sum(pre_labels == class_id)),
                    "rows_after_representation_filter": int(np.sum(post_labels == class_id)),
                    "rows_dropped_by_representation_filter": int(
                        np.sum(pre_labels == class_id) - np.sum(post_labels == class_id)
                    ),
                }
            )
    dropped = (
        pd.concat(dropped_parts, ignore_index=True)
        if dropped_parts
        else pd.DataFrame(
            columns=[
                "source_file",
                "source_row",
                "class_id",
                "original_split",
                "reason",
                "raw_fingerprint_sha256",
                "tensor_fingerprint_exact_sha256",
                "tensor_fingerprint_round5_sha256",
            ]
        )
    )
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    duplicate_groups = [group for group in feature_groups if len(group["members"]) > 1]
    preprocessing_summary = {
        "fit_partition": "train_only",
        "input_feature_count": len(frame.feature_names),
        "selected_feature_count": len(selected_features),
        "dropped_train_constant_features": sorted(
            set(frame.feature_names) - set(variance_features.tolist())
        ),
        "dropped_train_exact_duplicate_feature_count": int(
            len(variance_features) - len(selected_features)
        ),
        "train_exact_duplicate_feature_groups": duplicate_groups,
        "imputer": {
            "type": type(imputer).__name__,
            "strategy": imputer.strategy,
            "statistics": np.asarray(imputer.statistics_).tolist(),
        },
        "selector": {
            "type": type(selector).__name__,
            "threshold": float(selector.threshold),
            "support": selector.get_support().astype(bool).tolist(),
        },
        "deduplicator": {
            "type": type(deduplicator).__name__,
            "fit_partition": "train_only",
            "enabled": bool(deduplicator.enabled),
            "support_after_variance_filter": deduplicator.get_support().astype(bool).tolist(),
            "duplicate_of_after_variance_filter": deduplicator.duplicate_of_.astype(int).tolist(),
        },
        "scaler": {
            "type": type(scaler).__name__,
            "mean": np.asarray(scaler.mean_).tolist(),
            "scale": np.asarray(scaler.scale_).tolist(),
        },
        "output_dtype": "float32",
        "representation_guard": config["representation_guard"],
    }
    return PreparedTask(
        task_id=frame.task_id,
        task=frame.task,
        class_names=frame.class_names,
        input_feature_names=frame.feature_names,
        feature_names=selected_features,
        feature_groups=feature_groups,
        pipeline=pipeline,
        splits=final,
        dropped_rows=dropped,
        validation=validation,
        distribution=pd.DataFrame(distribution_rows),
        representation_summary=representation_summary.assign(task_id=frame.task_id),
        preprocessing_summary=preprocessing_summary,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_output_namespace_available(results_root: Path, tables_root: Path) -> None:
    decision = results_root / "decision.json"
    protected = [decision, tables_root / "split_validation.csv"]
    protected.extend(results_root.glob("processed/*/cache_metadata.json"))
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an invalid/existing leakage-free preparation namespace: "
            + ", ".join(path.as_posix() for path in existing[:5])
        )


def validate_cache(config_path: Path = DEFAULT_CONFIG, *, quiet: bool = False) -> bool:
    config = read_config(config_path)
    results_root = ROOT / str(config["outputs"]["results_root"])
    decision_path = results_root / "decision.json"
    if not decision_path.is_file():
        if not quiet:
            print("leakage-free data cache: missing decision.json")
        return False
    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        expected = stable_hash(stage_payload(config))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    valid = (
        decision.get("status") == "complete"
        and decision.get("signature") == expected
        and records_valid(decision.get("outputs", []))
    )
    for task_meta_record in decision.get("task_metadata", []):
        path = ROOT / str(task_meta_record.get("path", ""))
        if not path.is_file() or path.stat().st_size != int(task_meta_record.get("size_bytes", -1)):
            valid = False
            break
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            valid = False
            break
        if metadata.get("signature") != expected or not records_valid(metadata.get("outputs", [])):
            valid = False
            break
    if not quiet:
        print(f"leakage-free data cache: {'valid' if valid else 'invalid'}")
    return bool(valid)


def _write_task(
    prepared: PreparedTask,
    out_dir: Path,
    *,
    signature: str,
    payload: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=False)
    written: list[Path] = []
    for split in SPLITS:
        split_payload = prepared.splits[split]
        suffix = "val" if split == "validation" else split
        arrays = {
            f"X_{suffix}.npy": split_payload["X"],
            f"y_{suffix}.npy": split_payload["y"],
            f"source_file_{suffix}.npy": split_payload["source_file"],
            f"source_row_{suffix}.npy": split_payload["source_row"],
            f"raw_fingerprint_{suffix}.npy": split_payload["raw_fingerprint"],
            f"tensor_fingerprint_exact_{suffix}.npy": split_payload["tensor_fingerprint_exact"],
            f"tensor_fingerprint_round5_{suffix}.npy": split_payload["tensor_fingerprint_round5"],
        }
        for name, values in arrays.items():
            path = out_dir / name
            np.save(path, values, allow_pickle=False)
            written.append(path)
    baseline = np.median(prepared.splits["train"]["X"], axis=0).astype("<f4")
    np.save(out_dir / "baseline.npy", baseline, allow_pickle=False)
    written.append(out_dir / "baseline.npy")
    _write_json(out_dir / "feature_names.json", prepared.feature_names)
    written.append(out_dir / "feature_names.json")
    _write_json(out_dir / "feature_groups.json", prepared.feature_groups)
    written.append(out_dir / "feature_groups.json")
    _write_json(out_dir / "input_feature_names.json", prepared.input_feature_names)
    written.append(out_dir / "input_feature_names.json")
    mapping = {name: index for index, name in enumerate(prepared.class_names)}
    _write_json(out_dir / "label_mapping.json", mapping)
    written.append(out_dir / "label_mapping.json")
    _write_json(out_dir / "preprocessor_metadata.json", prepared.preprocessing_summary)
    written.append(out_dir / "preprocessor_metadata.json")
    joblib.dump(prepared.pipeline, out_dir / "preprocessor.joblib")
    written.append(out_dir / "preprocessor.joblib")
    # Compatibility alias is explicit; it is the scaler fitted within the same
    # train-only pipeline, not a second fit.
    joblib.dump(prepared.pipeline.named_steps["scaler"], out_dir / "scaler.joblib")
    written.append(out_dir / "scaler.joblib")
    prepared.validation.to_csv(out_dir / "split_validation.csv", index=False)
    written.append(out_dir / "split_validation.csv")
    prepared.dropped_rows.to_csv(out_dir / "dropped_rows.csv", index=False)
    written.append(out_dir / "dropped_rows.csv")
    records = [file_record(path) for path in written]
    metadata = {
        "pipeline_version": str(payload["config"]["pipeline_version"]),
        "stage": "leakage_free_data_preparation",
        "status": "complete",
        "created_at": utc_now(),
        "task_id": prepared.task_id,
        "dataset": str(payload["config"]["dataset"]["name"]),
        "task": prepared.task,
        "signature": signature,
        "config_signature": payload["config_signature"],
        "implementation_hash": payload["implementation_hash"],
        "raw_inputs": payload["raw_inputs"],
        "implementation": payload["implementation"],
        "outputs": records,
        "invariants": {
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
        },
    }
    metadata_path = out_dir / "cache_metadata.json"
    _write_json(metadata_path, metadata)
    return metadata_path, records


def prepare_all(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_config(config_path)
    if validate_cache(config_path, quiet=True):
        return json.loads(
            (ROOT / str(config["outputs"]["results_root"]) / "decision.json").read_text(encoding="utf-8")
        )
    results_root = ROOT / str(config["outputs"]["results_root"])
    tables_root = ROOT / str(config["outputs"]["tables_root"])
    _assert_output_namespace_available(results_root, tables_root)
    payload = stage_payload(config)
    signature = stable_hash(payload)
    task_frames, inventory, caps, nonfinite = load_finite_capped_tasks(config)
    prepared_tasks = [prepare_task(task_frames[str(task["task_id"])], config) for task in config["tasks"]]

    processed_root = results_root / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    task_metadata: list[dict[str, Any]] = []
    output_records: list[dict[str, Any]] = []
    for prepared in prepared_tasks:
        metadata_path, records = _write_task(
            prepared,
            processed_root / prepared.task_id,
            signature=signature,
            payload=payload,
        )
        task_metadata.append(file_record(metadata_path))
        output_records.extend(records)

    table_frames = {
        "source_cleaning.csv": inventory,
        "class_caps.csv": caps,
        "dropped_nonfinite_rows.csv": nonfinite,
        "split_distribution.csv": pd.concat([task.distribution for task in prepared_tasks], ignore_index=True),
        "representation_filter.csv": pd.concat(
            [task.representation_summary for task in prepared_tasks], ignore_index=True
        ),
        "split_validation.csv": pd.concat(
            [task.validation.assign(task_id=task.task_id) for task in prepared_tasks], ignore_index=True
        ),
        "feature_preprocessing.csv": pd.DataFrame(
            [
                {
                    "task_id": task.task_id,
                    "fit_partition": "train_only",
                    "input_feature_count": len(task.input_feature_names),
                    "selected_feature_count": len(task.feature_names),
                    "dropped_train_constant_feature_count": len(
                        task.preprocessing_summary["dropped_train_constant_features"]
                    ),
                    "dropped_train_exact_duplicate_feature_count": task.preprocessing_summary[
                        "dropped_train_exact_duplicate_feature_count"
                    ],
                    "imputer": "SimpleImputer(median)",
                    "selector": "VarianceThreshold(0.0) + train-exact duplicate grouping",
                    "scaler": "StandardScaler",
                    "output_dtype": "float32",
                }
                for task in prepared_tasks
            ]
        ),
    }
    for name, table in table_frames.items():
        path = tables_root / name
        table.to_csv(path, index=False)
        output_records.append(file_record(path))

    decision = {
        "pipeline_version": str(config["pipeline_version"]),
        "stage": "leakage_free_data_preparation",
        "status": "complete",
        "created_at": utc_now(),
        "signature": signature,
        "config_signature": payload["config_signature"],
        "implementation_hash": payload["implementation_hash"],
        "raw_inputs": payload["raw_inputs"],
        "implementation": payload["implementation"],
        "task_metadata": task_metadata,
        "outputs": output_records,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "claim_scope": {
            "exact_and_round5_model_tensor_cross_split_overlap": 0,
            "train_exact_duplicate_model_coordinate_pairs": 0,
            "session_independent": False,
            "capture_day_independent": False,
        },
    }
    results_root.mkdir(parents=True, exist_ok=True)
    _write_json(results_root / "decision.json", decision)
    return decision


__all__ = [
    "DEFAULT_CONFIG",
    "SPLITS",
    "build_preprocessor",
    "fit_transform_train_only",
    "group_safe_split",
    "prepare_all",
    "representation_unique_holdouts",
    "row_fingerprints",
    "tensor_fingerprints",
    "validate_cache",
]
