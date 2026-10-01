"""Study 1: temporal and host/session evaluation splits.

The baseline splits each task once at random, 70/15/15, with group safety on
the exact feature-vector hash.  Its own preparation decision records the
consequence explicitly - ``session_independent: false`` and
``capture_day_independent: false`` - so flows from the same host, the same
5-tuple session, or the same minute of capture can sit on both sides of the
split.  Predictive metrics and explanation-reliability estimates measured under
that split can therefore be optimistic in a way no feature-hash guard detects.

This module rebuilds the split under four declared policies, re-runs the frozen
preparation, training and audit code against each of them, and compares the
resulting reliability profile with the baseline.

CICIDS2017 only: NF-ToN-IoT-V2 as distributed here has neither a timestamp
column nor IPv4 addresses, so no temporal or host policy is constructible for
it.  That is recorded as an infeasibility.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.leakage_free_tabular import prepare, training
from src.leakage_free_tabular.cache import (
    artifact_records,
    cache_hit,
    sha256_file,
    signature,
    write_cache_metadata,
)
from src.leakage_free_tabular.epistemic_audit import (
    ALLOWED_MODELS,
    AuditPaths,
    ProbeSpec,
    _as_bool,
    _compute_probe,
    _flow_id,
    _load_frozen_model,
    _prediction_margin,
    _processed_probe_inputs,
    functional_checks_for_arrays,
    load_condition_data,
    require_valid_training_condition,
    summarize_metrics,
)
from src.leakage_free_tabular_ext.common import (
    ExtensionContext,
    similarity_matrix_at,
    write_decision,
    write_extension_table,
)

STUDY = "study1_deployment_splits"
BASELINE_PREPARE_CONFIG = Path("configs/leakage_free_tabular.yaml")
METADATA_COLUMNS = (
    "timestamp",
    "source_address",
    "destination_address",
    "source_port",
    "destination_port",
    "protocol",
)
DETERMINISTIC_SPECS = {
    "gradient_x_input": ProbeSpec("gradient_x_input", "input_zero", role="primary"),
    "integrated_gradients": ProbeSpec(
        "integrated_gradients", "training_median", 64, role="primary"
    ),
    "occlusion": ProbeSpec("occlusion", "training_median", role="primary"),
}
REPEATABILITY_SPECS = {
    "integrated_gradients": (
        ProbeSpec("integrated_gradients", "training_median", 64, role="primary"),
        ProbeSpec(
            "integrated_gradients", "training_median", 128, role="numerical_sensitivity"
        ),
        "steps_64_vs_128",
    ),
    "occlusion": (
        ProbeSpec("occlusion", "training_median", role="primary"),
        ProbeSpec("occlusion", "training_median", run_id=1, role="determinism_replicate"),
        "fixed_rule_repeat",
    ),
}
EXTRA_SPECS = (
    ProbeSpec("integrated_gradients", "training_median", 128, role="numerical_sensitivity"),
    ProbeSpec("occlusion", "training_median", run_id=1, role="determinism_replicate"),
)
METRICS = ("spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1")


def _policy_root(context: ExtensionContext, policy_id: str) -> Path:
    return context.results_root / STUDY / "policies" / policy_id


def _policies(context: ExtensionContext, policies: Sequence[str] | None) -> list[dict[str, Any]]:
    declared = {str(item["policy_id"]): dict(item) for item in context.study(STUDY)["policies"]}
    names = list(policies or declared)
    unknown = sorted(set(names) - set(declared))
    if unknown:
        raise ValueError(f"Policies are not declared in the extension config: {unknown}")
    return [declared[name] for name in names]


# --------------------------------------------------------------------------- #
# Stage 1 - flow metadata
# --------------------------------------------------------------------------- #
def build_flow_metadata(
    context: ExtensionContext, *, force: bool = False, logger=None
) -> pd.DataFrame:
    """Recover capture time and endpoints per source row, from the raw parquet.

    These columns are split metadata only.  They are dropped from the model
    input by the frozen preparation recipe (``dataset.drop_columns``) and never
    reach a feature vector, so using them to constrain the split cannot leak
    identifiers into the models.
    """

    study = context.study(STUDY)
    columns = study["metadata_columns"]
    source_root = context.project_root / str(study["metadata_source"]["cicids2017"])
    directory = context.results_dir(STUDY, "metadata")
    inventory: list[dict[str, Any]] = []
    for path in sorted(source_root.glob("*.parquet")):
        output = directory / f"{path.stem}.parquet"
        if output.is_file() and not force:
            frame = pd.read_parquet(output)
        else:
            wanted = [str(columns[key]) for key in METADATA_COLUMNS]
            raw = pd.read_parquet(path, columns=wanted)
            raw.columns = [str(column).strip() for column in raw.columns]
            frame = pd.DataFrame(
                {
                    "source_file": path.name,
                    "source_row": np.arange(len(raw), dtype=np.int64),
                    "timestamp": pd.to_datetime(
                        raw[str(columns["timestamp"])], errors="coerce"
                    ),
                    "source_address": raw[str(columns["source_address"])].astype(str),
                    "destination_address": raw[
                        str(columns["destination_address"])
                    ].astype(str),
                    "source_port": pd.to_numeric(
                        raw[str(columns["source_port"])], errors="coerce"
                    ).astype("Int64"),
                    "destination_port": pd.to_numeric(
                        raw[str(columns["destination_port"])], errors="coerce"
                    ).astype("Int64"),
                    "protocol": pd.to_numeric(
                        raw[str(columns["protocol"])], errors="coerce"
                    ).astype("Int64"),
                }
            )
            frame["session_five_tuple"] = _session_keys(frame)
            frame.to_parquet(output, index=False)
        inventory.append(
            {
                "source_file": path.name,
                "rows": int(len(frame)),
                "timestamp_parsed_fraction": float(frame["timestamp"].notna().mean()),
                "distinct_source_addresses": int(frame["source_address"].nunique()),
                "distinct_sessions": int(frame["session_five_tuple"].nunique()),
                "earliest_timestamp": str(frame["timestamp"].min()),
                "latest_timestamp": str(frame["timestamp"].max()),
            }
        )
        if logger is not None:
            logger.info(
                "%s: %d rows, %.4f timestamps parsed, %d hosts, %d sessions",
                path.name,
                len(frame),
                float(frame["timestamp"].notna().mean()),
                int(frame["source_address"].nunique()),
                int(frame["session_five_tuple"].nunique()),
            )
    frame = pd.DataFrame(inventory)
    frame["dataset"] = "cicids2017"
    unavailable = pd.DataFrame(
        [
            {
                "dataset": "ton_iot",
                "source_file": "NF-ToN-IoT-V2.parquet",
                "rows": np.nan,
                "timestamp_parsed_fraction": 0.0,
                "distinct_source_addresses": 0,
                "distinct_sessions": np.nan,
                "earliest_timestamp": "",
                "latest_timestamp": "",
                "status": str(context.study(STUDY)["metadata_source"]["ton_iot"]),
            }
        ]
    )
    frame["status"] = "available"
    combined = pd.concat([frame, unavailable], ignore_index=True, sort=False)
    write_extension_table(
        combined, context.tables_dir(STUDY) / "flow_metadata_inventory.csv"
    )
    return combined


def _session_keys(frame: pd.DataFrame) -> pd.Series:
    """Direction-insensitive 5-tuple: sorted endpoints plus protocol."""

    left = frame["source_address"].astype(str)
    right = frame["destination_address"].astype(str)
    left_port = frame["source_port"].astype("Int64").astype(str)
    right_port = frame["destination_port"].astype("Int64").astype(str)
    low_first = left <= right
    endpoint_a = np.where(low_first, left, right)
    endpoint_b = np.where(low_first, right, left)
    port_a = np.where(low_first, left_port, right_port)
    port_b = np.where(low_first, right_port, left_port)
    return pd.Series(
        [
            f"{a}|{b}|{pa}|{pb}|{proto}"
            for a, b, pa, pb, proto in zip(
                endpoint_a, endpoint_b, port_a, port_b, frame["protocol"].astype(str)
            )
        ],
        index=frame.index,
        dtype=object,
    )


def load_flow_metadata(context: ExtensionContext) -> pd.DataFrame:
    directory = context.results_root / STUDY / "metadata"
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise RuntimeError("Run the study 1 'metadata' stage first")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


# --------------------------------------------------------------------------- #
# Stage 2 - policy split assignments
# --------------------------------------------------------------------------- #
def merged_components(key_arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Connected components over several row-keyed grouping variables.

    Both the frozen feature-hash groups and the policy's own groups have to
    stay inside one partition.  When they overlap - one host produces two rows
    with the same feature vector, or one feature-hash group spans two hosts -
    the only assignment that respects both is the union of the two groupings.
    The union is computed as connected components of the bipartite
    row-to-group-value graph, which is linear in the number of rows.
    """

    if not key_arrays:
        raise ValueError("At least one grouping key is required")
    length = len(key_arrays[0])
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    offset = length
    for keys in key_arrays:
        if len(keys) != length:
            raise ValueError("Grouping keys must align by row")
        codes = pd.factorize(pd.Series(np.asarray(keys)), sort=False)[0]
        rows.append(np.arange(length, dtype=np.int64))
        columns.append(codes.astype(np.int64) + offset)
        offset += int(codes.max()) + 1
    row_index = np.concatenate(rows)
    column_index = np.concatenate(columns)
    graph = coo_matrix(
        (np.ones(len(row_index), dtype=np.int8), (row_index, column_index)),
        shape=(offset, offset),
    )
    _, labels = connected_components(graph, directed=False)
    _, codes = np.unique(labels[:length], return_inverse=True)
    return codes.astype(int)


def _class_row_matrix(components: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.zeros((int(components.max()) + 1, int(n_classes)), dtype=np.int64)
    np.add.at(counts, (components, labels), 1)
    return counts


def _temporal_assignment(
    components: np.ndarray,
    order_value: np.ndarray,
    labels: np.ndarray,
    *,
    proportions: np.ndarray,
    per_class: bool,
    n_classes: int,
) -> np.ndarray:
    """Assign whole components in time order, cutting at the row quantiles."""

    n_components = int(components.max()) + 1
    component_time = np.full(n_components, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(component_time, components, np.asarray(order_value, dtype=np.int64))
    assignment = np.full(len(components), -1, dtype=int)
    boundaries = np.cumsum(proportions)
    groups = (
        [np.flatnonzero(labels == class_id) for class_id in range(int(n_classes))]
        if per_class
        else [np.arange(len(components))]
    )
    for rows in groups:
        if not len(rows):
            continue
        local = components[rows]
        unique_local, inverse = np.unique(local, return_inverse=True)
        sizes = np.bincount(inverse, minlength=len(unique_local))
        order = np.argsort(component_time[unique_local], kind="stable")
        cumulative = np.cumsum(sizes[order]) / max(1, int(sizes.sum()))
        split_of_ordered = np.where(
            cumulative <= boundaries[0], 0, np.where(cumulative <= boundaries[1], 1, 2)
        )
        split_of_local = np.empty(len(unique_local), dtype=int)
        split_of_local[order] = split_of_ordered
        assignment[rows] = split_of_local[inverse]
    return assignment


def _balanced_group_assignment(
    components: np.ndarray,
    labels: np.ndarray,
    *,
    proportions: np.ndarray,
    n_classes: int,
    seed: int,
) -> np.ndarray:
    """Largest component first, minimising squared deviation per class.

    A generalisation of the frozen ``group_safe_split`` heuristic: the frozen
    version assigns single-label feature-hash groups and drops groups whose
    label is ambiguous, whereas a host or a session normally carries several
    classes and cannot be dropped without discarding most of the capture.  The
    deviation is therefore measured against a per-class row target vector.

    Choosing the partition that minimises the squared deviation reduces to
    minimising ``counts . (allocated - target)`` over the three partitions,
    because the component's own squared norm is the same wherever it goes.
    """

    counts = _class_row_matrix(components, labels, n_classes)
    targets = np.outer(proportions, counts.sum(axis=0)).astype(float)
    allocated = np.zeros_like(targets, dtype=float)
    order = np.lexsort(
        (
            np.random.default_rng(int(seed)).permutation(len(counts)),
            -counts.sum(axis=1),
        )
    )
    component_split = np.full(len(counts), -1, dtype=int)
    for component in order:
        row = counts[component].astype(float)
        best = int(np.argmin((allocated - targets) @ row))
        component_split[component] = best
        allocated[best] += row
    component_split = _repair_class_coverage(counts, component_split, n_classes)
    return component_split[components]


def _repair_class_coverage(
    counts: np.ndarray,
    component_split: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Move the smallest donor component into any partition missing a class."""

    sizes = counts.sum(axis=1)
    for _ in range(3 * int(n_classes)):
        per_split = np.zeros((3, int(n_classes)), dtype=np.int64)
        for split_index in range(3):
            members = component_split == split_index
            if members.any():
                per_split[split_index] = counts[members].sum(axis=0)
        deficits = np.argwhere(per_split == 0)
        if not len(deficits):
            return component_split
        split_index, class_id = (int(value) for value in deficits[0])
        donor_mask = (component_split != split_index) & (counts[:, class_id] > 0)
        # A donor may not empty its own partition of that class.
        remaining = per_split[component_split, class_id] - counts[:, class_id]
        donor_mask &= remaining > 0
        if not donor_mask.any():
            return component_split
        candidates = np.flatnonzero(donor_mask)
        component_split[candidates[np.argmin(sizes[candidates])]] = split_index
    return component_split


def _assignment_seed(split_cfg: Mapping[str, Any]) -> int:
    """The policy's own tie-break seed, never the reservoir/cap seed."""

    return int(split_cfg.get("assignment_seed", split_cfg["seed"]))


def _policy_split(
    policy: Mapping[str, Any],
    *,
    fingerprints: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    n_classes: int,
    split_cfg: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Split indices, dropped-group mask and diagnostics for one policy."""

    proportions = np.asarray(
        [
            float(split_cfg["train_size"]),
            float(split_cfg["validation_size"]),
            float(split_cfg["test_size"]),
        ],
        dtype=float,
    )
    conflict = (
        pd.DataFrame({"fingerprint": fingerprints, "label": labels})
        .groupby("fingerprint", sort=False)["label"]
        .nunique()
        .gt(1)
    )
    conflict_values = conflict[conflict].index.to_numpy()
    conflict_mask = (
        np.isin(fingerprints, conflict_values)
        if len(conflict_values)
        else np.zeros(len(labels), dtype=bool)
    )
    eligible = np.flatnonzero(~conflict_mask)
    if not len(eligible):
        raise RuntimeError("Every feature-hash group is label conflicting")

    # Fingerprints are raw 32-byte SHA-256 digests (dtype S32); decoding them
    # as text fails on non-ASCII bytes, and the grouping only needs equality.
    fingerprint_keys = np.asarray(fingerprints)[eligible]
    if "group_key" in policy and bool(policy.get("drop_crossing_groups", False)):
        # Keep the policy grouping exactly disjoint by removing the rows whose
        # feature-hash group spans several policy groups, instead of merging
        # those groups into one component.  The removed rows are reported.
        group_values = (
            metadata[str(policy["group_key"])].astype(str).to_numpy()
        )
        frame = pd.DataFrame(
            {
                "fingerprint": pd.factorize(pd.Series(np.asarray(fingerprints)), sort=False)[0],
                "group": group_values,
            }
        )
        spanning = frame.groupby("fingerprint")["group"].nunique().gt(1)
        crossing_mask = frame["fingerprint"].map(spanning).fillna(False).to_numpy()
        conflict_mask = conflict_mask | crossing_mask
        eligible = np.flatnonzero(~conflict_mask)
        if not len(eligible):
            raise RuntimeError(
                "Every feature-hash group spans several policy groups; a disjoint "
                "split would discard the whole task"
            )
        components = merged_components([group_values[eligible]])
        assignment_local = _balanced_group_assignment(
            components,
            labels[eligible],
            proportions=proportions,
            n_classes=n_classes,
            seed=_assignment_seed(split_cfg),
        )
    elif "group_key" in policy:
        group_values = metadata.iloc[eligible][str(policy["group_key"])].astype(str).to_numpy()
        components = merged_components([fingerprint_keys, group_values])
        assignment_local = _balanced_group_assignment(
            components,
            labels[eligible],
            proportions=proportions,
            n_classes=n_classes,
            seed=_assignment_seed(split_cfg),
        )
    else:
        components = merged_components([fingerprint_keys])
        order_value = (
            metadata.iloc[eligible]["timestamp"].astype("int64").to_numpy()
        )
        assignment_local = _temporal_assignment(
            components,
            order_value,
            labels[eligible],
            proportions=proportions,
            per_class=bool(policy.get("per_class", False)),
            n_classes=n_classes,
        )
    assignment = np.full(len(labels), -1, dtype=int)
    assignment[eligible] = assignment_local
    split_indices = {
        split: np.flatnonzero(assignment == index)
        for index, split in enumerate(prepare.SPLITS)
    }
    diagnostics = _split_diagnostics(
        policy,
        split_indices=split_indices,
        labels=labels,
        metadata=metadata,
        proportions=proportions,
        n_classes=n_classes,
        conflict_rows=int(conflict_mask.sum()),
        components=components,
        eligible=eligible,
    )
    return split_indices, conflict_mask, diagnostics


def _split_diagnostics(
    policy: Mapping[str, Any],
    *,
    split_indices: Mapping[str, np.ndarray],
    labels: np.ndarray,
    metadata: pd.DataFrame,
    proportions: np.ndarray,
    n_classes: int,
    conflict_rows: int,
    components: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, Any]:
    total = sum(len(index) for index in split_indices.values())
    record: dict[str, Any] = {
        "policy_id": str(policy["policy_id"]),
        "rows_assigned": int(total),
        "rows_dropped_label_conflicting": int(conflict_rows),
        "components": int(len(np.unique(components))),
    }
    for position, split in enumerate(prepare.SPLITS):
        index = split_indices[split]
        record[f"{split}_rows"] = int(len(index))
        record[f"{split}_fraction"] = float(len(index) / max(1, total))
        record[f"{split}_fraction_deviation"] = float(
            abs(len(index) / max(1, total) - proportions[position])
        )
        record[f"{split}_classes_present"] = int(len(np.unique(labels[index]))) if len(index) else 0
        if len(index):
            times = metadata.iloc[index]["timestamp"]
            record[f"{split}_earliest"] = str(times.min())
            record[f"{split}_latest"] = str(times.max())
    record["all_classes_each_split"] = all(
        record[f"{split}_classes_present"] == int(n_classes) for split in prepare.SPLITS
    )
    train_times = metadata.iloc[split_indices["train"]]["timestamp"]
    test_times = metadata.iloc[split_indices["test"]]["timestamp"]
    if len(train_times) and len(test_times):
        # A whole feature-hash group has to stay in one partition, and a group
        # of identical feature vectors can span capture days.  The strict
        # "no test row precedes the last train row" test is therefore dominated
        # by those groups; the quantile-based measures below say how much of
        # the partition is actually ordered in time.
        record["train_timestamp_median"] = str(train_times.median())
        record["test_timestamp_median"] = str(test_times.median())
        record["train_timestamp_p95"] = str(train_times.quantile(0.95))
        record["test_timestamp_p05"] = str(test_times.quantile(0.05))
        record["test_rows_before_last_train_row"] = int(
            (test_times < train_times.max()).sum()
        )
        record["test_rows_before_train_median"] = int(
            (test_times < train_times.median()).sum()
        )
        record["fraction_test_before_train_median"] = float(
            (test_times < train_times.median()).mean()
        )
        record["fraction_test_before_train_p95"] = float(
            (test_times < train_times.quantile(0.95)).mean()
        )
        record["fraction_train_after_test_p05"] = float(
            (train_times > test_times.quantile(0.05)).mean()
        )
    assigned_rows = np.concatenate([split_indices[split] for split in prepare.SPLITS])
    day = metadata.iloc[assigned_rows]["timestamp"].dt.date.astype(str).to_numpy()
    component_of_row = components[
        np.searchsorted(eligible, assigned_rows)
    ]
    day_frame = pd.DataFrame({"component": component_of_row, "day": day})
    days_per_component = day_frame.groupby("component")["day"].nunique()
    record["components_spanning_several_capture_days"] = int(
        (days_per_component > 1).sum()
    )
    record["rows_in_components_spanning_several_days"] = int(
        day_frame["component"].isin(days_per_component[days_per_component > 1].index).sum()
    )
    sizes = np.bincount(component_of_row)
    record["largest_component_rows"] = int(sizes.max()) if len(sizes) else 0
    record["largest_component_row_fraction"] = (
        float(sizes.max() / max(1, len(component_of_row))) if len(sizes) else 0.0
    )
    for key, column in (
        ("host", "source_address"),
        ("session", "session_five_tuple"),
    ):
        groups = {
            split: set(metadata.iloc[split_indices[split]][column].astype(str))
            for split in prepare.SPLITS
        }
        record[f"shared_{key}s_train_test"] = int(
            len(groups["train"] & groups["test"])
        )
        record[f"shared_{key}s_train_validation"] = int(
            len(groups["train"] & groups["validation"])
        )
        record[f"distinct_{key}s"] = int(
            len(set().union(*groups.values())) if groups else 0
        )
    return record


def build_splits(
    context: ExtensionContext,
    *,
    policies: Sequence[str] | None = None,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Compute and record each policy's assignment, with a feasibility verdict."""

    study = context.study(STUDY)
    split_cfg = study["split"]
    prepare_config = prepare.read_config(context.project_root / BASELINE_PREPARE_CONFIG)
    task_frames, _, _, _ = prepare.load_finite_capped_tasks(prepare_config)
    metadata = load_flow_metadata(context)
    rows: list[dict[str, Any]] = []
    for policy in _policies(context, policies):
        policy_id = str(policy["policy_id"])
        for task_id in study["tasks"]:
            frame = task_frames[str(task_id)]
            joined = _aligned_metadata(frame, metadata)
            try:
                split_indices, conflict_mask, diagnostics = _policy_split(
                    policy,
                    fingerprints=frame.raw_fingerprint,
                    labels=frame.y,
                    metadata=joined,
                    n_classes=len(frame.class_names),
                    split_cfg=split_cfg,
                )
            except Exception as error:  # noqa: BLE001 - recorded, not silenced
                rows.append(
                    {
                        "policy_id": policy_id,
                        "task_id": task_id,
                        "feasible": False,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                if logger is not None:
                    logger.warning("%s/%s infeasible: %s", policy_id, task_id, error)
                continue
            maximum_deviation = max(
                diagnostics[f"{split}_fraction_deviation"] for split in prepare.SPLITS
            )
            minimum_class_rows = min(
                int(np.min(np.bincount(frame.y[split_indices[split]], minlength=len(frame.class_names))))
                if len(split_indices[split])
                else 0
                for split in prepare.SPLITS
            )
            feasible = bool(
                diagnostics["all_classes_each_split"]
                and maximum_deviation
                <= float(split_cfg["max_absolute_split_fraction_deviation"])
                and minimum_class_rows
                >= int(split_cfg["minimum_rows_per_class_per_split"])
            )
            reason = "feasible"
            if not diagnostics["all_classes_each_split"]:
                reason = "a partition is missing at least one class"
            elif maximum_deviation > float(
                split_cfg["max_absolute_split_fraction_deviation"]
            ):
                reason = (
                    f"split fractions deviate by {maximum_deviation:.3f}, above the "
                    f"declared {float(split_cfg['max_absolute_split_fraction_deviation']):.3f}"
                )
            elif minimum_class_rows < int(split_cfg["minimum_rows_per_class_per_split"]):
                reason = (
                    f"smallest class has {minimum_class_rows} rows in a partition, "
                    f"below the declared minimum"
                )
            record = {
                "policy_id": policy_id,
                "task_id": task_id,
                "feasible": feasible,
                "reason": reason,
                "maximum_fraction_deviation": maximum_deviation,
                "minimum_class_rows_per_split": int(minimum_class_rows),
                **diagnostics,
            }
            rows.append(record)
            directory = _policy_root(context, policy_id) / "splits"
            directory.mkdir(parents=True, exist_ok=True)
            assignment = np.full(len(frame.y), "dropped", dtype=object)
            for split, index in split_indices.items():
                assignment[index] = split
            pd.DataFrame(
                {
                    "task_id": task_id,
                    "row": np.arange(len(frame.y)),
                    "source_file": [
                        frame.source_files[int(value)] for value in frame.source_file_id
                    ],
                    "source_row": frame.source_row,
                    "class_id": frame.y,
                    "split": assignment,
                    "label_conflicting": conflict_mask,
                    "timestamp": joined["timestamp"].astype(str).to_numpy(),
                    "source_address": joined["source_address"].to_numpy(),
                    "session_five_tuple": joined["session_five_tuple"].to_numpy(),
                }
            ).to_parquet(directory / f"{task_id}_assignment.parquet", index=False)
            if logger is not None:
                logger.info(
                    "%s/%s: %s (train %d / val %d / test %d, shared hosts train-test %d, "
                    "test rows before last train row %d)",
                    policy_id,
                    task_id,
                    "feasible" if feasible else f"INFEASIBLE - {reason}",
                    record["train_rows"],
                    record["validation_rows"],
                    record["test_rows"],
                    record["shared_hosts_train_test"],
                    record.get("test_rows_before_last_train_row", -1),
                )
    frame = pd.DataFrame(rows)
    write_extension_table(
        frame, context.tables_dir(STUDY) / "split_policy_diagnostics.csv"
    )
    return frame


def _aligned_metadata(frame: Any, metadata: pd.DataFrame) -> pd.DataFrame:
    """Join capture metadata onto a task frame's rows, preserving row order."""

    keys = pd.DataFrame(
        {
            "source_file": [
                frame.source_files[int(value)] for value in frame.source_file_id
            ],
            "source_row": np.asarray(frame.source_row, dtype=np.int64),
        }
    )
    joined = keys.merge(metadata, on=["source_file", "source_row"], how="left")
    if len(joined) != len(keys):
        raise RuntimeError("Metadata join changed the task frame row count")
    missing = int(joined["timestamp"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} task rows have no capture metadata")
    return joined


# --------------------------------------------------------------------------- #
# Stage 3 - preparation under a policy split
# --------------------------------------------------------------------------- #
def _policy_prepare_config_path(context: ExtensionContext, policy_id: str) -> Path:
    return _policy_root(context, policy_id) / "configs" / "prepare.yaml"


def _policy_training_config_path(context: ExtensionContext, policy_id: str) -> Path:
    return _policy_root(context, policy_id) / "configs" / "training.yaml"


def _policy_audit_config_path(context: ExtensionContext, policy_id: str) -> Path:
    return _policy_root(context, policy_id) / "configs" / "audit.yaml"


def _relative(context: ExtensionContext, path: Path) -> str:
    return path.resolve().relative_to(context.project_root).as_posix()


def feasible_tasks(context: ExtensionContext, policy_id: str) -> list[str]:
    """Tasks whose split is feasible under this policy, from the frozen verdict."""

    path = context.tables_root / STUDY / "split_policy_diagnostics.csv"
    if not path.is_file():
        raise RuntimeError("Run the study 1 'splits' stage first")
    frame = pd.read_csv(path)
    frame = frame[frame["policy_id"].astype(str).eq(policy_id)]
    return [
        str(row.task_id)
        for row in frame.itertuples()
        if _as_bool(row.feasible)
    ]


def write_policy_configs(
    context: ExtensionContext,
    policy: Mapping[str, Any],
    *,
    tasks: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Derive per-policy configs from the frozen ones, changing only what must change.

    The preprocessing recipe, representation guard, class caps, candidate
    architecture, seeds and gate thresholds are inherited byte-for-byte from the
    camera-ready configs; only the output namespace, the task list (CICIDS only)
    and the split policy marker differ.
    """

    policy_id = str(policy["policy_id"])
    study = context.study(STUDY)
    selected_tasks = set(tasks if tasks is not None else study["tasks"])
    root = _policy_root(context, policy_id)
    (root / "configs").mkdir(parents=True, exist_ok=True)

    prepare_config = prepare.read_config(context.project_root / BASELINE_PREPARE_CONFIG)
    prepare_config = copy.deepcopy(prepare_config)
    prepare_config["pipeline_version"] = (
        f"{prepare_config['pipeline_version']}__ext_{policy_id}"
    )
    prepare_config["tasks"] = [
        task for task in prepare_config["tasks"] if str(task["task_id"]) in selected_tasks
    ]
    if not prepare_config["tasks"]:
        raise RuntimeError(f"No feasible task remains for policy {policy_id!r}")
    prepare_config["split"] = {
        **prepare_config["split"],
        "train_size": float(study["split"]["train_size"]),
        "validation_size": float(study["split"]["validation_size"]),
        "test_size": float(study["split"]["test_size"]),
        # split.seed is deliberately NOT changed: it also seeds the per-class
        # reservoir that caps the task rows, so changing it would give the
        # policy a different row population and confound the split effect with
        # a sampling effect.  The policy's own assignment seed is separate.
        "assignment_seed": int(study["split"]["seed"]),
        "max_absolute_split_fraction_deviation": float(
            study["split"]["max_absolute_split_fraction_deviation"]
        ),
        "extension_policy_id": policy_id,
        "extension_policy": dict(policy),
    }
    prepare_config["outputs"] = {
        "results_root": _relative(context, root / "data"),
        "tables_root": _relative(context, root / "tables"),
    }
    prepare_path = _policy_prepare_config_path(context, policy_id)
    prepare_path.write_text(
        yaml.safe_dump(prepare_config, sort_keys=False), encoding="utf-8"
    )

    training_config = training.load_config(
        context.project_root / "configs/leakage_free_tabular_training.yaml"
    )
    training_config = copy.deepcopy(training_config)
    training_config["pipeline_id"] = f"{training_config['pipeline_id']}__ext_{policy_id}"
    training_config["tasks"] = {
        task_id: value
        for task_id, value in training_config["tasks"].items()
        if task_id in selected_tasks
    }
    training_config["paths"] = {
        "processed_root": _relative(context, root / "data" / "processed"),
        "results_root": _relative(context, root / "models"),
        "tables_root": _relative(context, root / "tables"),
    }
    grids: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task_id in training_config["tasks"]:
        grids[task_id] = {}
        for model in ALLOWED_MODELS:
            lock_path = (
                context.paths.training_results_root
                / "model_selection"
                / task_id
                / model
                / "selected_config.lock.json"
            )
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            grids[task_id][model] = [dict(lock["selected_candidate"])]
    training_config["grids"] = grids
    training_config["split_policy"] = {
        **training_config["split_policy"],
        "extension_policy_id": policy_id,
    }
    training_path = _policy_training_config_path(context, policy_id)
    training_path.write_text(
        yaml.safe_dump(training_config, sort_keys=False), encoding="utf-8"
    )

    audit_config = copy.deepcopy(dict(context.audit_config))
    audit_config["protocol_id"] = f"{audit_config['protocol_id']}__ext_{policy_id}"
    audit_config["data_guard"] = {
        **audit_config["data_guard"],
        "processed_root": _relative(context, root / "data" / "processed"),
    }
    audit_path = _policy_audit_config_path(context, policy_id)
    audit_path.write_text(yaml.safe_dump(audit_config, sort_keys=False), encoding="utf-8")
    return {
        "prepare": prepare_path,
        "training": training_path,
        "audit": audit_path,
    }


def prepare_partitions(
    context: ExtensionContext,
    *,
    policies: Sequence[str] | None = None,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Run the frozen preparation with the policy split injected.

    ``prepare.group_safe_split`` is temporarily replaced by the policy
    splitter for the duration of this call.  Everything else - the train-only
    imputer/variance/duplicate/scaler pipeline, the float32 representation
    guard, the leakage invariants and the artefact layout - is the frozen code
    running unmodified, which is the point: only the split changes.
    """

    study = context.study(STUDY)
    prepare_config = prepare.read_config(context.project_root / BASELINE_PREPARE_CONFIG)
    task_frames, _, _, _ = prepare.load_finite_capped_tasks(prepare_config)
    metadata = load_flow_metadata(context)
    aligned = {
        task_id: _aligned_metadata(frame, metadata)
        for task_id, frame in task_frames.items()
        if task_id in set(study["tasks"])
    }
    if logger is not None:
        logger.info("capture metadata aligned for %s", sorted(aligned))
    feasibility = pd.read_csv(
        context.tables_root / STUDY / "split_policy_diagnostics.csv"
    )
    rows: list[dict[str, Any]] = []
    for policy in _policies(context, policies):
        policy_id = str(policy["policy_id"])
        verdict = feasibility[feasibility["policy_id"].eq(policy_id)]
        infeasible = verdict[~verdict["feasible"].map(_as_bool)]
        eligible_tasks = feasible_tasks(context, policy_id)
        if not eligible_tasks:
            rows.append(
                {
                    "policy_id": policy_id,
                    "status": "skipped_infeasible_split",
                    "tasks": "",
                    "reason": "; ".join(
                        f"{row.task_id}: {row.reason}" for row in infeasible.itertuples()
                    ),
                }
            )
            if logger is not None:
                logger.warning("%s skipped: no feasible task", policy_id)
            continue
        paths = write_policy_configs(context, policy, tasks=eligible_tasks)
        started = time.perf_counter()
        with _policy_split_patch(context, policy, task_frames, aligned):
            decision = prepare.prepare_all(paths["prepare"])
        rows.append(
            {
                "policy_id": policy_id,
                "status": str(decision.get("status", "unknown")),
                "tasks": ",".join(eligible_tasks),
                "skipped_tasks": ";".join(
                    f"{row.task_id}: {row.reason}" for row in infeasible.itertuples()
                ),
                "signature": str(decision.get("signature", "")),
                "runtime_seconds": float(time.perf_counter() - started),
                "prepare_config": _relative(context, paths["prepare"]),
                "training_config": _relative(context, paths["training"]),
                "audit_config": _relative(context, paths["audit"]),
            }
        )
        if logger is not None:
            logger.info(
                "%s prepared in %.1fs", policy_id, float(time.perf_counter() - started)
            )
    frame = pd.DataFrame(rows)
    write_extension_table(frame, context.tables_dir(STUDY) / "prepare_status.csv")
    return frame


class _policy_split_patch:
    """Scoped replacement of the frozen random group split by a policy split."""

    def __init__(
        self,
        context: ExtensionContext,
        policy: Mapping[str, Any],
        task_frames: Mapping[str, Any],
        aligned: Mapping[str, pd.DataFrame],
    ) -> None:
        self._context = context
        self._policy = policy
        self._task_frames = task_frames
        self._aligned = aligned
        self._original = None

    def __enter__(self) -> "_policy_split_patch":
        self._original = prepare.group_safe_split
        policy = self._policy
        task_frames = self._task_frames
        aligned = self._aligned
        split_cfg = self._context.study(STUDY)["split"]

        def patched(
            fingerprints: np.ndarray,
            labels: np.ndarray,
            *,
            train_size: float,
            validation_size: float,
            test_size: float,
            seed: int,
        ):
            matched: str | None = None
            for task_id, frame in task_frames.items():
                if task_id not in aligned:
                    continue
                if len(frame.raw_fingerprint) != len(fingerprints):
                    continue
                if np.array_equal(frame.raw_fingerprint, fingerprints) and np.array_equal(
                    frame.y, labels
                ):
                    matched = task_id
                    break
            if matched is None:
                raise RuntimeError(
                    "Policy split cannot identify the task frame it was called for; "
                    "refusing to guess a split"
                )
            split_indices, conflict_mask, _ = _policy_split(
                policy,
                fingerprints=fingerprints,
                labels=labels,
                metadata=aligned[matched],
                n_classes=len(task_frames[matched].class_names),
                split_cfg={
                    **split_cfg,
                    "train_size": float(train_size),
                    "validation_size": float(validation_size),
                    "test_size": float(test_size),
                },
            )
            # Re-assert the invariants the replaced function guaranteed.
            for split in prepare.SPLITS:
                index = split_indices[split]
                if not len(index):
                    raise RuntimeError(f"Policy split produced an empty {split} partition")
            for left, right in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            ):
                overlap = np.intersect1d(
                    fingerprints[split_indices[left]],
                    fingerprints[split_indices[right]],
                ).size
                if overlap:
                    raise RuntimeError(
                        f"Policy split leaks {overlap} feature-hash groups between "
                        f"{left} and {right}"
                    )
            assigned = sum(len(index) for index in split_indices.values())
            if assigned + int(conflict_mask.sum()) != len(labels):
                raise RuntimeError("Policy split did not assign every eligible row")
            return split_indices, conflict_mask

        prepare.group_safe_split = patched
        return self

    def __exit__(self, *exception: object) -> None:
        if self._original is not None:
            prepare.group_safe_split = self._original


# --------------------------------------------------------------------------- #
# Stage 4 - training under a policy split
# --------------------------------------------------------------------------- #
def train_models(
    context: ExtensionContext,
    *,
    policies: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Re-select the threshold and refit the frozen architecture per policy."""

    rows: list[dict[str, Any]] = []
    for policy in _policies(context, policies):
        policy_id = str(policy["policy_id"])
        config_path = _policy_training_config_path(context, policy_id)
        if not config_path.is_file():
            if logger is not None:
                logger.warning("%s has no training config; skipping", policy_id)
            continue
        config = training.load_config(config_path)
        selection_tables: list[pd.DataFrame] = []
        final_tables: list[pd.DataFrame] = []
        for task_id in config["tasks"]:
            for model in ALLOWED_MODELS:
                started = time.perf_counter()
                table, lock = training.run_model_selection(
                    config,
                    project_root=context.project_root,
                    config_path=config_path,
                    task_id=task_id,
                    model_name=model,
                    force=force,
                )
                selection_tables.append(table)
                if logger is not None:
                    logger.info(
                        "%s selection %s/%s: candidate %s, threshold %.6g, %.1fs",
                        policy_id,
                        task_id,
                        model,
                        lock["selected_candidate"]["candidate_id"],
                        float(lock["selection_metrics"].get("selected_threshold", float("nan"))),
                        time.perf_counter() - started,
                    )
                for seed in seeds or training.FINAL_SEEDS:
                    started = time.perf_counter()
                    result = training.run_final_condition(
                        config,
                        project_root=context.project_root,
                        config_path=config_path,
                        task_id=task_id,
                        model_name=model,
                        seed=int(seed),
                        force=force,
                    )
                    final_tables.append(result["metrics"])
                    rows.append(
                        {
                            "policy_id": policy_id,
                            "task_id": task_id,
                            "model": model,
                            "seed": int(seed),
                            "runtime_seconds": float(time.perf_counter() - started),
                        }
                    )
                    if logger is not None:
                        metrics = result["metrics"].iloc[0]
                        logger.info(
                            "%s final %s/%s/seed_%d: f2=%.4f fpr=%.4f fnr=%.4f, %.1fs",
                            policy_id, task_id, model, seed,
                            float(metrics.get("f2", float("nan"))),
                            float(metrics.get("fpr", float("nan"))),
                            float(metrics.get("fnr", float("nan"))),
                            float(time.perf_counter() - started),
                        )
        if selection_tables and final_tables:
            training.write_summary_tables(
                config,
                project_root=context.project_root,
                selection_tables=selection_tables,
                final_metric_tables=final_tables,
            )
    frame = pd.DataFrame(rows)
    if len(frame):
        write_extension_table(frame, context.tables_dir(STUDY) / "training_runtime.csv")
    return frame


# --------------------------------------------------------------------------- #
# Stage 5 - bounded reliability audit inside a policy tree
# --------------------------------------------------------------------------- #
def policy_audit_paths(context: ExtensionContext, policy_id: str) -> AuditPaths:
    root = _policy_root(context, policy_id)
    return AuditPaths(
        project_root=context.project_root,
        config_path=_policy_audit_config_path(context, policy_id).resolve(),
        training_config_path=_policy_training_config_path(context, policy_id).resolve(),
        processed_root=(root / "data" / "processed").resolve(),
        training_results_root=(root / "models").resolve(),
        audit_root=(root / "models" / "epistemic_audit").resolve(),
        tables_root=(root / "tables" / "epistemic_audit").resolve(),
    )


def build_policy_panel(
    context: ExtensionContext,
    policy_id: str,
    *,
    tasks: Sequence[str] | None = None,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Per-seed central panel under the policy's own test partition.

    Same rule as the baseline central panel - flows both architectures classify
    correctly at that fitted seed, capped per true class and seed, chosen before
    any attribution is computed - so the reliability numbers are comparable.
    """

    study = context.study(STUDY)
    spec = study["audit"]["panel"]
    paths = policy_audit_paths(context, policy_id)
    output = paths.audit_root / "panel" / "central_panel.csv"
    if output.is_file() and not force:
        return pd.read_csv(output)
    training_config = training.load_config(
        _policy_training_config_path(context, policy_id)
    )
    frames: list[pd.DataFrame] = []
    for task_id in tasks or list(training_config["tasks"]):
        for seed in training.FINAL_SEEDS:
            predictions = {}
            for model in ALLOWED_MODELS:
                condition = require_valid_training_condition(
                    paths,
                    training_config,
                    task_id=task_id,
                    model=model,
                    seed=int(seed),
                )
                predictions[model] = pd.read_csv(condition["predictions"])
            base = predictions[ALLOWED_MODELS[0]]
            correct = np.ones(len(base), dtype=bool)
            for frame in predictions.values():
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
            pool["seed"] = int(seed)
            # The panel is shared by both architectures, so each model's own
            # confidence and decision margin are carried per model and selected
            # at probe time; the functional stage needs them per audited model.
            for model, frame in predictions.items():
                pool[f"confidence_{model}"] = frame.loc[correct, "confidence"].to_numpy()
                pool[f"margin_{model}"] = _prediction_margin(frame)[correct]
            pool["flow_id"] = [_flow_id(row) for row in pool.to_dict("records")]
            rng = np.random.default_rng(
                int(spec["sampling_seed"]) + 1000 * int(seed)
            )
            cap = int(spec["maximum_per_true_class_per_seed"])
            stochastic_cap = int(study["audit"]["stochastic_subset_per_true_class_per_seed"])
            for class_id, group in pool.groupby("true_class_id", sort=True):
                ordered = group.sort_values("sample_order", kind="mergesort").reset_index(
                    drop=True
                )
                take = min(cap, len(ordered))
                chosen = ordered.iloc[
                    np.sort(rng.choice(len(ordered), size=take, replace=False))
                ].copy()
                chosen["eligible_pool_support"] = int(len(ordered))
                chosen["central_class_seed_support"] = int(take)
                stochastic = np.zeros(len(chosen), dtype=bool)
                stochastic[
                    np.sort(
                        rng.choice(
                            len(chosen),
                            size=min(stochastic_cap, len(chosen)),
                            replace=False,
                        )
                    )
                ] = True
                chosen["in_stochastic_subset"] = stochastic
                frames.append(chosen)
    panel = pd.concat(frames, ignore_index=True)
    minimum = int(context.study(STUDY)["audit"]["panel"].get(
        "minimum_class_support_for_class_specific_inference", 20
    ))
    panel["central_sample_support_sufficient"] = panel[
        "central_class_seed_support"
    ].ge(minimum)
    panel["class_specific_inference_eligible"] = panel[
        "central_sample_support_sufficient"
    ]
    panel["target_class_id"] = panel["true_class_id"]
    panel["target_class"] = panel["true_class"]
    panel["target_basis"] = "true_class_correct_for_both"
    panel["in_central"] = True
    panel["policy_id"] = policy_id
    panel = panel.sort_values(
        ["task_id", "seed", "sample_order"], kind="mergesort"
    ).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    if logger is not None:
        logger.info(
            "%s panel: %d flows (%d stochastic)",
            policy_id,
            len(panel),
            int(panel["in_stochastic_subset"].sum()),
        )
    return panel


def _policy_probe_directory(
    context: ExtensionContext,
    policy_id: str,
    task_id: str,
    model: str,
    seed: int,
    spec: ProbeSpec,
) -> Path:
    return (
        policy_audit_paths(context, policy_id).audit_root
        / "probes"
        / task_id
        / model
        / f"seed_{seed}"
        / spec.method
        / spec.spec_id
    )


def _policy_probe_paths(directory: Path) -> dict[str, Path]:
    return {
        "array": directory / "attributions.npz",
        "index": directory / "per_flow_index.csv",
        "decision": directory / "probe_decision.json",
        "metadata": directory / "cache_metadata.json",
    }


def load_policy_probe(
    context: ExtensionContext,
    policy_id: str,
    *,
    task_id: str,
    model: str,
    seed: int,
    spec: ProbeSpec,
) -> tuple[pd.DataFrame, np.ndarray]:
    paths = _policy_probe_paths(
        _policy_probe_directory(context, policy_id, task_id, model, int(seed), spec)
    )
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    hit, reason = cache_hit(
        paths["metadata"],
        expected_signature=str(metadata.get("signature", "")),
        expected_outputs=[paths["array"], paths["index"], paths["decision"]],
    )
    if not hit:
        raise RuntimeError(f"Policy probe outputs failed their digests: {reason}")
    index = pd.read_csv(paths["index"])
    with np.load(paths["array"], allow_pickle=False) as archive:
        attributions = archive["attributions"].astype(np.float32)
    return index, attributions


def run_audit(
    context: ExtensionContext,
    *,
    policies: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Panel, deterministic probes and functional checks inside each policy tree."""

    study = context.study(STUDY)
    audit_spec = study["audit"]
    methods = list(audit_spec["probes"])
    specs = [DETERMINISTIC_SPECS[method] for method in methods] + [
        spec for spec in EXTRA_SPECS if spec.method in methods
    ]
    rows: list[dict[str, Any]] = []
    for policy in _policies(context, policies):
        policy_id = str(policy["policy_id"])
        training_config_path = _policy_training_config_path(context, policy_id)
        if not training_config_path.is_file():
            continue
        paths = policy_audit_paths(context, policy_id)
        training_config = training.load_config(training_config_path)
        audit_config = _policy_audit_config(context, policy_id)
        panel = build_policy_panel(context, policy_id, logger=logger, force=force)
        for task_id in list(training_config["tasks"]):
            data = load_condition_data(paths, task_id)
            for model in ALLOWED_MODELS:
                for seed in seeds or training.FINAL_SEEDS:
                    condition = require_valid_training_condition(
                        paths,
                        training_config,
                        task_id=task_id,
                        model=model,
                        seed=int(seed),
                    )
                    units = panel[
                        panel["task_id"].astype(str).eq(task_id)
                        & pd.to_numeric(panel["seed"], errors="raise")
                        .astype(int)
                        .eq(int(seed))
                    ].reset_index(drop=True)
                    if units.empty:
                        continue
                    units["confidence"] = units[f"confidence_{model}"]
                    units["model_margin"] = units[f"margin_{model}"]
                    frozen_model = None
                    for spec in specs:
                        directory = _policy_probe_directory(
                            context, policy_id, task_id, model, int(seed), spec
                        )
                        probe_paths = _policy_probe_paths(directory)
                        inputs = artifact_records(
                            [
                                context.extension_config_path,
                                paths.config_path,
                                paths.training_config_path,
                                paths.audit_root / "panel" / "central_panel.csv",
                                condition["model"],
                                condition["metadata"],
                                *_processed_probe_inputs(paths.processed_root / task_id),
                            ]
                        )
                        implementation_hash = signature(
                            {"files": context.implementation_records()}
                        )
                        stage_signature = signature(
                            {
                                "extension_study": STUDY,
                                "stage": "policy_probe",
                                "policy_id": policy_id,
                                "implementation_hash": implementation_hash,
                                "inputs": inputs,
                                "task_id": task_id,
                                "model": model,
                                "seed": int(seed),
                                "probe": spec.payload(),
                            }
                        )
                        expected = [
                            probe_paths["array"],
                            probe_paths["index"],
                            probe_paths["decision"],
                        ]
                        hit, _ = cache_hit(
                            probe_paths["metadata"],
                            expected_signature=stage_signature,
                            expected_outputs=expected,
                        )
                        if hit and not force:
                            continue
                        if frozen_model is None:
                            frozen_model = _load_frozen_model(condition["model"])
                        started = time.perf_counter()
                        attributions, probabilities = _compute_probe(
                            audit_config, data, frozen_model, units, spec
                        )
                        runtime = float(time.perf_counter() - started)
                        index = units.copy()
                        index["model"] = model
                        index["method"] = spec.method
                        index["spec_id"] = spec.spec_id
                        index["reference"] = spec.reference
                        index["steps"] = spec.steps
                        index["probe_run"] = int(spec.run_id)
                        index["target_probability"] = probabilities
                        directory.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            probe_paths["array"],
                            attributions=attributions,
                            target_probability=probabilities,
                            sample_order=index["sample_order"].to_numpy(np.int64),
                            target_class_id=index["target_class_id"].to_numpy(np.int64),
                            feature_names=np.asarray(data.feature_names, dtype="S128"),
                        )
                        index.to_csv(probe_paths["index"], index=False)
                        decision = {
                            "extension_study": STUDY,
                            "policy_id": policy_id,
                            "status": "complete",
                            "task_id": task_id,
                            "model": model,
                            "seed": int(seed),
                            "probe": spec.payload(),
                            "rows": int(len(index)),
                            "runtime_seconds": runtime,
                            "stage_signature": stage_signature,
                        }
                        probe_paths["decision"].write_text(
                            json.dumps(decision, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                        write_cache_metadata(
                            probe_paths["metadata"],
                            stage="policy_probe",
                            signature_value=stage_signature,
                            config_hash=sha256_file(context.extension_config_path),
                            implementation_hash=implementation_hash,
                            inputs=inputs,
                            outputs=expected,
                            extra=decision,
                        )
                        rows.append(
                            {
                                "policy_id": policy_id,
                                "task_id": task_id,
                                "model": model,
                                "seed": int(seed),
                                "method": spec.method,
                                "spec_id": spec.spec_id,
                                "stage": "probe",
                                "rows": int(len(index)),
                                "runtime_seconds": runtime,
                            }
                        )
                        if logger is not None:
                            logger.info(
                                "%s probe %s/%s/seed_%d/%s/%s: rows=%d, %.1fs",
                                policy_id, task_id, model, seed,
                                spec.method, spec.spec_id, len(index), runtime,
                            )
                    for method in methods:
                        record = _run_policy_functional(
                            context,
                            policy_id,
                            paths=paths,
                            audit_config=audit_config,
                            data=data,
                            condition=condition,
                            task_id=task_id,
                            model=model,
                            seed=int(seed),
                            method=method,
                            force=force,
                            logger=logger,
                        )
                        if record is not None:
                            rows.append(record)
    frame = pd.DataFrame(rows)
    if len(frame):
        write_extension_table(
            frame, context.tables_dir(STUDY) / "policy_audit_runtime.csv"
        )
    return frame


def _policy_audit_config(context: ExtensionContext, policy_id: str) -> dict[str, Any]:
    path = _policy_audit_config_path(context, policy_id)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _policy_functional_config(
    context: ExtensionContext, policy_id: str
) -> dict[str, Any]:
    config = copy.deepcopy(_policy_audit_config(context, policy_id))
    study = context.study(STUDY)["audit"]
    functional = config["functional_checks"]
    functional["panel_scope"] = "central_stochastic_subset"
    deletion = functional["deletion_and_insertion"]
    deletion["evaluators"] = list(study["functional_evaluators"])
    deletion["feature_fractions"] = [float(v) for v in study["functional_fractions"]]
    deletion["random_feature_repetitions"] = int(study["functional_random_repetitions"])
    functional["signed_intervention"]["strengths"] = [1.0]
    functional["signed_intervention"]["status"] = "secondary_descriptive_reduced"
    return config


def _run_policy_functional(
    context: ExtensionContext,
    policy_id: str,
    *,
    paths: AuditPaths,
    audit_config: Mapping[str, Any],
    data: Any,
    condition: Mapping[str, Path],
    task_id: str,
    model: str,
    seed: int,
    method: str,
    force: bool,
    logger=None,
) -> dict[str, Any] | None:
    config = _policy_functional_config(context, policy_id)
    spec = DETERMINISTIC_SPECS[method]
    directory = (
        paths.audit_root / "functional" / task_id / model / f"seed_{seed}" / method
    )
    outputs = {
        "curves": directory / "deletion_insertion_per_flow.csv",
        "aopc": directory / "functional_aopc_per_flow.csv",
        "signed": directory / "signed_intervention_per_flow.csv",
        "decision": directory / "functional_decision.json",
    }
    metadata_path = directory / "cache_metadata.json"
    probe_metadata = _policy_probe_paths(
        _policy_probe_directory(context, policy_id, task_id, model, int(seed), spec)
    )["metadata"]
    inputs = artifact_records(
        [
            context.extension_config_path,
            paths.config_path,
            paths.training_config_path,
            probe_metadata,
            condition["model"],
            condition["metadata"],
            *_processed_probe_inputs(paths.processed_root / task_id),
        ]
    )
    implementation_hash = signature({"files": context.implementation_records()})
    stage_signature = signature(
        {
            "extension_study": STUDY,
            "stage": "policy_functional",
            "policy_id": policy_id,
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
        metadata_path, expected_signature=stage_signature, expected_outputs=expected
    )
    if hit and not force:
        return None
    index, attributions = load_policy_probe(
        context, policy_id, task_id=task_id, model=model, seed=int(seed), spec=spec
    )
    mask = index["in_stochastic_subset"].map(_as_bool).to_numpy()
    index = index.loc[mask].reset_index(drop=True)
    attributions = attributions[mask]
    identity = index.copy()
    identity["in_outcome_correct"] = True
    identity["in_outcome_error"] = False
    orders = pd.to_numeric(index["sample_order"], errors="raise").to_numpy(int)
    frozen_model = _load_frozen_model(condition["model"])
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
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in (("curves", curves), ("aopc", aopc), ("signed", signed)):
        frame.to_csv(outputs[name], index=False)
    decision = {
        "extension_study": STUDY,
        "policy_id": policy_id,
        "status": "complete",
        "task_id": task_id,
        "model": model,
        "seed": int(seed),
        "method": method,
        "flows": int(aopc["flow_id"].nunique()),
        "runtime_seconds": runtime,
        "stage_signature": stage_signature,
    }
    outputs["decision"].write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_cache_metadata(
        metadata_path,
        stage="policy_functional",
        signature_value=stage_signature,
        config_hash=sha256_file(context.extension_config_path),
        implementation_hash=implementation_hash,
        inputs=inputs,
        outputs=expected,
        extra=decision,
    )
    if logger is not None:
        logger.info(
            "%s functional %s/%s/seed_%d/%s: flows=%d, %.1fs",
            policy_id, task_id, model, seed, method,
            aopc["flow_id"].nunique(), runtime,
        )
    return {
        "policy_id": policy_id,
        "task_id": task_id,
        "model": model,
        "seed": int(seed),
        "method": method,
        "stage": "functional",
        "rows": int(aopc["flow_id"].nunique()),
        "runtime_seconds": runtime,
    }


# --------------------------------------------------------------------------- #
# Stage 6 - summaries against the baseline split
# --------------------------------------------------------------------------- #
def _policy_repeatability_and_cross_model(
    context: ExtensionContext, policy_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_k = int(context.shared["representation"]["top_k"])
    training_config = training.load_config(
        _policy_training_config_path(context, policy_id)
    )
    repeat_frames: list[pd.DataFrame] = []
    cross_frames: list[pd.DataFrame] = []
    for task_id in list(training_config["tasks"]):
        for seed in training.FINAL_SEEDS:
            per_model: dict[str, dict[str, tuple[pd.DataFrame, np.ndarray]]] = {}
            for model in ALLOWED_MODELS:
                per_model[model] = {}
                for method, (left, right, label) in REPEATABILITY_SPECS.items():
                    left_index, left_attrs = load_policy_probe(
                        context, policy_id, task_id=task_id, model=model,
                        seed=int(seed), spec=left,
                    )
                    _, right_attrs = load_policy_probe(
                        context, policy_id, task_id=task_id, model=model,
                        seed=int(seed), spec=right,
                    )
                    metrics = similarity_matrix_at(
                        left_attrs, right_attrs, top_k=top_k
                    )
                    frame = left_index[
                        [
                            "task_id",
                            "seed",
                            "flow_id",
                            "true_class_id",
                            "true_class",
                            "class_specific_inference_eligible",
                        ]
                    ].copy()
                    frame["policy_id"] = policy_id
                    frame["model"] = model
                    frame["method"] = method
                    frame["comparison"] = label
                    frame["analysis_role"] = (
                        "primary_main_to_128"
                        if label == "steps_64_vs_128"
                        else "primary_fixed_rule_repeat"
                    )
                    for key, value in metrics.items():
                        frame[key] = value
                    repeat_frames.append(frame)
                for method, spec in DETERMINISTIC_SPECS.items():
                    per_model[model][method] = load_policy_probe(
                        context, policy_id, task_id=task_id, model=model,
                        seed=int(seed), spec=spec,
                    )
            for method in DETERMINISTIC_SPECS:
                left_index, left_attrs = per_model[ALLOWED_MODELS[0]][method]
                right_index, right_attrs = per_model[ALLOWED_MODELS[1]][method]
                lookup = {
                    str(flow): position
                    for position, flow in enumerate(right_index["flow_id"])
                }
                order = np.asarray(
                    [lookup[str(flow)] for flow in left_index["flow_id"]], dtype=int
                )
                metrics = similarity_matrix_at(
                    left_attrs, right_attrs[order], top_k=top_k
                )
                frame = left_index[
                    [
                        "task_id",
                        "seed",
                        "flow_id",
                        "true_class_id",
                        "true_class",
                        "class_specific_inference_eligible",
                    ]
                ].copy()
                frame["policy_id"] = policy_id
                frame["method"] = method
                frame["model_left"] = ALLOWED_MODELS[0]
                frame["model_right"] = ALLOWED_MODELS[1]
                frame["probe_run"] = 0
                for key, value in metrics.items():
                    frame[key] = value
                cross_frames.append(frame)
    repeatability = (
        pd.concat(repeat_frames, ignore_index=True) if repeat_frames else pd.DataFrame()
    )
    cross_model = (
        pd.concat(cross_frames, ignore_index=True) if cross_frames else pd.DataFrame()
    )
    return repeatability, cross_model


def run_summaries(
    context: ExtensionContext,
    *,
    policies: Sequence[str] | None = None,
    logger=None,
) -> dict[str, pd.DataFrame]:
    """Predictive and reliability comparison between each policy and the baseline."""

    from src.leakage_free_tabular_ext.study5_thresholds import (
        _config_for,
        summarize_inventory,
    )
    from src.leakage_free_tabular_ext.detail_recompute import RepresentationSetting

    baseline_setting = RepresentationSetting(setting_id="baseline", axis="baseline")
    predictive_rows: list[pd.DataFrame] = []
    repeat_summaries: list[pd.DataFrame] = []
    cross_summaries: list[pd.DataFrame] = []
    functional_summaries: list[pd.DataFrame] = []
    available: list[str] = []
    for policy in _policies(context, policies):
        policy_id = str(policy["policy_id"])
        paths = policy_audit_paths(context, policy_id)
        metrics_path = (
            _policy_root(context, policy_id) / "tables" / "predictive_metrics_by_seed.csv"
        )
        if not metrics_path.is_file():
            continue
        available.append(policy_id)
        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, "policy_id", policy_id)
        predictive_rows.append(metrics)
        repeatability, cross_model = _policy_repeatability_and_cross_model(
            context, policy_id
        )
        config = _config_for(_policy_audit_config(context, policy_id), baseline_setting)
        if len(repeatability):
            summary = summarize_inventory(repeatability, "repeatability", config)
            summary.insert(0, "policy_id", policy_id)
            repeat_summaries.append(summary)
        if len(cross_model):
            summary = summarize_inventory(cross_model, "cross_model", config)
            summary.insert(0, "policy_id", policy_id)
            cross_summaries.append(summary)
        aopc_paths = sorted(
            (paths.audit_root / "functional").glob(
                "*/*/seed_*/*/functional_aopc_per_flow.csv"
            )
        )
        if aopc_paths:
            aopc = pd.concat(
                [pd.read_csv(path) for path in aopc_paths], ignore_index=True
            )
            eligible = aopc[aopc["class_specific_inference_eligible"].map(_as_bool)]
            summary = summarize_metrics(
                eligible,
                group_columns=["task_id", "model", "method", "functional_check", "evaluator"],
                metric_columns=["attribution_minus_random_aopc"],
                config=config,
                analysis_id="ext_policy_functional",
            )
            summary.insert(0, "policy_id", policy_id)
            summary["positive_against_matched_random"] = summary["ci_low"].gt(0.0)
            functional_summaries.append(summary)

    baseline_root = context.project_root / "tables/leakage_free_tabular"
    baseline_predictive = pd.read_csv(baseline_root / "predictive_metrics_by_seed.csv")
    baseline_predictive.insert(0, "policy_id", "baseline_random_split")
    predictive = pd.concat(
        [baseline_predictive] + predictive_rows, ignore_index=True, sort=False
    )
    shared_columns = [
        column
        for column in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "f2",
            "fpr",
            "fnr",
            "pr_auc",
            "roc_auc",
            "log_loss",
            "ece",
            "validation_selected_threshold",
        )
        if column in predictive.columns
    ]
    predictive_summary = (
        predictive.groupby(["policy_id", "task_id", "model"], dropna=False)[shared_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    predictive_summary.columns = [
        "_".join(part for part in column if part).strip("_")
        for column in predictive_summary.columns
    ]

    baseline_repeat = pd.read_csv(
        baseline_root / "epistemic_audit/repeatability_summary.csv"
    )
    baseline_repeat.insert(0, "policy_id", "baseline_random_split")
    baseline_cross = pd.read_csv(
        baseline_root / "epistemic_audit/cross_model_summary.csv"
    )
    baseline_cross.insert(0, "policy_id", "baseline_random_split")
    repeatability_summary = pd.concat(
        [baseline_repeat] + repeat_summaries, ignore_index=True, sort=False
    )
    cross_model_summary = pd.concat(
        [baseline_cross] + cross_summaries, ignore_index=True, sort=False
    )
    functional_summary = (
        pd.concat(functional_summaries, ignore_index=True, sort=False)
        if functional_summaries
        else pd.DataFrame()
    )

    mix_rows: list[pd.DataFrame] = []
    for policy_id in available:
        distribution_path = (
            _policy_root(context, policy_id) / "tables" / "split_distribution.csv"
        )
        if distribution_path.is_file():
            frame = pd.read_csv(distribution_path)
            frame.insert(0, "policy_id", policy_id)
            mix_rows.append(frame)
    baseline_distribution = (
        context.project_root / "tables/leakage_free_tabular/split_distribution.csv"
    )
    if baseline_distribution.is_file():
        frame = pd.read_csv(baseline_distribution)
        frame.insert(0, "policy_id", "baseline_random_split")
        mix_rows.append(frame)
    class_mix = (
        pd.concat(mix_rows, ignore_index=True, sort=False) if mix_rows else pd.DataFrame()
    )
    if len(class_mix):
        totals = class_mix.groupby(["policy_id", "task_id", "split"])[
            "rows_after_representation_filter"
        ].transform("sum")
        class_mix["class_share_of_split"] = (
            class_mix["rows_after_representation_filter"] / totals.replace(0, np.nan)
        )

    tables_dir = context.tables_dir(STUDY)
    outputs = {
        "class_mix": tables_dir / "policy_split_class_mix.csv",
        "predictive_by_seed": tables_dir / "policy_predictive_by_seed.csv",
        "predictive_summary": tables_dir / "policy_predictive_summary.csv",
        "repeatability_summary": tables_dir / "policy_repeatability_summary.csv",
        "cross_model_summary": tables_dir / "policy_cross_model_summary.csv",
        "functional_summary": tables_dir / "policy_functional_summary.csv",
    }
    frames = {
        "class_mix": class_mix,
        "predictive_by_seed": predictive,
        "predictive_summary": predictive_summary,
        "repeatability_summary": repeatability_summary,
        "cross_model_summary": cross_model_summary,
        "functional_summary": functional_summary,
    }
    for name, path in outputs.items():
        write_extension_table(frames[name], path)
    write_decision(
        context.results_dir(STUDY) / "study1_decision.json",
        {
            "study": STUDY,
            "status": "complete" if available else "no_policy_completed",
            "policies_available": available,
            "policies_declared": [
                str(item["policy_id"]) for item in context.study(STUDY)["policies"]
            ],
            "tasks": list(context.study(STUDY)["tasks"]),
            "ton_iot_status": str(
                context.study(STUDY)["metadata_source"]["ton_iot"]
            ),
            "probes": list(context.study(STUDY)["audit"]["probes"]),
            "stochastic_probes_status": "pending_compute_bound",
            "role": "exploratory_split_stress_test",
            "outputs": {name: str(path) for name, path in outputs.items()},
        },
    )
    return frames
