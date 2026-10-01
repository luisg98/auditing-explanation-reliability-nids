"""Study 2: dependency-preserving and group-wise perturbation operators.

The baseline measures functional faithfulness by replacing the top-ranked
coordinates with a marginal constant - the training median or the target-class
median.  That intervention destroys the dependence between the replaced
coordinates and everything else, so a "faithful" or "unfaithful" verdict can be
a statement about off-manifold inputs rather than about the explanation.

This module keeps the baseline's estimand (attribution-minus-random AOPC over
the same fractions, the same displacement-bin-matched random control, the same
50 frozen draws) and the baseline's attributions, and varies only the
replacement operator:

* two baseline marginal constants, recomputed here for a like-for-like row;
* two joint donor operators, which copy the replaced block from one training
  row, preserving its joint distribution;
* two Gaussian conditional operators, which replace by E[x_S | x_R] with and
  without conditional noise;
* one nearest-neighbour operator, a nonparametric draw from x_S | x_R;
* two group-wise operators, which move a whole correlation group at once
  instead of leaving a top feature's correlated partners untouched.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

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
    ConditionData,
    _as_bool,
    _displacement_bin_matched_feature_matrix,
    _load_frozen_model,
    _processed_probe_inputs,
    _random_feature_matrix,
    load_condition_data,
    probe_specs,
    require_valid_training_condition,
    stable_seed,
    summarize_metrics,
    target_probabilities,
)
from src.leakage_free_tabular_ext.common import (
    ExtensionContext,
    write_decision,
    write_extension_table,
)
from src.leakage_free_tabular_ext.detail_recompute import ProbeCache, align_probe_arrays

STUDY = "study2_dependency_perturbations"
GROUP_OPERATORS = ("correlation_group_median", "correlation_group_joint_donor")


# --------------------------------------------------------------------------- #
# Correlation groups
# --------------------------------------------------------------------------- #
def _absolute_spearman_distance(X: np.ndarray) -> np.ndarray:
    ranks = rankdata(X, axis=0)
    centred = ranks - ranks.mean(axis=0, keepdims=True)
    norms = np.sqrt((centred**2).sum(axis=0))
    norms = np.where(norms > 0, norms, 1.0)
    correlation = (centred.T @ centred) / np.outer(norms, norms)
    distance = 1.0 - np.abs(np.clip(correlation, -1.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    return 0.5 * (distance + distance.T)


def _cluster_with_cap(
    distance: np.ndarray,
    *,
    threshold: float,
    maximum_size: int,
    depth: int = 0,
) -> np.ndarray:
    """Average-linkage partition, re-split while any group exceeds the cap."""

    n = distance.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    labels = fcluster(tree, t=float(threshold), criterion="distance")
    if depth >= 8:
        return labels.astype(int) - 1
    result = np.asarray(labels, dtype=int) - 1
    next_label = int(result.max()) + 1
    for label in np.unique(result):
        members = np.flatnonzero(result == label)
        if len(members) <= int(maximum_size) or len(members) <= 1:
            continue
        sub = _cluster_with_cap(
            distance[np.ix_(members, members)],
            threshold=float(threshold) * 0.5,
            maximum_size=int(maximum_size),
            depth=depth + 1,
        )
        for sub_label in np.unique(sub):
            if sub_label == sub.min():
                continue
            result[members[sub == sub_label]] = next_label
            next_label += 1
    return result


def build_feature_groups(
    context: ExtensionContext,
    *,
    tasks: Sequence[str] | None = None,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Partition each task's feature space into correlation groups."""

    study = context.study(STUDY)
    spec = study["group_definition"]
    rows: list[dict[str, Any]] = []
    for task_id in tasks or ALLOWED_TASKS:
        data = load_condition_data(context.paths, task_id)
        X_train = np.asarray(data.X_train, dtype=np.float64)
        cap = int(spec["correlation_max_rows"])
        if len(X_train) > cap:
            rng = np.random.default_rng(stable_seed(20260908, "groups", task_id))
            X_train = X_train[rng.choice(len(X_train), size=cap, replace=False)]
        distance = _absolute_spearman_distance(X_train)
        maximum_size = max(
            2,
            int(math.floor(float(spec["maximum_group_size_fraction"]) * distance.shape[0])),
        )
        labels = _cluster_with_cap(
            distance,
            threshold=float(spec["distance_threshold"]),
            maximum_size=maximum_size,
        )
        sizes = pd.Series(labels).value_counts()
        for index, name in enumerate(data.feature_names):
            rows.append(
                {
                    "task_id": task_id,
                    "feature_index": int(index),
                    "feature_name": str(name),
                    "group_id": int(labels[index]),
                    "group_size": int(sizes[labels[index]]),
                    "grouping_method": str(spec["method"]),
                    "distance_threshold": float(spec["distance_threshold"]),
                    "maximum_group_size": int(maximum_size),
                    "correlation_rows_used": int(len(X_train)),
                }
            )
        if logger is not None:
            logger.info(
                "%s: %d features -> %d correlation groups (largest %d)",
                task_id,
                distance.shape[0],
                int(len(np.unique(labels))),
                int(sizes.max()),
            )
    frame = pd.DataFrame(rows)
    write_extension_table(frame, context.tables_dir(STUDY) / "feature_groups.csv")
    return frame


def load_feature_groups(context: ExtensionContext, task_id: str) -> np.ndarray:
    path = context.tables_root / STUDY / "feature_groups.csv"
    if not path.is_file():
        raise RuntimeError("Run the study 2 'groups' stage first")
    frame = pd.read_csv(path)
    frame = frame[frame["task_id"].astype(str).eq(task_id)].sort_values("feature_index")
    if frame.empty:
        raise RuntimeError(f"No correlation groups recorded for {task_id}")
    return frame["group_id"].to_numpy(dtype=int)


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #
@dataclass
class OperatorContext:
    """Everything the replacement operators draw on, fitted on training data."""

    data: ConditionData
    donor_pool: np.ndarray
    donor_labels: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    feature_std: np.ndarray
    training_median: np.ndarray
    class_medians: np.ndarray
    groups: np.ndarray


def build_operator_context(
    context: ExtensionContext, task_id: str
) -> OperatorContext:
    study = context.study(STUDY)
    data = load_condition_data(context.paths, task_id)
    X_train = np.asarray(data.X_train, dtype=np.float32)
    y_train = np.asarray(data.y_train, dtype=int)
    rng = np.random.default_rng(
        stable_seed(int(study["donor_pool_seed"]), "donor_pool", task_id)
    )
    pool_cap = int(study["donor_pool_max_rows"])
    if len(X_train) > pool_cap:
        selection = np.sort(rng.choice(len(X_train), size=pool_cap, replace=False))
    else:
        selection = np.arange(len(X_train))
    donor_pool = X_train[selection]
    donor_labels = y_train[selection]
    covariance_rows = min(int(study["covariance_max_rows"]), len(X_train))
    covariance_sample = X_train[:covariance_rows].astype(np.float64)
    mean = covariance_sample.mean(axis=0)
    covariance = np.cov(covariance_sample, rowvar=False)
    covariance = covariance + float(study["covariance_ridge"]) * np.eye(
        covariance.shape[0]
    )
    feature_std = np.sqrt(np.diag(covariance))
    feature_std = np.where(feature_std > 1e-12, feature_std, 1.0)
    return OperatorContext(
        data=data,
        donor_pool=donor_pool,
        donor_labels=donor_labels,
        mean=mean,
        covariance=covariance,
        feature_std=feature_std,
        training_median=np.asarray(data.references["training_median"], dtype=np.float32),
        class_medians=np.asarray(data.class_medians, dtype=np.float32),
        groups=load_feature_groups(context, task_id),
    )


def _conditional_gaussian(
    operator: OperatorContext,
    X: np.ndarray,
    selected: np.ndarray,
    *,
    draw: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """E[x_S | x_R] (optionally plus a conditional draw), row by row.

    Rows share a selected set only when the attribution ranking coincides, so
    the Schur complement is solved per distinct selection rather than per row.
    """

    n_rows, n_features = X.shape
    values = np.empty((n_rows, selected.shape[1]), dtype=np.float64)
    keys: dict[bytes, list[int]] = {}
    for row in range(n_rows):
        key = np.sort(selected[row]).astype(np.int32).tobytes()
        keys.setdefault(key, []).append(row)
    for key, rows in keys.items():
        columns = np.frombuffer(key, dtype=np.int32).astype(int)
        retained = np.setdiff1d(np.arange(n_features), columns, assume_unique=False)
        if len(retained) == 0:
            block = np.broadcast_to(
                operator.mean[columns], (len(rows), len(columns))
            ).copy()
        else:
            sigma_sr = operator.covariance[np.ix_(columns, retained)]
            sigma_rr = operator.covariance[np.ix_(retained, retained)]
            solved = np.linalg.solve(sigma_rr, sigma_sr.T).T
            deviation = X[np.ix_(rows, retained)].astype(np.float64) - operator.mean[retained]
            block = operator.mean[columns] + deviation @ solved.T
            if draw:
                sigma_ss = operator.covariance[np.ix_(columns, columns)]
                conditional = sigma_ss - solved @ sigma_sr.T
                conditional = 0.5 * (conditional + conditional.T)
                eigenvalues, eigenvectors = np.linalg.eigh(conditional)
                eigenvalues = np.clip(eigenvalues, 0.0, None)
                factor = eigenvectors * np.sqrt(eigenvalues)
                noise = rng.standard_normal((len(rows), len(columns)))
                block = block + noise @ factor.T
        # Restore the caller's column order for this row group.
        order = np.argsort(np.argsort(selected[rows[0]]))
        values[rows] = block[:, order]
    return values.astype(np.float32)


def _nearest_neighbour_values(
    operator: OperatorContext,
    X: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    """Donor row closest in the retained coordinates, standardised distance."""

    n_rows, n_features = X.shape
    values = np.empty((n_rows, selected.shape[1]), dtype=np.float32)
    pool = operator.donor_pool / operator.feature_std.astype(np.float32)
    scaled = np.asarray(X, dtype=np.float32) / operator.feature_std.astype(np.float32)
    for row in range(n_rows):
        columns = selected[row]
        retained = np.setdiff1d(np.arange(n_features), columns, assume_unique=False)
        if len(retained) == 0:
            values[row] = operator.donor_pool[0, columns]
            continue
        difference = pool[:, retained] - scaled[row, retained]
        distance = np.einsum("ij,ij->i", difference, difference)
        values[row] = operator.donor_pool[int(np.argmin(distance)), columns]
    return values


def operator_values(
    operator_id: str,
    operator: OperatorContext,
    X: np.ndarray,
    targets: np.ndarray,
    true_classes: np.ndarray,
    selected: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replacement values for the selected coordinates, one row per flow."""

    rows = np.arange(len(X))[:, None]
    if operator_id in {"training_median", "correlation_group_median"}:
        matrix = np.broadcast_to(operator.training_median, X.shape)
        return np.asarray(matrix)[rows, selected]
    if operator_id == "target_class_median":
        return operator.class_medians[np.asarray(targets, dtype=int)][rows, selected]
    if operator_id in {"joint_donor", "correlation_group_joint_donor"}:
        donors = rng.integers(0, len(operator.donor_pool), size=len(X))
        return operator.donor_pool[donors[:, None], selected]
    if operator_id == "target_class_donor":
        donors = np.empty(len(X), dtype=int)
        for row, target in enumerate(np.asarray(targets, dtype=int)):
            candidates = np.flatnonzero(operator.donor_labels == int(target))
            if len(candidates) == 0:
                candidates = np.arange(len(operator.donor_pool))
            donors[row] = int(rng.choice(candidates))
        return operator.donor_pool[donors[:, None], selected]
    if operator_id == "conditional_gaussian_mean":
        return _conditional_gaussian(operator, X, selected, draw=False, rng=rng)
    if operator_id == "conditional_gaussian_draw":
        return _conditional_gaussian(operator, X, selected, draw=True, rng=rng)
    if operator_id == "nearest_neighbour_donor":
        return _nearest_neighbour_values(operator, X, selected)
    raise ValueError(f"Unknown perturbation operator {operator_id!r}")


def unconditional_reference(
    operator_id: str,
    operator: OperatorContext,
    X: np.ndarray,
    targets: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """The operator's complete-replacement row, used as the insertion origin."""

    n_rows, n_features = X.shape
    everything = np.broadcast_to(np.arange(n_features), (n_rows, n_features))
    if operator_id in {
        "training_median",
        "correlation_group_median",
        "target_class_median",
        "joint_donor",
        "correlation_group_joint_donor",
        "target_class_donor",
    }:
        return np.asarray(
            operator_values(
                operator_id,
                operator,
                X,
                targets,
                targets,
                np.asarray(everything),
                rng=rng,
            ),
            dtype=np.float32,
        )
    # With no retained coordinates the conditional operators degenerate to
    # their unconditional law; the Gaussian mean and the donor pool mean are
    # the declared fallbacks.
    if operator_id in {"conditional_gaussian_mean", "conditional_gaussian_draw"}:
        return np.broadcast_to(
            operator.mean.astype(np.float32), (n_rows, n_features)
        ).copy()
    if operator_id == "nearest_neighbour_donor":
        return np.broadcast_to(
            operator.donor_pool.mean(axis=0).astype(np.float32), (n_rows, n_features)
        ).copy()
    raise ValueError(f"Unknown perturbation operator {operator_id!r}")


def _selected_coordinates(
    attributes: np.ndarray, k: int, groups: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k coordinates, expanded to whole groups when grouping is active."""

    absolute = np.abs(np.asarray(attributes, dtype=float))
    order = np.argsort(-absolute, axis=1, kind="stable")
    top = order[:, : min(max(int(k), 1), absolute.shape[1])]
    if groups is None:
        return top, np.ones(len(top), dtype=int) * top.shape[1]
    masks = np.zeros(absolute.shape, dtype=bool)
    group_counts = np.zeros(len(absolute), dtype=int)
    for row in range(len(absolute)):
        selected_groups = np.unique(groups[top[row]])
        masks[row] = np.isin(groups, selected_groups)
        group_counts[row] = len(selected_groups)
    # Rows can select different coordinate counts; pad with repeats of the
    # first selected coordinate, which `_replace_selected` handles idempotently.
    widths = masks.sum(axis=1)
    width = int(widths.max())
    padded = np.empty((len(absolute), width), dtype=int)
    for row in range(len(absolute)):
        members = np.flatnonzero(masks[row])
        padded[row, : len(members)] = members
        if len(members) < width:
            padded[row, len(members) :] = members[0]
    return padded, group_counts


def _group_matched_control(
    groups: np.ndarray,
    displacement: np.ndarray,
    selected: np.ndarray,
    *,
    bins: int,
    seed: int,
    flow_ids: Sequence[str],
) -> np.ndarray:
    """Sample the same number of groups from matched group-displacement bins."""

    unique_groups = np.unique(groups)
    group_displacement = np.vstack(
        [displacement[:, groups == group].mean(axis=1) for group in unique_groups]
    ).T
    result_masks = np.zeros(displacement.shape, dtype=bool)
    for row, flow_id in enumerate(flow_ids):
        order = np.argsort(group_displacement[row], kind="stable")
        bin_id = np.empty(len(unique_groups), dtype=int)
        bin_id[order] = np.minimum(
            int(bins) - 1,
            np.floor(np.arange(len(unique_groups)) * int(bins) / len(unique_groups)).astype(int),
        )
        chosen_groups = np.unique(groups[selected[row]])
        chosen_positions = np.searchsorted(unique_groups, chosen_groups)
        rng = np.random.default_rng(stable_seed(seed, flow_id, "group_control"))
        picked: list[int] = []
        for current_bin in range(int(bins)):
            count = int(np.sum(bin_id[chosen_positions] == current_bin))
            if count == 0:
                continue
            candidates = np.flatnonzero(bin_id == current_bin)
            replace = count > len(candidates)
            picked.extend(
                rng.choice(candidates, size=count, replace=replace).astype(int).tolist()
            )
        result_masks[row] = np.isin(groups, unique_groups[np.asarray(picked, dtype=int)])
    widths = result_masks.sum(axis=1)
    width = int(max(widths.max(), selected.shape[1]))
    padded = np.empty((len(displacement), width), dtype=int)
    for row in range(len(displacement)):
        members = np.flatnonzero(result_masks[row])
        padded[row, : len(members)] = members
        padded[row, len(members) :] = members[0]
    return padded


# --------------------------------------------------------------------------- #
# Perturbation runner
# --------------------------------------------------------------------------- #
def _complement(selected: np.ndarray, n_features: int) -> np.ndarray:
    """Per-row complement of a (possibly padded) selection, padded to a rectangle."""

    complements = [
        np.setdiff1d(np.arange(n_features), np.unique(row), assume_unique=False)
        for row in selected
    ]
    width = max(1, max(len(item) for item in complements))
    padded = np.empty((len(selected), width), dtype=int)
    for row, members in enumerate(complements):
        if len(members) == 0:
            padded[row, :] = int(np.unique(selected[row])[0])
            continue
        padded[row, : len(members)] = members
        padded[row, len(members) :] = members[0]
    return padded


def _replace(X: np.ndarray, selected: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.asarray(X, dtype=np.float32).copy()
    rows = np.arange(len(out))[:, None]
    out[rows, selected] = np.asarray(values, dtype=np.float32)
    return out


def _normalized_area(fractions: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Baseline AOPC: trapezoid from the origin, normalised by the last fraction."""

    x = np.concatenate([[0.0], fractions])
    y = np.concatenate([np.zeros((len(values), 1)), values], axis=1)
    return np.trapezoid(y, x, axis=1) / fractions[-1]


def _operator_repetitions(study: Mapping[str, Any], operator_id: str) -> int:
    overrides = study.get("operator_random_repetitions") or {}
    if operator_id in overrides:
        return int(overrides[operator_id])
    return int(study["random_feature_repetitions"])


def _primary_attributions(
    cache: ProbeCache,
    config: Mapping[str, Any],
    *,
    task_id: str,
    model: str,
    seed: int,
    method: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    specs = probe_specs(config, method)
    if method != "lime":
        spec = next(item for item in specs if item.role == "primary")
        result = cache.get(task_id, model, int(seed), spec)
        return result.index, result.attributions
    results = [cache.get(task_id, model, int(seed), spec) for spec in specs]
    index, arrays = align_probe_arrays(results)
    return index, np.mean(np.stack(arrays, axis=0), axis=0)


def _condition_arrays(
    context: ExtensionContext,
    cache: ProbeCache,
    *,
    task_id: str,
    model: str,
    seed: int,
    method: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    index, attributions = _primary_attributions(
        cache,
        context.audit_config,
        task_id=task_id,
        model=model,
        seed=int(seed),
        method=method,
    )
    mask = index["in_stochastic_subset"].map(_as_bool).to_numpy()
    return index.loc[mask].reset_index(drop=True), np.asarray(attributions)[mask]


def run_perturbations(
    context: ExtensionContext,
    *,
    tasks: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    methods: Sequence[str] | None = None,
    operators: Sequence[str] | None = None,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Evaluate every declared operator on the baseline functional panel."""

    study = context.study(STUDY)
    declared = {str(item["operator_id"]): item for item in study["operators"]}
    operator_ids = list(operators or declared)
    unknown = sorted(set(operator_ids) - set(declared))
    if unknown:
        raise ValueError(f"Operators are not declared in the extension config: {unknown}")
    fractions = np.asarray(
        sorted(float(value) for value in study["feature_fractions"]), dtype=float
    )
    bins = int(study["displacement_bins"])
    method_list = list(methods or study["methods"])
    cache = ProbeCache(context.audit_config, context.training_config, context.paths)
    runtime_rows: list[dict[str, Any]] = []

    for task_id in tasks or ALLOWED_TASKS:
        operator_context = build_operator_context(context, task_id)
        nearest_cap = int(study["nearest_neighbour_donor_pool_max_rows"])
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
                for method in method_list:
                    directory = (
                        context.results_root
                        / STUDY
                        / "operators"
                        / task_id
                        / model
                        / f"seed_{seed}"
                        / method
                    )
                    outputs = {
                        "curves": directory / "operator_curves_per_flow.csv",
                        "aopc": directory / "operator_aopc_per_flow.csv",
                        "decision": directory / "operator_decision.json",
                    }
                    metadata_path = directory / "cache_metadata.json"
                    inputs = artifact_records(
                        [
                            context.extension_config_path,
                            context.paths.config_path,
                            context.paths.training_config_path,
                            context.paths.central_panel,
                            condition["model"],
                            condition["metadata"],
                            context.tables_root / STUDY / "feature_groups.csv",
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
                            "stage": "dependency_perturbations",
                            "implementation_hash": implementation_hash,
                            "inputs": inputs,
                            "task_id": task_id,
                            "model": model,
                            "seed": int(seed),
                            "method": method,
                            "operators": operator_ids,
                            "fractions": fractions.tolist(),
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
                    identity, attributions = _condition_arrays(
                        context,
                        cache,
                        task_id=task_id,
                        model=model,
                        seed=int(seed),
                        method=method,
                    )
                    orders = pd.to_numeric(
                        identity["sample_order"], errors="raise"
                    ).to_numpy(int)
                    X = np.asarray(
                        operator_context.data.X_test[orders], dtype=np.float32
                    )
                    targets = pd.to_numeric(
                        identity["target_class_id"], errors="raise"
                    ).to_numpy(int)
                    true_classes = pd.to_numeric(
                        identity["true_class_id"], errors="raise"
                    ).to_numpy(int)
                    if frozen_model is None:
                        frozen_model = _load_frozen_model(condition["model"])
                    started = time.perf_counter()
                    curves, aopc = _evaluate_operators(
                        frozen_model,
                        X,
                        targets,
                        true_classes,
                        attributions,
                        identity,
                        operator_context,
                        study=study,
                        operator_ids=operator_ids,
                        fractions=fractions,
                        bins=bins,
                        task_id=task_id,
                        model_name=model,
                        seed=int(seed),
                        method=method,
                        nearest_cap=nearest_cap,
                    )
                    runtime = float(time.perf_counter() - started)
                    directory.mkdir(parents=True, exist_ok=True)
                    curves.to_csv(outputs["curves"], index=False)
                    aopc.to_csv(outputs["aopc"], index=False)
                    decision = {
                        "extension_study": STUDY,
                        "status": "complete",
                        "task_id": task_id,
                        "model": model,
                        "seed": int(seed),
                        "method": method,
                        "operators": operator_ids,
                        "panel": "baseline_central_stochastic_subset",
                        "flows": int(identity["flow_id"].nunique()),
                        "runtime_seconds": runtime,
                        "stage_signature": stage_signature,
                    }
                    outputs["decision"].write_text(
                        json.dumps(decision, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    write_cache_metadata(
                        metadata_path,
                        stage="dependency_perturbations",
                        signature_value=stage_signature,
                        config_hash=sha256_file(context.extension_config_path),
                        implementation_hash=implementation_hash,
                        inputs=inputs,
                        outputs=expected,
                        extra=decision,
                    )
                    runtime_rows.append(
                        {
                            "task_id": task_id,
                            "model": model,
                            "seed": int(seed),
                            "method": method,
                            "flows": int(identity["flow_id"].nunique()),
                            "runtime_seconds": runtime,
                        }
                    )
                    if logger is not None:
                        logger.info(
                            "operators %s/%s/seed_%d/%s: flows=%d, %.1fs",
                            task_id, model, seed, method,
                            identity["flow_id"].nunique(), runtime,
                        )
                cache.clear()
    frame = pd.DataFrame(runtime_rows)
    if len(frame):
        write_extension_table(
            frame, context.tables_dir(STUDY) / "operator_runtime.csv"
        )
    return frame


def _evaluate_operators(
    model: Any,
    X: np.ndarray,
    targets: np.ndarray,
    true_classes: np.ndarray,
    attributions: np.ndarray,
    identity: pd.DataFrame,
    operator_context: OperatorContext,
    *,
    study: Mapping[str, Any],
    operator_ids: Sequence[str],
    fractions: np.ndarray,
    bins: int,
    task_id: str,
    model_name: str,
    seed: int,
    method: str,
    nearest_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    declared = {str(item["operator_id"]): item for item in study["operators"]}
    n_features = X.shape[1]
    flow_ids = identity["flow_id"].astype(str).tolist()
    clean = target_probabilities(model, X, targets)
    identity_columns = list(IDENTITY_COLUMNS) + [
        "class_specific_inference_eligible",
        "target_class_id",
        "target_class",
        "confidence",
        "model_margin",
    ]
    curve_rows: list[pd.DataFrame] = []

    for operator_id in operator_ids:
        family = str(declared[operator_id]["family"])
        grouped = operator_id in GROUP_OPERATORS
        repetitions = _operator_repetitions(study, operator_id)
        pool_backup = operator_context.donor_pool
        label_backup = operator_context.donor_labels
        if operator_id == "nearest_neighbour_donor" and len(pool_backup) > nearest_cap:
            operator_context.donor_pool = pool_backup[:nearest_cap]
            operator_context.donor_labels = label_backup[:nearest_cap]
        reference_rng = np.random.default_rng(
            stable_seed(20260908, "reference", task_id, model_name, seed, method, operator_id)
        )
        reference = unconditional_reference(
            operator_id, operator_context, X, targets, rng=reference_rng
        )
        reference_probability = target_probabilities(model, reference, targets)
        displacement = np.abs(X - reference)

        for fraction in fractions:
            k = min(n_features, max(1, int(math.ceil(float(fraction) * n_features))))
            selected, group_counts = _selected_coordinates(
                attributions, k, operator_context.groups if grouped else None
            )
            arm_rng = np.random.default_rng(
                stable_seed(
                    20260908, "attribution", task_id, model_name, seed, method,
                    operator_id, float(fraction),
                )
            )
            deletion_effect, insertion_effect, changed = _arm_effects(
                model,
                X,
                targets,
                true_classes,
                selected,
                operator_id,
                operator_context,
                clean,
                reference_probability,
                rng=arm_rng,
            )
            attribution_displacement = displacement[
                np.arange(len(X))[:, None], selected
            ].mean(axis=1)
            controls: dict[str, dict[str, np.ndarray]] = {
                "matched": {
                    "deletion": np.empty((repetitions, len(X))),
                    "insertion": np.empty((repetitions, len(X))),
                    "displacement": np.empty((repetitions, len(X))),
                    "coordinates": np.empty((repetitions, len(X))),
                },
                "uniform": {
                    "deletion": np.empty((repetitions, len(X))),
                    "insertion": np.empty((repetitions, len(X))),
                    "displacement": np.empty((repetitions, len(X))),
                    "coordinates": np.empty((repetitions, len(X))),
                },
            }
            for repetition in range(repetitions):
                control_seed = stable_seed(
                    20260908, "control", task_id, model_name, seed, method,
                    operator_id, float(fraction), repetition,
                )
                if grouped:
                    matched_selected = _group_matched_control(
                        operator_context.groups,
                        displacement,
                        selected,
                        bins=bins,
                        seed=control_seed,
                        flow_ids=flow_ids,
                    )
                else:
                    matched_selected = _displacement_bin_matched_feature_matrix(
                        displacement,
                        selected,
                        bins=bins,
                        seed=control_seed,
                        flow_ids=flow_ids,
                    )
                uniform_selected = _random_feature_matrix(
                    len(X),
                    n_features,
                    selected.shape[1],
                    seed=stable_seed(control_seed, "uniform"),
                    flow_ids=flow_ids,
                )
                for label, control_selected in (
                    ("matched", matched_selected),
                    ("uniform", uniform_selected),
                ):
                    control_rng = np.random.default_rng(
                        stable_seed(control_seed, label, "values")
                    )
                    control_deletion, control_insertion, control_changed = _arm_effects(
                        model,
                        X,
                        targets,
                        true_classes,
                        control_selected,
                        operator_id,
                        operator_context,
                        clean,
                        reference_probability,
                        rng=control_rng,
                    )
                    controls[label]["deletion"][repetition] = control_deletion
                    controls[label]["insertion"][repetition] = control_insertion
                    controls[label]["displacement"][repetition] = displacement[
                        np.arange(len(X))[:, None], control_selected
                    ].mean(axis=1)
                    controls[label]["coordinates"][repetition] = control_changed

            for check, attribution_effect in (
                ("deletion", deletion_effect),
                ("insertion", insertion_effect),
            ):
                frame = identity[identity_columns].copy()
                frame["model"] = model_name
                frame["method"] = method
                frame["operator"] = operator_id
                frame["operator_family"] = family
                frame["functional_check"] = check
                frame["feature_fraction"] = float(fraction)
                frame["requested_top_k"] = int(k)
                frame["coordinates_changed"] = changed
                frame["groups_changed"] = group_counts if grouped else np.nan
                frame["random_feature_repetitions"] = int(repetitions)
                frame["attribution_effect"] = attribution_effect
                matched = controls["matched"][check]
                uniform = controls["uniform"][check]
                frame["random_effect_mean"] = matched.mean(axis=0)
                frame["random_effect_std"] = matched.std(
                    axis=0, ddof=1 if repetitions > 1 else 0
                )
                frame["random_effect_standard_error"] = frame[
                    "random_effect_std"
                ] / math.sqrt(repetitions)
                frame["uniform_random_effect_mean"] = uniform.mean(axis=0)
                frame["attribution_minus_random"] = (
                    frame["attribution_effect"] - frame["random_effect_mean"]
                )
                frame["attribution_minus_uniform_random"] = (
                    frame["attribution_effect"] - frame["uniform_random_effect_mean"]
                )
                frame["attribution_displacement_mean"] = attribution_displacement
                frame["matched_random_displacement_mean"] = controls["matched"][
                    "displacement"
                ].mean(axis=0)
                frame["matched_random_coordinates_mean"] = controls["matched"][
                    "coordinates"
                ].mean(axis=0)
                frame["clean_target_probability"] = clean
                frame["reference_target_probability"] = reference_probability
                curve_rows.append(frame)

        operator_context.donor_pool = pool_backup
        operator_context.donor_labels = label_backup

    curves = pd.concat(curve_rows, ignore_index=True)
    aopc = _aopc_from_curves(curves, fractions)
    return curves, aopc


def _arm_effects(
    model: Any,
    X: np.ndarray,
    targets: np.ndarray,
    true_classes: np.ndarray,
    selected: np.ndarray,
    operator_id: str,
    operator_context: OperatorContext,
    clean: np.ndarray,
    reference_probability: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deletion and insertion effects for one selected set under one operator.

    Deletion replaces the selected coordinates.  Insertion replaces their
    complement, so a conditional operator conditions on exactly the
    coordinates the explanation claims matter - the mirror image of deletion.
    """

    n_features = X.shape[1]
    deletion_values = operator_values(
        operator_id, operator_context, X, targets, true_classes, selected, rng=rng
    )
    deleted = _replace(X, selected, deletion_values)
    complement = _complement(selected, n_features)
    insertion_values = operator_values(
        operator_id, operator_context, X, targets, true_classes, complement, rng=rng
    )
    inserted = _replace(X, complement, insertion_values)
    deletion_effect = clean - target_probabilities(model, deleted, targets)
    insertion_effect = target_probabilities(model, inserted, targets) - reference_probability
    changed = np.asarray([len(np.unique(row)) for row in selected], dtype=float)
    return deletion_effect, insertion_effect, changed


def _aopc_from_curves(curves: pd.DataFrame, fractions: np.ndarray) -> pd.DataFrame:
    keys = [
        "task_id",
        "seed",
        "flow_id",
        "model",
        "method",
        "operator",
        "operator_family",
        "functional_check",
    ]
    carried = [
        "sample_order",
        "source_file",
        "source_row",
        "true_class_id",
        "true_class",
        "class_specific_inference_eligible",
        "target_class_id",
        "target_class",
        "confidence",
        "model_margin",
        "random_feature_repetitions",
    ]
    value_columns = [
        "attribution_effect",
        "random_effect_mean",
        "uniform_random_effect_mean",
        "attribution_minus_random",
        "attribution_minus_uniform_random",
        "random_effect_standard_error",
    ]
    rows: list[dict[str, Any]] = []
    for group_keys, group in curves.groupby(keys, dropna=False, sort=False):
        ordered = group.sort_values("feature_fraction", kind="mergesort")
        observed = ordered["feature_fraction"].to_numpy(dtype=float)
        if not np.array_equal(observed, fractions):
            raise RuntimeError(
                "Operator curve does not contain every declared feature fraction"
            )
        record = dict(zip(keys, group_keys))
        record.update({column: ordered.iloc[0][column] for column in carried})
        matrix = ordered[value_columns].to_numpy(dtype=float).T
        areas = _normalized_area(fractions, matrix)
        for column, area in zip(value_columns, areas):
            record[f"{column}_aopc"] = float(area)
        # Independent per-fraction control means: propagate their standard
        # errors through the same trapezoid weights the baseline uses.
        weights = np.empty(len(fractions))
        for index, fraction in enumerate(fractions):
            previous = 0.0 if index == 0 else fractions[index - 1]
            left = 0.5 * (fraction - previous)
            right = (
                0.5 * (fractions[index + 1] - fraction)
                if index + 1 < len(fractions)
                else 0.0
            )
            weights[index] = (left + right) / fractions[-1]
        standard_errors = ordered["random_effect_standard_error"].to_numpy(dtype=float)
        monte_carlo = float(np.sqrt(np.sum(np.square(weights * standard_errors))))
        record["random_aopc_monte_carlo_standard_error"] = monte_carlo
        record["attribution_minus_random_aopc_mc_lower95"] = (
            record["attribution_minus_random_aopc"] - 1.96 * monte_carlo
        )
        record["attribution_displacement_mean"] = float(
            ordered["attribution_displacement_mean"].mean()
        )
        record["matched_random_displacement_mean"] = float(
            ordered["matched_random_displacement_mean"].mean()
        )
        record["attribution_minus_matched_displacement"] = (
            record["attribution_displacement_mean"]
            - record["matched_random_displacement_mean"]
        )
        record["coordinates_changed_mean"] = float(
            ordered["coordinates_changed"].mean()
        )
        record["matched_random_coordinates_mean"] = float(
            ordered["matched_random_coordinates_mean"].mean()
        )
        record["aopc_definition"] = "trapezoid_from_zero_normalized_by_max_fraction"
        rows.append(record)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def _collect_aopc(context: ExtensionContext) -> pd.DataFrame:
    root = context.results_root / STUDY / "operators"
    paths = sorted(root.glob("*/*/seed_*/*/operator_aopc_per_flow.csv"))
    if not paths:
        raise RuntimeError("Run the study 2 'perturb' stage first")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def run_summaries(context: ExtensionContext, *, logger=None) -> dict[str, pd.DataFrame]:
    """Seed-balanced operator comparison against the baseline's own estimand."""

    study = context.study(STUDY)
    aopc = _collect_aopc(context)
    eligible = aopc[aopc["class_specific_inference_eligible"].map(_as_bool)]
    group_columns = ["task_id", "model", "method", "operator", "operator_family", "functional_check"]
    summary = summarize_metrics(
        eligible,
        group_columns=group_columns,
        metric_columns=[
            "attribution_minus_random_aopc",
            "attribution_minus_uniform_random_aopc",
            "attribution_effect_aopc",
            "random_effect_mean_aopc",
        ],
        config=context.audit_config,
        analysis_id="ext_dependency_perturbations",
    )
    summary["positive_against_matched_random"] = summary["metric"].eq(
        "attribution_minus_random_aopc"
    ) & summary["ci_low"].gt(0.0)
    monte_carlo = (
        eligible.groupby(group_columns, dropna=False)[
            [
                "random_aopc_monte_carlo_standard_error",
                "attribution_minus_random_aopc_mc_lower95",
                "coordinates_changed_mean",
                "matched_random_coordinates_mean",
                "attribution_displacement_mean",
                "matched_random_displacement_mean",
                "random_feature_repetitions",
            ]
        ]
        .mean()
        .reset_index()
    )
    summary = summary.merge(monte_carlo, on=group_columns, how="left")

    primary = summary[summary["metric"].eq("attribution_minus_random_aopc")].copy()
    counts = (
        primary.groupby(["method", "operator", "operator_family"], dropna=False)
        .agg(
            conditions=("positive_against_matched_random", "size"),
            positive_conditions=("positive_against_matched_random", "sum"),
            mean_effect=("estimate", "mean"),
            minimum_ci_low=("ci_low", "min"),
            mean_ci_low=("ci_low", "mean"),
        )
        .reset_index()
    )
    counts["positive_fraction"] = counts["positive_conditions"] / counts["conditions"]

    baseline_operator = "training_median"
    pivot_keys = ["task_id", "model", "method", "functional_check"]
    baseline_rows = primary[primary["operator"].eq(baseline_operator)][
        pivot_keys + ["estimate", "ci_low", "ci_high"]
    ].rename(
        columns={
            "estimate": "baseline_estimate",
            "ci_low": "baseline_ci_low",
            "ci_high": "baseline_ci_high",
        }
    )
    contrast = primary.merge(baseline_rows, on=pivot_keys, how="left")
    contrast["estimate_minus_baseline_operator"] = (
        contrast["estimate"] - contrast["baseline_estimate"]
    )
    contrast["verdict_baseline_operator"] = np.where(
        contrast["baseline_ci_low"] > 0.0, "positive", "not_positive"
    )
    contrast["verdict_this_operator"] = np.where(
        contrast["ci_low"] > 0.0, "positive", "not_positive"
    )
    contrast["verdict_changed"] = contrast["verdict_this_operator"].ne(
        contrast["verdict_baseline_operator"]
    )

    tables_dir = context.tables_dir(STUDY)
    outputs = {
        "operator_summary": tables_dir / "operator_summary.csv",
        "operator_positive_counts": tables_dir / "operator_positive_counts.csv",
        "operator_vs_baseline_contrast": tables_dir / "operator_vs_baseline_contrast.csv",
    }
    frames = {
        "operator_summary": summary,
        "operator_positive_counts": counts,
        "operator_vs_baseline_contrast": contrast,
    }
    for name, path in outputs.items():
        write_extension_table(frames[name], path)
    write_decision(
        context.results_dir(STUDY) / "study2_decision.json",
        {
            "study": STUDY,
            "status": "complete",
            "panel": "baseline_central_stochastic_subset",
            "operators": [str(item["operator_id"]) for item in study["operators"]],
            "estimand": str(study["primary_estimand"]),
            "conditions_per_method": int(
                primary.groupby("method")["operator"].size().max()
            )
            if len(primary)
            else 0,
            "role": "exploratory_operator_sensitivity",
            "outputs": {name: str(path) for name, path in outputs.items()},
        },
    )
    if logger is not None:
        for _, row in counts.iterrows():
            logger.info(
                "%-20s %-32s positive %d/%d, mean effect %+.4f",
                row["method"], row["operator"],
                int(row["positive_conditions"]), int(row["conditions"]),
                float(row["mean_effect"]),
            )
    return frames
