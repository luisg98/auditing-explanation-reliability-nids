"""Re-derive the baseline's three agreement inventories under swept settings.

The frozen module computes repeatability, cross-model and cross-probe
similarity with the top-k cut and identification tolerances taken from module
constants and from the frozen protocol file.  Study 5 has to vary exactly those
choices, so the pairing logic is restated here and the measurement is delegated
to :func:`common.similarity_matrix_at`.

The restatement is verified rather than assumed: :func:`reproduces_baseline`
recomputes every inventory at the baseline settings and compares it row by row
with the frozen per-flow tables the manuscript rests on.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.leakage_free_tabular import training
from src.leakage_free_tabular.cache import cache_hit
from src.leakage_free_tabular.epistemic_audit import (
    ALLOWED_MODELS,
    ALLOWED_PROBES,
    ALLOWED_TASKS,
    IDENTITY_COLUMNS,
    STOCHASTIC_SUBSET_PROBES,
    AuditPaths,
    ProbeSpec,
    _as_bool,
    _probe_directory,
    _probe_output_paths,
    _representative_stochastic_spec,
    load_probe_result,
    probe_specs,
)
from src.leakage_free_tabular_ext.common import similarity_matrix_at

METRIC_COLUMNS = ("spearman", "jaccard_10", "weighted_jaccard", "normalized_mass_l1")
GATED_METRICS = {
    "repeatability": ("spearman", "jaccard_10"),
    "cross_model": METRIC_COLUMNS,
    "cross_probe": METRIC_COLUMNS,
}


@dataclass(frozen=True)
class RepresentationSetting:
    """One point of study 5's measurement-choice sweep."""

    setting_id: str
    axis: str
    top_k: int = 10
    minimum_account_l1: float = 1.0e-12
    rank_tie_relative_tolerance: float = 1.0e-8
    minimum_identifiable_fraction: float = 0.95

    @property
    def is_baseline(self) -> bool:
        return (
            self.top_k == 10
            and self.minimum_account_l1 == 1.0e-12
            and self.rank_tie_relative_tolerance == 1.0e-8
            and self.minimum_identifiable_fraction == 0.95
        )

    def tolerances(self) -> dict[str, float | int]:
        return {
            "top_k": int(self.top_k),
            "minimum_account_l1": float(self.minimum_account_l1),
            "rank_tie_relative_tolerance": float(self.rank_tie_relative_tolerance),
        }

    def payload(self) -> dict[str, Any]:
        return {
            "setting_id": self.setting_id,
            "axis": self.axis,
            "top_k": int(self.top_k),
            "minimum_account_l1": float(self.minimum_account_l1),
            "rank_tie_relative_tolerance": float(self.rank_tie_relative_tolerance),
            "minimum_identifiable_fraction": float(self.minimum_identifiable_fraction),
            "is_baseline_setting": bool(self.is_baseline),
        }


class ProbeCache:
    """Per-condition probe loader with a two-level integrity check.

    ``epistemic_audit.load_probe_result`` re-hashes every stage input on each
    call, including the 100 MB+ processed training arrays, which costs about
    3.4 s per probe; a single sweep setting needs roughly two thousand probe
    reads.  This loader keeps the same guarantee at a fraction of the cost:

    * once per fitted condition it calls the frozen loader for one spec, so the
      full input chain (config bytes, implementation bytes, processed arrays,
      frozen model) is verified exactly as the baseline verifies it;
    * every other spec of that condition is validated against the signature and
      output digests recorded in its own ``cache_metadata.json``, so tampered or
      truncated outputs still fail closed.

    A condition whose full guard fails raises, exactly as the baseline does.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        training_config: Mapping[str, Any],
        paths: AuditPaths,
    ) -> None:
        self._config = config
        self._training_config = training_config
        self._paths = paths
        self._key: tuple[str, str, int] | None = None
        self._store: dict[tuple[str, str], ProbeArrays] = {}
        self._verified: set[tuple[str, str, int]] = set()

    def _verify_condition(self, task_id: str, model: str, seed: int) -> None:
        key = (task_id, model, int(seed))
        if key in self._verified:
            return
        witness = probe_specs(self._config, "integrated_gradients")[0]
        load_probe_result(
            self._config,
            self._training_config,
            self._paths,
            task_id=task_id,
            model=model,
            seed=int(seed),
            spec=witness,
        )
        self._verified.add(key)

    def get(self, task_id: str, model: str, seed: int, spec: ProbeSpec) -> "ProbeArrays":
        key = (task_id, model, int(seed))
        if self._key != key:
            self._key = key
            self._store = {}
        self._verify_condition(task_id, model, int(seed))
        store_key = (spec.method, spec.spec_id)
        if store_key not in self._store:
            self._store[store_key] = _read_probe_outputs(
                self._paths, task_id, model, int(seed), spec
            )
        return self._store[store_key]

    def clear(self) -> None:
        self._key = None
        self._store = {}


@dataclass(frozen=True)
class ProbeArrays:
    """The subset of ``ProbeResult`` the recomputation needs."""

    index: pd.DataFrame
    attributions: np.ndarray


def _read_probe_outputs(
    paths: AuditPaths,
    task_id: str,
    model: str,
    seed: int,
    spec: ProbeSpec,
) -> ProbeArrays:
    directory = _probe_directory(paths, task_id, model, int(seed), spec)
    outputs = _probe_output_paths(directory)
    metadata_path = outputs["metadata"]
    if not metadata_path.is_file():
        raise RuntimeError(f"Probe cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded = str(metadata.get("signature", ""))
    if not recorded:
        raise RuntimeError(f"Probe cache metadata records no signature: {metadata_path}")
    expected = [outputs["array"], outputs["index"], outputs["decision"]]
    hit, reason = cache_hit(
        metadata_path, expected_signature=recorded, expected_outputs=expected
    )
    if not hit:
        raise RuntimeError(
            f"Probe outputs failed their recorded digests for {task_id}/{model}/"
            f"seed_{seed}/{spec.method}/{spec.spec_id}: {reason}"
        )
    index = pd.read_csv(outputs["index"])
    with np.load(outputs["array"], allow_pickle=False) as archive:
        attributions = archive["attributions"].astype(np.float32)
        sample_order = archive["sample_order"].astype(np.int64)
    if (
        attributions.ndim != 2
        or len(attributions) != len(index)
        or not np.array_equal(
            sample_order,
            pd.to_numeric(index["sample_order"], errors="raise").to_numpy(np.int64),
        )
        or not np.isfinite(attributions).all()
    ):
        raise RuntimeError("Probe arrays and per-flow identity table disagree")
    return ProbeArrays(index=index, attributions=attributions)


def align_probe_arrays(
    results: Sequence["ProbeArrays"],
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Flow-aligned stack, mirroring ``epistemic_audit._align_probe_results``."""

    if not results:
        raise ValueError("At least one probe result is required")
    base = results[0].index.reset_index(drop=True)
    flow_ids = base["flow_id"].astype(str).to_numpy()
    arrays = [results[0].attributions]
    for result in results[1:]:
        lookup = {
            str(flow): position
            for position, flow in enumerate(result.index["flow_id"])
        }
        missing = [flow for flow in flow_ids if flow not in lookup]
        if missing:
            raise RuntimeError(
                f"Probe results do not share {len(missing)} panel flows"
            )
        order = np.asarray([lookup[flow] for flow in flow_ids], dtype=int)
        arrays.append(np.asarray(result.attributions)[order])
    return base, arrays


def _identity_frame(index: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    return index.loc[mask, list(IDENTITY_COLUMNS)].reset_index(drop=True)


def _attach(
    identity: pd.DataFrame,
    metrics: Mapping[str, np.ndarray],
    extra: Mapping[str, Any],
) -> pd.DataFrame:
    frame = identity.copy()
    for key, value in extra.items():
        frame[key] = value
    for key, value in metrics.items():
        frame[key] = value
    return frame


def _primary_or_mean(
    cache: ProbeCache,
    config: Mapping[str, Any],
    *,
    task_id: str,
    model: str,
    seed: int,
    method: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Frozen convention: LIME is represented by the mean over its runs."""

    specs = probe_specs(config, method)
    if method != "lime":
        spec = next(item for item in specs if item.role == "primary")
        result = cache.get(task_id, model, int(seed), spec)
        return result.index, result.attributions
    results = [cache.get(task_id, model, int(seed), spec) for spec in specs]
    index, arrays = align_probe_arrays(results)
    return index, np.mean(np.stack(arrays, axis=0), axis=0)


# --------------------------------------------------------------------------- #
# Repeatability
# --------------------------------------------------------------------------- #
def repeatability_comparisons(
    config: Mapping[str, Any], method: str
) -> list[tuple[ProbeSpec, ProbeSpec, str, str]]:
    """The frozen repeatability roles, restated with their analysis labels."""

    specs = probe_specs(config, method)
    if method == "lime":
        return [
            (
                left,
                right,
                f"run_{left.run_id:02d}_vs_{right.run_id:02d}",
                "primary_mean_pairwise",
            )
            for left, right in itertools.combinations(specs, 2)
        ]
    if method == "integrated_gradients":
        main = next(spec for spec in specs if spec.role == "primary")
        reference = next(
            spec
            for spec in specs
            if spec.reference == main.reference and int(spec.steps or -1) == 128
        )
        comparisons = []
        for candidate in specs:
            if (
                candidate.reference == main.reference
                and candidate.steps != reference.steps
                and candidate.role in {"primary", "numerical_sensitivity"}
            ):
                comparison = f"steps_{candidate.steps}_vs_{reference.steps}"
                role = (
                    "primary_main_to_128"
                    if comparison == "steps_64_vs_128"
                    else "numerical_sensitivity"
                )
                comparisons.append((candidate, reference, comparison, role))
        return comparisons
    if method == "occlusion":
        primary = next(spec for spec in specs if spec.role == "primary")
        replicate = next(spec for spec in specs if spec.role == "determinism_replicate")
        return [(primary, replicate, "fixed_rule_repeat", "primary_fixed_rule_repeat")]
    if method == "shap":
        primary = next(spec for spec in specs if spec.role == "primary")
        replicate = next(
            spec for spec in specs if spec.role == "stochastic_repeatability_replicate"
        )
        return [(primary, replicate, "stochastic_repeat", "primary_stochastic_repeat")]
    return []


def repeatability_detail_at(
    cache: ProbeCache,
    config: Mapping[str, Any],
    setting: RepresentationSetting,
    *,
    task_id: str,
    model: str,
    seed: int,
    method: str,
) -> pd.DataFrame:
    comparisons = repeatability_comparisons(config, method)
    if not comparisons:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for left_spec, right_spec, comparison, role in comparisons:
        index, arrays = align_probe_arrays(
            [
                cache.get(task_id, model, int(seed), left_spec),
                cache.get(task_id, model, int(seed), right_spec),
            ]
        )
        mask = index["in_central"].map(_as_bool).to_numpy()
        if not mask.any():
            continue
        identity = _identity_frame(index, mask)
        metrics = similarity_matrix_at(
            arrays[0][mask], arrays[1][mask], **setting.tolerances()
        )
        frames.append(
            _attach(
                identity,
                metrics,
                {
                    "model": model,
                    "method": method,
                    "comparison": comparison,
                    "left_run": int(left_spec.run_id),
                    "right_run": int(right_spec.run_id),
                    "probe_pair": f"{left_spec.spec_id}::{right_spec.spec_id}",
                    "analysis_role": role,
                    "class_specific_inference_eligible": index.loc[
                        mask, "class_specific_inference_eligible"
                    ]
                    .map(_as_bool)
                    .to_numpy(),
                },
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Cross-model equivalence
# --------------------------------------------------------------------------- #
def cross_model_detail_at(
    cache: ProbeCache,
    config: Mapping[str, Any],
    setting: RepresentationSetting,
    paths: AuditPaths,
    *,
    task_id: str,
    seed: int,
    method: str,
    central_panel: pd.DataFrame,
) -> pd.DataFrame:
    if method in STOCHASTIC_SUBSET_PROBES:
        specs = [
            spec
            for spec in probe_specs(config, method)
            if spec.role
            in {"primary", "independent_run", "stochastic_repeatability_replicate"}
        ]
        comparisons = [
            (
                spec.run_id,
                cache.get(task_id, ALLOWED_MODELS[0], int(seed), spec),
                cache.get(task_id, ALLOWED_MODELS[1], int(seed), spec),
            )
            for spec in specs
        ]
        pairs = [
            (run_id, left.index, left.attributions, right.index, right.attributions)
            for run_id, left, right in comparisons
        ]
    else:
        left_index, left_attrs = _primary_or_mean(
            cache, config, task_id=task_id, model=ALLOWED_MODELS[0], seed=seed, method=method
        )
        right_index, right_attrs = _primary_or_mean(
            cache, config, task_id=task_id, model=ALLOWED_MODELS[1], seed=seed, method=method
        )
        pairs = [(0, left_index, left_attrs, right_index, right_attrs)]

    panel = central_panel[
        central_panel["task_id"].astype(str).eq(task_id)
        & pd.to_numeric(central_panel["seed"], errors="raise").astype(int).eq(int(seed))
    ]
    if method in STOCHASTIC_SUBSET_PROBES:
        panel = panel[panel["in_stochastic_subset"].map(_as_bool)]
    flows = panel["flow_id"].astype(str).to_numpy()
    frames: list[pd.DataFrame] = []
    for run_id, left_index, left_attrs, right_index, right_attrs in pairs:
        left_lookup = {
            str(flow): position for position, flow in enumerate(left_index["flow_id"])
        }
        right_lookup = {
            str(flow): position for position, flow in enumerate(right_index["flow_id"])
        }
        missing = [
            flow for flow in flows if flow not in left_lookup or flow not in right_lookup
        ]
        if missing:
            raise RuntimeError(
                f"Cross-model recomputation is missing {len(missing)} panel flows"
            )
        left_rows = np.asarray([left_lookup[flow] for flow in flows], dtype=int)
        right_rows = np.asarray([right_lookup[flow] for flow in flows], dtype=int)
        identity = panel[list(IDENTITY_COLUMNS)].reset_index(drop=True)
        metrics = similarity_matrix_at(
            np.asarray(left_attrs)[left_rows],
            np.asarray(right_attrs)[right_rows],
            **setting.tolerances(),
        )
        frames.append(
            _attach(
                identity,
                metrics,
                {
                    "method": method,
                    "model_left": ALLOWED_MODELS[0],
                    "model_right": ALLOWED_MODELS[1],
                    "probe_run": int(run_id),
                    "class_specific_inference_eligible": panel[
                        "class_specific_inference_eligible"
                    ]
                    .map(_as_bool)
                    .to_numpy(),
                },
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Cross-probe triangulation
# --------------------------------------------------------------------------- #
def cross_probe_detail_at(
    cache: ProbeCache,
    config: Mapping[str, Any],
    setting: RepresentationSetting,
    *,
    task_id: str,
    model: str,
    seed: int,
    central_panel: pd.DataFrame,
) -> pd.DataFrame:
    triangulation = config["cross_probe_triangulation"]
    deterministic_methods = tuple(triangulation["deterministic_central_probes"])
    all_methods = tuple(triangulation["all_probe_stochastic_subset"])
    stochastic_methods = tuple(
        method for method in all_methods if method in STOCHASTIC_SUBSET_PROBES
    )
    deterministic: dict[str, tuple[pd.DataFrame, np.ndarray]] = {
        method: _primary_or_mean(
            cache, config, task_id=task_id, model=model, seed=seed, method=method
        )
        for method in deterministic_methods
    }
    stochastic_results = {
        method: [cache.get(task_id, model, int(seed), spec) for spec in probe_specs(config, method)]
        for method in stochastic_methods
    }
    panel = central_panel[
        central_panel["task_id"].astype(str).eq(task_id)
        & pd.to_numeric(central_panel["seed"], errors="raise").astype(int).eq(int(seed))
    ]
    stochastic_panel = panel[panel["in_stochastic_subset"].map(_as_bool)]
    frames: list[pd.DataFrame] = []

    def append(
        target_panel: pd.DataFrame,
        panel_scope: str,
        left_method: str,
        right_method: str,
        left_index: pd.DataFrame,
        left_attrs: np.ndarray,
        right_index: pd.DataFrame,
        right_attrs: np.ndarray,
        probe_run: int,
    ) -> None:
        flows = target_panel["flow_id"].astype(str).to_numpy()
        left_lookup = {
            str(flow): position for position, flow in enumerate(left_index["flow_id"])
        }
        right_lookup = {
            str(flow): position for position, flow in enumerate(right_index["flow_id"])
        }
        if any(
            flow not in left_lookup or flow not in right_lookup for flow in flows
        ):
            raise RuntimeError("Cross-probe recomputation lacks a registered panel flow")
        left_rows = np.asarray([left_lookup[flow] for flow in flows], dtype=int)
        right_rows = np.asarray([right_lookup[flow] for flow in flows], dtype=int)
        metrics = similarity_matrix_at(
            np.asarray(left_attrs)[left_rows],
            np.asarray(right_attrs)[right_rows],
            **setting.tolerances(),
        )
        frames.append(
            _attach(
                target_panel[list(IDENTITY_COLUMNS)].reset_index(drop=True),
                metrics,
                {
                    "model": model,
                    "panel_scope": panel_scope,
                    "probe_left": left_method,
                    "probe_right": right_method,
                    "probe_pair": f"{left_method}::{right_method}",
                    "probe_run": int(probe_run),
                    "class_specific_inference_eligible": target_panel[
                        "class_specific_inference_eligible"
                    ]
                    .map(_as_bool)
                    .to_numpy(),
                },
            )
        )

    for left_method, right_method in itertools.combinations(deterministic_methods, 2):
        append(
            panel,
            "deterministic_central",
            left_method,
            right_method,
            *deterministic[left_method],
            *deterministic[right_method],
            0,
        )
    for left_method, right_method in itertools.combinations(all_methods, 2):
        left_stochastic = left_method in stochastic_methods
        right_stochastic = right_method in stochastic_methods
        if not left_stochastic and not right_stochastic:
            append(
                stochastic_panel,
                "all_probe_stochastic_subset",
                left_method,
                right_method,
                *deterministic[left_method],
                *deterministic[right_method],
                0,
            )
        elif left_stochastic and right_stochastic:
            left_spec = _representative_stochastic_spec(config, left_method)
            right_spec = _representative_stochastic_spec(config, right_method)
            left_result = cache.get(task_id, model, int(seed), left_spec)
            right_result = cache.get(task_id, model, int(seed), right_spec)
            append(
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
            for spec, result in zip(
                probe_specs(config, stochastic_method),
                stochastic_results[stochastic_method],
            ):
                if left_stochastic:
                    arguments = (
                        result.index,
                        result.attributions,
                        deterministic_index,
                        deterministic_attrs,
                    )
                else:
                    arguments = (
                        deterministic_index,
                        deterministic_attrs,
                        result.index,
                        result.attributions,
                    )
                append(
                    stochastic_panel,
                    "all_probe_stochastic_subset",
                    left_method,
                    right_method,
                    *arguments,
                    int(spec.run_id),
                )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_inventories_at(
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    paths: AuditPaths,
    setting: RepresentationSetting,
    *,
    tasks: Sequence[str] = ALLOWED_TASKS,
    models: Sequence[str] = ALLOWED_MODELS,
    seeds: Sequence[int] = training.FINAL_SEEDS,
    methods: Sequence[str] = ALLOWED_PROBES,
    inventories: Sequence[str] = ("repeatability", "cross_model", "cross_probe"),
) -> dict[str, pd.DataFrame]:
    """Recompute the requested per-flow inventories at one sweep setting."""

    cache = ProbeCache(config, training_config, paths)
    central_panel = pd.read_csv(paths.central_panel)
    collected: dict[str, list[pd.DataFrame]] = {name: [] for name in inventories}
    for task_id in tasks:
        for seed in seeds:
            if "cross_probe" in collected:
                for model in models:
                    collected["cross_probe"].append(
                        cross_probe_detail_at(
                            cache,
                            config,
                            setting,
                            task_id=task_id,
                            model=model,
                            seed=int(seed),
                            central_panel=central_panel,
                        )
                    )
            for method in methods:
                if "cross_model" in collected:
                    collected["cross_model"].append(
                        cross_model_detail_at(
                            cache,
                            config,
                            setting,
                            paths,
                            task_id=task_id,
                            seed=int(seed),
                            method=method,
                            central_panel=central_panel,
                        )
                    )
                if "repeatability" in collected:
                    for model in models:
                        frame = repeatability_detail_at(
                            cache,
                            config,
                            setting,
                            task_id=task_id,
                            model=model,
                            seed=int(seed),
                            method=method,
                        )
                        if len(frame):
                            collected["repeatability"].append(frame)
            cache.clear()
    return {
        name: (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
        for name, frames in collected.items()
    }


BASELINE_PER_FLOW_TABLES = {
    "repeatability": "repeatability_per_flow.csv",
    "cross_model": "cross_model_per_flow.csv",
    "cross_probe": "cross_probe_per_flow.csv",
}

_JOIN_KEYS = {
    "repeatability": [
        "task_id",
        "seed",
        "flow_id",
        "model",
        "method",
        "comparison",
        "probe_pair",
    ],
    "cross_model": ["task_id", "seed", "flow_id", "method", "probe_run"],
    "cross_probe": [
        "task_id",
        "seed",
        "flow_id",
        "model",
        "panel_scope",
        "probe_pair",
        "probe_run",
    ],
}


def reproduces_baseline(
    recomputed: Mapping[str, pd.DataFrame],
    baseline_tables_root: Path,
    *,
    tolerance: float = 1.0e-9,
) -> pd.DataFrame:
    """Compare a baseline-setting recomputation with the frozen per-flow tables.

    The frozen cross-probe table labels several reference variants of the same
    probe with the same ``probe_run``, so its rows are not unique on their own
    labels.  The comparison is therefore made on per-key group means plus the
    sorted metric multiset, both of which are invariant to that ambiguity and
    still detect any changed value.
    """

    rows: list[dict[str, Any]] = []
    for name, frame in recomputed.items():
        baseline = pd.read_csv(baseline_tables_root / BASELINE_PER_FLOW_TABLES[name])
        scope = frame[["task_id", "seed"]].drop_duplicates()
        baseline = baseline.merge(scope, on=["task_id", "seed"], how="inner")
        keys = _JOIN_KEYS[name]
        record: dict[str, Any] = {
            "inventory": name,
            "recomputed_rows": int(len(frame)),
            "baseline_rows": int(len(baseline)),
            "row_count_matches": bool(len(frame) == len(baseline)),
        }
        left = frame.groupby(keys, dropna=False)[list(METRIC_COLUMNS)].mean().sort_index()
        right = (
            baseline.groupby(keys, dropna=False)[list(METRIC_COLUMNS)].mean().sort_index()
        )
        record["group_index_matches"] = bool(left.index.equals(right.index))
        for metric in METRIC_COLUMNS:
            if record["group_index_matches"]:
                record[f"{metric}_max_abs_group_diff"] = float(
                    (left[metric] - right[metric]).abs().max()
                )
            else:
                record[f"{metric}_max_abs_group_diff"] = float("nan")
            if record["row_count_matches"]:
                record[f"{metric}_max_abs_multiset_diff"] = float(
                    np.abs(
                        np.sort(frame[metric].to_numpy(dtype=float))
                        - np.sort(baseline[metric].to_numpy(dtype=float))
                    ).max()
                )
            else:
                record[f"{metric}_max_abs_multiset_diff"] = float("nan")
        differences = [
            record[f"{metric}_max_abs_{kind}_diff"]
            for metric in METRIC_COLUMNS
            for kind in ("group", "multiset")
        ]
        record["within_tolerance"] = bool(
            record["row_count_matches"]
            and record["group_index_matches"]
            and all(
                np.isfinite(value) and value <= float(tolerance) for value in differences
            )
        )
        record["tolerance"] = float(tolerance)
        rows.append(record)
    return pd.DataFrame(rows)
