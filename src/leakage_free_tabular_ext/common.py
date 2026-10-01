"""Shared context, metrics and uncertainty helpers for the extension studies.

The measurement definitions here are deliberate re-implementations of the
frozen ones in ``src.leakage_free_tabular.epistemic_audit``, parameterised on
the choices the extensions need to vary (top-k, identification tolerances,
resampling unit).  Where a definition is *not* varied it is imported from the
frozen module instead of being copied, so the baseline stays the single source
of truth for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Must precede `import pandas`; see the note in epistemic_audit.py.
import tensorflow as _tensorflow_import_order_guard  # noqa: F401

import numpy as np
import pandas as pd
import yaml
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
    ALLOWED_PROBES,
    ALLOWED_TASKS,
    AuditPaths,
    _as_bool,
    load_context,
    stable_seed,
)
from src.leakage_free_tabular_ext import EXTENSION_VERSION

__all__ = [
    "ExtensionContext",
    "load_extension_context",
    "similarity_at",
    "pairwise_similarity_frame",
    "crossed_source_seed_bootstrap_fast",
    "cluster_bootstrap",
    "paired_cluster_bootstrap",
    "apply_gate",
    "quantile_bins",
    "error_type_labels",
    "write_extension_table",
    "extension_stage_signature",
    "cached_stage",
]


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExtensionContext:
    """Everything an extension study needs to locate baseline and new outputs."""

    project_root: Path
    extension_config_path: Path
    extension_config: dict[str, Any]
    audit_config: dict[str, Any]
    training_config: dict[str, Any]
    paths: AuditPaths
    results_root: Path
    tables_root: Path
    reports_root: Path

    def study(self, key: str) -> dict[str, Any]:
        section = self.extension_config.get(key)
        if not isinstance(section, Mapping):
            raise KeyError(f"Extension config has no study section {key!r}")
        return dict(section)

    @property
    def shared(self) -> dict[str, Any]:
        return dict(self.extension_config["shared"])

    def results_dir(self, *parts: str) -> Path:
        path = self.results_root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def tables_dir(self, *parts: str) -> Path:
        path = self.tables_root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def implementation_records(self) -> list[dict[str, Any]]:
        """Hash the extension implementation, not only the frozen baseline."""

        package = self.project_root / "src/leakage_free_tabular_ext"
        files = sorted(package.glob("*.py"))
        return artifact_records(
            [
                self.project_root / "src/leakage_free_tabular/cache.py",
                self.project_root / "src/leakage_free_tabular/epistemic_audit.py",
                self.project_root / "src/leakage_free_tabular/training.py",
                *files,
            ]
        )


def load_extension_context(
    project_root: Path,
    extension_config_path: Path = Path("configs/audit_extensions.yaml"),
    audit_config_path: Path = Path("configs/leakage_free_epistemic_audit.yaml"),
    training_config_path: Path = Path("configs/leakage_free_tabular_training.yaml"),
) -> ExtensionContext:
    root = project_root.resolve()
    ext_path = (
        extension_config_path
        if extension_config_path.is_absolute()
        else root / extension_config_path
    )
    with ext_path.open("r", encoding="utf-8") as handle:
        extension_config = yaml.safe_load(handle) or {}
    if not isinstance(extension_config, dict):
        raise ValueError(f"Expected a YAML mapping in {ext_path}")
    _validate_extension_config(extension_config)
    audit_config, training_config, paths = load_context(
        root, audit_config_path, training_config_path
    )
    outputs = extension_config["outputs"]
    return ExtensionContext(
        project_root=root,
        extension_config_path=ext_path,
        extension_config=extension_config,
        audit_config=audit_config,
        training_config=training_config,
        paths=paths,
        results_root=root / str(outputs["results_root"]),
        tables_root=root / str(outputs["tables_root"]),
        reports_root=root / str(outputs["reports_root"]),
    )


def _validate_extension_config(config: Mapping[str, Any]) -> None:
    required = {"protocol_id", "baseline_protocol", "shared", "outputs"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Extension config lacks {missing}")
    if str(config["baseline_protocol"]) != "leakage_free_epistemic_audit_v2":
        raise ValueError("Extensions are declared against the v2 baseline protocol only")
    shared = config["shared"]
    if tuple(shared["tasks"]) != ALLOWED_TASKS:
        raise ValueError("Extension task scope must match the frozen audit scope")
    if tuple(shared["models"]) != ALLOWED_MODELS:
        raise ValueError("Extension model scope must match the frozen audit scope")
    if tuple(shared["seeds"]) != tuple(training.FINAL_SEEDS):
        raise ValueError("Extension seed scope must match the frozen fitted seeds")
    representation = shared["representation"]
    for key in (
        "top_k",
        "minimum_account_l1",
        "rank_tie_relative_tolerance",
        "minimum_identifiable_fraction_for_gate",
    ):
        if key not in representation:
            raise ValueError(f"Extension representation section lacks {key!r}")


# --------------------------------------------------------------------------- #
# Parameterised similarity
# --------------------------------------------------------------------------- #
def _account_diagnostics_at(
    absolute: np.ndarray,
    *,
    top_k: int,
    minimum_account_l1: float,
    rank_tie_relative_tolerance: float,
) -> dict[str, Any]:
    """Identification diagnostics with the tolerances exposed as arguments.

    Mirrors ``epistemic_audit._account_diagnostics``, whose tolerances are
    module constants and therefore cannot be swept without editing the frozen
    file.
    """

    total = float(absolute.sum())
    maximum = float(absolute.max(initial=0.0))
    tolerance = max(
        float(minimum_account_l1), maximum * float(rank_tie_relative_tolerance)
    )
    nonzero = bool(total >= float(minimum_account_l1))
    rank_identifiable = bool(nonzero and float(np.ptp(absolute)) > tolerance)
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


def _topk_at(absolute: np.ndarray, k: int) -> np.ndarray:
    effective = min(max(int(k), 1), len(absolute))
    return np.lexsort((np.arange(len(absolute)), -absolute))[:effective]


def _spearman_pair(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, right, rtol=0.0, atol=1e-12):
        return 1.0
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return 0.0
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    left_centred = left_rank - left_rank.mean()
    right_centred = right_rank - right_rank.mean()
    denominator = float(
        np.sqrt(np.sum(left_centred**2) * np.sum(right_centred**2))
    )
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(left_centred * right_centred) / denominator)


def similarity_at(
    left: np.ndarray,
    right: np.ndarray,
    *,
    top_k: int,
    minimum_account_l1: float = 1.0e-12,
    rank_tie_relative_tolerance: float = 1.0e-8,
) -> dict[str, float]:
    """Absolute-attribution similarity, with the baseline's fail-closed rules.

    Identical in definition to ``epistemic_audit.attribution_similarity`` at
    the baseline tolerances; the tolerances are arguments so study 5 can sweep
    them.  ``tests/test_audit_extensions.py`` pins the two against each other.
    """

    left_abs = np.abs(np.asarray(left, dtype=float).reshape(-1))
    right_abs = np.abs(np.asarray(right, dtype=float).reshape(-1))
    if left_abs.shape != right_abs.shape or not (
        np.isfinite(left_abs).all() and np.isfinite(right_abs).all()
    ):
        raise ValueError("Attribution vectors must have equal finite geometry")
    tolerances = {
        "top_k": int(top_k),
        "minimum_account_l1": float(minimum_account_l1),
        "rank_tie_relative_tolerance": float(rank_tie_relative_tolerance),
    }
    left_diagnostics = _account_diagnostics_at(left_abs, **tolerances)
    right_diagnostics = _account_diagnostics_at(right_abs, **tolerances)
    pair_nonzero = bool(left_diagnostics["nonzero"] and right_diagnostics["nonzero"])
    pair_rank = bool(
        left_diagnostics["rank_identifiable"] and right_diagnostics["rank_identifiable"]
    )
    pair_top_k = bool(
        left_diagnostics["top_k_identifiable"]
        and right_diagnostics["top_k_identifiable"]
    )
    left_top = set(_topk_at(left_abs, int(top_k)).tolist())
    right_top = set(_topk_at(right_abs, int(top_k)).tolist())
    union = left_top | right_top
    left_total = float(left_abs.sum())
    right_total = float(right_abs.sum())
    if pair_nonzero:
        left_mass = left_abs / left_total
        right_mass = right_abs / right_total
        mass_l1 = float(0.5 * np.abs(left_mass - right_mass).sum())
        weighted_denominator = float(np.maximum(left_mass, right_mass).sum())
        weighted = (
            float(np.minimum(left_mass, right_mass).sum() / weighted_denominator)
            if weighted_denominator >= float(minimum_account_l1)
            else 0.0
        )
    else:
        # Fail closed: a missing account must not read as perfect agreement.
        mass_l1 = 1.0
        weighted = 0.0
    return {
        "spearman": _spearman_pair(left_abs, right_abs) if pair_rank else 0.0,
        "jaccard_10": (
            float(len(left_top & right_top) / len(union))
            if union and pair_top_k
            else 0.0
        ),
        "weighted_jaccard": weighted,
        "normalized_mass_l1": mass_l1,
        "left_account_mass": float(left_diagnostics["mass"]),
        "right_account_mass": float(right_diagnostics["mass"]),
        "left_account_nonzero": bool(left_diagnostics["nonzero"]),
        "right_account_nonzero": bool(right_diagnostics["nonzero"]),
        "account_pair_nonzero": pair_nonzero,
        "left_rank_identifiable": bool(left_diagnostics["rank_identifiable"]),
        "right_rank_identifiable": bool(right_diagnostics["rank_identifiable"]),
        "rank_pair_identifiable": pair_rank,
        "left_top_k_identifiable": bool(left_diagnostics["top_k_identifiable"]),
        "right_top_k_identifiable": bool(right_diagnostics["top_k_identifiable"]),
        "top_k_pair_identifiable": pair_top_k,
        "left_top_k_boundary_gap": float(left_diagnostics["top_k_boundary_gap"]),
        "right_top_k_boundary_gap": float(right_diagnostics["top_k_boundary_gap"]),
        "agreement_on_null_accounts": bool(
            not left_diagnostics["nonzero"] and not right_diagnostics["nonzero"]
        ),
    }


def pairwise_similarity_frame(
    left: np.ndarray,
    right: np.ndarray,
    identity: pd.DataFrame,
    *,
    top_k: int,
    minimum_account_l1: float = 1.0e-12,
    rank_tie_relative_tolerance: float = 1.0e-8,
    extra: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Row-wise ``similarity_at`` over two aligned attribution matrices."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape or len(identity) != len(left_array):
        raise ValueError("Similarity inputs must align by row")
    rows: list[dict[str, Any]] = []
    identity_records = identity.reset_index(drop=True).to_dict("records")
    for index, record in enumerate(identity_records):
        rows.append(
            {
                **record,
                **dict(extra or {}),
                **similarity_at(
                    left_array[index],
                    right_array[index],
                    top_k=int(top_k),
                    minimum_account_l1=float(minimum_account_l1),
                    rank_tie_relative_tolerance=float(rank_tie_relative_tolerance),
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
def crossed_source_seed_bootstrap_fast(
    frame: pd.DataFrame,
    value_column: str,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    seed_column: str = "seed",
    flow_column: str = "flow_id",
    block_size: int = 256,
) -> dict[str, Any]:
    """Vectorised crossed source/seed equal-seed multinomial bootstrap.

    Same estimator and same resampling design as the frozen
    ``crossed_source_seed_equal_seed_bootstrap`` (fitted seeds resampled with
    replacement; one global multinomial source draw shared across every
    selected seed occurrence; equal weight per fitted seed), re-implemented in
    array form because study 5 needs the interval recomputed for thousands of
    (cell, setting) combinations.  It does not reproduce the frozen routine's
    RNG stream, so it agrees with it only up to Monte Carlo error; the
    agreement is asserted in the test suite and reported by study 5.
    """

    clean = frame[[seed_column, flow_column, value_column]].copy()
    clean[value_column] = pd.to_numeric(clean[value_column], errors="coerce")
    clean = clean.dropna(subset=[value_column])
    if clean.empty:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_std": float("nan"),
            "n_rows": 0,
            "n_flows": 0,
            "n_seeds": 0,
            "bootstrap_design": "vectorised_crossed_source_seed_equal_seed_bootstrap",
        }
    cell = (
        clean.groupby([seed_column, flow_column], sort=True)[value_column]
        .mean()
        .reset_index()
    )
    seed_codes, seed_values = pd.factorize(cell[seed_column], sort=True)
    flow_codes, flow_values = pd.factorize(cell[flow_column], sort=True)
    n_seeds = len(seed_values)
    n_flows = len(flow_values)
    values = cell[value_column].to_numpy(dtype=float)
    # Dense (seed x flow) value/presence matrices; absent cells contribute
    # nothing and are excluded from their seed's weighted mean.
    value_matrix = np.zeros((n_seeds, n_flows), dtype=float)
    present = np.zeros((n_seeds, n_flows), dtype=float)
    value_matrix[seed_codes, flow_codes] = values
    present[seed_codes, flow_codes] = 1.0
    per_seed_mean = np.divide(
        value_matrix.sum(axis=1),
        np.maximum(present.sum(axis=1), 1.0),
        out=np.zeros(n_seeds),
        where=present.sum(axis=1) > 0,
    )
    estimate = float(per_seed_mean.mean())
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_resamples), dtype=float)
    written = 0
    probabilities = np.full(n_flows, 1.0 / n_flows, dtype=float)
    while written < int(n_resamples):
        size = min(int(block_size), int(n_resamples) - written)
        # One shared source draw per replicate, reused across selected seeds.
        counts = rng.multinomial(n_flows, probabilities, size=size).astype(float)
        selected = rng.integers(0, n_seeds, size=(size, n_seeds))
        numerators = counts @ value_matrix.T  # (size, n_seeds)
        denominators = counts @ present.T
        with np.errstate(invalid="ignore", divide="ignore"):
            seed_means = np.where(denominators > 0, numerators / denominators, np.nan)
        gathered = np.take_along_axis(seed_means, selected, axis=1)
        replicate = np.nanmean(gathered, axis=1)
        valid = np.isfinite(replicate)
        accepted = replicate[valid]
        take = min(len(accepted), size)
        draws[written : written + take] = accepted[:take]
        written += take
        if take == 0:
            raise RuntimeError("Crossed bootstrap could not draw a usable replicate")
    tail = (1.0 - float(confidence_level)) / 2.0
    ci_low, ci_high = np.quantile(draws, [tail, 1.0 - tail])
    return {
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_std": float(np.std(draws, ddof=1)),
        "n_rows": int(len(clean)),
        "n_flows": int(n_flows),
        "n_seeds": int(n_seeds),
        "bootstrap_design": "vectorised_crossed_source_seed_equal_seed_bootstrap",
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    cluster_column: str,
    stratum_column: str | None,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Resample clusters (source flows); average with equal weight per stratum.

    Study 3's compared object is a *pair* of fitted models, so fitted seeds are
    no longer exchangeable replicates of one estimand.  Uncertainty therefore
    comes from resampling source flows, with each compared pair given equal
    weight (``stratum_column``).
    """

    columns = [cluster_column, value_column] + (
        [stratum_column] if stratum_column else []
    )
    clean = frame[columns].copy()
    clean[value_column] = pd.to_numeric(clean[value_column], errors="coerce")
    clean = clean.dropna(subset=[value_column])
    if clean.empty:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_std": float("nan"),
            "n_rows": 0,
            "n_clusters": 0,
            "n_strata": 0,
            "bootstrap_design": "source_flow_cluster_bootstrap_equal_stratum_weight",
        }
    if stratum_column is None:
        clean["__stratum"] = "all"
        stratum = "__stratum"
    else:
        stratum = stratum_column
    cluster_codes, cluster_values = pd.factorize(clean[cluster_column], sort=True)
    stratum_codes, stratum_values = pd.factorize(clean[stratum], sort=True)
    n_clusters = len(cluster_values)
    n_strata = len(stratum_values)
    values = clean[value_column].to_numpy(dtype=float)
    totals = np.zeros((n_strata, n_clusters), dtype=float)
    counts = np.zeros((n_strata, n_clusters), dtype=float)
    np.add.at(totals, (stratum_codes, cluster_codes), values)
    np.add.at(counts, (stratum_codes, cluster_codes), 1.0)
    per_stratum_mean = np.divide(
        totals.sum(axis=1),
        np.maximum(counts.sum(axis=1), 1.0),
        out=np.zeros(n_strata),
        where=counts.sum(axis=1) > 0,
    )
    estimate = float(per_stratum_mean.mean())
    rng = np.random.default_rng(int(seed))
    probabilities = np.full(n_clusters, 1.0 / n_clusters, dtype=float)
    draws = np.empty(int(n_resamples), dtype=float)
    written = 0
    while written < int(n_resamples):
        size = min(256, int(n_resamples) - written)
        weights = rng.multinomial(n_clusters, probabilities, size=size).astype(float)
        numerators = weights @ totals.T
        denominators = weights @ counts.T
        with np.errstate(invalid="ignore", divide="ignore"):
            stratum_means = np.where(
                denominators > 0, numerators / denominators, np.nan
            )
        replicate = np.nanmean(stratum_means, axis=1)
        valid = np.isfinite(replicate)
        accepted = replicate[valid]
        take = min(len(accepted), size)
        draws[written : written + take] = accepted[:take]
        written += take
        if take == 0:
            raise RuntimeError("Cluster bootstrap could not draw a usable replicate")
    tail = (1.0 - float(confidence_level)) / 2.0
    ci_low, ci_high = np.quantile(draws, [tail, 1.0 - tail])
    return {
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_std": float(np.std(draws, ddof=1)),
        "n_rows": int(len(clean)),
        "n_clusters": int(n_clusters),
        "n_strata": int(n_strata),
        "bootstrap_design": "source_flow_cluster_bootstrap_equal_stratum_weight",
    }


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    left_column: str,
    right_column: str,
    *,
    cluster_column: str,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Interval for ``left - right`` with the same source flows in both arms."""

    clean = frame[[cluster_column, left_column, right_column]].copy()
    for column in (left_column, right_column):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=[left_column, right_column])
    if clean.empty:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_clusters": 0,
            "bootstrap_design": "paired_source_flow_cluster_bootstrap",
        }
    cluster_codes, cluster_values = pd.factorize(clean[cluster_column], sort=True)
    n_clusters = len(cluster_values)
    difference = (
        clean[left_column].to_numpy(dtype=float)
        - clean[right_column].to_numpy(dtype=float)
    )
    totals = np.zeros(n_clusters, dtype=float)
    counts = np.zeros(n_clusters, dtype=float)
    np.add.at(totals, cluster_codes, difference)
    np.add.at(counts, cluster_codes, 1.0)
    estimate = float(totals.sum() / counts.sum())
    rng = np.random.default_rng(int(seed))
    probabilities = np.full(n_clusters, 1.0 / n_clusters, dtype=float)
    draws = np.empty(int(n_resamples), dtype=float)
    written = 0
    while written < int(n_resamples):
        size = min(512, int(n_resamples) - written)
        weights = rng.multinomial(n_clusters, probabilities, size=size).astype(float)
        numerators = weights @ totals
        denominators = weights @ counts
        with np.errstate(invalid="ignore", divide="ignore"):
            replicate = np.where(denominators > 0, numerators / denominators, np.nan)
        accepted = replicate[np.isfinite(replicate)]
        take = min(len(accepted), size)
        draws[written : written + take] = accepted[:take]
        written += take
        if take == 0:
            raise RuntimeError("Paired bootstrap could not draw a usable replicate")
    tail = (1.0 - float(confidence_level)) / 2.0
    ci_low, ci_high = np.quantile(draws, [tail, 1.0 - tail])
    return {
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_std": float(np.std(draws, ddof=1)),
        "n_clusters": int(n_clusters),
        "bootstrap_design": "paired_source_flow_cluster_bootstrap",
    }


def apply_gate(
    metric: str,
    *,
    ci_low: float,
    ci_high: float,
    gates: Mapping[str, Any],
    coverage_sufficient: bool,
) -> tuple[str, float | None]:
    """Return ``pass``/``fail``/``not_evaluable`` under the baseline gate rules."""

    lookup = {
        "spearman": ("spearman_ci_lower", "lower"),
        "jaccard_10": ("jaccard_10_ci_lower", "lower"),
        "weighted_jaccard": ("weighted_jaccard_ci_lower", "lower"),
        "normalized_mass_l1": ("normalized_mass_l1_ci_upper", "upper"),
    }
    if metric not in lookup:
        return "not_gated", None
    key, side = lookup[metric]
    if key not in gates:
        return "not_gated", None
    threshold = float(gates[key])
    if not coverage_sufficient:
        return "not_evaluable", threshold
    if side == "lower":
        if not np.isfinite(ci_low):
            return "not_evaluable", threshold
        return ("pass" if ci_low >= threshold else "fail"), threshold
    if not np.isfinite(ci_high):
        return "not_evaluable", threshold
    return ("pass" if ci_high <= threshold else "fail"), threshold


# --------------------------------------------------------------------------- #
# Stratification helpers
# --------------------------------------------------------------------------- #
def quantile_bins(
    values: pd.Series,
    *,
    bins: int,
    labels: Sequence[str],
) -> pd.Series:
    """Rank-based quantile bins that tolerate heavy ties and tiny groups."""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), "", dtype=object)
    finite = np.isfinite(numeric)
    if not finite.any():
        return pd.Series(result, index=values.index, dtype=object)
    ranks = np.full(len(numeric), np.nan)
    ranks[finite] = rankdata(numeric[finite], method="average") / finite.sum()
    edges = np.linspace(0.0, 1.0, int(bins) + 1)[1:]
    for index in np.flatnonzero(finite):
        position = int(np.searchsorted(edges, ranks[index], side="left"))
        result[index] = str(labels[min(position, len(labels) - 1)])
    return pd.Series(result, index=values.index, dtype=object)


def error_type_labels(
    frame: pd.DataFrame,
    *,
    task: str,
    benign_class: str = "BENIGN",
) -> pd.Series:
    """Operational error taxonomy for the outcome panel.

    ``target_class`` is the audited model's predicted class (the outcome
    panel's declared target), so the label is derived from the (true,
    predicted) pair rather than from a probability.
    """

    true_class = frame["true_class"].astype(str)
    predicted = frame["target_class"].astype(str)
    correct = frame["correct"].map(_as_bool) if "correct" in frame else true_class.eq(predicted)
    if str(task) == "binary":
        labels = np.where(
            correct,
            np.where(true_class.eq(benign_class), "true_negative", "true_positive"),
            np.where(
                true_class.eq(benign_class), "false_positive", "false_negative"
            ),
        )
        return pd.Series(labels, index=frame.index, dtype=object)
    labels = np.where(
        correct,
        "correct",
        np.where(
            true_class.eq(benign_class),
            "benign_predicted_as_attack",
            np.where(
                predicted.eq(benign_class),
                "attack_predicted_as_benign",
                "attack_predicted_as_other_attack",
            ),
        ),
    )
    return pd.Series(labels, index=frame.index, dtype=object)


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def write_extension_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def extension_stage_signature(
    context: ExtensionContext,
    *,
    stage: str,
    inputs: Iterable[Path],
    payload: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], str]:
    input_records = artifact_records(list(inputs))
    implementation_hash = signature({"files": context.implementation_records()})
    stage_signature = signature(
        {
            "extension_version": EXTENSION_VERSION,
            "stage": stage,
            "extension_config_hash": sha256_file(context.extension_config_path),
            "audit_config_hash": sha256_file(context.paths.config_path),
            "training_config_hash": sha256_file(context.paths.training_config_path),
            "implementation_hash": implementation_hash,
            "inputs": input_records,
            "payload": dict(payload),
        }
    )
    return stage_signature, input_records, implementation_hash


def cached_stage(
    context: ExtensionContext,
    *,
    stage: str,
    directory: Path,
    outputs: Mapping[str, Path],
    inputs: Iterable[Path],
    payload: Mapping[str, Any],
    compute,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run ``compute`` unless every declared output already matches its hash."""

    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "cache_metadata.json"
    expected = [Path(value) for value in outputs.values()]
    stage_signature, input_records, implementation_hash = extension_stage_signature(
        context, stage=stage, inputs=inputs, payload=payload
    )
    hit, _ = cache_hit(
        metadata_path, expected_signature=stage_signature, expected_outputs=expected
    )
    if hit and not force:
        return {key: pd.read_csv(path) for key, path in outputs.items()}
    frames = compute()
    missing = sorted(set(outputs) - set(frames))
    if missing:
        raise RuntimeError(f"Stage {stage!r} did not produce {missing}")
    for key, path in outputs.items():
        write_extension_table(frames[key], Path(path))
    write_cache_metadata(
        metadata_path,
        stage=stage,
        signature_value=stage_signature,
        config_hash=sha256_file(context.extension_config_path),
        implementation_hash=implementation_hash,
        inputs=input_records,
        outputs=expected,
        extra={
            "extension_version": EXTENSION_VERSION,
            "baseline_protocol": str(context.extension_config["baseline_protocol"]),
            "payload": dict(payload),
        },
    )
    return frames


def write_decision(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def baseline_probe_methods() -> tuple[str, ...]:
    return ALLOWED_PROBES


def stable_extension_seed(*parts: object) -> int:
    return stable_seed(EXTENSION_VERSION, *parts)


# --------------------------------------------------------------------------- #
# Vectorised similarity (row-wise, identical definitions to `similarity_at`)
# --------------------------------------------------------------------------- #
def _row_diagnostics_at(
    absolute: np.ndarray,
    *,
    top_k: int,
    minimum_account_l1: float,
    rank_tie_relative_tolerance: float,
) -> dict[str, np.ndarray]:
    mass = absolute.sum(axis=1)
    maximum = absolute.max(axis=1)
    tolerance = np.maximum(
        float(minimum_account_l1), maximum * float(rank_tie_relative_tolerance)
    )
    nonzero = mass >= float(minimum_account_l1)
    spread = absolute.max(axis=1) - absolute.min(axis=1)
    rank_identifiable = nonzero & (spread > tolerance)
    features = absolute.shape[1]
    effective = min(max(int(top_k), 1), features)
    if effective >= features:
        # No boundary exists when the whole vector is inside the cut; a null
        # account still reports a zero gap, matching the frozen scalar rule.
        boundary_gap = np.where(nonzero, np.inf, 0.0)
        top_k_identifiable = nonzero.copy()
    else:
        ordered = -np.sort(-absolute, axis=1)
        gap = ordered[:, effective - 1] - ordered[:, effective]
        boundary_gap = np.where(nonzero, gap, 0.0)
        top_k_identifiable = nonzero & (gap > tolerance)
    return {
        "mass": mass,
        "nonzero": nonzero,
        "rank_identifiable": rank_identifiable,
        "top_k_identifiable": top_k_identifiable,
        "top_k_boundary_gap": boundary_gap,
    }


def similarity_matrix_at(
    left: np.ndarray,
    right: np.ndarray,
    *,
    top_k: int,
    minimum_account_l1: float = 1.0e-12,
    rank_tie_relative_tolerance: float = 1.0e-8,
) -> dict[str, np.ndarray]:
    """Row-wise similarity for two aligned attribution matrices.

    Same definitions as :func:`similarity_at`, evaluated for every row at once.
    The test suite pins this against the frozen scalar implementation.
    """

    left_abs = np.abs(np.asarray(left, dtype=float))
    right_abs = np.abs(np.asarray(right, dtype=float))
    if left_abs.shape != right_abs.shape or left_abs.ndim != 2:
        raise ValueError("Similarity matrices must be two-dimensional and aligned")
    if not (np.isfinite(left_abs).all() and np.isfinite(right_abs).all()):
        raise ValueError("Attribution matrices must be finite")
    tolerances = {
        "top_k": int(top_k),
        "minimum_account_l1": float(minimum_account_l1),
        "rank_tie_relative_tolerance": float(rank_tie_relative_tolerance),
    }
    left_diag = _row_diagnostics_at(left_abs, **tolerances)
    right_diag = _row_diagnostics_at(right_abs, **tolerances)
    pair_nonzero = left_diag["nonzero"] & right_diag["nonzero"]
    pair_rank = left_diag["rank_identifiable"] & right_diag["rank_identifiable"]
    pair_top_k = left_diag["top_k_identifiable"] & right_diag["top_k_identifiable"]

    rows, features = left_abs.shape
    effective = min(max(int(top_k), 1), features)
    # A stable descending sort reproduces the frozen feature-index tie-break.
    left_top = np.argsort(-left_abs, axis=1, kind="stable")[:, :effective]
    right_top = np.argsort(-right_abs, axis=1, kind="stable")[:, :effective]
    left_mask = np.zeros((rows, features), dtype=bool)
    right_mask = np.zeros((rows, features), dtype=bool)
    row_index = np.arange(rows)[:, None]
    left_mask[row_index, left_top] = True
    right_mask[row_index, right_top] = True
    intersection = np.count_nonzero(left_mask & right_mask, axis=1)
    union = np.count_nonzero(left_mask | right_mask, axis=1)
    jaccard = np.where(
        pair_top_k & (union > 0), intersection / np.maximum(union, 1), 0.0
    )

    left_total = np.where(left_diag["mass"] > 0, left_diag["mass"], 1.0)
    right_total = np.where(right_diag["mass"] > 0, right_diag["mass"], 1.0)
    left_norm = left_abs / left_total[:, None]
    right_norm = right_abs / right_total[:, None]
    minimum_sum = np.minimum(left_norm, right_norm).sum(axis=1)
    maximum_sum = np.maximum(left_norm, right_norm).sum(axis=1)
    weighted = np.where(
        pair_nonzero & (maximum_sum >= float(minimum_account_l1)),
        minimum_sum / np.where(maximum_sum > 0, maximum_sum, 1.0),
        0.0,
    )
    mass_l1 = np.where(
        pair_nonzero, 0.5 * np.abs(left_norm - right_norm).sum(axis=1), 1.0
    )

    left_rank = rankdata(left_abs, axis=1)
    right_rank = rankdata(right_abs, axis=1)
    left_centred = left_rank - left_rank.mean(axis=1, keepdims=True)
    right_centred = right_rank - right_rank.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        (left_centred**2).sum(axis=1) * (right_centred**2).sum(axis=1)
    )
    correlation = np.divide(
        (left_centred * right_centred).sum(axis=1),
        denominator,
        out=np.zeros(rows),
        where=denominator > 0,
    )
    # Reproduce `_safe_spearman`'s short-circuits in their original order:
    # numerically identical vectors score 1.0, and a vector that is constant
    # to within numpy's default `allclose` tolerance scores 0.0 even when its
    # spread still clears the much smaller rank-tie tolerance.  Dropping the
    # second check silently turns near-flat accounts into strong agreement.
    identical = np.all(np.abs(left_abs - right_abs) <= 1e-12, axis=1)
    left_flat = np.all(
        np.abs(left_abs - left_abs[:, :1])
        <= (1e-8 + 1e-5 * np.abs(left_abs[:, :1])),
        axis=1,
    )
    right_flat = np.all(
        np.abs(right_abs - right_abs[:, :1])
        <= (1e-8 + 1e-5 * np.abs(right_abs[:, :1])),
        axis=1,
    )
    spearman = np.where(
        pair_rank,
        np.where(
            identical,
            1.0,
            np.where(left_flat | right_flat, 0.0, correlation),
        ),
        0.0,
    )

    return {
        "spearman": spearman,
        "jaccard_10": jaccard,
        "weighted_jaccard": weighted,
        "normalized_mass_l1": mass_l1,
        "left_account_mass": left_diag["mass"],
        "right_account_mass": right_diag["mass"],
        "left_account_nonzero": left_diag["nonzero"],
        "right_account_nonzero": right_diag["nonzero"],
        "account_pair_nonzero": pair_nonzero,
        "left_rank_identifiable": left_diag["rank_identifiable"],
        "right_rank_identifiable": right_diag["rank_identifiable"],
        "rank_pair_identifiable": pair_rank,
        "left_top_k_identifiable": left_diag["top_k_identifiable"],
        "right_top_k_identifiable": right_diag["top_k_identifiable"],
        "top_k_pair_identifiable": pair_top_k,
        "left_top_k_boundary_gap": left_diag["top_k_boundary_gap"],
        "right_top_k_boundary_gap": right_diag["top_k_boundary_gap"],
        "agreement_on_null_accounts": (~left_diag["nonzero"]) & (~right_diag["nonzero"]),
    }
