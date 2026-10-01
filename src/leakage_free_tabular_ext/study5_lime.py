"""Study 5a/5b: LIME convergence curves and kernel/regularisation ablations.

The manuscript reports that LIME fails the repeatability gate at 32,000
perturbation samples over 50 runs, and that raising the budget by more than an
order of magnitude improved repeatability without removing the failure.  It
does not show the shape of that curve, nor whether the failure is specific to
LIME's two other free measurement choices - the exponential kernel width that
weights the perturbed neighbourhood, and the ridge penalty of the local
surrogate.

5a traces repeatability against the sampling budget up to the manuscript's own
budget.  5b holds the budget fixed and sweeps kernel width and ridge penalty
around LIME's defaults.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.leakage_free_tabular.epistemic_audit import (
    _as_bool,
    _load_frozen_model,
    _stratified_background_indices,
    lime_prediction_matrix,
    load_condition_data,
    require_valid_training_condition,
    stable_seed,
)
from src.leakage_free_tabular_ext.common import (
    ExtensionContext,
    similarity_matrix_at,
    write_decision,
    write_extension_table,
)

STUDY = "study5_lime_and_thresholds"
OUTPUT = "study5_lime"


def _panel(context: ExtensionContext, part: Mapping[str, Any]) -> pd.DataFrame:
    """A small subset of the baseline stochastic subset, chosen by declared seed."""

    central = pd.read_csv(context.paths.central_panel)
    stochastic = central[central["in_stochastic_subset"].map(_as_bool)].copy()
    cap = int(part["panel_cap_per_task_model_seed"])
    rng = np.random.default_rng(int(part["panel_selection_seed"]))
    frames: list[pd.DataFrame] = []
    for task_id in part["tasks"]:
        for seed in part["seeds"]:
            group = stochastic[
                stochastic["task_id"].astype(str).eq(task_id)
                & pd.to_numeric(stochastic["seed"], errors="raise").astype(int).eq(int(seed))
            ]
            if group.empty:
                continue
            ordered = group.sort_values("sample_order", kind="mergesort").reset_index(
                drop=True
            )
            take = min(cap, len(ordered))
            frames.append(
                ordered.iloc[np.sort(rng.choice(len(ordered), size=take, replace=False))]
            )
    if not frames:
        raise RuntimeError("Study 5 LIME panel selection produced no rows")
    return pd.concat(frames, ignore_index=True)


def _explain_lime_configured(
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
    kernel_width: float | None,
    ridge_alpha: float,
    feature_selection: str,
    discretize_continuous: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """The frozen LIME call with kernel width and ridge penalty exposed.

    Identical to ``epistemic_audit.explain_lime`` when ``kernel_width`` is None
    and ``ridge_alpha`` is 1.0 (LIME's own defaults); the surrogate's R^2 is
    returned alongside the coefficients so the sweep can show whether a
    configuration that is more repeatable also fits the neighbourhood better.
    """

    from lime.lime_tabular import LimeTabularExplainer

    background_indices = _stratified_background_indices(
        y_train, int(background_max_rows), seed=stable_seed(seed, "background")
    )
    background = np.asarray(X_train[background_indices], dtype=np.float32)
    explainer = LimeTabularExplainer(
        background,
        feature_names=list(feature_names),
        discretize_continuous=bool(discretize_continuous),
        mode="classification",
        random_state=int(seed),
        kernel_width=kernel_width,
        feature_selection=str(feature_selection),
    )
    attributes = np.zeros((len(X_panel), len(feature_names)), dtype=np.float32)
    scores = np.zeros(len(X_panel), dtype=float)

    def predict_fn(values: np.ndarray) -> np.ndarray:
        return lime_prediction_matrix(model, values)

    for row_index, row in enumerate(np.asarray(X_panel, dtype=np.float32)):
        target = int(targets[row_index])
        explanation = explainer.explain_instance(
            row,
            predict_fn,
            labels=[target],
            num_features=len(feature_names),
            num_samples=int(num_samples),
            model_regressor=Ridge(
                alpha=float(ridge_alpha), fit_intercept=True, random_state=int(seed)
            ),
        )
        for feature_index, weight in explanation.as_map().get(target, []):
            attributes[row_index, int(feature_index)] = float(weight)
        scores[row_index] = float(getattr(explanation, "score", np.nan))
    return attributes, scores


def _run_estimands(
    runs: np.ndarray,
    reference_mean: np.ndarray | None,
    *,
    top_k: int,
) -> dict[str, np.ndarray]:
    """Mean pairwise agreement across runs, plus stability of the run mean."""

    n_runs, n_rows, _ = runs.shape
    spearman = np.zeros(n_rows)
    jaccard = np.zeros(n_rows)
    identifiable = np.zeros(n_rows)
    pairs = 0
    for left, right in itertools.combinations(range(n_runs), 2):
        metrics = similarity_matrix_at(runs[left], runs[right], top_k=top_k)
        spearman += metrics["spearman"]
        jaccard += metrics["jaccard_10"]
        identifiable += metrics["rank_pair_identifiable"].astype(float)
        pairs += 1
    spearman /= max(pairs, 1)
    jaccard /= max(pairs, 1)
    identifiable /= max(pairs, 1)
    run_mean = runs.mean(axis=0)
    absolute = np.abs(runs)
    top_one = absolute.max(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        coefficient_of_variation = np.where(
            top_one.mean(axis=0) > 0,
            top_one.std(axis=0, ddof=1 if n_runs > 1 else 0) / top_one.mean(axis=0),
            np.nan,
        )
    result = {
        "mean_pairwise_spearman": spearman,
        "mean_pairwise_jaccard_10": jaccard,
        "rank_pair_identifiable_fraction": identifiable,
        "top_1_coefficient_of_variation": coefficient_of_variation,
    }
    if reference_mean is not None:
        against_reference = similarity_matrix_at(
            run_mean, reference_mean, top_k=top_k
        )
        result["spearman_to_reference_budget"] = against_reference["spearman"]
        result["jaccard_10_to_reference_budget"] = against_reference["jaccard_10"]
    else:
        result["spearman_to_reference_budget"] = np.full(n_rows, np.nan)
        result["jaccard_10_to_reference_budget"] = np.full(n_rows, np.nan)
    return result


def _condition_inputs(
    context: ExtensionContext, task_id: str, model: str, seed: int, panel: pd.DataFrame
):
    condition = require_valid_training_condition(
        context.paths,
        context.training_config,
        task_id=task_id,
        model=model,
        seed=int(seed),
    )
    data = load_condition_data(context.paths, task_id)
    units = panel[
        panel["task_id"].astype(str).eq(task_id)
        & pd.to_numeric(panel["seed"], errors="raise").astype(int).eq(int(seed))
    ].reset_index(drop=True)
    orders = pd.to_numeric(units["sample_order"], errors="raise").to_numpy(int)
    X = np.asarray(data.X_test[orders], dtype=np.float32)
    targets = np.asarray(data.y_test[orders], dtype=int)
    return condition, data, units, X, targets


def run_convergence(
    context: ExtensionContext, *, force: bool = False, logger=None
) -> dict[str, pd.DataFrame]:
    """Repeatability against the sampling budget, up to the manuscript's budget."""

    study = context.study(STUDY)
    part = study["parts"]["lime_convergence"]
    if not bool(part.get("enabled", False)):
        raise RuntimeError("lime_convergence is disabled in the config")
    method_cfg = context.audit_config["measurement_probes"]["lime"]
    top_k = int(context.shared["representation"]["top_k"])
    panel = _panel(context, part)
    budgets = [int(value) for value in part["num_samples"]]
    reference_budget = int(part["reference_budget"])
    runs_per_budget = int(part["independent_runs"])
    rows: list[dict[str, Any]] = []
    for task_id in part["tasks"]:
        for model in part["models"]:
            for seed in part["seeds"]:
                condition, data, units, X, targets = _condition_inputs(
                    context, task_id, model, int(seed), panel
                )
                if units.empty:
                    continue
                frozen_model = _load_frozen_model(condition["model"])
                cache: dict[int, np.ndarray] = {}
                started = time.perf_counter()
                for budget in sorted(set(budgets + [reference_budget])):
                    runs = np.stack(
                        [
                            _explain_lime_configured(
                                frozen_model,
                                data.X_train,
                                data.y_train,
                                X,
                                targets,
                                data.feature_names,
                                num_samples=budget,
                                background_max_rows=int(
                                    method_cfg["training_background_max_rows"]
                                ),
                                seed=stable_seed(
                                    20260908, "convergence", task_id, model, seed,
                                    budget, run,
                                ),
                                kernel_width=None,
                                ridge_alpha=1.0,
                                feature_selection=str(
                                    study["parts"][
                                        "lime_kernel_regularization_ablation"
                                    ]["feature_selection"]
                                ),
                                discretize_continuous=False,
                            )[0]
                            for run in range(runs_per_budget)
                        ],
                        axis=0,
                    )
                    cache[budget] = runs
                reference_mean = cache[reference_budget].mean(axis=0)
                for budget in budgets:
                    estimands = _run_estimands(
                        cache[budget], reference_mean, top_k=top_k
                    )
                    for row_index in range(len(units)):
                        rows.append(
                            {
                                "task_id": task_id,
                                "model": model,
                                "seed": int(seed),
                                "flow_id": str(units.loc[row_index, "flow_id"]),
                                "true_class": str(units.loc[row_index, "true_class"]),
                                "num_samples": int(budget),
                                "independent_runs": runs_per_budget,
                                "reference_budget": reference_budget,
                                "is_manuscript_budget": bool(
                                    budget == int(part["baseline_budget_marker"])
                                ),
                                **{
                                    key: float(value[row_index])
                                    for key, value in estimands.items()
                                },
                            }
                        )
                if logger is not None:
                    logger.info(
                        "lime convergence %s/%s/seed_%d: %d flows x %d budgets, %.1fs",
                        task_id, model, seed, len(units), len(budgets),
                        time.perf_counter() - started,
                    )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["num_samples", "task_id", "model"], as_index=False)
        .agg(
            flows=("flow_id", "nunique"),
            mean_pairwise_spearman=("mean_pairwise_spearman", "mean"),
            mean_pairwise_jaccard_10=("mean_pairwise_jaccard_10", "mean"),
            spearman_to_reference_budget=("spearman_to_reference_budget", "mean"),
            top_1_coefficient_of_variation=("top_1_coefficient_of_variation", "mean"),
        )
        .sort_values(["task_id", "model", "num_samples"])
    )
    overall = (
        detail.groupby("num_samples", as_index=False)
        .agg(
            flows=("flow_id", "nunique"),
            mean_pairwise_spearman=("mean_pairwise_spearman", "mean"),
            mean_pairwise_jaccard_10=("mean_pairwise_jaccard_10", "mean"),
            spearman_to_reference_budget=("spearman_to_reference_budget", "mean"),
            top_1_coefficient_of_variation=("top_1_coefficient_of_variation", "mean"),
        )
        .sort_values("num_samples")
    )
    tables_dir = context.tables_dir(OUTPUT)
    outputs = {
        "convergence_per_flow": tables_dir / "lime_convergence_per_flow.csv",
        "convergence_summary": tables_dir / "lime_convergence_summary.csv",
        "convergence_overall": tables_dir / "lime_convergence_overall.csv",
    }
    frames = {
        "convergence_per_flow": detail,
        "convergence_summary": summary,
        "convergence_overall": overall,
    }
    for name, path in outputs.items():
        write_extension_table(frames[name], path)
    write_decision(
        context.results_dir(OUTPUT) / "lime_convergence_decision.json",
        {
            "study": "study5_lime_convergence",
            "status": "complete",
            "panel_flows": int(detail["flow_id"].nunique()),
            "budgets": budgets,
            "independent_runs": runs_per_budget,
            "reference_budget": reference_budget,
            "role": "exploratory_estimator_convergence",
            "outputs": {name: str(path) for name, path in outputs.items()},
        },
    )
    if logger is not None:
        for _, row in overall.iterrows():
            logger.info(
                "budget %6d: pairwise spearman %.4f, jaccard@10 %.4f, to-reference %.4f",
                int(row["num_samples"]),
                float(row["mean_pairwise_spearman"]),
                float(row["mean_pairwise_jaccard_10"]),
                float(row["spearman_to_reference_budget"]),
            )
    return frames


def run_kernel_ablation(
    context: ExtensionContext, *, force: bool = False, logger=None
) -> dict[str, pd.DataFrame]:
    """Kernel width and ridge penalty at a fixed budget."""

    study = context.study(STUDY)
    part = study["parts"]["lime_kernel_regularization_ablation"]
    convergence_part = study["parts"]["lime_convergence"]
    if not bool(part.get("enabled", False)):
        raise RuntimeError("lime_kernel_regularization_ablation is disabled")
    method_cfg = context.audit_config["measurement_probes"]["lime"]
    top_k = int(context.shared["representation"]["top_k"])
    panel = _panel(context, convergence_part)
    budget = int(part["num_samples"])
    runs_per_config = int(part["independent_runs"])
    multipliers = [float(value) for value in part["kernel_width_multipliers"]]
    alphas = [float(value) for value in part["ridge_alpha"]]
    default_alpha = float(part["ridge_alpha_default"])
    rows: list[dict[str, Any]] = []
    for task_id in convergence_part["tasks"]:
        for model in convergence_part["models"]:
            for seed in convergence_part["seeds"]:
                condition, data, units, X, targets = _condition_inputs(
                    context, task_id, model, int(seed), panel
                )
                if units.empty:
                    continue
                frozen_model = _load_frozen_model(condition["model"])
                # LIME's own default: sqrt(n_features) * 0.75.
                default_width = float(np.sqrt(X.shape[1]) * 0.75)
                started = time.perf_counter()
                cache: dict[tuple[float, float], np.ndarray] = {}
                scores: dict[tuple[float, float], np.ndarray] = {}
                configurations = [
                    (multiplier, default_alpha) for multiplier in multipliers
                ] + [(1.0, alpha) for alpha in alphas if alpha != default_alpha]
                for multiplier, alpha in configurations:
                    width = default_width * multiplier
                    stacked = []
                    stacked_scores = []
                    for run in range(runs_per_config):
                        attributes, run_scores = _explain_lime_configured(
                            frozen_model,
                            data.X_train,
                            data.y_train,
                            X,
                            targets,
                            data.feature_names,
                            num_samples=budget,
                            background_max_rows=int(
                                method_cfg["training_background_max_rows"]
                            ),
                            seed=stable_seed(
                                20260908, "ablation", task_id, model, seed,
                                multiplier, alpha, run,
                            ),
                            kernel_width=width,
                            ridge_alpha=alpha,
                            feature_selection=str(part["feature_selection"]),
                            discretize_continuous=bool(part["discretize_continuous"]),
                        )
                        stacked.append(attributes)
                        stacked_scores.append(run_scores)
                    cache[(multiplier, alpha)] = np.stack(stacked, axis=0)
                    scores[(multiplier, alpha)] = np.mean(
                        np.stack(stacked_scores, axis=0), axis=0
                    )
                default_mean = cache[(1.0, default_alpha)].mean(axis=0)
                for (multiplier, alpha), runs in cache.items():
                    estimands = _run_estimands(runs, default_mean, top_k=top_k)
                    for row_index in range(len(units)):
                        rows.append(
                            {
                                "task_id": task_id,
                                "model": model,
                                "seed": int(seed),
                                "flow_id": str(units.loc[row_index, "flow_id"]),
                                "true_class": str(units.loc[row_index, "true_class"]),
                                "num_samples": budget,
                                "independent_runs": runs_per_config,
                                "kernel_width_multiplier": multiplier,
                                "kernel_width": float(default_width * multiplier),
                                "ridge_alpha": alpha,
                                "is_default_configuration": bool(
                                    multiplier == 1.0 and alpha == default_alpha
                                ),
                                "local_surrogate_r2": float(
                                    scores[(multiplier, alpha)][row_index]
                                ),
                                "mean_pairwise_spearman": float(
                                    estimands["mean_pairwise_spearman"][row_index]
                                ),
                                "mean_pairwise_jaccard_10": float(
                                    estimands["mean_pairwise_jaccard_10"][row_index]
                                ),
                                "top_1_coefficient_of_variation": float(
                                    estimands["top_1_coefficient_of_variation"][row_index]
                                ),
                                "spearman_to_default_configuration": float(
                                    estimands["spearman_to_reference_budget"][row_index]
                                ),
                                "jaccard_10_to_default_configuration": float(
                                    estimands["jaccard_10_to_reference_budget"][row_index]
                                ),
                            }
                        )
                if logger is not None:
                    logger.info(
                        "lime ablation %s/%s/seed_%d: %d flows x %d configs, %.1fs",
                        task_id, model, seed, len(units), len(configurations),
                        time.perf_counter() - started,
                    )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(
            ["kernel_width_multiplier", "ridge_alpha", "is_default_configuration"],
            as_index=False,
        )
        .agg(
            flows=("flow_id", "nunique"),
            mean_pairwise_spearman=("mean_pairwise_spearman", "mean"),
            mean_pairwise_jaccard_10=("mean_pairwise_jaccard_10", "mean"),
            local_surrogate_r2=("local_surrogate_r2", "mean"),
            spearman_to_default_configuration=(
                "spearman_to_default_configuration",
                "mean",
            ),
            top_1_coefficient_of_variation=("top_1_coefficient_of_variation", "mean"),
        )
        .sort_values(["kernel_width_multiplier", "ridge_alpha"])
    )
    tables_dir = context.tables_dir(OUTPUT)
    outputs = {
        "ablation_per_flow": tables_dir / "lime_kernel_ablation_per_flow.csv",
        "ablation_summary": tables_dir / "lime_kernel_ablation_summary.csv",
    }
    frames = {"ablation_per_flow": detail, "ablation_summary": summary}
    for name, path in outputs.items():
        write_extension_table(frames[name], path)
    write_decision(
        context.results_dir(OUTPUT) / "lime_kernel_ablation_decision.json",
        {
            "study": "study5_lime_kernel_regularization_ablation",
            "status": "complete",
            "panel_flows": int(detail["flow_id"].nunique()) if len(detail) else 0,
            "num_samples": budget,
            "independent_runs": runs_per_config,
            "kernel_width_multipliers": multipliers,
            "ridge_alpha": alphas,
            "role": "exploratory_measurement_sensitivity",
            "outputs": {name: str(path) for name, path in outputs.items()},
        },
    )
    if logger is not None and len(summary):
        for _, row in summary.iterrows():
            logger.info(
                "kernel x%.2f alpha %.3g: pairwise spearman %.4f, R2 %.4f",
                float(row["kernel_width_multiplier"]),
                float(row["ridge_alpha"]),
                float(row["mean_pairwise_spearman"]),
                float(row["local_surrogate_r2"]),
            )
    return frames
