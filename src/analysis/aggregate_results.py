"""Aggregate pipeline outputs into paper-ready tables."""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.compare_xai import compare_explanation_matrices
from src.utils.io import ensure_dir, load_yaml, processed_dir, resolve_path, results_root
from src.xai.methods import configured_xai_methods


def _concat_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _allowed(config_path: str | Path) -> tuple[set[str], set[str]]:
    cfg = load_yaml(config_path)
    return set(cfg.get("datasets", [])), set(cfg.get("models", []))


def _is_allowed(path: Path, datasets: set[str], models: set[str]) -> bool:
    if len(path.parts) < 3:
        return True
    dataset, model = path.parts[-3], path.parts[-2]
    return dataset in datasets and model in models


def _feature_counts(datasets: set[str]) -> dict[str, int]:
    counts = {}
    for dataset in datasets:
        paths = [
            processed_dir(dataset) / "feature_names.json",
            *sorted(resolve_path("data/processed/seeds").glob(f"seed_*/{dataset}/feature_names.json")),
            resolve_path(f"data/processed/{dataset}/feature_names.json"),
        ]
        for path in paths:
            if path.exists():
                counts[dataset] = len(json.loads(path.read_text(encoding="utf-8")))
                break
    return counts


def _has_valid_topk(path: Path, feature_counts: dict[str, int]) -> bool:
    match = re.search(r"(?:top-k|random-k)_(\d+)", path.name)
    if not match or len(path.parts) < 3:
        return True
    dataset = path.parts[-3]
    n_features = feature_counts.get(dataset)
    return n_features is None or int(match.group(1)) < n_features


def _read_with_context(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "results" in path.parts:
        dataset, model = path.parts[-3], path.parts[-2]
        df.insert(0, "model", model)
        df.insert(0, "dataset", dataset)
    if "xai" not in df.columns:
        for marker in ["_runtime_", "_top_features_", "_global_importance_"]:
            if marker in path.name:
                df.insert(2, "xai", path.name.split(marker, 1)[0])
                break
    return df


def maybe_to_latex(df: pd.DataFrame, path: Path, export_latex: bool, float_format: str = "%.4f") -> None:
    """Optionally export a dataframe to LaTeX."""
    if export_latex:
        df.to_latex(path, index=False, float_format=float_format)


def generate_tables(config_path: str | Path = "configs/experiment.yaml") -> None:
    """Generate CSV tables from existing result artifacts."""
    root = results_root()
    tables_dir = ensure_dir(root / "tables")
    stability_root = ensure_dir(root / "stability")
    cfg = load_yaml(config_path)
    export_latex = bool(cfg.get("outputs", {}).get("latex", False))
    datasets, models = _allowed(config_path)
    feature_counts = _feature_counts(datasets)

    agreement_frames = []
    pairwise_agreement_frames = []
    configured_xais = configured_xai_methods(cfg)
    for model_dir in (root / "explanations").glob("*/*"):
        if model_dir.parts[-2] not in datasets or model_dir.parts[-1] not in models:
            continue
        dataset = model_dir.parts[-2]
        ks = [
            int(k)
            for k in cfg.get("stability", {}).get("topk_values", [3, 5, 10])
            if int(k) < feature_counts.get(dataset, 10**9)
        ]
        shap_path = model_dir / "shap_clean.npy"
        lime_path = model_dir / "lime_clean.npy"
        if shap_path.exists() and lime_path.exists():
            agreement_frames.append(
                compare_explanation_matrices(
                    np.load(shap_path),
                    np.load(lime_path),
                    ks or [3],
                    model_dir.parts[-2],
                    model_dir.parts[-1],
                )
            )
        available = [
            (xai, model_dir / f"{xai}_clean.npy")
            for xai in configured_xais
            if (model_dir / f"{xai}_clean.npy").exists()
        ]
        for (method_a, path_a), (method_b, path_b) in combinations(available, 2):
            pairwise_agreement_frames.append(
                compare_explanation_matrices(
                    np.load(path_a),
                    np.load(path_b),
                    ks or [3],
                    model_dir.parts[-2],
                    model_dir.parts[-1],
                    method_a=method_a,
                    method_b=method_b,
                )
            )
    if agreement_frames:
        pd.concat(agreement_frames, ignore_index=True).to_csv(
            stability_root / "shap_lime_agreement_clean.csv", index=False
        )
    if pairwise_agreement_frames:
        pd.concat(pairwise_agreement_frames, ignore_index=True).to_csv(
            stability_root / "xai_pair_agreement_clean.csv", index=False
        )

    for name in ["dataset_summary", "class_distribution"]:
        path = tables_dir / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "dataset" in df.columns:
                df = df[df["dataset"].isin(datasets)]
                df.to_csv(path, index=False)
            maybe_to_latex(df, tables_dir / f"{name}.tex", export_latex)

    perf_paths = [
        p for p in (root / "performance").glob("*/*/performance_metrics.csv")
        if _is_allowed(p, datasets, models)
    ]
    performance = _concat_csvs(perf_paths)
    if not performance.empty:
        performance.to_csv(tables_dir / "model_performance.csv", index=False)
        maybe_to_latex(performance, tables_dir / "model_performance.tex", export_latex)

    runtime_paths = [
        p for p in (root / "explanations").glob("*/*/*_runtime_*.csv")
        if _is_allowed(p, datasets, models) and _has_valid_topk(p, feature_counts)
    ]
    runtimes = pd.concat([_read_with_context(p) for p in runtime_paths], ignore_index=True) if runtime_paths else pd.DataFrame()
    if not runtimes.empty:
        runtimes.to_csv(tables_dir / "xai_runtime.csv", index=False)
        maybe_to_latex(runtimes, tables_dir / "xai_runtime.tex", export_latex)

    top_paths = [
        p for p in (root / "explanations").glob("*/*/*_top_features_clean.csv")
        if _is_allowed(p, datasets, models)
    ]
    top_features = pd.concat([_read_with_context(p) for p in top_paths], ignore_index=True) if top_paths else pd.DataFrame()
    if not top_features.empty:
        top_features.to_csv(tables_dir / "top_features.csv", index=False)
        maybe_to_latex(top_features, tables_dir / "top_features.tex", export_latex, "%.6f")

    stability_paths = [
        p for p in (root / "stability").glob("*/*/*_summary.csv")
        if _is_allowed(p, datasets, models) and _has_valid_topk(p, feature_counts)
    ]
    stability = _concat_csvs(stability_paths)
    if not stability.empty:
        stability.to_csv(stability_root / "stability_summary_all.csv", index=False)
        stability.to_csv(tables_dir / "stability_under_perturbations.csv", index=False)
        maybe_to_latex(stability, tables_dir / "stability_under_perturbations.tex", export_latex)

    agreement = stability_root / "shap_lime_agreement_clean.csv"
    if agreement.exists():
        df = pd.read_csv(agreement)
        df.to_csv(tables_dir / "shap_lime_agreement.csv", index=False)
        maybe_to_latex(df, tables_dir / "shap_lime_agreement.tex", export_latex)

    pairwise_agreement = stability_root / "xai_pair_agreement_clean.csv"
    if pairwise_agreement.exists():
        df = pd.read_csv(pairwise_agreement)
        df.to_csv(tables_dir / "xai_pair_agreement.csv", index=False)
        maybe_to_latex(df, tables_dir / "xai_pair_agreement.tex", export_latex)

    per_sample_paths = [
        p for p in (root / "stability").glob("*/*/*_per_sample.csv")
        if _is_allowed(p, datasets, models) and _has_valid_topk(p, feature_counts)
    ]
    drift_frames = []
    for path in per_sample_paths:
        df = pd.read_csv(path)
        if "prediction_changed" not in df.columns:
            continue
        metric_mask = df["spearman"].fillna(-1) < 0.5
        if "jaccard_10" in df.columns:
            metric_mask |= df["jaccard_10"].fillna(-1) < 0.5
        drift_frames.append(df[(df["prediction_changed"] == False) & metric_mask])
    if drift_frames:
        df = pd.concat(drift_frames, ignore_index=True)
        df.to_csv(stability_root / "same_prediction_different_explanation.csv", index=False)
        df.to_csv(tables_dir / "same_prediction_different_explanation.csv", index=False)
        maybe_to_latex(df, tables_dir / "same_prediction_different_explanation.tex", export_latex)

    if not performance.empty and not stability.empty:
        spearman = stability[stability["metric"].eq("spearman") & stability["group"].eq("all")]
        stability_by_xai = (
            spearman.groupby(["dataset", "model", "xai"], as_index=False)["mean"]
            .mean()
            .rename(columns={"mean": "mean_spearman_stability"})
        )
        if not stability_by_xai.empty:
            perf_stability = performance.merge(stability_by_xai, on=["dataset", "model"], how="inner")
            perf_stability.to_csv(tables_dir / "performance_vs_stability.csv", index=False)
            maybe_to_latex(perf_stability, tables_dir / "performance_vs_stability.tex", export_latex)

            combined_stability = (
                stability_by_xai.groupby(["dataset", "model"], as_index=False)["mean_spearman_stability"]
                .mean()
                .rename(columns={"mean_spearman_stability": "combined_mean_spearman_stability"})
            )
            perf_combined = performance.merge(combined_stability, on=["dataset", "model"], how="inner")
            perf_combined.to_csv(tables_dir / "performance_vs_combined_stability.csv", index=False)
            maybe_to_latex(perf_combined, tables_dir / "performance_vs_combined_stability.tex", export_latex)

        f1 = performance.sort_values("f1", ascending=False).head(1)
        stable = spearman.sort_values("mean", ascending=False).head(1)
        combined = pd.concat(
            [
                f1.assign(selection="best_f1_model"),
                stable.assign(selection="most_stable_model"),
            ],
            ignore_index=True,
            sort=False,
        )
        combined.to_csv(tables_dir / "best_f1_vs_most_stable.csv", index=False)
        maybe_to_latex(combined, tables_dir / "best_f1_vs_most_stable.tex", export_latex)
