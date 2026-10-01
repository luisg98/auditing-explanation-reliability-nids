"""Feature importance plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_top_feature_importance(importance_csv: Path, output_dir: Path) -> None:
    """Plot top-10 global feature importance."""
    if not importance_csv.exists():
        return
    import matplotlib.pyplot as plt

    df = pd.read_csv(importance_csv).head(10)
    if df.empty:
        return
    stem = importance_csv.stem
    plt.figure(figsize=(9, 5))
    plt.barh(df["feature"][::-1], df["importance"][::-1])
    plt.xlabel("Mean absolute importance")
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}.png", dpi=200)
    plt.savefig(output_dir / f"{stem}.pdf")
    plt.close()


def plot_rank_agreement(shap_csv: Path, lime_csv: Path, output_dir: Path, dataset: str, model: str) -> None:
    """Scatter SHAP vs LIME feature ranks."""
    if not shap_csv.exists() or not lime_csv.exists():
        return
    import matplotlib.pyplot as plt

    shap = pd.read_csv(shap_csv).assign(shap_rank=lambda d: d["importance"].rank(ascending=False))
    lime = pd.read_csv(lime_csv).assign(lime_rank=lambda d: d["importance"].rank(ascending=False))
    merged = shap.merge(lime, on="feature", suffixes=("_shap", "_lime"))
    if merged.empty:
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(merged["shap_rank"], merged["lime_rank"], alpha=0.7)
    plt.xlabel("SHAP rank")
    plt.ylabel("LIME rank")
    plt.tight_layout()
    name = f"shap_lime_rank_agreement_{dataset}_{model}"
    plt.savefig(output_dir / f"{name}.png", dpi=200)
    plt.savefig(output_dir / f"{name}.pdf")
    plt.close()
