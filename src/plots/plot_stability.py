"""Stability plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_stability_lines(stability_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot stability metrics across perturbation ids."""
    if stability_df.empty or "metric" not in stability_df.columns:
        return
    import matplotlib.pyplot as plt

    for metric in ["spearman", "pearson", "kendall_tau", "emd"]:
        df = stability_df[(stability_df["metric"] == metric) & (stability_df["group"] == "all")]
        if df.empty:
            continue
        labels = df["dataset"].astype(str) + "/" + df["model"].astype(str) + "/" + df["xai"].astype(str) + "/" + df["perturbation"].astype(str)
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(df)), df["mean"], marker="o")
        plt.xticks(range(len(df)), labels, rotation=45, ha="right")
        plt.ylabel(metric)
        plt.tight_layout()
        name = f"{metric}_vs_perturbation"
        plt.savefig(output_dir / f"{name}.png", dpi=200)
        plt.savefig(output_dir / f"{name}.pdf")
        plt.close()


def plot_stability_heatmap(stability_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot a simple stability heatmap for Spearman means."""
    df = stability_df[(stability_df["metric"] == "spearman") & (stability_df["group"] == "all")]
    if df.empty:
        return
    import matplotlib.pyplot as plt

    df = df.copy()
    df["row"] = df["dataset"].astype(str) + "/" + df["model"].astype(str) + "/" + df["xai"].astype(str)
    pivot = df.pivot_table(index="row", columns="perturbation", values="mean", aggfunc="mean")
    plt.figure(figsize=(max(8, pivot.shape[1] * 1.4), max(4, pivot.shape[0] * 0.4)))
    plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Mean Spearman")
    plt.tight_layout()
    plt.savefig(output_dir / "stability_heatmap.png", dpi=200)
    plt.savefig(output_dir / "stability_heatmap.pdf")
    plt.close()
