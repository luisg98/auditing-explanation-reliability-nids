"""Performance versus stability plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_performance_vs_stability(performance_df: pd.DataFrame, stability_df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter F1 against mean Spearman stability."""
    if performance_df.empty or stability_df.empty:
        return
    import matplotlib.pyplot as plt

    stab = stability_df[(stability_df["metric"] == "spearman") & (stability_df["group"] == "all")]
    stab = stab.groupby(["dataset", "model"], as_index=False)["mean"].mean().rename(columns={"mean": "mean_spearman_stability"})
    merged = performance_df.merge(stab, on=["dataset", "model"], how="inner")
    if merged.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(merged["f1"], merged["mean_spearman_stability"])
    for _, row in merged.iterrows():
        plt.annotate(f"{row['dataset']}/{row['model']}", (row["f1"], row["mean_spearman_stability"]))
    plt.xlabel("F1-score")
    plt.ylabel("Mean Spearman stability")
    plt.tight_layout()
    plt.savefig(output_dir / "performance_vs_stability.png", dpi=200)
    plt.savefig(output_dir / "performance_vs_stability.pdf")
    plt.close()
