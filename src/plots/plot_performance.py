"""Performance plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_f1_comparison(performance_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot F1-score comparison across models."""
    if performance_df.empty or "f1" not in performance_df.columns:
        return
    import matplotlib.pyplot as plt

    labels = performance_df["dataset"].astype(str) + "/" + performance_df["model"].astype(str)
    plt.figure(figsize=(9, 5))
    plt.bar(labels, performance_df["f1"])
    plt.ylabel("F1-score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "f1_score_comparison.png", dpi=200)
    plt.savefig(output_dir / "f1_score_comparison.pdf")
    plt.close()


def plot_learning_curve(history_csv: Path, output_dir: Path, dataset: str, model: str) -> None:
    """Plot train/validation loss curves."""
    if not history_csv.exists():
        return
    import matplotlib.pyplot as plt

    hist = pd.read_csv(history_csv)
    if "loss" not in hist.columns:
        return
    plt.figure(figsize=(7, 5))
    plt.plot(hist["epoch"], hist["loss"], label="Train loss")
    if "val_loss" in hist.columns:
        plt.plot(hist["epoch"], hist["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    name = f"learning_curve_{dataset}_{model}"
    plt.savefig(output_dir / f"{name}.png", dpi=200)
    plt.savefig(output_dir / f"{name}.pdf")
    plt.close()
