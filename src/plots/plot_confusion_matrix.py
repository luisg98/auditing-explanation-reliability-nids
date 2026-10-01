"""Confusion matrix plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_confusion_matrix_csv(cm_csv: Path, output_dir: Path, dataset: str, model: str) -> None:
    """Plot a saved confusion matrix CSV."""
    if not cm_csv.exists():
        return
    import matplotlib.pyplot as plt

    cm = pd.read_csv(cm_csv, index_col=0)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm.values, cmap="Blues")
    plt.xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
    plt.yticks(range(len(cm.index)), cm.index)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm.values[i, j]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    name = f"confusion_matrix_{dataset}_{model}"
    plt.savefig(output_dir / f"{name}.png", dpi=200)
    plt.savefig(output_dir / f"{name}.pdf")
    plt.close()
