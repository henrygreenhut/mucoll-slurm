#!/usr/bin/env python3
"""Plot held-out ROC and score distributions for a preliminary track PFN."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def plot(results, output):
    results, output = Path(results), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    summary = json.loads((results / "summary.json").read_text())
    if summary["unit"] != "track":
        raise ValueError("this diagnostic expects a single-track classifier")
    with np.load(results / "test_predictions.npz") as data:
        scores, labels, weights = (data[name] for name in ("scores", "labels", "weights"))
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
    auc = summary["results"]["test"]["auc"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].plot(fpr, tpr, lw=2, label=f"PFN, AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="0.55", linestyle="--", label="chance")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="SIM false-positive rate",
                ylabel="COUNT true-positive rate", title="Held-out track ROC")
    axes[0].legend(frameon=False)
    lo, hi = float(scores.min()), float(scores.max())
    bins = np.linspace(lo - 1e-6, hi + 1e-6, 24)
    for label, name, color in ((0, "SIM", "#2369a8"), (1, "COUNT", "#c95b2c")):
        mask = labels == label
        axes[1].hist(scores[mask], bins=bins, weights=weights[mask], density=True,
                     histtype="step", linewidth=2, label=f"{name} ({mask.sum()} tracks)",
                     color=color)
    axes[1].set(xlabel="PFN COUNT score", ylabel="Event-weighted density",
                title="Held-out score distribution")
    axes[1].legend(frameon=False)
    f_dropout = float(summary["architecture"]["F_dropout"])
    fig.suptitle(
        f"Reservoir preliminary: F dropout = {f_dropout:g}, "
        "6/2/2 paired-event split (2 test pairs)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.results, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
