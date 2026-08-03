#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = arguments()

    with (args.result_dir / "history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    with (args.result_dir / "state.json").open() as handle:
        state = json.load(handle)

    epochs = [int(row["epoch"]) for row in history]
    auc = [float(row["val_auc"]) for row in history]
    maximum_index = max(range(len(auc)), key=auc.__getitem__)
    selected_epoch = int(state["best_epoch"])
    selected_index = epochs.index(selected_epoch)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, auc, linewidth=2, label="validation AUC")
    axis.scatter(
        epochs[maximum_index], auc[maximum_index], s=70, color="#D55E00",
        zorder=3,
        label="maximum {:.3f} (epoch {})".format(
            auc[maximum_index], epochs[maximum_index]),
    )
    axis.scatter(
        selected_epoch, auc[selected_index], s=65, facecolors="white",
        edgecolors="black", linewidths=1.5, zorder=3,
        label="loss-selected checkpoint (epoch {})".format(selected_epoch),
    )
    axis.axhline(0.5, color="0.45", linewidth=1, linestyle="--")
    axis.set_xlabel("epoch")
    axis.set_ylabel("validation AUC")
    axis.set_title(args.title)
    axis.set_xlim(min(epochs) - 1, max(epochs) + 1)
    axis.set_ylim(min(0.5, min(auc)) - 0.01, max(0.5, max(auc)) + 0.01)
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("epochs: {}".format(len(epochs)))
    print("maximum validation AUC: {:.6f} at epoch {}".format(
        auc[maximum_index], epochs[maximum_index]))
    print("plot -> {}".format(args.output.with_suffix(".pdf")))
    print("plot -> {}".format(args.output.with_suffix(".png")))


if __name__ == "__main__":
    main()
