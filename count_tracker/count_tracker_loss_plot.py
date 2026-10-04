#!/usr/bin/env python3
"""Plot training and validation loss from a COUNT tracker PFN fit."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def plot(results, output_prefix):
    results = Path(results)
    output_prefix = Path(output_prefix)
    outputs = [output_prefix.with_suffix(suffix) for suffix in (".png", ".pdf")]
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"Refusing to replace an existing plot: {outputs}")

    summary = json.loads((results / "summary.json").read_text())
    with (results / "history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    if not history or any(not {"epoch", "loss", "val_loss"} <= row.keys()
                          for row in history):
        raise ValueError("history.csv must contain epoch, loss, and val_loss")

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    if epochs != list(range(1, len(history) + 1)):
        raise ValueError("history.csv epochs are not consecutive, starting at one")
    best_epoch = int(summary["selection"]["best_epoch"])
    if best_epoch < 1 or best_epoch > len(epochs):
        raise ValueError("selected epoch lies outside the saved training history")

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(epochs, train_loss, color="#2369a8", linewidth=1.8,
            label="Training loss")
    ax.plot(epochs, val_loss, color="#c95b2c", linewidth=2.2,
            label="Validation loss")
    ax.axhline(math.log(2), color="0.55", linestyle=":", linewidth=1.4,
               label="Balanced chance loss (ln 2)")
    ax.axvline(best_epoch, color="0.3", linestyle="--", linewidth=1.1,
               label=f"Selected epoch {best_epoch}")
    ax.scatter([best_epoch], [val_loss[best_epoch - 1]], color="#c95b2c",
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.set(xlim=(1, len(epochs)), xlabel="Epoch", ylabel="Cross-entropy loss")
    f_dropout = float(summary["architecture"]["F_dropout"])
    ax.set_title(f"Reservoir preliminary: single-track PFN (F dropout = {f_dropout:g})")
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs[0], dpi=200)
    fig.savefig(outputs[1])
    plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    for path in plot(args.results, args.output_prefix):
        print(path)


if __name__ == "__main__":
    main()
