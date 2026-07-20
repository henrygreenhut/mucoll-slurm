#!/usr/bin/env python3
"""Plot the completed N=420 GEN k=1 versus k=10 reuse study."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INPUT = Path("plots/gen_n420_k1_vs_k10")
OUTPUT = Path("plots")


def history(name):
    with (INPUT / (name + "_history.csv")).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in ("epoch", "train_loss", "val_macro_auc")
    }


def test_auc(name):
    with (INPUT / (name + "_summary.json")).open() as handle:
        return float(json.load(handle)["test_macro_auc"])


def finish(axis):
    axis.set_xlabel("epoch")
    axis.grid(alpha=0.25, lw=0.5)
    axis.spines[["top", "right"]].set_visible(False)


def save(fig, stem):
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        path = OUTPUT / (stem + "." + suffix)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("plot -> {}".format(path))
    plt.close(fig)


def plot_validation_auc(main, null, main_auc, null_auc):
    fig, axis = plt.subplots(figsize=(5.3, 4.0))
    axis.plot(main["epoch"], main["val_macro_auc"], color="#0072B2",
              lw=2, label="k=1 vs k=10")
    axis.plot(null["epoch"], null["val_macro_auc"], color="#D55E00",
              lw=2, label="null")
    axis.axhline(0.5, color="0.5", lw=1, ls=":")
    axis.scatter(main["epoch"][-1], main_auc, color="#0072B2", s=25,
                 zorder=3, label="test AUC {:.3f}".format(main_auc))
    axis.scatter(null["epoch"][-1], null_auc, color="#D55E00", s=25,
                 zorder=3, label="null test AUC {:.3f}".format(null_auc))
    axis.set_ylabel("validation AUC")
    axis.set_title("N=420 GEN PFN: no reuse vs 10x reuse")
    axis.legend(frameon=False, fontsize=8)
    finish(axis)
    save(fig, "gen_n420_k1_vs_k10_validation_auc")


def plot_training_loss(main, null):
    fig, axis = plt.subplots(figsize=(5.3, 4.0))
    axis.plot(main["epoch"], main["train_loss"], color="#0072B2",
              lw=2, label="k=1 vs k=10")
    axis.plot(null["epoch"], null["train_loss"], color="#D55E00",
              lw=2, label="null")
    axis.axhline(np.log(2), color="0.5", lw=1, ls=":", label="ln 2")
    axis.set_ylabel("training loss")
    axis.set_title("N=420 GEN PFN: no reuse vs 10x reuse")
    axis.legend(frameon=False, fontsize=8)
    finish(axis)
    save(fig, "gen_n420_k1_vs_k10_training_loss")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    main_history = history("main")
    null_history = history("null")
    plot_validation_auc(
        main_history, null_history, test_auc("main"), test_auc("null"))
    plot_training_loss(main_history, null_history)


if __name__ == "__main__":
    main()
