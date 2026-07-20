#!/usr/bin/env python3
"""Plot the recorded N=420 EnergyFlow PFN training histories."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    1: ("U_vs_R_seed1", "null_seed1"),
    2: ("U_vs_R_seed2", "null_seed2"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="plots/reco_n420_energyflow_histories")
    parser.add_argument("--output-dir", default="plots")
    return parser.parse_args()


def read_history(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in ("epoch", "loss", "val_loss")
    }


def read_auc(path):
    with path.open() as handle:
        return json.load(handle)["results"]["combined"]["auc"]


def finish_axis(axis):
    axis.axhline(np.log(2), color="0.5", lw=1, ls=":", label="ln 2")
    axis.set_xlabel("epoch")
    axis.grid(alpha=0.25, lw=0.5)
    axis.spines[["top", "right"]].set_visible(False)


def save(fig, output_dir, stem):
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        path = output_dir / "{}.{}".format(stem, suffix)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("plot -> {}".format(path))
    plt.close(fig)


def validation_plot(input_dir, output_dir):
    for seed in sorted(RUNS):
        fig, axis = plt.subplots(figsize=(5.2, 4.0))
        main_name, null_name = RUNS[seed]
        main = read_history(input_dir / (main_name + ".csv"))
        null = read_history(input_dir / (null_name + ".csv"))
        main_auc = read_auc(input_dir / (main_name + ".json"))
        null_auc = read_auc(input_dir / (null_name + ".json"))
        axis.plot(main["epoch"], main["val_loss"], color="#0072B2", lw=2,
                  label="unique vs 42x reuse (AUC {:.3f})".format(main_auc))
        axis.plot(null["epoch"], null["val_loss"], color="#D55E00", lw=2,
                  ls="--", label="null (AUC {:.3f})".format(null_auc))
        finish_axis(axis)
        axis.set_title("N=420 RECO EnergyFlow PFN validation loss\nseed {}".format(seed))
        axis.set_ylabel("validation loss")
        axis.legend(frameon=False, fontsize=8)
        save(fig, output_dir,
             "reco_n420_energyflow_validation_loss_seed{}".format(seed))


def diagnostic_plot(input_dir, output_dir):
    for seed in sorted(RUNS):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), sharex=True,
                                 sharey=True)
        for axis, (name, title) in zip(axes, zip(
                RUNS[seed], ("unique vs 42x reuse", "null"))):
            history = read_history(input_dir / (name + ".csv"))
            auc = read_auc(input_dir / (name + ".json"))
            axis.plot(history["epoch"], history["loss"], color="#0072B2",
                      lw=2, label="training")
            axis.plot(history["epoch"], history["val_loss"], color="#D55E00",
                      lw=2, label="validation")
            finish_axis(axis)
            axis.set_title("{}; seed {}; test AUC {:.3f}".format(
                title, seed, auc), fontsize=10)
            axis.legend(frameon=False, fontsize=8)
        axes[0].set_ylabel("loss")
        fig.suptitle("N=420 RECO EnergyFlow PFN training history — seed {}".format(seed))
        save(fig, output_dir,
             "reco_n420_energyflow_training_history_seed{}".format(seed))


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_plot(input_dir, output_dir)
    diagnostic_plot(input_dir, output_dir)


if __name__ == "__main__":
    main()
