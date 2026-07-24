#!/usr/bin/env python3
"""Single-run train+val loss overlay, with test AUC reported in the legend.

Unlike pfn_libtest_compare.py (multi-run, loss OR AUC per panel), this is
for showing one run's training and validation loss together on one axis --
built for the raw-sum / scaled-sum + warmup writeup plots.

    python pfn_libtest_plot_overlay.py \
        pfn_results/oscar_n420_halfphi_raw_seed1_w1_c0 \
        --title "raw sum + warmup, no gradient clipping" \
        --out plots/raw_warmup_overlay.pdf
"""

import argparse
import csv
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# Okabe-Ito, colorblind-safe -- same palette as pfn_libtest_compare.py.
TRAIN_COLOR = "#0072B2"
VAL_COLOR = "#D55E00"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("rundir", help="results dir holding history.csv / point_summary.json")
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_history(rundir):
    epochs, train_loss, val_loss = [], [], []
    with open(os.path.join(rundir, "history.csv")) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
    order = np.argsort(epochs)
    return (np.asarray(epochs)[order], np.asarray(train_loss)[order],
            np.asarray(val_loss)[order])


def load_test_auc(rundir):
    path = os.path.join(rundir, "point_summary.json")
    with open(path) as f:
        return json.load(f)["auc"]


def load_best_epoch(rundir):
    # state["best_epoch"] is authoritative regardless of --select-metric
    # (auc or loss) -- it's whichever epoch's weights actually got saved to
    # best.weights.h5 and reloaded for the point/bootstrap test evaluation.
    path = os.path.join(rundir, "state.json")
    with open(path) as f:
        return json.load(f)["best_epoch"]


def main():
    args = parse_args()
    epochs, train_loss, val_loss = load_history(args.rundir)
    test_auc = load_test_auc(args.rundir)
    best_epoch = load_best_epoch(args.rundir)

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(epochs, train_loss, "-", lw=2, color=TRAIN_COLOR, label="train loss")
    ax.plot(epochs, val_loss, "-", lw=2, color=VAL_COLOR, label="val loss")
    ax.axhline(np.log(2), ls="--", lw=1, color="#888888")
    # The epoch whose weights the reported test AUC actually came from --
    # not necessarily the last epoch plotted (training continues past it
    # until patience runs out).
    ax.axvline(best_epoch, ls=":", lw=1.5, color="#666666",
               label=f"best epoch ({best_epoch})")

    # Same log-vs-linear scale question as pfn_libtest_compare.py: raw-sum's
    # loss spans orders of magnitude (log needed), scaled-sum's sits in a
    # narrow band near ln 2 (log degenerates the tick locator -- see that
    # script's history for why). The ln2-label offset is scale-aware for
    # the same reason: a multiplicative offset only makes sense in log-space.
    combined = np.concatenate([train_loss, val_loss, [np.log(2)]])
    spans_decade = combined.max() / max(combined.min(), 1e-12) >= 10
    if spans_decade:
        ax.set_yscale("log")
        ln2_label_y = np.log(2) * 1.15
    else:
        ln2_label_y = np.log(2) + 0.03 * (combined.max() - combined.min())
    ax.text(0.02, ln2_label_y, "ln 2", transform=ax.get_yaxis_transform(),
            fontsize=9, color="#666666", va="bottom")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Test AUC as its own legend entry (no error bar -- point estimate
    # only): an invisible proxy handle carries the label into the legend.
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="None"))
    labels.append(f"test AUC = {test_auc:.3f}")
    ax.legend(handles, labels, frameon=False, fontsize=9)

    ax.set_title(args.title, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"chart -> {args.out}")


if __name__ == "__main__":
    main()
