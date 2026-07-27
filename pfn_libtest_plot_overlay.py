#!/usr/bin/env python3
"""Single-run train+val loss overlay, with test AUC reported in the legend.

Unlike pfn_libtest_compare.py (multi-run, loss OR AUC per panel), this is
for showing one run's training and validation loss together on one axis --
built for the raw-sum / scaled-sum + warmup writeup plots. It accepts both
the shared GEN training-engine history schema and the Keras RECO schema.

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
            train_loss.append(float(
                row["train_loss"] if "train_loss" in row else row["loss"]))
            val_loss.append(float(row["val_loss"]))
    order = np.argsort(epochs)
    return (np.asarray(epochs)[order], np.asarray(train_loss)[order],
            np.asarray(val_loss)[order])


def load_test_auc(rundir):
    for name in ("point_summary.json", "summary.json", "auc_summary.json"):
        path = os.path.join(rundir, name)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            payload = json.load(f)
        if name == "point_summary.json":
            auc = payload.get("auc")
        elif name == "auc_summary.json":
            auc = payload.get("test_auc", payload.get("auc"))
        else:
            auc = payload.get("test_auc", payload.get("auc"))
            if auc is None:
                auc = payload.get("results", {}).get("test", {}).get("auc")
        if auc is not None:
            return float(auc)
    return None


def load_best_epoch(rundir, epochs, val_loss):
    # state["best_epoch"] is authoritative regardless of --select-metric
    # (auc or loss) -- it's whichever epoch's weights actually got saved to
    # best.weights.h5 and reloaded for the point/bootstrap test evaluation.
    path = os.path.join(rundir, "state.json")
    if os.path.isfile(path):
        with open(path) as f:
            return int(json.load(f)["best_epoch"])
    return int(epochs[np.argmin(val_loss)])


def make_plot(rundir, title, output):
    epochs, train_loss, val_loss = load_history(rundir)
    test_auc = load_test_auc(rundir)
    best_epoch = load_best_epoch(rundir, epochs, val_loss)

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
    if test_auc is not None:
        handles.append(Line2D([], [], linestyle="None"))
        labels.append(f"test AUC = {test_auc:.3f}")
    ax.legend(handles, labels, frameon=False, fontsize=9)

    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"chart -> {output}")


def main():
    args = parse_args()
    make_plot(args.rundir, args.title, args.out)


if __name__ == "__main__":
    main()
