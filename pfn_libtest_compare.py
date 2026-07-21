#!/usr/bin/env python3
"""Compare training histories of pfn_libtest_train.py runs.

Loss vs epoch (log y, ln2 reference) and validation AUC vs epoch (0.5
reference). Any number of runs; built for the latent-scale A/B:

    python pfn_libtest_compare.py \
        "scaled sum=pfn_results/A0_n42_paper" \
        "raw sum=pfn_results/A0_n42_paper_rawsum" \
        --out latent_scale_ab.pdf

Default: two separate PDFs, <stem>_loss.pdf and <stem>_auc.pdf.
Pass --combined for the original single two-panel figure.
Needs numpy + matplotlib (mucoll-inspect env or laptop).
"""

import argparse
import csv
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Okabe-Ito, colorblind-safe, assigned in fixed order
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+",
                        help="name=path/to/results_dir (dir holding history.csv)")
    parser.add_argument("--out", default="pfn_libtest_compare.pdf")
    parser.add_argument("--title", default="")
    parser.add_argument("--combined", action="store_true",
                        help="single two-panel figure instead of separate PDFs")
    parser.add_argument("--max-epoch", type=int, default=0,
                        help="truncate curves at this epoch (0 = all)")
    parser.add_argument("--loss-metric", choices=("train", "val"), default="train",
                        help="'train' plots raw Keras training loss "
                             "(unbounded -- can blow up on badly-miscalibrated "
                             "raw-sum outputs); 'val' plots the trainer's own "
                             "clipped-probability validation cross entropy "
                             "(bounded at -log(1e-7)=~16.1, computed on the "
                             "fixed validation set) -- rejects any run whose "
                             "history.csv predates val_loss being recorded")
    return parser.parse_args()


def load_history(dirpath, max_epoch=0):
    rows = []
    with open(os.path.join(dirpath, "history.csv")) as f:
        for row in csv.DictReader(f):
            val_loss = float(row["val_loss"]) if "val_loss" in row and row["val_loss"] != "" else None
            rows.append((int(row["epoch"]), float(row["train_loss"]),
                         float(row["val_auc"]), val_loss))
    rows.sort()
    if max_epoch:
        rows = [r for r in rows if r[0] <= max_epoch]
    epochs = np.asarray([r[0] for r in rows])
    val_loss_col = [r[3] for r in rows]
    val_loss = (np.asarray(val_loss_col) if all(v is not None for v in val_loss_col)
                else None)
    return (epochs, np.asarray([r[1] for r in rows]),
            np.asarray([r[2] for r in rows]), val_loss)


def load_test_result(dirpath):
    path = os.path.join(dirpath, "auc_summary.json")
    if os.path.isfile(path):
        with open(path) as f:
            d = json.load(f)
        return {
            "auc": d.get("test_auc"),
            "std": d.get("bootstrap_std"),
            # Backwards compatibility: results written before test_mode was
            # added used mutually disjoint blocked units.
            "mode": d.get("test_mode", "disjoint"),
            "near_constant": d.get("near_constant_test_scores", False),
        }
    return {"auc": None, "std": None, "mode": None,
            "near_constant": False}


def style_axis(ax):
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    # Epoch counts are integers; matplotlib's default locator can pick
    # half-integer ticks on a short run (e.g. 0, 0.5, 1, 1.5...), which
    # looks like fractional-epoch data that doesn't exist.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def draw_loss(ax, runs, metric="train"):
    for i, (name, path, epochs, train_loss, _, val_loss) in enumerate(runs):
        loss = train_loss if metric == "train" else val_loss
        ax.plot(epochs, loss, "-", lw=2, color=COLORS[i % len(COLORS)], label=name)
    ax.axhline(np.log(2), ls="--", lw=1, color="#888888")
    ax.text(0.02, np.log(2) * 1.15, "ln 2",
            transform=ax.get_yaxis_transform(),
            fontsize=9, color="#666666", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss" if metric == "train" else "validation loss")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax)


def draw_auc(ax, runs):
    for i, (name, path, epochs, _, val_auc, _) in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        ax.plot(epochs, val_auc, "-", lw=2, color=color, label=name)
        test_auc = load_test_result(path)["auc"]
        if test_auc is not None:
            ax.plot(epochs[-1], test_auc, "o", ms=7, color=color,
                    markeredgecolor="white", zorder=5)
            ax.annotate(f"test {test_auc:.3f}",
                        (epochs[-1], test_auc), textcoords="offset points",
                        xytext=(6, -4), fontsize=8, color="#444444")
    ax.axhline(0.5, ls="--", lw=1, color="#888888")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation AUC")
    lo = min(0.38, min(r[4].min() for r in runs) - 0.03)
    hi = max(1.02, max(r[4].max() for r in runs) + 0.03)
    ax.set_ylim(lo, hi)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style_axis(ax)


def main():
    args = parse_args()
    runs = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"run spec must be name=dir, got: {spec}")
        name, path = spec.split("=", 1)
        runs.append((name, path) + load_history(path, args.max_epoch))

    if args.loss_metric == "val":
        missing = [name for name, path, *_, val_loss in runs if val_loss is None]
        if missing:
            raise SystemExit(
                "--loss-metric val requires val_loss in every run's "
                "history.csv; missing for: " + ", ".join(missing))

    plt.rcParams["font.family"] = "serif"
    stem, ext = os.path.splitext(args.out)
    ext = ext or ".pdf"

    if args.combined:
        fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(9, 3.6),
                                              tight_layout=True)
        draw_loss(ax_loss, runs, args.loss_metric)
        draw_auc(ax_auc, runs)
        if args.title:
            fig.suptitle(args.title)
        fig.savefig(args.out)
        print(f"chart -> {args.out}")
    else:
        default_titles = {"loss": ("PFN training loss" if args.loss_metric == "train"
                                   else "PFN validation loss"),
                          "auc": "PFN validation AUC"}
        for tag, draw in [("loss", lambda ax, r: draw_loss(ax, r, args.loss_metric)),
                          ("auc", draw_auc)]:
            fig, ax = plt.subplots(figsize=(4.8, 3.6), tight_layout=True)
            draw(ax, runs)
            ax.set_title(args.title or default_titles[tag], fontsize=11)
            out = f"{stem}_{tag}{ext}"
            fig.savefig(out)
            plt.close(fig)
            print(f"chart -> {out}")

    for name, path, epochs, loss, val_auc, val_loss in runs:
        result = load_test_result(path)
        test_auc, test_std = result["auc"], result["std"]
        if test_auc is None:
            test = "test pending"
        elif result["mode"] in ("overlapping", "shared-blocked"):
            marker = " *" + (" †" if result["near_constant"] else "")
            test = (f"test AUC {test_auc:.4f}{marker} "
                    "(source-correlated; no error)")
        else:
            marker = " †" if result["near_constant"] else ""
            test = f"test AUC {test_auc:.4f} +- {test_std:.4f}{marker}"
        print(f"  {name:20s} epochs {len(epochs):3d} | first loss {loss[0]:9.3f}"
              f" | last loss {loss[-1]:7.4f} | best val AUC {val_auc.max():.4f}"
              f" | {test}")


if __name__ == "__main__":
    main()
