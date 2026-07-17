#!/usr/bin/env python3
"""Plot matched GEN reuse trials, with each null overlaid on its main run.

By default this plots validation cross entropy. Historical runs that did not
record ``val_loss`` are rejected rather than being mislabeled. Use
``--metric train-loss`` only when a plot of the historical training loss is
explicitly wanted.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRIALS = {
    "scaled": (
        (42, "A0_n42_scaled_clean", "A0_n42_null_shared_v2", None, None),
        (126, "A0_n126_scaled", "A0_n126_null_shared_v2", None, None),
        (210, "A0_n210_scaled_disjoint", "A0_n210_null_shared", None, None),
        # The old scaled N=420 model produced constant scores. Prefer the
        # corrected full-optimizer/minimum-epoch rerun once it has finished.
        (420, "A0_n420_scaled_disjoint", "A0_n420_null_shared",
         "gen_n420_scaled_20260717", "gen_n420_null_20260717"),
    ),
    "raw": (
        (42, "A0_n42_paper_rawsum", "null_n42_paper", None, None),
        (210, "A0_n210_rawsum_disjoint", "A0_n210_rawsum_null_shared", None, None),
        (420, "A0_n420_rawsum_disjoint", "A0_n420_rawsum_null_shared", None, None),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="pfn_results")
    parser.add_argument("--kind", choices=sorted(TRIALS), required=True)
    parser.add_argument("--metric", choices=("val-loss", "train-loss"),
                        default="val-loss")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def history(path, column):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or column not in rows[0]:
        raise ValueError(
            "{} does not contain {}. Historical GEN histories only recorded "
            "training loss; use --metric train-loss or rerun to obtain "
            "validation-loss curves.".format(path, column))
    return (np.asarray([int(row["epoch"]) for row in rows]),
            np.asarray([float(row[column]) for row in rows]))


def test_text(path):
    summary = path / "auc_summary.json"
    if not summary.is_file():
        return "pending"
    with summary.open() as handle:
        data = json.load(handle)
    auc = data["test_auc"]
    std = data.get("bootstrap_std")
    if std is not None:
        return "{:.3f} +/- {:.3f}".format(auc, std)
    return "{:.3f}*".format(auc)


def resolve_result(root, historical, corrected):
    """Use a corrected replacement only after it has produced a history."""
    if corrected:
        replacement = root / corrected
        if (replacement / "history.csv").is_file():
            return replacement, True
    return root / historical, False


def main():
    args = parse_args()
    column = "val_loss" if args.metric == "val-loss" else "train_loss"
    result_root = Path(args.results)
    trials = TRIALS[args.kind]
    fig, axes = plt.subplots(1, len(trials), figsize=(4.1 * len(trials), 3.7),
                             sharey=True)
    if len(trials) == 1:
        axes = [axes]

    footer = []
    for axis, (n_files, main_label, null_label,
               corrected_main, corrected_null) in zip(axes, trials):
        main, corrected = resolve_result(result_root, main_label, corrected_main)
        null, corrected_null_used = resolve_result(
            result_root, null_label, corrected_null)
        for path, color, label, style, zorder in (
                (main, "#0072B2", "reuse comparison", "-", 2),
                (null, "#D55E00", "matched null", "--", 3)):
            epochs, values = history(path / "history.csv", column)
            axis.plot(epochs, values, color=color, lw=2, ls=style,
                      label=label, zorder=zorder)
        axis.axhline(np.log(2), color="0.5", lw=1, ls=":")
        subtitle = "N={}".format(n_files)
        if corrected_main and not corrected:
            subtitle += "\n(old; corrected rerun pending)"
        axis.set_title(subtitle)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25, lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        tag = " corrected" if corrected and corrected_null_used else ""
        footer.append("N={}:{} main {}, null {}".format(
            n_files, tag, test_text(main), test_text(null)))

    axes[0].set_ylabel("validation cross entropy" if args.metric == "val-loss"
                       else "training cross entropy")
    axes[0].legend(frameon=False, fontsize=9)
    title = "GEN PFN: {} sum".format("scaled" if args.kind == "scaled" else "raw")
    fig.suptitle(title, y=1.02)
    fig.text(0.5, -0.02, "Final test AUC — " + " | ".join(footer),
             ha="center", va="top", fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print("plot -> {}".format(output))


if __name__ == "__main__":
    main()
