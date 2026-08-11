#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "plots"

LABELS = ("CaloConer enabled", "CaloConer disabled")
AUC = (0.7027625, 0.967853625)
COLORS = ("#0072B2", "#D55E00")


def main():
    OUTPUT.mkdir(exist_ok=True)

    with (OUTPUT / "reco_n420_caloconer_auc.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("caloconer", "test_samples_per_class", "test_auc"))
        writer.writerow(("enabled", 800, AUC[0]))
        writer.writerow(("disabled", 2000, AUC[1]))

    plt.rcParams["font.family"] = "serif"
    figure, axis = plt.subplots(figsize=(6.2, 4.6))
    bars = axis.bar((0, 1), AUC, width=0.58, color=COLORS)

    axis.axhline(0.5, color="0.45", linewidth=1, linestyle="--")
    axis.set_xticks((0, 1), LABELS)
    axis.set_ylabel("Test AUC")
    axis.set_title("RECO-Level PFN at N=420")
    axis.set_ylim(0.45, 1.02)
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.bar_label(bars, labels=["{:.3f}".format(value) for value in AUC],
                   padding=4, fontsize=11)

    figure.tight_layout()
    for suffix in ("pdf", "png"):
        path = OUTPUT / "reco_n420_caloconer_auc.{}".format(suffix)
        figure.savefig(path, dpi=220, bbox_inches="tight")
        print("plot -> {}".format(path))
    plt.close(figure)


if __name__ == "__main__":
    main()
