#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = ["Particle Flow\nNetwork", "Second-harmonic\n$C_2$"]
AUCS = [0.79825, 0.9797]
COLORS = ["#0072B2", "#D55E00"]


def main():
    output = Path(__file__).resolve().parent / "plots"
    output.mkdir(exist_ok=True)

    plt.rcParams["font.family"] = "serif"
    figure, axis = plt.subplots(figsize=(7.4, 5.8))
    bars = axis.bar(LABELS, AUCS, width=0.58, color=COLORS)

    axis.axhline(0.5, color="0.4", linestyle="--", linewidth=1.2)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Test AUC")
    axis.set_title("GEN-Level BIB at N=420: Unrotated vs Rotated")
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    for bar, auc in zip(bars, AUCS):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            auc + 0.018,
            f"{auc:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )

    figure.tight_layout()
    stem = output / "gen_n420_k1_pfn_vs_c2"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
