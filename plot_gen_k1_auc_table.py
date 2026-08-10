#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROWS = [
    ("Unrotated BIB vs BIB Rotated Once", 0.8242777778),
    ("BIB Rotated Once vs BIB Rotated Once", 0.4938944444),
    ("Unrotated BIB vs BIB Rotated Once (phi removed)", 0.4854666667),
]


def main():
    output = Path(__file__).resolve().parent / "plots"
    output.mkdir(exist_ok=True)

    labels = [
        "Unrotated BIB vs\nBIB rotated once",
        "BIB rotated once vs\nBIB rotated once",
        "Unrotated BIB vs BIB rotated once\n(φ removed)",
    ]
    aucs = [auc for _, auc in ROWS]

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bars = ax.bar(labels, aucs, width=0.62, color="#0072B2")
    ax.axhline(0.5, color="0.45", linestyle="--", linewidth=1)
    ax.text(2.43, 0.515, "0.5 AUC", color="0.35", ha="right", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test AUC")
    ax.set_title("GEN-Level K=1 PFN Tests at N=420")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, auc - 0.035,
                f"{auc:.3f}", ha="center", va="top", fontsize=11,
                color="white", weight="bold")

    fig.tight_layout()

    stem = output / "gen_n420_k1_auc_bars"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
