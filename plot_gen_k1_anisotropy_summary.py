#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = [
    "All particles\nand features",
    "Muons\nremoved",
    "Momentum $\\phi$ features\nremoved",
]

AUCS = [0.79825, 0.8432222222, 0.4854666667]
COLORS = ["#0072B2", "#D55E00", "#D55E00"]


def main():
    output = Path(__file__).resolve().parent / "plots"
    output.mkdir(exist_ok=True)

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    bars = ax.bar(LABELS, AUCS, width=0.62, color=COLORS)
    ax.axhline(0.5, color="0.4", linestyle="--", linewidth=1.2)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Test AUC")
    ax.set_title("GEN-Level PFN at N=420: Unrotated vs Rotated BIB")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, auc in zip(bars, AUCS):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            auc + 0.018,
            f"{auc:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    fig.tight_layout()

    stem = output / "gen_n420_k1_anisotropy_summary"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
