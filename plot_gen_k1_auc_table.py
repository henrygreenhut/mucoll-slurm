#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROWS = [
    ("Unrotated vs rotated once\nseed 1", 0.8242777778, "main"),
    ("Unrotated vs rotated once\n5,000-event evaluation", 0.78425398, "evaluation"),
    ("Unrotated vs rotated once\nmodel seed 2", 0.8354111111, "main"),
    ("Unrotated vs rotated once\nOSCAR repeat", 0.79825, "main"),
    ("Unrotated vs rotated once\nremove muons above 5 GeV", 0.8242888889, "ablation"),
    ("Unrotated vs rotated once\nremove all muons", 0.8432222222, "ablation"),
    ("Unrotated vs rotated once\n" + r"$\phi$ features removed", 0.4854666667, "ablation"),
    ("Rotated once vs rotated once\nmatched null", 0.4938944444, "null"),
    ("No-muon matched null", 0.4918111111, "null"),
]

COLORS = {
    "main": "#0072B2",
    "evaluation": "#56B4E9",
    "ablation": "#D55E00",
    "null": "#999999",
}


def main():
    output = Path(__file__).resolve().parent / "plots"
    output.mkdir(exist_ok=True)

    labels = [label for label, _, _ in ROWS]
    aucs = [auc for _, auc, _ in ROWS]
    colors = [COLORS[kind] for _, _, kind in ROWS]

    plt.rcParams["font.family"] = "serif"
    positions = list(range(len(ROWS)))
    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    bars = ax.barh(positions, aucs, height=0.7, color=colors)
    ax.axvline(0.5, color="0.45", linestyle="--", linewidth=1)
    ax.set_xlim(0.4, 0.9)
    ax.set_xlabel("Test AUC")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_title("GEN-Level PFN: Unrotated BIB vs BIB Rotated Once at N=420")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)

    for bar, auc in zip(bars, aucs):
        ax.text(auc + 0.006, bar.get_y() + bar.get_height() / 2,
                f"{auc:.3f}", ha="left", va="center", fontsize=10,
                weight="bold")

    fig.tight_layout()

    stem = output / "gen_n420_k1_auc_bars"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
