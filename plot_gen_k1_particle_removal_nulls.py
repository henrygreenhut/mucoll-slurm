#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = [
    "All particles",
    "Muons removed",
    "Photons removed",
    "Muons and photons\nremoved",
]
MAIN_AUCS = [0.79825, 0.8432222222, 0.6039555556, 0.5701444444]
NULL_AUCS = [0.4938944444, 0.4918111111, 0.5070222222, 0.5064]


def main():
    output = Path(__file__).resolve().parent / "plots"
    output.mkdir(exist_ok=True)

    plt.rcParams["font.family"] = "serif"
    positions = np.arange(len(LABELS))
    width = 0.34

    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    main_bars = axis.bar(
        positions - width / 2,
        MAIN_AUCS,
        width,
        color="#0072B2",
        label="Unrotated BIB vs rotated BIB",
    )
    null_bars = axis.bar(
        positions + width / 2,
        NULL_AUCS,
        width,
        color="#999999",
        label="Matched null",
    )

    axis.axhline(0.5, color="0.35", linestyle="--", linewidth=1.2)
    axis.set_ylim(0, 0.9)
    axis.set_ylabel("Test AUC")
    axis.set_xticks(positions, LABELS)
    axis.set_title("GEN-Level PFN at N=420: Unrotated vs Rotated BIB")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    for bars in (main_bars, null_bars):
        for bar in bars:
            auc = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                auc + 0.015,
                f"{auc:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
            )

    figure.tight_layout()
    stem = output / "gen_n420_k1_particle_removal_nulls"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
