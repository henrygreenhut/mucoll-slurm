#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = [
    "Unmodified\ntest inputs",
    "Shuffle photon\n$\\phi$",
    "Shuffle non-photon\n$\\phi$",
]

AUCS = [0.8242777778, 0.5131555556, 0.8311222222]
COLORS = ["#0072B2", "#D55E00", "#56B4E9"]


def main():
    output = Path(__file__).resolve().parent / "plots"
    output.mkdir(exist_ok=True)

    plt.rcParams["font.family"] = "serif"
    figure, axis = plt.subplots(figsize=(8.2, 5.8))

    bars = axis.bar(LABELS, AUCS, width=0.62, color=COLORS)
    axis.axhline(0.5, color="0.4", linestyle="--", linewidth=1.2)
    axis.set_ylim(0, 0.9)
    axis.set_ylabel("Test AUC")
    axis.set_title("GEN-Level N=420 Frozen-PFN Evaluation")
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
            fontsize=11,
            weight="bold",
        )

    figure.tight_layout()

    stem = output / "gen_n420_k1_selective_phi_audit"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
