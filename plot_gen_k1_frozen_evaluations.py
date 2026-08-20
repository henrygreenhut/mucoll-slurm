#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROWS = [
    ("Unmodified test inputs", 0.8242777778, "baseline"),
    (r"Shuffle $\phi$ pair together", 0.5365777778, "group"),
    (r"Rotate each construction by a random $\phi$", 0.5141333333, "azimuth"),
    (r"Replace $\phi$ with uniform values", 0.5370888889, "azimuth"),
    (r"Add $\pi/2$ to $\phi$", 0.2744666667, "azimuth"),
    (r"Shuffle $\cos(\phi)$", 0.6686222222, "feature"),
    (r"Shuffle $\sin(\phi)$", 0.7204444444, "feature"),
    (r"Shuffle $\log(p_T)$", 0.6375777778, "feature"),
    (r"Shuffle $\theta$", 0.7990444444, "feature"),
    (r"Shuffle $\log(E)$", 0.5890444444, "feature"),
    ("Shuffle time", 0.7238555556, "feature"),
    (r"Shuffle particle $z$ position", 0.8414444444, "feature"),
    (r"Shuffle particle $r$ position ($r=\sqrt{x^2+y^2}$)", 0.5395888889, "feature"),
    ("Shuffle photon indicator", 0.6048333333, "feature"),
    ("Shuffle neutron indicator", 0.7800000000, "feature"),
    ("Shuffle electron indicator", 0.8268111111, "feature"),
    ("Shuffle muon indicator", 0.8217555556, "feature"),
    ("Shuffle other-particle indicator", 0.8161000000, "feature"),
    ("Shuffle particle indicators together", 0.5676111111, "group"),
]

COLORS = {
    "baseline": "#0072B2",
    "azimuth": "#D55E00",
    "feature": "#56B4E9",
    "group": "#CC79A7",
}


def main():
    output = Path(__file__).resolve().parent / "plots"
    output.mkdir(exist_ok=True)

    labels = [label for label, _, _ in ROWS]
    aucs = [auc for _, auc, _ in ROWS]
    colors = [COLORS[kind] for _, _, kind in ROWS]
    positions = list(range(len(ROWS)))

    plt.rcParams["font.family"] = "serif"
    figure, axis = plt.subplots(figsize=(11.5, 11.5))
    bars = axis.barh(positions, aucs, height=0.68, color=colors)
    axis.axvline(0.5, color="0.45", linestyle="--", linewidth=1)
    axis.set_xlim(0.2, 0.9)
    axis.set_xlabel("Test AUC")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_title("GEN-Level N=420 Frozen-PFN Evaluations")
    axis.grid(axis="x", alpha=0.25, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    for bar, auc in zip(bars, aucs):
        axis.text(
            auc + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{auc:.3f}",
            ha="left",
            va="center",
            fontsize=9,
            weight="bold",
        )

    figure.tight_layout()
    stem = output / "gen_n420_k1_frozen_evaluations"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
