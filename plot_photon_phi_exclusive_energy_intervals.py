#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / "phi_anisotropy_out"
UPPER_INPUT = DIRECTORY / "photon_phi_upper_cut_counts.json"
LOWER_INPUT = DIRECTORY / "photon_phi_lower_cut_counts.json"
OUTPUT = DIRECTORY / "photon_phi_exclusive_energy_intervals"


def threshold_row(data, threshold):
    index = np.flatnonzero(np.isclose(data["thresholds"], threshold))
    if len(index) != 1:
        raise ValueError(f"expected one histogram for threshold {threshold}")
    return np.asarray(data["unrotated"][index[0]], dtype=float)


def main():
    with open(UPPER_INPUT) as source:
        upper = json.load(source)
    with open(LOWER_INPUT) as source:
        lower = json.load(source)

    if upper["bins"] != lower["bins"]:
        raise ValueError("upper- and lower-cut histograms use different phi bins")

    intervals = [
        ("E_lt_0p5_MeV", r"$E<0.5$ MeV", threshold_row(upper, 0.0005)),
        ("E_0p5_to_2_MeV", r"$0.5\leq E<2$ MeV", threshold_row(upper, 0.002) - threshold_row(upper, 0.0005)),
        ("E_2_to_5_MeV", r"$2\leq E<5$ MeV", threshold_row(upper, 0.005) - threshold_row(upper, 0.002)),
        ("E_ge_5_MeV", r"$E\geq5$ MeV", threshold_row(lower, 0.005)),
    ]

    bins = np.asarray(upper["bins"])[::2]
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, 6)[[0, 2, 4, 5]])
    distributions = []
    for tag, label, counts in intervals:
        if np.any(counts < 0):
            raise ValueError(f"negative exclusive count in {label}")
        counts = counts.reshape(-1, 2).sum(axis=1)
        distributions.append((tag, label, counts, counts / counts.sum()))

    minimum = min(fraction.min() for _, _, _, fraction in distributions)
    maximum = max(fraction.max() for _, _, _, fraction in distributions)
    padding = 0.06 * (maximum - minimum)

    plt.rcParams["font.family"] = "serif"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for (tag, label, counts, fraction), color in zip(distributions, colors):
        figure, axis = plt.subplots(figsize=(11.5, 7.2))
        axis.stairs(
            fraction,
            bins,
            baseline=None,
            linewidth=1.9,
            color=color,
        )
        axis.set_xlim(-np.pi, np.pi)
        axis.set_ylim(minimum - padding, maximum + padding)
        axis.set_xticks(
            [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
        )
        axis.set_xlabel(r"Photon momentum $\phi$ [rad]", fontsize=17)
        axis.set_ylabel("Fraction of photons per bin", fontsize=17)
        axis.set_title(
            fr"Unrotated BIB photon $\phi$: {label} ({int(counts.sum()):,} photons)",
            fontsize=20,
            pad=18,
        )
        axis.tick_params(labelsize=14)
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)

        figure.tight_layout()
        figure.savefig(OUTPUT / f"photon_phi_{tag}.pdf", bbox_inches="tight")
        figure.savefig(OUTPUT / f"photon_phi_{tag}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
