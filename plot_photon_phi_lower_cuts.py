#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "photon_phi_lower_cut_counts.json"
OUTPUT = ROOT / "phi_anisotropy_out" / "photon_phi_lower_energy_cuts"


def main():
    with open(INPUT) as source:
        data = json.load(source)

    bins = np.asarray(data["bins"])[::2]
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(data["thresholds"])))

    plt.rcParams["font.family"] = "serif"
    figure, axis = plt.subplots(figsize=(11.5, 7.2))

    for threshold, counts, color in zip(
        data["thresholds"],
        data["unrotated"],
        colors,
    ):
        counts = np.asarray(counts, dtype=float).reshape(-1, 2).sum(axis=1)
        fraction = counts / counts.sum()
        axis.stairs(
            fraction,
            bins,
            baseline=None,
            linewidth=1.9,
            color=color,
            label=fr"$E>{threshold * 1000:g}$ MeV ({int(counts.sum()):,})",
        )

    axis.set_xlim(-np.pi, np.pi)
    axis.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    axis.set_xlabel(r"Photon momentum $\phi$ [rad]", fontsize=17)
    axis.set_ylabel("Fraction of photons per bin", fontsize=17)
    axis.set_title(r"Unrotated BIB photon $\phi$ above energy thresholds", fontsize=20, pad=18)
    axis.tick_params(labelsize=14)
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend(
        frameon=False,
        fontsize=11,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )
    axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
