#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "photon_phi_energy_cut_counts.json"
OUTPUT = ROOT / "phi_anisotropy_out"


def draw(axis, bins, unrotated, rotated, threshold):
    photons = int(unrotated.sum())
    unrotated = unrotated / unrotated.sum()
    rotated = rotated / rotated.sum()
    uniform = 1 / len(unrotated)

    axis.stairs(
        unrotated,
        bins,
        baseline=None,
        color="#0072B2",
        linewidth=2,
        label="Unrotated",
    )
    axis.stairs(
        rotated,
        bins,
        baseline=None,
        color="#D55E00",
        linewidth=2,
        label="Rotated once",
    )
    axis.axhline(
        uniform,
        color="0.45",
        linestyle="--",
        linewidth=1.2,
        label="Uniform",
    )

    lower = min(unrotated.min(), rotated.min(), uniform)
    upper = max(unrotated.max(), rotated.max(), uniform)
    padding = 0.12 * (upper - lower)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_xlim(-np.pi, np.pi)
    axis.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    axis.set_xlabel(r"Photon momentum $\phi$ [rad]")
    axis.set_ylabel("Fraction of photons per bin")
    axis.set_title(
        fr"BIB photon momentum $\phi$, $E>{threshold * 1000:g}$ MeV"
        + f"\n{photons:,} photons per sample"
    )
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def main():
    with open(INPUT) as source:
        data = json.load(source)

    bins = np.asarray(data["bins"])
    thresholds = data["thresholds"]
    unrotated = np.asarray(data["unrotated"], dtype=float)
    rotated = np.asarray(data["rotated"], dtype=float)

    plt.rcParams["font.family"] = "serif"
    OUTPUT.mkdir(exist_ok=True)

    for threshold, native_counts, rotated_counts in zip(
        thresholds,
        unrotated,
        rotated,
    ):
        figure, axis = plt.subplots(figsize=(10, 6.5))
        draw(axis, bins, native_counts, rotated_counts, threshold)
        axis.legend(frameon=False, loc="upper right")
        figure.tight_layout()

        mev = int(round(threshold * 1000))
        stem = OUTPUT / f"photon_phi_Egt{mev}MeV_unrotated_vs_rotated_fine"
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        figure.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(1, len(thresholds), figsize=(15, 5.4), sharex=True)
    for axis, threshold, native_counts, rotated_counts in zip(
        axes,
        thresholds,
        unrotated,
        rotated,
    ):
        draw(axis, bins, native_counts, rotated_counts, threshold)

    axes[0].legend(frameon=False, loc="upper right")
    figure.tight_layout()
    stem = OUTPUT / "photon_phi_energy_cuts_unrotated_vs_rotated_fine"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
