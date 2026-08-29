#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "particle_phi_histograms_MUPLUS.npz"
OUTPUT = ROOT / "phi_anisotropy_out" / "photon_phi_anisotropy_summary"


def main():
    histograms = np.load(INPUT)
    categories = list(histograms["categories"])
    photon = categories.index("photons")
    bins = histograms["bins"]

    unrotated = histograms["native"][photon].astype(float)
    rotated = histograms["rotated"][photon].astype(float)
    unrotated /= unrotated.sum()
    rotated /= rotated.sum()
    uniform = 1 / len(unrotated)

    plt.rcParams["font.family"] = "serif"
    figure, (distribution, residual) = plt.subplots(
        2,
        1,
        figsize=(10, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.08},
        layout="constrained",
    )

    distribution.stairs(
        unrotated,
        bins,
        linewidth=2.5,
        color="#0072B2",
        baseline=None,
        label=f"Unrotated ({int(histograms['native'][photon].sum()):,} photons)",
    )
    distribution.stairs(
        rotated,
        bins,
        linewidth=2.5,
        color="#D55E00",
        baseline=None,
        label="Rotated once",
    )
    distribution.axhline(
        uniform,
        color="0.45",
        linestyle="--",
        linewidth=1.2,
        label="Uniform",
    )
    distribution.set_ylabel("Fraction of photons per bin")
    distribution.set_title(r"BIB photon momentum $\phi$")
    distribution.legend(frameon=False, loc="upper right")
    distribution.grid(axis="y", alpha=0.25, linewidth=0.5)
    distribution.spines[["top", "right"]].set_visible(False)

    unrotated_residual = 100 * (unrotated / uniform - 1)
    rotated_residual = 100 * (rotated / uniform - 1)
    residual.stairs(
        unrotated_residual,
        bins,
        linewidth=2.5,
        color="#0072B2",
        baseline=None,
    )
    residual.stairs(
        rotated_residual,
        bins,
        linewidth=2.5,
        color="#D55E00",
        baseline=None,
    )
    residual.axhline(0, color="0.45", linestyle="--", linewidth=1.2)
    residual.set_ylabel("Deviation from\nuniform [%]")
    residual.set_xlabel(r"Photon momentum $\phi$ [rad]")
    residual.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    residual.set_xlim(-np.pi, np.pi)
    residual.grid(axis="y", alpha=0.25, linewidth=0.5)
    residual.spines[["top", "right"]].set_visible(False)

    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
