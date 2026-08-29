#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "photon_phi_all_fine_counts.json"
OUTPUT = ROOT / "phi_anisotropy_out" / "photon_phi_all_unrotated_vs_rotated_fine"


def rebin(counts, factor):
    return counts.reshape(-1, factor).sum(axis=1)


def draw(bins, unrotated_counts, rotated_counts, output):
    unrotated = unrotated_counts / unrotated_counts.sum()
    rotated = rotated_counts / rotated_counts.sum()
    uniform = 1 / len(unrotated)

    figure, axis = plt.subplots(figsize=(10, 6.5))

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
        r"BIB photon momentum $\phi$"
        + f"\n{int(unrotated_counts.sum()):,} unrotated and "
        + f"{int(rotated_counts.sum()):,} rotated photons"
    )
    axis.legend(frameon=False, loc="upper right")
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    with open(INPUT) as source:
        data = json.load(source)

    bins = np.asarray(data["bins"])
    unrotated_counts = np.asarray(data["unrotated"][0], dtype=float)
    rotated_counts = np.asarray(data["rotated"][0], dtype=float)
    plt.rcParams["font.family"] = "serif"
    draw(bins, unrotated_counts, rotated_counts, OUTPUT)

    draw(
        bins[::2],
        rebin(unrotated_counts, 2),
        rebin(rotated_counts, 2),
        ROOT / "phi_anisotropy_out" / "photon_phi_all_unrotated_vs_rotated_zoom",
    )


if __name__ == "__main__":
    main()
