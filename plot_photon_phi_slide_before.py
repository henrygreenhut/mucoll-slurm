#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "photon_phi_all_fine_counts.json"
OUTPUT = ROOT / "phi_anisotropy_out" / "phi_all_photons"


def main():
    with open(INPUT) as source:
        data = json.load(source)

    bins = np.asarray(data["bins"])[::4]
    unrotated = np.asarray(data["unrotated"][0]).reshape(-1, 4).sum(axis=1)
    rotated = np.asarray(data["rotated"][0]).reshape(-1, 4).sum(axis=1)
    center = (unrotated.sum() + rotated.sum()) / (2 * len(unrotated))

    plt.rcParams["font.family"] = "serif"
    figure, axis = plt.subplots(figsize=(10.5, 7.15))
    axis.stairs(
        unrotated,
        bins,
        baseline=None,
        linewidth=2.6,
        color="#0072B2",
        label="Unrotated",
    )
    axis.stairs(
        rotated,
        bins,
        baseline=None,
        linewidth=2.6,
        color="#D55E00",
        label="Rotated once",
    )

    axis.set_xlim(-np.pi, np.pi)
    axis.set_ylim(0.975 * center, 1.025 * center)
    axis.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    axis.set_xlabel(r"Photon momentum $\phi$ [rad]", fontsize=18)
    axis.set_ylabel("Photon count", fontsize=18)
    axis.set_title(r"Photon momentum $\phi$", fontsize=20, pad=20)
    axis.tick_params(labelsize=16)
    axis.legend(frameon=False, fontsize=16, loc="upper left")
    axis.grid(alpha=0.2, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
