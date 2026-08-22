#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "photon_phi_Egt0p05_counts.json"
OUTPUT = ROOT / "phi_anisotropy_out" / "photon_phi_Egt0p05_unrotated_vs_rotated"


def main():
    with open(INPUT) as source:
        data = json.load(source)

    bins = np.asarray(data["bins"])[::2]
    unrotated = np.asarray(data["unrotated"]).reshape(-1, 2).sum(axis=1)
    rotated = np.asarray(data["rotated"]).reshape(-1, 2).sum(axis=1)
    unrotated = unrotated / unrotated.sum()
    rotated = rotated / rotated.sum()

    plt.rcParams["font.family"] = "serif"
    figure, axis = plt.subplots(figsize=(10.5, 7.2))
    axis.stairs(
        unrotated,
        bins,
        color="#0072B2",
        linewidth=2.6,
        baseline=None,
        label="Unrotated",
    )
    axis.stairs(
        rotated,
        bins,
        color="#D55E00",
        linewidth=2.6,
        baseline=None,
        label="Rotated once",
    )

    axis.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    axis.set_xlim(-np.pi, np.pi)
    lower = min(unrotated.min(), rotated.min())
    upper = max(unrotated.max(), rotated.max())
    padding = 0.12 * (upper - lower)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_xlabel(r"Photon momentum $\phi$ [rad]", fontsize=17)
    axis.set_ylabel("Fraction of photons per bin", fontsize=17)
    axis.set_title(
        r"BIB photon $\phi$, $E>0.05$ GeV",
        fontsize=20,
        pad=18,
    )
    axis.tick_params(labelsize=14)
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend(frameon=False, fontsize=15, loc="upper right")
    axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"unrotated photons: {sum(data['unrotated']):,}")
    print(f"rotated photons: {sum(data['rotated']):,}")
    print(OUTPUT.with_suffix(".pdf"))
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
