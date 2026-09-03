#!/usr/bin/env python3

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parent
PHOTON_INPUT = (
    ROOT
    / "phi_anisotropy_out"
    / "photon_energy_binned_anisotropy"
    / "photon_energy_binned_anisotropy.npz"
)
MUON_INPUT = ROOT / "phi_anisotropy_out" / "gen_phi_samples_MUPLUS.npz"
OUTPUT = ROOT / "phi_anisotropy_out"


def draw(counts, edges, particle, output):
    difference = 100.0 * (counts / counts.mean() - 1.0)
    limit = np.max(np.abs(difference))
    normalization = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    colormap = plt.get_cmap("RdBu_r")

    figure, axis = plt.subplots(figsize=(9.2, 7.2))
    for left, right, value in zip(edges[:-1], edges[1:], difference):
        angle = np.linspace(left, right, 32)
        axis.plot(
            np.cos(angle),
            np.sin(angle),
            color=colormap(normalization(value)),
            linewidth=8,
            solid_capstyle="butt",
        )

    axis.text(1.08, 0, r"$+x$  ($\phi=0$)", ha="left", va="center", fontsize=13)
    axis.text(0, 1.08, r"$+y$  ($\phi=\pi/2$)", ha="center", va="bottom", fontsize=13)
    axis.text(-1.08, 0, r"$-x$  ($\phi=\pi$)", ha="right", va="center", fontsize=13)
    axis.text(0, -1.08, r"$-y$  ($\phi=-\pi/2$)", ha="center", va="top", fontsize=13)
    axis.set_xlim(-1.35, 1.35)
    axis.set_ylim(-1.28, 1.28)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(
        f"BIB {particle} momentum direction in the x–y plane\n"
        f"{counts.sum():,.0f} {particle}s",
        fontsize=18,
        pad=20,
    )

    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=colormap),
        ax=axis,
        pad=0.06,
        fraction=0.045,
    )
    colorbar.set_label("Deviation from uniform [percent]", fontsize=12)
    colorbar.ax.tick_params(labelsize=11)

    figure.tight_layout()
    figure.savefig(OUTPUT / f"{output}.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / f"{output}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    photons = np.load(PHOTON_INPUT)
    photon_counts = photons["hist_unrotated"].sum(axis=0)
    photon_counts = photon_counts.reshape(16, 4).sum(axis=1)
    edges = photons["phi_edges"][::4]
    draw(photon_counts, edges, "photon", "photon_xy_momentum_projection")

    muons = np.load(MUON_INPUT)
    muon_counts, _ = np.histogram(muons["phi_muons"], bins=edges)
    draw(muon_counts, edges, "muon", "muon_xy_momentum_projection")


if __name__ == "__main__":
    main()
