#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parent
PHOTON_INPUT = ROOT / "phi_anisotropy_out" / "photon_energy_binned_anisotropy" / "photon_energy_binned_anisotropy.npz"
MUON_INPUT = ROOT / "phi_anisotropy_out" / "gen_phi_samples_MUPLUS.npz"
OUTPUT = ROOT / "phi_anisotropy_out"


def draw(counts, edges, particle, output):
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    difference = 100.0 * (counts / counts.mean() - 1.0)
    limit = np.max(np.abs(difference))
    normalization = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    colormap = plt.get_cmap("RdBu_r")

    figure = plt.figure(figsize=(10.4, 7.2))
    axis = figure.add_subplot(111, projection="polar")
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(-1)
    axis.bar(
        centers,
        np.full_like(centers, 0.62),
        bottom=0.38,
        width=width,
        align="center",
        color=colormap(normalization(difference)),
        edgecolor="white",
        linewidth=1.0,
    )

    axis.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    axis.set_xticklabels([])
    axis.text(1.02, 0.50, r"$+x$  ($\phi=0$)", transform=axis.transAxes,
              ha="left", va="center", fontsize=13)
    axis.text(0.50, -0.06, r"$+y$  ($\phi=\pi/2$)", transform=axis.transAxes,
              ha="center", va="top", fontsize=13)
    axis.text(-0.02, 0.50, r"$-x$  ($\phi=\pi$)", transform=axis.transAxes,
              ha="right", va="center", fontsize=13)
    axis.text(0.50, 1.02, r"$-y$  ($\phi=-\pi/2$)", transform=axis.transAxes,
              ha="center", va="bottom", fontsize=13)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([])
    axis.set_title(
        f"BIB {particle} momentum in the x–y plane  ({counts.sum():,.0f} {particle}s)",
        fontsize=18,
        pad=58,
    )
    axis.text(
        0,
        0,
        "⊗ +z    ⊙ −z\nbeam axis",
        ha="center",
        va="center",
        fontsize=10,
        zorder=5,
        bbox={
            "boxstyle": "circle,pad=0.55",
            "facecolor": "white",
            "edgecolor": "0.25",
            "linewidth": 1.2,
        },
    )
    axis.grid(color="0.65", linewidth=0.7, alpha=0.55)

    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=colormap),
        ax=axis,
        pad=0.15,
        fraction=0.045,
    )
    colorbar.set_label("Deviation from uniform [percent]", fontsize=12)
    colorbar.ax.tick_params(labelsize=11)

    figure.tight_layout()
    figure.savefig(OUTPUT / f"{output}.pdf", bbox_inches="tight")
    figure.savefig(
        OUTPUT / f"{output}.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    photons = np.load(PHOTON_INPUT)
    source_edges = photons["phi_edges"]
    photon_counts = photons["hist_unrotated"].sum(axis=0)
    photon_counts = photon_counts.reshape(16, 4).sum(axis=1)
    edges = source_edges[::4]
    draw(photon_counts, edges, "photon", "photon_xy_momentum_projection")

    muons = np.load(MUON_INPUT)
    muon_counts, _ = np.histogram(muons["phi_muons"], bins=edges)
    draw(muon_counts, edges, "muon", "muon_xy_momentum_projection")


if __name__ == "__main__":
    main()
