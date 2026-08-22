#!/usr/bin/env python3

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "stores_oscar" / "gen_norm1_reconstructed_MUPLUS.h5"
OUTPUT = ROOT / "phi_anisotropy_out" / "unrotated_photon_phi_vs_energy"
THRESHOLD_OUTPUT = ROOT / "phi_anisotropy_out" / "unrotated_photon_phi_energy_thresholds"
ENERGY_THRESHOLDS = (
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
)


def photon_histogram(path):
    phi_edges = np.linspace(-np.pi, np.pi, 65)
    log_energy_edges = np.linspace(-4.0, 0.35, 70)
    counts = np.zeros((len(log_energy_edges) - 1, len(phi_edges) - 1))
    threshold_counts = np.zeros((len(ENERGY_THRESHOLDS), len(phi_edges) - 1))
    photons = 0

    with h5py.File(path, "r") as source:
        particles = source["particles"]
        chunk_size = 1_000_000

        for start in range(0, len(particles["pdg"]), chunk_size):
            stop = min(start + chunk_size, len(particles["pdg"]))
            pdg = particles["pdg"][start:stop]
            selected = pdg == 22

            px = particles["px"][start:stop][selected]
            py = particles["py"][start:stop][selected]
            energy = particles["E"][start:stop][selected]
            positive = energy > 0

            phi = np.arctan2(py[positive], px[positive])
            log_energy = np.log10(energy[positive])
            counts += np.histogram2d(
                log_energy,
                phi,
                bins=(log_energy_edges, phi_edges),
            )[0]
            for row, threshold in enumerate(ENERGY_THRESHOLDS):
                threshold_counts[row] += np.histogram(
                    phi[energy[positive] > threshold],
                    bins=phi_edges,
                )[0]
            photons += len(phi)

    row_totals = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts),
        where=row_totals > 0,
    )
    return (
        phi_edges,
        log_energy_edges,
        fractions,
        counts.sum(axis=1),
        threshold_counts,
        photons,
    )


def plot_thresholds(phi_edges, threshold_counts):
    centers = (phi_edges[:-1] + phi_edges[1:]) / 2
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(ENERGY_THRESHOLDS)))

    figure, axis = plt.subplots(figsize=(11.5, 7.2))
    for threshold, counts, color in zip(
        ENERGY_THRESHOLDS,
        threshold_counts,
        colors,
    ):
        total = int(counts.sum())
        fraction = counts / total if total else np.zeros_like(counts)
        axis.step(
            centers,
            fraction,
            where="mid",
            linewidth=1.9,
            color=color,
            label=fr"$E>{threshold:g}$ GeV ($N={total:,}$)",
        )

    axis.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    axis.set_xlim(-np.pi, np.pi)
    axis.set_xlabel(r"Photon momentum $\phi$ [rad]", fontsize=17)
    axis.set_ylabel("Fraction of photons per bin", fontsize=17)
    axis.set_title(
        r"Unrotated BIB photon $\phi$ by energy threshold",
        fontsize=20,
        pad=18,
    )
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
    figure.savefig(THRESHOLD_OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        THRESHOLD_OUTPUT.with_suffix(".png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    (
        phi_edges,
        energy_edges,
        fractions,
        energy_counts,
        threshold_counts,
        photons,
    ) = photon_histogram(INPUT)
    reliable = energy_counts >= 500
    fractions[~reliable] = np.nan
    highest_reliable = np.flatnonzero(reliable)[-1]

    plt.rcParams["font.family"] = "serif"
    figure, axis = plt.subplots(figsize=(10.5, 7.2))
    colormap = plt.get_cmap("magma").copy()
    colormap.set_bad("0.88")
    image = axis.pcolormesh(
        phi_edges,
        energy_edges,
        fractions,
        shading="auto",
        cmap=colormap,
        vmin=0,
        vmax=0.03,
    )

    axis.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    axis.set_xlim(-np.pi, np.pi)
    axis.set_ylim(energy_edges[0], energy_edges[highest_reliable + 1])
    axis.set_xlabel(r"Photon momentum $\phi$ [rad]", fontsize=17)
    axis.set_ylabel(r"$\log_{10}(E/\mathrm{GeV})$", fontsize=17)
    axis.set_title(r"Unrotated BIB photon $\phi$ versus energy", fontsize=20, pad=18)
    axis.tick_params(labelsize=14)

    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Fraction of photons in each energy bin", fontsize=14)
    colorbar.ax.tick_params(labelsize=12)

    figure.tight_layout()
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)
    plot_thresholds(phi_edges, threshold_counts)
    print(f"plotted {photons:,} photons")
    print(OUTPUT.with_suffix(".pdf"))
    print(OUTPUT.with_suffix(".png"))
    print(THRESHOLD_OUTPUT.with_suffix(".pdf"))
    print(THRESHOLD_OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
