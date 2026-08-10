#!/usr/bin/env python3

from pathlib import Path

import h5py
import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "stores_oscar" / "gen_norm1_reconstructed_MUPLUS.h5"
OUTPUT = ROOT / "plots" / "all_bib_theta_eta_vs_phi"
CHUNK_SIZE = 1_000_000
EXCLUDED_CYCLES = {6291}


def kept_ranges(handle):
    cycle_ids = handle["cycle_ids"][:]
    offsets = handle["offsets"][:]
    start = None
    for index, cycle in enumerate(cycle_ids):
        if int(cycle) in EXCLUDED_CYCLES:
            if start is not None:
                yield start, int(offsets[index])
                start = None
        elif start is None:
            start = int(offsets[index])
    if start is not None:
        yield start, int(offsets[-1])


def pieces(start, stop):
    for first in range(start, stop, CHUNK_SIZE):
        yield first, min(first + CHUNK_SIZE, stop)


def angular_histograms(path):
    phi_edges = np.linspace(-np.pi, np.pi, 181)
    theta_edges = np.linspace(0.0, 180.0, 181)
    eta_edges = np.linspace(-8.0, 8.0, 241)
    theta_counts = np.zeros((len(phi_edges) - 1, len(theta_edges) - 1))
    eta_counts = np.zeros((len(phi_edges) - 1, len(eta_edges) - 1))
    particles = 0
    eta_outside = 0

    with h5py.File(path) as handle:
        group = handle["particles"]
        for start, stop in kept_ranges(handle):
            for first, last in pieces(start, stop):
                px = group["px"][first:last]
                py = group["py"][first:last]
                pz = group["pz"][first:last]
                pt = np.hypot(px, py)
                phi = np.arctan2(py, px)
                theta = np.degrees(np.arctan2(pt, pz))
                eta = np.arcsinh(pz / pt)

                theta_counts += np.histogram2d(
                    phi, theta, bins=(phi_edges, theta_edges)
                )[0]
                eta_counts += np.histogram2d(
                    phi, eta, bins=(phi_edges, eta_edges)
                )[0]
                particles += len(phi)
                eta_outside += np.count_nonzero(
                    (eta < eta_edges[0]) | (eta > eta_edges[-1])
                )

    return (
        phi_edges,
        theta_edges,
        eta_edges,
        theta_counts,
        eta_counts,
        particles,
        eta_outside,
    )


def format_phi_axis(axis):
    axis.set_xlim(-np.pi, np.pi)
    axis.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axis.set_xticklabels(
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    )
    axis.set_xlabel(r"$\phi$ [rad]")


def main():
    (
        phi_edges,
        theta_edges,
        eta_edges,
        theta_counts,
        eta_counts,
        particles,
        eta_outside,
    ) = angular_histograms(INPUT)

    maximum = max(theta_counts.max(), eta_counts.max())
    norm = LogNorm(vmin=1, vmax=maximum)
    plt.rcParams["font.family"] = "serif"
    figure = plt.figure(figsize=(12.2, 4.6))
    grid = figure.add_gridspec(1, 3, width_ratios=(1, 1, 0.035), wspace=0.30)
    axes = [figure.add_subplot(grid[0]), figure.add_subplot(grid[1])]
    color_axis = figure.add_subplot(grid[2])

    theta_mesh = axes[0].pcolormesh(
        phi_edges, theta_edges, theta_counts.T,
        cmap="viridis", norm=norm, shading="auto"
    )
    axes[1].pcolormesh(
        phi_edges, eta_edges, eta_counts.T,
        cmap="viridis", norm=norm, shading="auto"
    )

    axes[0].set_ylabel(r"Polar angle $\theta$ [deg]")
    axes[1].set_ylabel(r"Pseudorapidity $\eta$")
    axes[0].set_title(r"$\theta$ versus $\phi$")
    axes[1].set_title(r"$\eta$ versus $\phi$")
    axes[0].set_ylim(0, 180)
    axes[1].set_ylim(eta_edges[0], eta_edges[-1])
    for axis in axes:
        format_phi_axis(axis)

    figure.colorbar(theta_mesh, cax=color_axis, label="Particles per bin")
    figure.suptitle(
        f"Native MUPLUS BIB: all {particles:,} particles"
    )
    figure.subplots_adjust(left=0.07, right=0.94, bottom=0.14, top=0.82)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(f"plotted {particles:,} particles")
    print(f"particles outside eta range: {eta_outside:,}")
    print(f"wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"wrote {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
