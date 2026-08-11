#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "spatial_phi_E5_counts.json"
OUTPUT = ROOT / "phi_anisotropy_out"


def make_plot(edges, counts, rotated_counts, particle, filename, ylim=54,
              title=None):
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.stairs(
        counts, edges, linewidth=2, color="#0072B2",
        label="unrotated BIB")
    axis.stairs(
        rotated_counts, edges, linewidth=2, color="#D55E00",
        label="rotated BIB")
    axis.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    axis.set_xlabel(r"Position angle around the beam axis $\phi$ [rad]")
    axis.set_ylabel("Muon count")
    axis.set_title(
        title or fr"MUPLUS GEN BIB: {particle} position angle ($E>5$ GeV)")
    axis.set_xlim(-np.pi, np.pi)
    axis.set_ylim(0, ylim)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(OUTPUT / f"{filename}.pdf")
    figure.savefig(OUTPUT / f"{filename}.png", dpi=180)
    plt.close(figure)


def main():
    with INPUT.open() as handle:
        data = json.load(handle)
    edges = np.asarray(data["edges"])
    make_plot(
        edges, np.asarray(data["counts"]["muplus"]),
        np.asarray(data["rotated_counts"]["muplus"]),
        r"$\mu^+$", "spatial_phi_E5_muplus")
    make_plot(
        edges, np.asarray(data["counts"]["muminus"]),
        np.asarray(data["rotated_counts"]["muminus"]),
        r"$\mu^-$", "spatial_phi_E5_muminus")
    all_muons = (
        np.asarray(data["counts"]["muplus"])
        + np.asarray(data["counts"]["muminus"])
    )
    all_rotated_muons = (
        np.asarray(data["rotated_counts"]["muplus"])
        + np.asarray(data["rotated_counts"]["muminus"])
    )
    make_plot(
        edges, all_muons, all_rotated_muons,
        "Muons", "spatial_phi_E5_all_muons",
        ylim=82,
        title="High-energy muon position in MUPLUS BIB ($E>5$ GeV)")


if __name__ == "__main__":
    main()
