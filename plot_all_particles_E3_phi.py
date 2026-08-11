#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "gen_phi_samples_MUPLUS.npz"
OUTPUT = ROOT / "phi_anisotropy_out"


def format_axis(axis, title, ylabel):
    axis.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    axis.set_xlim(-np.pi, np.pi)
    axis.set_xlabel(r"$\phi$ [rad]", fontsize=18)
    axis.set_ylabel(ylabel, fontsize=18)
    axis.set_title(title, fontsize=20, pad=28)
    axis.tick_params(labelsize=16)
    axis.legend(frameon=False, fontsize=14, loc="upper left")


def save(figure, name):
    figure.tight_layout()
    figure.savefig(OUTPUT / f"{name}.pdf")
    figure.savefig(OUTPUT / f"{name}.png", dpi=100)
    plt.close(figure)


def main():
    particles = np.load(INPUT)
    phi = particles["phi_E3"]
    rotated = particles["phi_E3_rotated"]
    muon = particles["is_muon_E3"]
    bins = np.linspace(-np.pi, np.pi, 17)

    figure, axis = plt.subplots(figsize=(10.5, 7.15))
    axis.hist(phi, bins=bins, histtype="step", linewidth=3, color="#0072B2", label="all particles, native")
    axis.hist(rotated, bins=bins, histtype="step", linewidth=3, color="#D55E00", label="all particles, rotated")
    axis.hist(phi[muon], bins=bins, histtype="step", linestyle="--", linewidth=2.5, color="#0072B2", label="muons, native")
    axis.hist(rotated[muon], bins=bins, histtype="step", linestyle="--", linewidth=2.5, color="#D55E00", label="muons, rotated")
    format_axis(axis, r"Particle momentum $\phi$ ($E > 3$ GeV)", "Particle count")
    save(figure, "phi_E3_all_particles")

    figure, axis = plt.subplots(figsize=(10.5, 7.15))
    fine_bins = np.linspace(-np.pi, np.pi, 33)
    axis.hist(particles["phi_muons"], bins=fine_bins, histtype="step", linewidth=3, color="#0072B2", label="native")
    axis.hist(particles["phi_muons_rotated"], bins=fine_bins, histtype="step", linewidth=3, color="#D55E00", label="rotated")
    format_axis(axis, r"Muon momentum $\phi$", "Muon count")
    save(figure, "phi_all_muons")

    figure, axis = plt.subplots(figsize=(10.5, 7.15))
    axis.hist(particles["phi_muons_below_1"], bins=bins, histtype="step", linewidth=3, color="#0072B2", label="native")
    axis.hist(particles["phi_muons_below_1_rotated"], bins=bins, histtype="step", linewidth=3, color="#D55E00", label="rotated")
    format_axis(axis, r"Muon momentum $\phi$ ($E < 1$ GeV)", "Muon count")
    save(figure, "phi_muons_below_1GeV")

    print(f"particles with E > 3 GeV: {len(phi)}")
    print(f"muons with E > 3 GeV: {muon.sum()}")
    print(f"muons at all energies: {len(particles['phi_muons'])}")
    print(f"muons below 1 GeV: {len(particles['phi_muons_below_1'])}")


if __name__ == "__main__":
    main()
