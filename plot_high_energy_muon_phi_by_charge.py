#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "energetic_bib_MUPLUS.npz"
OUTPUT = ROOT / "phi_anisotropy_out"
ENERGY_CUT = 5.0
COLORS = {"native": "#0072B2", "rotated": "#D55E00"}


def wrap_phi(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def make_plot(phi, particle, filename, seed):
    rng = np.random.default_rng(seed)
    rotated = wrap_phi(phi + rng.uniform(0, 2 * np.pi, len(phi)))
    bins = np.linspace(-np.pi, np.pi, 21)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.hist(
        phi, bins=bins, histtype="step", linewidth=2,
        color=COLORS["native"], label="unrotated BIB")
    axis.hist(
        rotated, bins=bins, histtype="step", linewidth=2,
        color=COLORS["rotated"], label="uniform-rotation reference")
    axis.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    axis.set_xlabel(r"Momentum azimuth $\phi=\operatorname{atan2}(p_y,p_x)$ [rad]")
    axis.set_ylabel(fr"{particle} with $E>{ENERGY_CUT:g}$ GeV per bin")
    axis.set_title(fr"MUPLUS GEN BIB: {particle} momentum azimuth")
    axis.set_ylim(0, 52)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(OUTPUT / f"{filename}.pdf")
    figure.savefig(OUTPUT / f"{filename}.png", dpi=180)
    plt.close(figure)


def summarize(phi, particle):
    print(
        f"{particle}: {len(phi)} particles; "
        f"px<0={np.mean(np.cos(phi) < 0):.3f}; "
        f"<cos(phi)>={np.mean(np.cos(phi)):.3f}; "
        f"<cos(2phi)>={np.mean(np.cos(2 * phi)):.3f}")


def main():
    arrays = np.load(INPUT)
    phi = np.arctan2(arrays["py"], arrays["px"])
    energetic = arrays["E"] > ENERGY_CUT

    muplus = phi[energetic & (arrays["pdg"] == -13)]
    muminus = phi[energetic & (arrays["pdg"] == 13)]

    make_plot(muplus, r"$\mu^+$", "phi_E5_muplus", 1701)
    make_plot(muminus, r"$\mu^-$", "phi_E5_muminus", 1702)
    summarize(muplus, "mu+")
    summarize(muminus, "mu-")


if __name__ == "__main__":
    main()
