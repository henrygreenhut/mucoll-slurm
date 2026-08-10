#!/usr/bin/env python3

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "energetic_bib_MUPLUS.npz"
OUTPUT = ROOT / "plots" / "high_energy_muon_theta_eta_vs_phi"
ENERGY_MIN = 5.0


def angular_coordinates(px, py, pz):
    pt = np.hypot(px, py)
    phi = np.arctan2(py, px)
    theta = np.degrees(np.arctan2(pt, pz))
    eta = np.arcsinh(np.divide(pz, pt, out=np.zeros_like(pz), where=pt > 0))
    return phi, theta, eta


def format_phi_axis(axis):
    axis.set_xlim(-np.pi, np.pi)
    axis.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axis.set_xticklabels(
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    )
    axis.set_xlabel(r"$\phi$ [rad]")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)


def main():
    data = np.load(INPUT)
    selected = (data["E"] > ENERGY_MIN) & (np.abs(data["pdg"]) == 13)
    phi, theta, eta = angular_coordinates(
        data["px"][selected], data["py"][selected], data["pz"][selected]
    )
    pdg = data["pdg"][selected]

    plt.rcParams["font.family"] = "serif"
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True)

    samples = [
        (pdg == -13, r"$\mu^+$", "#D55E00"),
        (pdg == 13, r"$\mu^-$", "#0072B2"),
    ]
    for mask, label, color in samples:
        axes[0].scatter(
            phi[mask], theta[mask], s=24, alpha=0.65,
            color=color, edgecolors="none", label=f"{label} ({mask.sum()})"
        )
        axes[1].scatter(
            phi[mask], eta[mask], s=24, alpha=0.65,
            color=color, edgecolors="none"
        )

    axes[0].set_ylabel(r"Polar angle $\theta$ [deg]")
    axes[1].set_ylabel(r"Pseudorapidity $\eta$")
    axes[0].set_title(r"$\theta$ versus $\phi$")
    axes[1].set_title(r"$\eta$ versus $\phi$")
    axes[0].legend(frameon=False, loc="upper center")
    for axis in axes:
        format_phi_axis(axis)

    figure.suptitle(
        rf"Native MUPLUS BIB muons with $E>{ENERGY_MIN:g}$ GeV"
    )
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(f"selected {len(phi)} muons")
    print(f"wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"wrote {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
