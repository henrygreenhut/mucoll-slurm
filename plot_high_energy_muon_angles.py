#!/usr/bin/env python3

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "energetic_bib_MUPLUS.npz"
OUTPUT = ROOT / "plots"
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


def save_plot(phi, vertical, pdg, ylabel, name):
    samples = [
        (pdg == -13, r"$\mu^+$", "#D55E00"),
        (pdg == 13, r"$\mu^-$", "#0072B2"),
    ]
    figure, axis = plt.subplots(figsize=(6.5, 4.6))
    for mask, label, color in samples:
        axis.scatter(
            phi[mask], vertical[mask], s=24, alpha=0.65,
            color=color, edgecolors="none", label=f"{label} ({mask.sum()})"
        )
    axis.set_ylabel(ylabel)
    axis.set_title(rf"MUPLUS GEN BIB muons with $E>{ENERGY_MIN:g}$ GeV")
    axis.legend(frameon=False)
    format_phi_axis(axis)
    figure.tight_layout()
    path = OUTPUT / name
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path.with_suffix('.pdf')}")
    print(f"wrote {path.with_suffix('.png')}")


def main():
    data = np.load(INPUT)
    selected = (data["E"] > ENERGY_MIN) & (np.abs(data["pdg"]) == 13)
    phi, theta, eta = angular_coordinates(
        data["px"][selected], data["py"][selected], data["pz"][selected]
    )
    pdg = data["pdg"][selected]

    plt.rcParams["font.family"] = "serif"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_plot(
        phi, theta, pdg, r"Polar angle $\theta$ [deg]",
        "high_energy_muon_theta_vs_phi"
    )
    save_plot(
        phi, eta, pdg, r"Pseudorapidity $\eta$",
        "high_energy_muon_eta_vs_phi"
    )
    print(f"selected {len(phi)} muons")


if __name__ == "__main__":
    main()
