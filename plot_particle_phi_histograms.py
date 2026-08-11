#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "particle_phi_histograms_MUPLUS.npz"
OUTPUT = ROOT / "phi_anisotropy_out"

TITLES = {
    "photons": "Photon",
    "neutrons": "Neutron",
    "electrons_positrons": "Electron and positron",
    "charged_pions": "Charged-pion",
    "protons": "Proton",
    "muons": "Muon",
    "kaons": "Kaon",
    "lambda_baryons": "Lambda-baryon",
    "light_nuclei": "Light-nucleus",
}


def main():
    histograms = np.load(INPUT)
    bins = histograms["bins"]

    for row, category in enumerate(histograms["categories"]):
        category = str(category)
        title = TITLES[category]

        figure, axis = plt.subplots(figsize=(10.5, 7.15))
        axis.stairs(
            histograms["native"][row], bins,
            linewidth=3, color="#0072B2", label="native",
        )
        axis.stairs(
            histograms["rotated"][row], bins,
            linewidth=3, color="#D55E00", label="rotated",
        )
        axis.set_xticks(
            [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
        )
        axis.set_xlim(-np.pi, np.pi)
        axis.set_xlabel(r"$\phi$ [rad]", fontsize=18)
        axis.set_ylabel(f"{title} count", fontsize=18)
        axis.set_title(fr"{title} momentum $\phi$", fontsize=20, pad=28)
        axis.tick_params(labelsize=16)
        axis.legend(frameon=False, fontsize=16, loc="upper left")
        figure.tight_layout()
        figure.savefig(OUTPUT / f"phi_all_{category}.pdf")
        figure.savefig(OUTPUT / f"phi_all_{category}.png", dpi=100)
        plt.close(figure)

        print(f"{category}: {histograms['native'][row].sum():,}")


if __name__ == "__main__":
    main()
