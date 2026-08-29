#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "particle_rayleigh_analysis.json"
OUTPUT = ROOT / "phi_anisotropy_out" / "particle_second_harmonic_by_species"

LABELS = {
    "photons": "Photons",
    "neutrons": "Neutrons",
    "electrons_positrons": "Electrons/\npositrons",
    "muons": "Muons",
    "charged_pions": "Charged\npions",
    "protons": "Protons",
    "kaons": "Kaons",
}


def harmonic(data, particle, index):
    return data[particle]["harmonics"][index]


def panel(axis, data, particles, limit):
    positions = np.arange(len(particles))
    width = 0.36

    for offset, source, color, label in [
        (-width / 2, "unrotated", "#0072B2", "Unrotated"),
        (width / 2, "rotated", "#D55E00", "Rotated once"),
    ]:
        rows = [harmonic(data[source], particle, 1) for particle in particles]
        values = [row["modulation_percent"] for row in rows]
        errors = [row["modulation_bootstrap_std_percent"] for row in rows]
        axis.bar(
            positions + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            color=color,
            label=label,
        )

    axis.set_xticks(positions, [LABELS[particle] for particle in particles])
    axis.set_ylim(0, limit)
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def main():
    with open(INPUT) as source:
        data = json.load(source)

    plt.rcParams["font.family"] = "serif"
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    panel(axes[0], data, ["photons", "neutrons", "electrons_positrons"], 1.8)
    panel(axes[1], data, ["muons", "charged_pions", "protons", "kaons"], 42)
    axes[0].set_ylabel(r"Second-harmonic modulation $2R_2$ [percent]")
    axes[0].legend(frameon=False)
    figure.suptitle(r"Second harmonic of BIB particle momentum $\phi$")
    figure.tight_layout()
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
