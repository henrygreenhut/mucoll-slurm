#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "photon_rayleigh_analysis.json"
OUTPUT = ROOT / "phi_anisotropy_out" / "photon_rayleigh_harmonics"


def main():
    with open(INPUT) as source:
        data = json.load(source)

    native = data["unrotated"]
    rotated = data["rotated"]
    harmonics = np.array([row["harmonic"] for row in native])
    native_amplitude = np.array([row["modulation_percent"] for row in native])
    rotated_amplitude = np.array([row["modulation_percent"] for row in rotated])
    native_error = 200 * np.array([row["radius_bootstrap_std"] for row in native])
    rotated_error = 200 * np.array([row["radius_bootstrap_std"] for row in rotated])

    plt.rcParams["font.family"] = "serif"
    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    width = 0.36
    axis.bar(
        harmonics - width / 2,
        native_amplitude,
        width,
        yerr=native_error,
        capsize=3,
        color="#0072B2",
        label="Unrotated",
    )
    axis.bar(
        harmonics + width / 2,
        rotated_amplitude,
        width,
        yerr=rotated_error,
        capsize=3,
        color="#D55E00",
        label="Rotated once",
    )

    axis.set_xticks(harmonics)
    axis.set_xlabel("Harmonic $n$")
    axis.set_ylabel(r"Sinusoidal modulation $2R_n$ [percent]")
    axis.set_title(r"Rayleigh analysis of BIB photon momentum $\phi$")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
