#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "phi_anisotropy_out" / "inclusive_phi_counts.json"
OUTPUT = ROOT / "plots" / "bib_example_unit_phi"


def main():
    with INPUT.open() as handle:
        data = json.load(handle)

    edges = np.linspace(-np.pi, np.pi, data["edge_count"])
    unrotated = np.asarray(data["counts"]["unrotated"])
    rotated = np.asarray(data["counts"]["rotated"])
    widths = np.diff(edges)
    unrotated = unrotated / (unrotated.sum() * widths)
    rotated = rotated / (rotated.sum() * widths)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.stairs(
        unrotated, edges, color="#0072B2", linewidth=2,
        label="unrotated BIB"
    )
    axis.stairs(
        rotated, edges, color="#D55E00", linewidth=2,
        linestyle="--", label="rotated BIB"
    )
    axis.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    )
    axis.set_xlabel(r"Momentum angle $\phi$ [rad]")
    axis.set_ylabel("Particle density")
    axis.set_title(r"GEN-level BIB momentum $\phi$")
    axis.set_xlim(-np.pi, np.pi)
    axis.set_ylim(0, 0.17)
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(OUTPUT.with_suffix(".pdf"))
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
