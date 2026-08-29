#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "phi_anisotropy_out" / "photon_second_harmonic_axis"

PHASE = np.deg2rad(88.7314)
MODULATION = 0.0032945
DISPLAY_SCALE = 60


def main():
    plt.rcParams["font.family"] = "serif"
    figure, axis = plt.subplots(figsize=(7.2, 7.2), subplot_kw={"projection": "polar"})

    phi = np.linspace(-np.pi, np.pi, 721)
    radius = 1 + DISPLAY_SCALE * MODULATION * np.cos(2 * (phi - PHASE))
    axis.fill(phi, radius, color="#0072B2", alpha=0.16)
    axis.plot(phi, radius, color="#0072B2", linewidth=2.2)
    axis.plot(phi, np.ones_like(phi), color="0.55", linestyle=":", linewidth=1.2)

    for angle in [PHASE, PHASE + np.pi]:
        axis.annotate(
            "",
            xy=(angle, 1.34),
            xytext=(angle, 0.05),
            arrowprops={"arrowstyle": "-|>", "color": "#0072B2", "lw": 2.5},
        )

    minimum = PHASE + np.pi / 2
    axis.plot(
        [minimum, minimum + np.pi],
        [1.16, 1.16],
        linestyle="--",
        color="#D55E00",
        linewidth=2,
    )

    axis.text(PHASE, 1.47, "preferred photon axis", ha="center", va="center", color="#0072B2")

    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_thetagrids(
        np.arange(0, 360, 45),
        [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$", r"$-3\pi/4$", r"$-\pi/2$", r"$-\pi/4$"],
    )
    axis.set_ylim(0, 1.55)
    axis.set_yticks([])
    axis.grid(alpha=0.2)
    axis.spines["polar"].set_alpha(0.3)
    axis.set_title(
        "Second-harmonic anisotropy of BIB photon momentum $\\phi$\n"
        r"preferred axis: $88.7^\circ$ and $-91.3^\circ$",
        pad=24,
    )

    figure.text(
        0.5,
        0.065,
        "Blue: preferred photon axis    Orange: relative deficit",
        ha="center",
        fontsize=10.5,
    )
    figure.text(
        0.5,
        0.035,
        r"Measured modulation: $2R_2=0.329\%$; radial deformation enlarged $60\times$ for visibility",
        ha="center",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
