#!/usr/bin/env python3

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from plot_gen_muon_energy_spectrum import INPUT, load_muons


OUTPUT = Path(__file__).resolve().parent / "plots" / "gen_native_muon_fraction_above_energy"


def main():
    energy, _ = load_muons(INPUT)
    thresholds = np.unique(np.sort(energy))
    counts = len(energy) - np.searchsorted(energy, thresholds, side="right", sorter=np.argsort(energy))
    fractions = counts / len(energy)

    threshold = 5.0
    count_above = int(np.count_nonzero(energy > threshold))
    fraction_above = count_above / len(energy)

    plt.rcParams["font.family"] = "serif"
    figure, axis = plt.subplots(figsize=(6.5, 4.6))
    axis.step(thresholds, fractions, where="post", color="#0072B2", linewidth=2)
    axis.axvline(threshold, color="0.45", linestyle="--", linewidth=1.2)
    axis.scatter([threshold], [fraction_above], color="#D55E00", s=35, zorder=3)
    axis.annotate(
        f"E > 5 GeV: {fraction_above:.1%} ({count_above}/{len(energy)})",
        xy=(threshold, fraction_above),
        xytext=(8, fraction_above + 0.10),
        arrowprops={"arrowstyle": "-", "color": "0.35"},
        fontsize=10,
    )
    axis.set_xscale("log")
    axis.set_xlim(energy.min(), energy.max())
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel(r"Energy threshold $E$ [GeV]")
    axis.set_ylabel(r"Fraction of muons with $E_\mu > E$")
    axis.set_title("Native MUPLUS GEN BIB muon energy survival fraction")
    axis.grid(alpha=0.2, linewidth=0.5, which="both")
    axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(f"muons: {len(energy)}")
    print(f"E > 5 GeV: {count_above} ({fraction_above:.6f})")
    print(f"wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"wrote {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
