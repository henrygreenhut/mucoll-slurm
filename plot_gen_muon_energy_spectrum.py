#!/usr/bin/env python3

from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "stores_oscar" / "gen_norm1_reconstructed_MUPLUS.h5"
OUTPUT = ROOT / "plots" / "gen_native_muon_energy_spectrum"
EXCLUDED_CYCLES = {6291}


def kept_ranges(handle):
    cycle_ids = handle["cycle_ids"][:]
    offsets = handle["offsets"][:]
    start = None
    for index, cycle in enumerate(cycle_ids):
        if int(cycle) in EXCLUDED_CYCLES:
            if start is not None:
                yield start, int(offsets[index])
                start = None
        elif start is None:
            start = int(offsets[index])
    if start is not None:
        yield start, int(offsets[-1])


def load_muons(path):
    energies = []
    pdgs = []
    with h5py.File(path) as handle:
        particles = handle["particles"]
        for start, stop in kept_ranges(handle):
            for first in range(start, stop, 1_000_000):
                last = min(first + 1_000_000, stop)
                pdg = particles["pdg"][first:last]
                selected = np.abs(pdg) == 13
                energies.append(particles["E"][first:last][selected])
                pdgs.append(pdg[selected])
    return np.concatenate(energies), np.concatenate(pdgs)


def main():
    energy, pdg = load_muons(INPUT)
    bins = np.logspace(
        np.log10(energy.min()), np.log10(energy.max()), 55
    )

    plt.rcParams["font.family"] = "serif"
    figure, axis = plt.subplots(figsize=(6.5, 4.6))
    axis.hist(
        energy, bins=bins, histtype="step", linewidth=2.2,
        color="black", label=rf"all muons ({len(energy)})"
    )
    axis.hist(
        energy[pdg == -13], bins=bins, histtype="step", linewidth=1.7,
        color="#D55E00", label=rf"$\mu^+$ ({np.count_nonzero(pdg == -13)})"
    )
    axis.hist(
        energy[pdg == 13], bins=bins, histtype="step", linewidth=1.7,
        color="#0072B2", label=rf"$\mu^-$ ({np.count_nonzero(pdg == 13)})"
    )
    axis.axvline(5.0, color="0.45", linestyle="--", linewidth=1.2,
                 label=r"$E=5$ GeV")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Muon energy $E$ [GeV]")
    axis.set_ylabel("Muons per bin")
    axis.set_title("Native MUPLUS GEN BIB muon energy spectrum")
    axis.grid(alpha=0.2, linewidth=0.5, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)

    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(f"muons: {len(energy)}")
    print(f"energy range: {energy.min():.6g} to {energy.max():.6g} GeV")
    print(f"E > 5 GeV: {np.count_nonzero(energy > 5.0)}")
    print(f"wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"wrote {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
