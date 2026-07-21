"""Diagnostic plots of ACTUAL TRAINING EXAMPLES ("units"), not raw library
files. A unit is exactly what the PFN trainer builds and feeds to the
network: n_files concatenated norm1 files (class 0, standard) or
n_files // clone_factor concatenated norm42 files (class 1, rotated/
cloned). Drawn via the SAME Store / common_positions / split_indices /
UnitSampler code the real trainer uses (imported, not reimplemented), from
the same train split and the same production store paths.

Contrast with bib_diagnostic_plots.py, which looks at individual raw
library files -- this script looks at what the network actually sees.

Usage (on a NERSC login node, after module load tensorflow):
    python bib_example_unit_plots.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import libtest_common as lc
from pfn_libtest_train import UnitSampler

# Current training config (config #1 baseline).
NORM1_STORE = "/pscratch/sd/h/hgreen/mucoll/libtest/stores/gen_norm1_MUPLUS.h5"
NORM42_STORE = "/pscratch/sd/h/hgreen/mucoll/libtest/stores/gen_norm42_MUPLUS.h5"
N_FILES = 420
CLONE_FACTOR = 42
SPLIT_FRACS = (0.50, 0.25, 0.25)
SPLIT = "train"

N_EXAMPLES = 20
SEED = 11
OUT_PNG = "plots/bib_example_unit_plots.png"

LABEL0 = "class 0 (norm1, standard)"
LABEL1 = "class 1 (norm42, rotated/cloned)"


def build_samplers():
    store1 = lc.Store(NORM1_STORE)
    store_b = lc.Store(NORM42_STORE)
    common, pos1, pos_b = lc.common_positions(store1, store_b)
    splits = lc.split_indices(len(common), SPLIT_FRACS)
    idx = splits[SPLIT]

    files_b = N_FILES // CLONE_FACTOR
    sampler0 = UnitSampler(store1, {SPLIT: pos1[idx]}, N_FILES)
    sampler1 = UnitSampler(store_b, {SPLIT: pos_b[idx]}, files_b)
    return store1, store_b, sampler0, sampler1


def example_unit_data(store, sampler, rng, n_examples):
    """Draw n_examples real training units; return per-unit multiplicities
    plus pooled RAW (unnormalized) E/phi across all drawn units. Uses
    store.file_arrays directly rather than UnitSampler.build, since build
    z-score-normalizes for network input -- we want physical values here."""
    mults = []
    e_parts, px_parts, py_parts = [], [], []
    for _ in range(n_examples):
        positions = sampler.random_unit(rng, SPLIT)
        raw = store.file_arrays(positions)
        mults.append(len(raw["E"]))
        e_parts.append(raw["E"])
        px_parts.append(raw["px"])
        py_parts.append(raw["py"])
    energy = np.concatenate(e_parts)
    phi = np.arctan2(np.concatenate(py_parts), np.concatenate(px_parts))
    return np.asarray(mults), energy, phi


def main():
    store1, store_b, sampler0, sampler1 = build_samplers()
    rng = np.random.default_rng(SEED)

    mult0, e0, phi0 = example_unit_data(store1, sampler0, rng, N_EXAMPLES)
    mult1, e1, phi1 = example_unit_data(store_b, sampler1, rng, N_EXAMPLES)

    print(f"{LABEL0}: {sampler0.files_per_unit} files/unit, {N_EXAMPLES} example "
          f"units -> per-unit N min={mult0.min()} median={int(np.median(mult0))} "
          f"mean={mult0.mean():.0f} max={mult0.max()}")
    print(f"{LABEL1}: {sampler1.files_per_unit} files/unit, {N_EXAMPLES} example "
          f"units -> per-unit N min={mult1.min()} median={int(np.median(mult1))} "
          f"mean={mult1.mean():.0f} max={mult1.max()}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {LABEL0: "#1f77b4", LABEL1: "#d62728"}

    ax = axes[0]
    e_all = np.concatenate([e0, e1])
    e_pos = e_all[e_all > 0]
    bins = np.logspace(np.log10(e_pos.min()), np.log10(e_pos.max()), 80)
    ax.hist(e0[e0 > 0], bins=bins, histtype="step", density=True, linewidth=1.8,
            color=colors[LABEL0], label=LABEL0)
    ax.hist(e1[e1 > 0], bins=bins, histtype="step", density=True, linewidth=1.8,
            color=colors[LABEL1], label=LABEL1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("particle energy E [GeV]")
    ax.set_ylabel("density")
    ax.set_title(f"Energy distribution\n({N_EXAMPLES} example training units/class, pooled)")
    ax.legend(fontsize=9)

    ax = axes[1]
    bins = np.linspace(-np.pi, np.pi, 80)
    ax.hist(phi0, bins=bins, histtype="step", density=True, linewidth=1.8,
            color=colors[LABEL0], label=LABEL0)
    ax.hist(phi1, bins=bins, histtype="step", density=True, linewidth=1.8,
            color=colors[LABEL1], label=LABEL1)
    ax.set_xlabel(r"particle $\phi$ [rad]")
    ax.set_ylabel("density")
    ax.set_title(f"$\\phi$ distribution\n({N_EXAMPLES} example training units/class, pooled)")
    ax.legend(fontsize=9)

    ax = axes[2]
    m_all = np.concatenate([mult0, mult1])
    bins = np.linspace(m_all.min(), m_all.max(), 20)
    ax.hist(mult0, bins=bins, histtype="step", density=True, linewidth=1.8,
            color=colors[LABEL0], label=LABEL0)
    ax.hist(mult1, bins=bins, histtype="step", density=True, linewidth=1.8,
            color=colors[LABEL1], label=LABEL1)
    ax.set_xlabel("particles per training unit")
    ax.set_ylabel("density")
    ax.set_title(f"Per-unit particle multiplicity\n({N_EXAMPLES} example units/class)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
