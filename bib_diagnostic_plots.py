"""BIB diagnostic plots: energy, phi, and per-file particle multiplicity,
compared between the standard (norm1, unique-mother) and rotated/cloned
(norm42, 42.64x RandomRot) BIB libraries.

Reads directly from the HDF5 stores via h5py hyperslab slicing -- never
loads a full store into RAM (norm42 alone is ~800M+ particles across 6666
files). Multiplicity uses the full per-file offsets diff (cheap, O(n_files)).
Energy/phi are pooled from a random sample of files per store.

Usage (on a NERSC login node, after module load tensorflow):
    python bib_diagnostic_plots.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import h5py
import numpy as np

STORES = {
    "norm1 (standard)": "/pscratch/sd/h/hgreen/mucoll/libtest/stores/gen_norm1_MUPLUS.h5",
    "norm42 (rotated/cloned 42.64x)": "/pscratch/sd/h/hgreen/mucoll/libtest/stores/gen_norm42_MUPLUS.h5",
}

N_SAMPLE_FILES = 150
SEED = 7
OUT_PNG = "bib_diagnostic_plots.png"


def sample_particles(path, n_sample_files, rng):
    with h5py.File(path, "r") as f:
        offsets = f["offsets"][:]
        n_files = len(offsets) - 1
        positions = rng.choice(n_files, size=min(n_sample_files, n_files), replace=False)
        positions.sort()  # ascending order plays nicer with HDF5 slicing
        e_parts, px_parts, py_parts = [], [], []
        for p in positions:
            a, b = offsets[p], offsets[p + 1]
            e_parts.append(f["particles"]["E"][a:b])
            px_parts.append(f["particles"]["px"][a:b])
            py_parts.append(f["particles"]["py"][a:b])
        per_file_mult = np.diff(offsets)  # full store, all n_files
    energy = np.concatenate(e_parts)
    phi = np.arctan2(np.concatenate(py_parts), np.concatenate(px_parts))
    return energy, phi, per_file_mult, len(positions)


def main():
    rng = np.random.default_rng(SEED)
    data = {}
    for label, path in STORES.items():
        energy, phi, mult, n_sampled = sample_particles(path, N_SAMPLE_FILES, rng)
        data[label] = {"energy": energy, "phi": phi, "mult": mult, "n_sampled": n_sampled}
        print(f"{label}: sampled {n_sampled} files, {len(energy)} particles pooled | "
              f"full store: {len(mult)} files, multiplicity min={mult.min()} "
              f"median={int(np.median(mult))} mean={mult.mean():.1f} max={mult.max()}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"norm1 (standard)": "#1f77b4", "norm42 (rotated/cloned 42.64x)": "#d62728"}

    ax = axes[0]
    e_all = np.concatenate([d["energy"] for d in data.values()])
    e_lo, e_hi = np.percentile(e_all[e_all > 0], [0.1, 99.9])
    bins = np.logspace(np.log10(max(e_lo, 1e-6)), np.log10(e_hi), 80)
    for label, d in data.items():
        e = d["energy"]
        e = e[e > 0]
        ax.hist(e, bins=bins, histtype="step", density=True, linewidth=1.8,
                color=colors[label], label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("particle energy E [GeV]")
    ax.set_ylabel("density")
    ax.set_title("Energy distribution")
    ax.legend(fontsize=9)

    ax = axes[1]
    bins = np.linspace(-np.pi, np.pi, 80)
    for label, d in data.items():
        ax.hist(d["phi"], bins=bins, histtype="step", density=True, linewidth=1.8,
                color=colors[label], label=label)
    ax.set_xlabel(r"particle $\phi$ [rad]")
    ax.set_ylabel("density")
    ax.set_title(r"$\phi$ distribution")
    ax.legend(fontsize=9)

    ax = axes[2]
    m_all = np.concatenate([d["mult"] for d in data.values()])
    bins = np.logspace(np.log10(m_all.min()), np.log10(m_all.max()), 60)
    for label, d in data.items():
        ax.hist(d["mult"], bins=bins, histtype="step", density=True, linewidth=1.8,
                color=colors[label], label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("particles per file (per cycle)")
    ax.set_ylabel("density")
    ax.set_title("Per-file particle multiplicity\n(full store, not sampled)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
