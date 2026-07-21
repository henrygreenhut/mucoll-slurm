"""BIB diagnostic plots: energy, phi, and per-file particle multiplicity,
compared between the standard (norm1, unique-mother) and rotated/cloned
(norm42, 42x RandomRot) BIB libraries.

Reads directly from the HDF5 stores via h5py hyperslab slicing -- never
loads a full store into RAM (norm42 alone is ~800M+ particles across 6666
files). Multiplicity uses the full per-file offsets diff (cheap, O(n_files)).

Energy/phi are pooled from as many randomly-ordered files as needed to
reach N_TARGET_PARTICLES per store -- the SAME particle count for both,
not the same file count. Caveat: norm1 and norm42 are paired 1:1 over the
same 6666 physical mother-events, and norm42 files are ~42x bigger (same
mother cloned), so matching particle count means norm42's sample draws
from far fewer independent files than norm1's -- this understates rare/
tail features (e.g. norm42's true high-energy tail, confirmed present via
a full-store scan, may not appear in this sample). Multiplicity (panels
3-4) is unaffected -- it always uses the full store, not this sample.

Usage (on a NERSC login node, after module load tensorflow):
    python bib_diagnostic_plots.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import h5py
import numpy as np

STORES = {
    "norm1 (standard)": "/pscratch/sd/h/hgreen/mucoll/libtest/stores/gen_norm1_MUPLUS.h5",
    "norm42 (rotated/cloned 42x)": "/pscratch/sd/h/hgreen/mucoll/libtest/stores/gen_norm42_MUPLUS.h5",
}

N_TARGET_PARTICLES = 3_000_000
SEED = 7
CLONE_FACTOR = 42
OUT_PNG = "plots/bib_diagnostic_plots.png"

# Current training config (config #1 baseline: phi_sizes=(200,200,256),
# batch_size=4, n_files=420, clone_factor=42). The TF/XLA int32 overflow
# bug requires batch_size * N * widest_Phi_layer_width < 2**31, where N is
# the padded sequence length of whichever single unit is largest in a
# batch -- F-layers only see the pooled (batch, width) tensor after the
# sum, so only Phi's widest layer matters, not F's. A norm42 ("classB")
# unit concatenates n_files // clone_factor = 10 files, so a per-file cap
# X is worst-case-safe (guaranteed, even if all 10 files in a unit hit the
# cap simultaneously) when batch_size * width * 10 * X < 2**31.
BATCH_SIZE = 4
WIDEST_PHI_LAYER = 256
N_FILES_PER_UNIT_NORM42 = 420 // CLONE_FACTOR
INT32_CEILING = 2**31 - 1
X_CAP = INT32_CEILING // (BATCH_SIZE * WIDEST_PHI_LAYER * N_FILES_PER_UNIT_NORM42)


def sample_particles(path, n_target_particles, rng):
    """Sample as many randomly-ordered files as needed to pool roughly
    n_target_particles from this store -- equal PARTICLE count between
    stores when called with the same target, not equal file count."""
    with h5py.File(path, "r") as f:
        offsets = f["offsets"][:]
        n_files = len(offsets) - 1
        per_file = np.diff(offsets)
        order = rng.permutation(n_files)
        cum = np.cumsum(per_file[order])
        n_needed = min(int(np.searchsorted(cum, n_target_particles) + 1), n_files)
        positions = np.sort(order[:n_needed])
        e_parts, px_parts, py_parts = [], [], []
        for p in positions:
            a, b = offsets[p], offsets[p + 1]
            e_parts.append(f["particles"]["E"][a:b])
            px_parts.append(f["particles"]["px"][a:b])
            py_parts.append(f["particles"]["py"][a:b])
    energy = np.concatenate(e_parts)
    phi = np.arctan2(np.concatenate(py_parts), np.concatenate(px_parts))
    return energy, phi, len(positions)


def full_store_multiplicity(path):
    """Per-file particle counts for the ENTIRE store (all 6666 files) --
    cheap, since it only needs the offsets array, not particle-level I/O.
    Independent of the energy/phi sampling above."""
    with h5py.File(path, "r") as f:
        offsets = f["offsets"][:]
    return np.diff(offsets)


def sample_excluded_energy(path, x_cap, n_sample_files, rng):
    """Energy of particles living in files that exceed x_cap -- the
    particles a per-file cap at x_cap would force us to disclude. Samples
    up to n_sample_files of the over-cap files (there may be far too many
    over-cap particles to pool all of them), reporting how many exist and
    how many were actually sampled."""
    with h5py.File(path, "r") as f:
        offsets = f["offsets"][:]
        per_file = np.diff(offsets)
        over_cap_positions = np.flatnonzero(per_file > x_cap)
        n_over_files = len(over_cap_positions)
        n_over_particles = int(per_file[over_cap_positions].sum())
        chosen = rng.choice(over_cap_positions,
                            size=min(n_sample_files, n_over_files), replace=False)
        chosen.sort()
        e_parts = [f["particles"]["E"][offsets[p]:offsets[p + 1]] for p in chosen]
    energy = np.concatenate(e_parts) if e_parts else np.array([])
    return energy, n_over_files, n_over_particles, len(chosen)


def main():
    rng = np.random.default_rng(SEED)
    data = {}
    for label, path in STORES.items():
        energy, phi, n_sampled = sample_particles(path, N_TARGET_PARTICLES, rng)
        mult = full_store_multiplicity(path)
        data[label] = {"energy": energy, "phi": phi, "mult": mult, "n_sampled": n_sampled}
        print(f"{label}: sampled {n_sampled} files -> {len(energy)} particles pooled "
              f"(target {N_TARGET_PARTICLES}) | full store: {len(mult)} files, "
              f"multiplicity min={mult.min()} median={int(np.median(mult))} "
              f"mean={mult.mean():.1f} max={mult.max()}")

    print(f"int32-safe per-file cap (batch={BATCH_SIZE}, phi_width={WIDEST_PHI_LAYER}, "
          f"{N_FILES_PER_UNIT_NORM42} files/norm42-unit): X_CAP={X_CAP} particles/file")

    fig, axes = plt.subplots(1, 6, figsize=(36, 5))
    colors = {"norm1 (standard)": "#1f77b4", "norm42 (rotated/cloned 42x)": "#d62728"}

    ax = axes[0]
    e_all = np.concatenate([d["energy"] for d in data.values()])
    e_pos = e_all[e_all > 0]
    e_lo, e_hi = e_pos.min(), e_pos.max()
    print(f"pooled energy range: min={e_lo:.6e} GeV max={e_hi:.6e} GeV")
    bins = np.logspace(np.log10(e_lo), np.log10(e_hi), 80)
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

    ax = axes[3]
    norm1_mult = data["norm1 (standard)"]["mult"]
    norm42_scaled = data["norm42 (rotated/cloned 42x)"]["mult"] / CLONE_FACTOR
    both = np.concatenate([norm1_mult, norm42_scaled])
    bins = np.logspace(np.log10(both.min()), np.log10(both.max()), 60)
    ax.hist(norm1_mult, bins=bins, histtype="step", density=True, linewidth=1.8,
            color=colors["norm1 (standard)"], label="norm1 (standard)")
    ax.hist(norm42_scaled, bins=bins, histtype="step", density=True, linewidth=1.8,
            color=colors["norm42 (rotated/cloned 42x)"],
            label=f"norm42 / {CLONE_FACTOR}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"particles per file (norm42 divided by {CLONE_FACTOR})")
    ax.set_ylabel("density")
    ax.set_title("Multiplicity, norm42 normalized to\nper-mother scale (full store)")
    ax.legend(fontsize=9)

    ax = axes[4]
    mult_bins = np.logspace(np.log10(m_all.min()), np.log10(m_all.max()), 60)
    for label, d in data.items():
        ax.hist(d["mult"], bins=mult_bins, histtype="step", density=True,
                linewidth=1.8, color=colors[label], label=label)
    ax.axvline(X_CAP, color="black", linestyle="--", linewidth=1.5,
               label=f"int32-safe cap = {X_CAP:,}/file")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("particles per file (per cycle)")
    ax.set_ylabel("density")
    ax.set_title(f"Multiplicity with int32-safe cap\n(batch={BATCH_SIZE}, "
                 f"phi_width={WIDEST_PHI_LAYER}, current config)")
    ax.legend(fontsize=8)

    ax = axes[5]
    norm42_path = STORES["norm42 (rotated/cloned 42x)"]
    excl_energy, n_over_files, n_over_particles, n_files_sampled = \
        sample_excluded_energy(norm42_path, X_CAP, 300, rng)
    n_total_files = len(data["norm42 (rotated/cloned 42x)"]["mult"])
    n_total_particles = int(data["norm42 (rotated/cloned 42x)"]["mult"].sum())
    print(f"excluded (over-cap) norm42 files: {n_over_files}/{n_total_files} "
          f"({100*n_over_files/n_total_files:.2f}%) | excluded particles: "
          f"{n_over_particles}/{n_total_particles} "
          f"({100*n_over_particles/n_total_particles:.2f}%) | "
          f"sampled {n_files_sampled} of those files -> {len(excl_energy)} "
          f"particles pooled for this panel")
    excl_pos = excl_energy[excl_energy > 0]
    bins = np.logspace(np.log10(excl_pos.min()), np.log10(excl_pos.max()), 80)
    ax.hist(excl_pos, bins=bins, histtype="step", density=True, linewidth=1.8,
            color="#d62728")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("particle energy E [GeV]")
    ax.set_ylabel("density")
    ax.set_title(f"Energy of particles in norm42 files\nover the cap "
                 f"({100*n_over_particles/n_total_particles:.1f}% of norm42)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
