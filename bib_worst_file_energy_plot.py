"""Energy distribution of the single norm42 file that drives the global
int32-safe energy cut: the max-multiplicity file (989,226 particles,
store position 6291), which needs the hardest per-file cut (0.940 GeV) of
any of the 612 over-cap files to bring its count down to X_CAP=209,715.

Usage (on a NERSC login node, after module load tensorflow):
    python bib_worst_file_energy_plot.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import h5py
import numpy as np

NORM42_STORE = "/pscratch/sd/h/hgreen/mucoll/libtest/stores/gen_norm42_MUPLUS.h5"
FILE_POSITION = 6291
X_CAP = 209715
E_CUT = 0.9404417  # exact per-file cut computed for this file
OUT_PNG = "plots/bib_worst_file_energy_plot.png"


def main():
    with h5py.File(NORM42_STORE, "r") as f:
        offsets = f["offsets"][:]
        cycle_id = int(f["cycle_ids"][FILE_POSITION])
        a, b = offsets[FILE_POSITION], offsets[FILE_POSITION + 1]
        n_particles = int(b - a)
        e = f["particles"]["E"][a:b]

    n_above_cut = int(np.sum(e > E_CUT))
    print(f"file position {FILE_POSITION} (cycle_id={cycle_id}): "
          f"{n_particles} particles | E range [{e.min():.6e}, {e.max():.6e}] GeV")
    print(f"applying E > {E_CUT} GeV keeps {n_above_cut} particles "
          f"(target <= {X_CAP})")

    fig, ax = plt.subplots(figsize=(8, 6))
    e_pos = e[e > 0]
    bins = np.logspace(np.log10(e_pos.min()), np.log10(e_pos.max()), 80)
    ax.hist(e_pos, bins=bins, histtype="step", linewidth=1.8, color="#d62728",
            label=f"file position {FILE_POSITION} (cycle_id={cycle_id})\n"
                  f"{n_particles} particles")
    ax.axvline(E_CUT, color="black", linestyle="--", linewidth=1.5,
               label=f"required cut E > {E_CUT:.4f} GeV\n"
                     f"(keeps {n_above_cut}/{n_particles} <= {X_CAP})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("particle energy E [GeV]")
    ax.set_ylabel("count")
    ax.set_title("Energy distribution of the single worst-case norm42 file\n"
                 "(drives the global int32-safe energy cut)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
