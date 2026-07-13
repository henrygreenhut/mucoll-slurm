#!/usr/bin/env python3
"""No-ML sanity check: is near-perfect unique/reuse separation possible?

Clones are identical in everything but phi, so in a reuse unit every
distinct pz value appears ~42 times, while in a unique unit values are
essentially unrepeated. This script computes, per unit, the maximum
repetition count of any single pz value ("max duplicate multiplicity") and
the fraction of particles whose pz appears more than once. If the two
classes separate perfectly on these hand-computed statistics, then AUC ~ 1
from the PFN is expected, not suspicious -- the information is in the data
by construction, independent of any ML pipeline.

Runs in the mucoll-inspect env (numpy + h5py), minutes on a login node:

    python libtest_duplicate_check.py
"""

import argparse
import os

import numpy as np

import libtest_common as lc


def parse_args():
    scratch = os.environ.get("PSCRATCH", ".")
    store_dir = os.path.join(scratch, "mucoll/libtest/stores")
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm1-store", default=os.path.join(store_dir, "gen_norm1_MUPLUS.h5"))
    parser.add_argument("--norm42-store", default=os.path.join(store_dir, "gen_norm42_MUPLUS.h5"))
    parser.add_argument("--n-files", type=int, default=42)
    parser.add_argument("--clone-factor", type=int, default=42)
    parser.add_argument("--units", type=int, default=100, help="per class")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def unit_duplicate_stats(store, positions, n_files, n_units, rng):
    max_mult, dup_frac = [], []
    for _ in range(n_units):
        pos = lc.sample_unit_positions(rng, positions, n_files)
        pz = store.file_arrays(pos)["pz"]
        _, counts = np.unique(pz, return_counts=True)
        max_mult.append(int(counts.max()))
        dup_frac.append(float(counts[counts > 1].sum() / len(pz)))
    return np.asarray(max_mult), np.asarray(dup_frac)


def main():
    args = parse_args()
    store1 = lc.Store(args.norm1_store)
    store42 = lc.Store(args.norm42_store)
    common, pos1, pos42 = lc.common_positions(store1, store42)
    splits = lc.split_indices(len(common))
    sel = splits[args.split]
    rng = np.random.default_rng(args.seed)

    print(f"{args.units} units/class at n={args.n_files},"
          f" '{args.split}' cycles, statistic = repetition of stored pz values")
    mm_u, df_u = unit_duplicate_stats(store1, pos1[sel], args.n_files,
                                      args.units, rng)
    mm_r, df_r = unit_duplicate_stats(store42, pos42[sel],
                                      args.n_files // args.clone_factor,
                                      args.units, rng)

    for name, u, r, fmt in [
        ("max multiplicity of one pz value", mm_u, mm_r, "d"),
        ("fraction of particles with duplicated pz", df_u, df_r, ".4f"),
    ]:
        print(f"\n{name}:")
        print(f"  unique: min {u.min():{fmt}} | median {np.median(u):.2f}"
              f" | max {u.max():{fmt}}")
        print(f"  reuse : min {r.min():{fmt}} | median {np.median(r):.2f}"
              f" | max {r.max():{fmt}}")
        y = np.concatenate([np.zeros(len(u)), np.ones(len(r))])
        auc = lc.auc_score(y, np.concatenate([u, r]).astype(float))
        print(f"  AUC of this single hand-made statistic: {auc:.4f}")

    print("\nReading: clones duplicate pz exactly (clone factor ~42), unique"
          "\nmothers don't. If these AUCs are ~1.0, perfect class separation is"
          "\npresent in the raw data by construction -- a near-1.0 PFN AUC is"
          "\nexpected, not evidence of a pipeline leak. (The leak test proper"
          "\nis the norm1-vs-norm1 null run.)")


if __name__ == "__main__":
    main()
