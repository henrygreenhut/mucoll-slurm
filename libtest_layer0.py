#!/usr/bin/env python3
"""Layer-0 fluctuation check: no ML, just count statistics per unit.

For matched-mother units (n norm1 files vs n/42 norm42 files), compares the
distributions of particles-per-unit N and sum(E)-per-unit between the two
classes. Cloning predicts identical means but Var_reuse/Var_unique ~ 42
(std ratio ~ sqrt(42) ~ 6.5) -- seeing that ratio validates both the physics
expectation and the unit-building pipeline before any training.

Runs in the mucoll-inspect env (numpy, h5py; matplotlib optional):

    python libtest_layer0.py \
        --norm1-store  $PSCRATCH/mucoll/libtest/stores/gen_norm1_MUPLUS.h5 \
        --norm42-store $PSCRATCH/mucoll/libtest/stores/gen_norm42_MUPLUS.h5
"""

import argparse
import csv
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
    parser.add_argument("--units", type=int, default=500, help="per class")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--e-min", type=float, default=0.0)
    parser.add_argument("--t-abs-max", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", default="libtest_layer0")
    return parser.parse_args()


def unit_stats(store, positions, n_files, n_units, rng, args):
    counts, sum_e = [], []
    for _ in range(n_units):
        pos = lc.sample_unit_positions(rng, positions, n_files)
        raw = store.file_arrays(pos)
        raw = lc.apply_cuts(raw, args.e_min, args.t_abs_max)
        counts.append(len(raw["E"]))
        sum_e.append(float(raw["E"].sum()))
    return np.asarray(counts, dtype=np.float64), np.asarray(sum_e)


def report(name, unique, reuse, clone_factor):
    var_ratio = reuse.var() / max(unique.var(), 1e-12)
    print(f"\n{name}:")
    print(f"  unique: mean {unique.mean():,.1f} | std {unique.std():,.1f}")
    print(f"  reuse : mean {reuse.mean():,.1f} | std {reuse.std():,.1f}")
    print(f"  mean ratio {reuse.mean() / max(unique.mean(), 1e-12):.3f} (expect ~1)")
    print(f"  variance ratio {var_ratio:.1f} (expect ~{clone_factor})"
          f" | std ratio {np.sqrt(var_ratio):.2f}"
          f" (expect ~{np.sqrt(clone_factor):.2f})")
    return var_ratio


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.n_files % args.clone_factor != 0:
        raise SystemExit("--n-files must be a multiple of --clone-factor")

    store1 = lc.Store(args.norm1_store)
    store42 = lc.Store(args.norm42_store)
    common, pos1, pos42 = lc.common_positions(store1, store42)
    splits = lc.split_indices(len(common))
    sel = splits[args.split]
    print(f"paired cycles: {len(common)}; using {len(sel)} '{args.split}' cycles;"
          f" {args.units} units/class at n={args.n_files}"
          f" (cuts: E>={args.e_min}, |t|<{args.t_abs_max or 'inf'})")

    rng = np.random.default_rng(args.seed)
    n_u, e_u = unit_stats(store1, pos1[sel], args.n_files, args.units, rng, args)
    n_r, e_r = unit_stats(store42, pos42[sel], args.n_files // args.clone_factor,
                          args.units, rng, args)

    report("particles per unit (N)", n_u, n_r, args.clone_factor)
    report("sum E per unit [GeV]", e_u, e_r, args.clone_factor)

    csv_path = os.path.join(args.outdir, f"units_n{args.n_files}_{args.split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "n_particles", "sum_E"])
        for n, e in zip(n_u, e_u):
            writer.writerow(["unique", int(n), f"{e:.3f}"])
        for n, e in zip(n_r, e_r):
            writer.writerow(["reuse", int(n), f"{e:.3f}"])
    print(f"\nper-unit values -> {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), tight_layout=True)
        for ax, (u, r, title) in zip(axes, [
            (n_u, n_r, "particles / unit"),
            (e_u, e_r, "sum E / unit [GeV]"),
        ]):
            lo = min(u.min(), r.min())
            hi = max(u.max(), r.max())
            bins = np.linspace(lo, hi, 40)
            ax.hist(u, bins=bins, histtype="step", label="unique (norm1)")
            ax.hist(r, bins=bins, histtype="step", label="reuse (norm42)")
            ax.set_xlabel(title)
            ax.set_ylabel("units")
            ax.legend(frameon=False, fontsize=8)
        pdf_path = os.path.join(args.outdir, f"layer0_n{args.n_files}_{args.split}.pdf")
        fig.savefig(pdf_path)
        print(f"histograms -> {pdf_path}")
    except ImportError:
        print("matplotlib not available; skipped plots")


if __name__ == "__main__":
    main()
