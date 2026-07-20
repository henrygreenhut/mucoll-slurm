#!/usr/bin/env python3
"""Measure val AUC's sampling noise empirically, for a frozen checkpoint.

Redraws fresh, INDEPENDENT batches of --val-units units/class from the same
val cycle pool many times, scores each with a frozen (already-trained)
model, and reports the empirical spread of AUC across redraws -- compared
to the theoretical spread a naive i.i.d.-sample formula (Hanley & McNeil
1982) would predict for that many "samples".

If empirical spread ~ naive prediction: units behave close to independent
samples: correlation from cycle-sharing between units isn't a big deal.
If empirical spread >> naive prediction: units are meaningfully correlated
(sharing source cycles with each other), and --val-units alone won't fix
val-AUC noise as efficiently as growing the val cycle pool would.

No training involved -- pure inference on an existing checkpoint, cheap.

Usage:
    python val_noise_diagnostic.py --source-label A0_n420_rawsum_disjoint
"""

import argparse
import json
import os

import numpy as np

import libtest_common as lc
from pfn_libtest_train import PHI_SIZES, F_SIZES, UnitSampler, predict_units


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-label", required=True,
                        help="trained run directory under --outdir to load "
                             "weights/config from (needs best.weights.h5, "
                             "config.json, norm_stats.json)")
    parser.add_argument("--outdir", default="pfn_results")
    parser.add_argument("--val-units", type=int, default=300,
                        help="units per class per redraw (match the config "
                             "being diagnosed)")
    parser.add_argument("--redraws", type=int, default=30,
                        help="number of independent redraws")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default: same as the source run's config")
    parser.add_argument("--seed", type=int, default=99)
    return parser.parse_args()


def hanley_mcneil_se(auc, n1, n2):
    """Standard error of an empirical AUC (Hanley & McNeil 1982), using the
    observed AUC rather than assuming the null (auc=0.5) case -- variance
    is generally smaller near 0/1 than near 0.5, so the null-case formula
    would overstate the naive baseline for a model with real separation."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2)
           + (n2 - 1) * (q2 - auc**2)) / (n1 * n2)
    return float(np.sqrt(max(var, 0.0)))


def main():
    args = parse_args()
    source_dir = os.path.join(args.outdir, args.source_label)
    with open(os.path.join(source_dir, "config.json")) as f:
        config = json.load(f)
    mean, std, latent_scale = lc.load_norm_stats(
        os.path.join(source_dir, "norm_stats.json"))

    n_files = config["n_files"]
    clone_factor = config.get("clone_factor", 42)
    split_fracs = tuple(config.get("split_fracs", (0.5, 0.25, 0.25)))
    null_test = config.get("null_test", False)
    arch = config.get("arch", "local")
    batch_size = args.batch_size or config.get("batch_size", 4)

    store1 = lc.Store(config["norm1_store"])
    store_b = store1 if null_test else lc.Store(config["norm42_store"])
    common, pos1, pos_b = lc.common_positions(store1, store_b)
    splits = lc.split_indices(len(common), split_fracs)
    val_idx = splits["val"]

    files_b = n_files if null_test else n_files // clone_factor
    samplers = [
        UnitSampler(store1, {"val": pos1[val_idx]}, n_files),
        UnitSampler(store_b, {"val": (pos1 if null_test else pos_b)[val_idx]},
                   files_b),
    ]

    if arch == "energyflow":
        model = lc.build_pfn_energyflow(len(mean), phi_sizes=PHI_SIZES,
                                        f_sizes=F_SIZES)
    else:
        model = lc.build_pfn(len(mean), latent_scale, phi_sizes=PHI_SIZES,
                             f_sizes=F_SIZES)
    model.load_weights(os.path.join(source_dir, "best.weights.h5"))

    n_val_cycles = len(val_idx)
    reuse_factor = args.val_units * n_files / n_val_cycles
    print(f"source: {args.source_label} (arch={arch}, n_files={n_files})")
    print(f"val pool: {n_val_cycles} cycles; {args.val_units} units/class/"
          f"redraw -> avg cycle reuse within one redraw: {reuse_factor:.1f}x")
    print(f"{args.redraws} independent redraws, batch_size={batch_size}\n")

    aucs = []
    for r in range(args.redraws):
        rng = np.random.default_rng(args.seed + r)
        defs = [(c, samplers[c].random_unit(rng, "val"))
                for c in (0, 1) for _ in range(args.val_units)]
        y, s = predict_units(model, defs, samplers, mean, std, batch_size)
        auc = lc.auc_score(y, s)
        aucs.append(auc)
        print(f"  redraw {r:2d}: AUC {auc:.4f}", flush=True)

    aucs = np.asarray(aucs)
    empirical_sd = float(aucs.std(ddof=1))
    naive_sd = hanley_mcneil_se(float(aucs.mean()), args.val_units, args.val_units)
    inflation = empirical_sd / naive_sd if naive_sd > 0 else float("nan")

    print(f"\nmean AUC across redraws: {aucs.mean():.4f}")
    print(f"empirical SD across redraws:      {empirical_sd:.4f}")
    print(f"naive i.i.d. SD (Hanley-McNeil):   {naive_sd:.4f}"
          f"  (as if {args.val_units}/class were independent samples)")
    print(f"inflation factor (empirical/naive): {inflation:.2f}x")
    if inflation > 1.5:
        print("=> correlation from cycle-sharing across units is INFLATING"
              " val AUC noise materially. Growing --val-units alone will"
              " have diminishing returns; the val cycle POOL needs to grow"
              " (larger val split) for a proportionate fix.")
    else:
        print("=> empirical noise is close to the naive i.i.d. expectation;"
              " correlation is not the dominant driver of val AUC noise"
              " here -- plain small-N is the main story.")


if __name__ == "__main__":
    main()
