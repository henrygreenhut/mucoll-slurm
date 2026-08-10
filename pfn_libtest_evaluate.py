#!/usr/bin/env python3
"""Post-hoc overlapping-event evaluation with paired cycle bootstrap.

The trained PFN and its train/validation/test mother split are frozen.  The
point AUC is estimated from independently constructed, potentially overlapping
events drawn from the held-out test pool.  Target-2 uncertainty is estimated
with a two-level bootstrap: resample matched norm1/norm42 test-cycle pairs,
then regenerate events from that bootstrap pool and score them.

The outer bootstrap is represented as multiplicity weights on the original
cycle IDs. Events sample distinct physical cycles without replacement, with
probability proportional to those weights. Thus the bootstrap represents
source uncertainty without introducing forbidden within-event duplicates.
"""

import argparse
import csv
import json
import os
import time

import numpy as np

import libtest_common as lc
from pfn_libtest_train import (
    PHI_SIZES, F_SIZES, UnitSampler, class_layout, predict_units)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True,
                        help="new evaluation label; source files are untouched")
    parser.add_argument("--source-label", required=True,
                        help="trained model label below --outdir")
    parser.add_argument("--outdir", default="pfn_results")
    parser.add_argument("--point-units", type=int, default=300,
                        help="overlapping held-out events per class for point AUC")
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--bootstrap-units", type=int, default=100,
                        help="regenerated events per class in each bootstrap pool")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--batch-size", type=int,
                        help="override source model evaluation batch size")
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="checkpoint between bootstrap replicates (0 = off)")
    return parser.parse_args()


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def source_config(source_dir):
    path = os.path.join(source_dir, "config.json")
    if os.path.isfile(path):
        return load_json(path)
    summary = load_json(os.path.join(source_dir, "auc_summary.json"))
    return summary["config"]


def cfg(config, key, default):
    value = config.get(key, default)
    return default if value is None else value


def append_bootstrap(path, replicate, auc):
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["replicate", "auc"])
        writer.writerow([replicate, "{:.12g}".format(auc)])


def load_bootstrap(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as handle:
        return [float(row["auc"]) for row in csv.DictReader(handle)]


def random_defs(rng, pools, files_per_unit, n_units, weights=None):
    definitions = []
    for class_id in (0, 1):
        pool = pools[class_id]
        size = files_per_unit[class_id]
        for _ in range(n_units):
            probability = None if weights is None else weights / weights.sum()
            if weights is not None and np.count_nonzero(weights) < size:
                raise ValueError("bootstrap has too few distinct cycles for a unit")
            slots = rng.choice(len(pool), size=size, replace=False, p=probability)
            definitions.append((class_id, pool[slots]))
    return definitions


def write_scores(path, labels, scores):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "score"])
        for label, score in zip(labels, scores):
            writer.writerow([int(label), "{:.12g}".format(score)])


def main():
    args = parse_args()
    if min(args.point_units, args.bootstrap_reps, args.bootstrap_units) < 1:
        raise SystemExit("point units, bootstrap reps, and bootstrap units must be positive")
    start_time = time.time()

    source_dir = os.path.join(args.outdir, args.source_label)
    eval_dir = os.path.join(args.outdir, args.label)
    os.makedirs(eval_dir, exist_ok=True)
    config = source_config(source_dir)
    weights = os.path.join(source_dir, "best.weights.h5")
    stats_path = os.path.join(source_dir, "norm_stats.json")
    if not os.path.isfile(weights) or not os.path.isfile(stats_path):
        raise SystemExit("source model is incomplete: {}".format(source_dir))

    n_files = int(cfg(config, "n_files", 42))
    clone_factor = int(cfg(config, "clone_factor", 42))
    split_fracs = tuple(cfg(config, "split_fracs", (0.60, 0.15, 0.25)))
    null_test = bool(cfg(config, "null_test", False))
    null_source = cfg(config, "null_source", "unique")
    if null_test and cfg(config, "null_partition", "shared") != "shared":
        raise SystemExit("paired-cycle evaluation requires a shared-pool null")

    store_unique = lc.Store(config["norm1_store"])
    store_reuse = lc.Store(config["norm42_store"])
    store_a, store_b, files_a, files_b = class_layout(
        store_unique, store_reuse, n_files, clone_factor,
        null_test, null_source)
    common, pos_a, pos_b = lc.common_positions(store_a, store_b)
    split_path = os.path.join(source_dir, "source_split.npz")
    if os.path.isfile(split_path):
        cycle_split = lc.load_or_create_cycle_split(
            split_path, common, split_fracs,
            int(cfg(config, "data_seed", cfg(config, "seed", 1))))
        splits = lc.cycle_split_positions(common, cycle_split)
    else:
        splits = lc.split_indices(len(common), split_fracs)
    test_indices = splits["test"]
    pool_a = pos_a[test_indices]
    pool_b = pos_b[test_indices]
    files_per_unit = (files_a, files_b)

    dummy_splits_a = {"test": pool_a}
    dummy_splits_b = {"test": pool_b}
    samplers = [
        UnitSampler(store_a, dummy_splits_a, files_a,
                    cfg(config, "features", "paper"),
                    cfg(config, "exclude_muons_above_gev", 0.0)),
        UnitSampler(store_b, dummy_splits_b, files_b,
                    cfg(config, "features", "paper"),
                    cfg(config, "exclude_muons_above_gev", 0.0)),
    ]
    mean, std, latent_scale = lc.load_norm_stats(stats_path)
    # Read the actual architecture used for this checkpoint, not the
    # module's current defaults -- a hyperparameter scan produces
    # checkpoints with varying Phi/F sizes.
    phi_sizes = tuple(cfg(config, "phi_sizes", PHI_SIZES))
    f_sizes = tuple(cfg(config, "f_sizes", F_SIZES))
    arch = cfg(config, "arch", "local")
    if arch == "energyflow":
        if latent_scale == 1.0:
            model = lc.build_pfn_energyflow(len(mean), phi_sizes=phi_sizes,
                                            f_sizes=f_sizes)
        else:
            model = lc.build_pfn_energyflow_scaled(
                len(mean), latent_scale, phi_sizes=phi_sizes, f_sizes=f_sizes)
    else:
        model = lc.build_pfn(len(mean), latent_scale,
                             phi_sizes=phi_sizes, f_sizes=f_sizes)
    model.load_weights(weights)
    batch_size = args.batch_size or int(cfg(config, "batch_size", 8))

    metadata = {
        "label": args.label,
        "source_label": args.source_label,
        "n_files": n_files,
        "clone_factor": clone_factor,
        "null_test": null_test,
        "null_source": null_source,
        "exclude_muons_above_gev": cfg(
            config, "exclude_muons_above_gev", 0.0),
        "split_fracs": split_fracs,
        "n_paired_test_cycles": int(len(test_indices)),
        "point_units_per_class": args.point_units,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_units_per_class": args.bootstrap_units,
        "seed": args.seed,
        "inference": "frozen best.weights.h5",
    }
    with open(os.path.join(eval_dir, "evaluation_config.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    point_path = os.path.join(eval_dir, "point_summary.json")
    if os.path.isfile(point_path):
        point = load_json(point_path)
    else:
        print("point estimate: {} overlapping events/class".format(args.point_units),
              flush=True)
        rng = np.random.default_rng(args.seed)
        definitions = random_defs(
            rng, (pool_a, pool_b), files_per_unit, args.point_units)
        y_point, s_point = predict_units(
            model, definitions, samplers, mean, std, batch_size)
        point = {
            "auc": lc.auc_score(y_point, s_point),
            "score_std": float(np.std(s_point)),
            "score_range": float(np.ptp(s_point)),
        }
        write_scores(os.path.join(eval_dir, "test_scores.csv"),
                     y_point, s_point)
        with open(point_path, "w") as handle:
            json.dump(point, handle, indent=2)
        print("point AUC = {:.6f}".format(point["auc"]), flush=True)

    bootstrap_path = os.path.join(eval_dir, "paired_cycle_bootstrap.csv")
    values = load_bootstrap(bootstrap_path)
    for replicate in range(len(values), args.bootstrap_reps):
        rng = np.random.default_rng(args.seed + 1000003 * (replicate + 1))
        # Jointly resample indices of matched norm1/norm42 cycle pairs.
        slots = rng.integers(0, len(test_indices), size=len(test_indices))
        weights = np.bincount(slots, minlength=len(test_indices)).astype(float)
        definitions = random_defs(
            rng, (pool_a, pool_b), files_per_unit, args.bootstrap_units,
            weights=weights)
        y_boot, s_boot = predict_units(
            model, definitions, samplers, mean, std, batch_size)
        auc = lc.auc_score(y_boot, s_boot)
        append_bootstrap(bootstrap_path, replicate, auc)
        values.append(auc)
        print("bootstrap {}/{}: AUC {:.6f}".format(
            replicate + 1, args.bootstrap_reps, auc), flush=True)
        if (args.max_minutes > 0 and
                (time.time() - start_time) / 60.0 > args.max_minutes):
            print("wall-clock limit reached; bootstrap checkpoint saved",
                  flush=True)
            return

    values = np.asarray(values, dtype=np.float64)
    near_constant = point["score_std"] < 1e-3
    summary = {
        "label": args.label,
        "source_label": args.source_label,
        "test_auc": point["auc"],
        "bootstrap_mean": float(np.mean(values)),
        "bootstrap_std": float(np.std(values, ddof=1)),
        "bootstrap_ci68": np.percentile(values, [16, 84]).tolist(),
        "bootstrap_ci95": np.percentile(values, [2.5, 97.5]).tolist(),
        "best_val_auc": None,
        "best_epoch": None,
        "epochs_run": None,
        "n_test_units": 2 * args.point_units,
        "test_mode": "overlapping-paired-cycle-bootstrap",
        "test_units_mutually_disjoint": False,
        "test_score_std": point["score_std"],
        "test_score_range": point["score_range"],
        "near_constant_test_scores": near_constant,
        "uncertainty_note": (
            "two-level nonparametric bootstrap over matched test-cycle pairs; "
            "events regenerated within each bootstrap pool; frozen classifier"),
        "config": metadata,
    }
    with open(os.path.join(eval_dir, "auc_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print("AUC {:.6f}; paired-cycle bootstrap SD {:.6f}; 95% CI [{:.6f}, {:.6f}]"
          .format(point["auc"], summary["bootstrap_std"],
                  summary["bootstrap_ci95"][0],
                  summary["bootstrap_ci95"][1]), flush=True)


if __name__ == "__main__":
    main()
