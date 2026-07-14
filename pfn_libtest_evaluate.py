#!/usr/bin/env python3
"""Post-hoc overlapping-event evaluation with paired cycle bootstrap.

The trained PFN and its train/validation/test mother split are frozen.  The
point AUC is estimated from independently constructed, potentially overlapping
events drawn from the held-out test pool.  Target-2 uncertainty is estimated
with a two-level bootstrap: resample matched norm1/norm42 test-cycle pairs,
then regenerate events from that bootstrap pool and score them.

Bootstrap duplicate cycles are retained as distinct pool entries.  An event
samples entries without replacement, so the same empirical cycle can occur
twice when it was selected twice by the outer bootstrap.  This is the usual
nonparametric-bootstrap analogue of drawing new cycles from the population.
"""

import argparse
import csv
import json
import os
import time

import numpy as np

import libtest_common as lc
from pfn_libtest_train import PHI_SIZES, F_SIZES, UnitSampler, predict_units


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


def random_defs(rng, pools, files_per_unit, n_units):
    definitions = []
    for class_id in (0, 1):
        pool = pools[class_id]
        size = files_per_unit[class_id]
        for _ in range(n_units):
            slots = rng.choice(len(pool), size=size, replace=False)
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
    null_partition = cfg(config, "null_partition", "halves")
    if null_test and null_partition != "shared":
        raise SystemExit("paired-cycle evaluation supports the corrected shared-pool null only")

    store1 = lc.Store(config["norm1_store"])
    store_b = store1 if null_test else lc.Store(config["norm42_store"])
    common, pos1, pos_b = lc.common_positions(store1, store_b)
    splits = lc.split_indices(len(common), split_fracs)
    test_indices = splits["test"]
    pool_a = pos1[test_indices]
    pool_b = pos1[test_indices] if null_test else pos_b[test_indices]
    files_b = n_files if null_test else n_files // clone_factor
    files_per_unit = (n_files, files_b)

    # UnitSampler needs only feature/cut fields during evaluation.
    eval_config = argparse.Namespace(
        e_min=float(cfg(config, "e_min", 0.0)),
        t_abs_max=float(cfg(config, "t_abs_max", 0.0)),
        features=cfg(config, "features", "paper"),
        drop_phi=bool(cfg(config, "drop_phi", False)),
    )
    dummy_splits_a = {"test": pool_a}
    dummy_splits_b = {"test": pool_b}
    samplers = [
        UnitSampler(store1, dummy_splits_a, n_files, eval_config),
        UnitSampler(store_b, dummy_splits_b, files_b, eval_config),
    ]
    mean, std, latent_scale = lc.load_norm_stats(stats_path)
    model = lc.build_pfn(len(mean), latent_scale,
                         phi_sizes=PHI_SIZES, f_sizes=F_SIZES)
    model.load_weights(weights)
    batch_size = args.batch_size or int(cfg(config, "batch_size", 8))

    metadata = {
        "label": args.label,
        "source_label": args.source_label,
        "n_files": n_files,
        "clone_factor": clone_factor,
        "null_test": null_test,
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
        bootstrap_pools = (pool_a[slots], pool_b[slots])
        definitions = random_defs(
            rng, bootstrap_pools, files_per_unit, args.bootstrap_units)
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
