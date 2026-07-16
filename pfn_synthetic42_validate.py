#!/usr/bin/env python3
"""PFN validation: on-the-fly synthetic k=42 versus production norm42 GEN."""

import argparse
import csv
import json
import os
import time

import numpy as np

import libtest_common as lc
from variable_reuse_common import MotherStore


PHI_SIZES = (200, 200, 256)
F_SIZES = (200, 200, 200)


def parse_args():
    scratch = os.environ.get("PSCRATCH", ".")
    stores = os.path.join(scratch, "mucoll/libtest/stores")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mother-store", default=os.path.join(
        stores, "gen_split_mothers_MUPLUS.h5"))
    parser.add_argument("--original-store", default=os.path.join(
        stores, "gen_norm42_MUPLUS.h5"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", default="synthetic42_results")
    parser.add_argument("--cycles-per-unit", type=int, default=10,
                        help="10 cycles matches the reuse side of old N=420")
    parser.add_argument("--clone-factor", type=int, default=42)
    parser.add_argument("--units-per-epoch", type=int, default=20,
                        help="paired units per class and epoch")
    parser.add_argument("--val-units", type=int, default=10,
                        help="fixed paired validation units per class")
    parser.add_argument("--test-units", type=int, default=30,
                        help="cycle-disjoint paired test units per class")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1,
                        help="cycle splits and pseudo-event sampling seed")
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--split-fracs", type=float, nargs=3,
                        default=[0.5, 0.25, 0.25])
    parser.add_argument("--features", choices=sorted(lc.FEATURE_SETS),
                        default="rotation",
                        help="rotation includes momentum and vertex azimuth")
    parser.add_argument("--drop-phi", action="store_true")
    parser.add_argument("--e-min", type=float, default=0.0)
    parser.add_argument("--t-abs-max", type=float, default=0.0)
    parser.add_argument("--latent-scale", default="auto")
    parser.add_argument("--norm-stat-pairs", type=int, default=1)
    parser.add_argument("--norm-particles-per-unit", type=int, default=100000)
    parser.add_argument("--null-test", action="store_true",
                        help="synthetic42 versus independent synthetic42")
    parser.add_argument("--audit-cycles", type=int, default=3,
                        help="matched cycles checked before training")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-minutes", type=float, default=25.0)
    return parser.parse_args()


def save_json(path, value):
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2)


def load_state(path):
    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)
    return {"epoch": 0, "best_val_auc": -1.0, "best_epoch": -1,
            "done": False}


def append_history(path, row):
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "epoch", "train_loss", "val_auc", "seconds"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def common_cycle_splits(mother_store, original_store, fractions, seed):
    fractions = np.asarray(fractions, dtype=np.float64)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("split fractions must be positive and sum to one")
    common = np.intersect1d(mother_store.cycle_ids, original_store.cycle_ids)
    order = np.random.default_rng(seed).permutation(common)
    n_train = int(round(len(order) * fractions[0]))
    n_val = int(round(len(order) * fractions[1]))
    return common, {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }


def paired_definitions(rng, cycle_pool, n_units, cycles_per_unit,
                       null_test=False, disjoint=False):
    """Make matched class pairs using identical cycles in both classes."""
    if disjoint:
        needed = n_units * cycles_per_unit
        if needed > len(cycle_pool):
            raise ValueError("{} test units require {} cycles; pool has {}"
                             .format(n_units, needed, len(cycle_pool)))
        selected = rng.permutation(cycle_pool)[:needed]
        groups = selected.reshape(n_units, cycles_per_unit)
    else:
        if cycles_per_unit > len(cycle_pool):
            raise ValueError("unit requires more cycles than its split contains")
        groups = [rng.choice(cycle_pool, cycles_per_unit, replace=False)
                  for _ in range(n_units)]

    definitions = []
    max_seed = np.iinfo(np.int64).max
    for pair_id, cycles in enumerate(groups):
        definitions.append((0, pair_id, {
            "kind": "synthetic", "cycles": np.asarray(cycles),
            "angle_seed": int(rng.integers(0, max_seed))}))
        definitions.append((1, pair_id, {
            "kind": "synthetic" if null_test else "original",
            "cycles": np.asarray(cycles),
            "angle_seed": int(rng.integers(0, max_seed)) if null_test else 0}))
    return definitions


class PairedSampler:
    def __init__(self, mother_store, original_store, args):
        self.mothers = mother_store
        self.original = original_store
        self.args = args

    def raw(self, definition):
        cycles = definition["cycles"]
        if definition["kind"] == "original":
            positions = np.searchsorted(self.original.cycle_ids, cycles)
            valid = positions < self.original.n_files
            valid[valid] &= self.original.cycle_ids[positions[valid]] == cycles[valid]
            if not np.all(valid):
                raise KeyError("cycle absent from original norm42 store")
            return self.original.file_arrays(positions)

        mothers = self.mothers.mothers_for_cycles(cycles)
        rng = np.random.default_rng(definition["angle_seed"])
        angles = rng.uniform(0.0, 2.0 * np.pi,
                             size=(len(mothers), self.args.clone_factor))
        return self.mothers.rotated_mothers(mothers, angles)

    def features(self, definition):
        raw = lc.apply_cuts(self.raw(definition), self.args.e_min,
                            self.args.t_abs_max)
        return lc.build_features(raw, self.args.features, self.args.drop_phi)


def audit_pairing(sampler, cycle_ids, n_cycles):
    """Require synthetic and original rotation-invariant content to match."""
    checked = min(n_cycles, len(cycle_ids))
    for offset, cycle in enumerate(cycle_ids[:checked]):
        synthetic = sampler.raw({
            "kind": "synthetic", "cycles": np.asarray([cycle]),
            "angle_seed": 20260716 + offset})
        original = sampler.raw({
            "kind": "original", "cycles": np.asarray([cycle]),
            "angle_seed": 0})
        if len(synthetic["pdg"]) != len(original["pdg"]):
            raise RuntimeError("cycle {} particle count differs: synthetic {} original {}"
                               .format(cycle, len(synthetic["pdg"]),
                                       len(original["pdg"])))
        for key in ("pz", "t", "vz", "pdg"):
            if not np.array_equal(np.sort(synthetic[key]), np.sort(original[key])):
                raise RuntimeError("cycle {} differs in rotation-invariant {}"
                                   .format(cycle, key))
    print("  pairing audit: {}/{} matched cycles passed particle-count and "
          "(pz,t,vz,pdg) checks".format(checked, checked))


def batches(definitions, sampler, mean, std, batch_size, rng=None):
    order = np.arange(len(definitions))
    if rng is not None:
        rng.shuffle(order)
    for first in range(0, len(order), batch_size):
        chosen = [definitions[index] for index in order[first:first + batch_size]]
        features = [sampler.features(definition) for _, _, definition in chosen]
        max_particles = max(len(values) for values in features)
        x = np.zeros((len(features), max_particles, len(mean)), dtype=np.float32)
        y = np.zeros((len(features), 2), dtype=np.float32)
        labels = []
        pair_ids = []
        for index, ((label, pair_id, _), values) in enumerate(zip(chosen, features)):
            x[index, :len(values)] = (values - mean) / std
            y[index, label] = 1.0
            labels.append(label)
            pair_ids.append(pair_id)
        yield x, y, np.asarray(labels), np.asarray(pair_ids)


def predict(model, definitions, sampler, mean, std, batch_size):
    labels, pair_ids, scores = [], [], []
    for x, _, batch_labels, batch_pairs in batches(
            definitions, sampler, mean, std, batch_size):
        probability = model.predict_on_batch(x)
        labels.extend(batch_labels.tolist())
        pair_ids.extend(batch_pairs.tolist())
        scores.extend(np.asarray(probability)[:, 1].tolist())
    return np.asarray(labels), np.asarray(pair_ids), np.asarray(scores)


def compute_stats(definitions, sampler, args):
    rng = np.random.default_rng(args.seed + 314159)
    sum1 = None
    sum2 = None
    count = 0
    particle_counts = []
    for _, _, definition in definitions:
        values = sampler.features(definition)
        particle_counts.append(len(values))
        if len(values) > args.norm_particles_per_unit:
            positions = rng.choice(len(values), args.norm_particles_per_unit,
                                   replace=False)
            values = values[positions]
        values = values.astype(np.float64)
        sum1 = values.sum(axis=0) if sum1 is None else sum1 + values.sum(axis=0)
        sum2 = ((values ** 2).sum(axis=0) if sum2 is None else
                sum2 + (values ** 2).sum(axis=0))
        count += len(values)
    mean = sum1 / count
    std = np.sqrt(np.maximum(sum2 / count - mean ** 2, 0.0))
    std[std < 1e-6] = 1.0
    if args.latent_scale == "auto":
        latent_scale = 1.0 / float(np.median(particle_counts))
    elif args.latent_scale == "none":
        latent_scale = 1.0
    else:
        latent_scale = float(args.latent_scale)
    return mean.astype(np.float32), std.astype(np.float32), latent_scale


def paired_bootstrap(scores0, scores1, n_boot, seed):
    labels = np.concatenate([np.zeros(len(scores0)), np.ones(len(scores1))])
    point = lc.auc_score(labels, np.concatenate([scores0, scores1]))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(scores0), size=len(scores0))
        values.append(lc.auc_score(
            labels, np.concatenate([scores0[indices], scores1[indices]])))
    return point, float(np.std(values)), [float(np.percentile(values, 2.5)),
                                          float(np.percentile(values, 97.5))]


def main():
    args = parse_args()
    if min(args.cycles_per_unit, args.clone_factor, args.units_per_epoch,
           args.val_units, args.test_units, args.batch_size,
           args.norm_stat_pairs, args.bootstrap) < 1:
        raise SystemExit("unit, clone, sample, and batch sizes must be positive")

    result_dir = os.path.join(args.outdir, args.label)
    os.makedirs(result_dir, exist_ok=True)
    summary_path = os.path.join(result_dir, "summary.json")
    if os.path.isfile(summary_path):
        print("[{}] complete: {}".format(args.label, summary_path))
        return
    save_json(os.path.join(result_dir, "config.json"), vars(args))
    state_path = os.path.join(result_dir, "state.json")
    history_path = os.path.join(result_dir, "history.csv")
    stats_path = os.path.join(result_dir, "norm_stats.json")
    best_weights = os.path.join(result_dir, "best.weights.h5")
    last_weights = os.path.join(result_dir, "last.weights.h5")
    state = load_state(state_path)
    start_time = time.time()

    print("[{}] loading synthetic and original stores".format(args.label),
          flush=True)
    mother_store = MotherStore(args.mother_store)
    original_store = lc.Store(args.original_store)
    common, pools = common_cycle_splits(
        mother_store, original_store, args.split_fracs, args.seed + 7001)
    print("  common cycles: {:,}; train/val/test = {}".format(
        len(common), "/".join("{:,}".format(len(pools[name]))
                               for name in ("train", "val", "test"))))
    sampler = PairedSampler(mother_store, original_store, args)
    audit_pairing(sampler, np.sort(common), args.audit_cycles)

    val_definitions = paired_definitions(
        np.random.default_rng(args.seed + 9001), pools["val"], args.val_units,
        args.cycles_per_unit, args.null_test)
    if os.path.isfile(stats_path):
        mean, std, latent_scale = lc.load_norm_stats(stats_path)
    else:
        stats_definitions = paired_definitions(
            np.random.default_rng(args.seed + 3141), pools["train"],
            args.norm_stat_pairs, args.cycles_per_unit, args.null_test)
        mean, std, latent_scale = compute_stats(stats_definitions, sampler, args)
        lc.save_norm_stats(stats_path, mean, std,
                           lc.feature_names(args.features, args.drop_phi),
                           latent_scale)
    print("  {} cycles/unit x {} clones; latent scale {:.4g}; null={}".format(
        args.cycles_per_unit, args.clone_factor, latent_scale, args.null_test))

    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.model_seed)
    model = lc.build_pfn(len(mean), latent_scale, PHI_SIZES, F_SIZES)
    if state["epoch"] > 0 and os.path.isfile(last_weights):
        model.load_weights(last_weights)
        print("  resumed epoch {} (best AUC {:.4f})".format(
            state["epoch"], state["best_val_auc"]))

    while not state["done"] and state["epoch"] < args.epochs:
        epoch = state["epoch"]
        rng = np.random.default_rng(args.seed * 100003 + epoch)
        train_definitions = paired_definitions(
            rng, pools["train"], args.units_per_epoch,
            args.cycles_per_unit, args.null_test)
        losses = []
        epoch_start = time.time()
        for x, y, _, _ in batches(
                train_definitions, sampler, mean, std, args.batch_size, rng):
            output = model.train_on_batch(x, y)
            losses.append(float(output[0] if isinstance(output, (list, tuple))
                                else output))
        labels, _, scores = predict(
            model, val_definitions, sampler, mean, std, args.batch_size)
        val_auc = lc.auc_score(labels, scores)
        seconds = time.time() - epoch_start
        improved = val_auc > state["best_val_auc"] + 1e-4
        state["epoch"] = epoch + 1
        if improved:
            state["best_val_auc"] = val_auc
            state["best_epoch"] = epoch
            model.save_weights(best_weights)
        model.save_weights(last_weights)
        append_history(history_path, {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "val_auc": val_auc, "seconds": round(seconds, 1)})
        save_json(state_path, state)
        print("epoch {}: loss {:.4f} | val AUC {:.4f}{} | {:.0f}s".format(
            epoch, np.mean(losses), val_auc, " *" if improved else "", seconds),
            flush=True)
        if epoch - state["best_epoch"] >= args.patience:
            state["done"] = True
            save_json(state_path, state)
        if (args.max_minutes > 0 and
                (time.time() - start_time) / 60.0 > args.max_minutes):
            print("wall-clock limit reached; checkpoint saved")
            return

    if state["epoch"] >= args.epochs:
        state["done"] = True
        save_json(state_path, state)
    if os.path.isfile(best_weights):
        model.load_weights(best_weights)

    test_definitions = paired_definitions(
        np.random.default_rng(args.seed + 202607), pools["test"],
        args.test_units, args.cycles_per_unit, args.null_test, disjoint=True)
    labels, pair_ids, scores = predict(
        model, test_definitions, sampler, mean, std, args.batch_size)
    scores0 = scores[labels == 0][np.argsort(pair_ids[labels == 0])]
    scores1 = scores[labels == 1][np.argsort(pair_ids[labels == 1])]
    auc, auc_std, auc_ci = paired_bootstrap(
        scores0, scores1, args.bootstrap, args.seed + 88)

    with open(os.path.join(result_dir, "test_scores.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "class", "source", "p_class1"])
        for label, pair_id, score in zip(labels, pair_ids, scores):
            source = "original" if label == 1 and not args.null_test else "synthetic"
            writer.writerow([int(pair_id), int(label), source, float(score)])
    save_json(summary_path, {
        "label": args.label,
        "class_0": "synthetic42",
        "class_1": "synthetic42" if args.null_test else "original_norm42",
        "null_test": args.null_test,
        "cycles_per_unit": args.cycles_per_unit,
        "clone_factor": args.clone_factor,
        "test_auc": auc,
        "paired_bootstrap_std": auc_std,
        "paired_bootstrap_95pct": auc_ci,
        "test_pairs": args.test_units,
        "test_cycles_mutually_disjoint": True,
        "best_val_auc": state["best_val_auc"],
        "best_epoch": state["best_epoch"],
        "epochs_run": state["epoch"],
        "config": vars(args),
    })
    print("TEST AUC {:.4f} +/- {:.4f}; outputs -> {}".format(
        auc, auc_std, result_dir))


if __name__ == "__main__":
    main()
