#!/usr/bin/env python3
"""Binary PFN study of two synthetic mother-reuse factors.

At fixed total mother-equivalents M, class k uses M/k distinct mothers and
k independent azimuthal copies of every selected mother. The only intended
class difference is therefore within-event mother reuse.
"""

import argparse
import csv
import json
import math
import os
import time

import numpy as np

import libtest_common as lc
from pfn_training_engine import (
    CONFIG_SCHEMA_VERSION,
    run_binary_pfn_training,
    write_or_validate_config,
)
from variable_reuse_common import MotherStore, sample_definition


# Fixed analysis recipe selected by the N=420 architecture comparison.
MOTHER_EQUIVALENTS = 29400
SOURCE_SPLIT = (0.50, 0.25, 0.25)
FEATURE_SET = "expanded"
PHI_SIZES = (100, 100, 128)
F_SIZES = (200, 200, 200)
BATCH_SIZE = 4
UNITS_PER_EPOCH = 500
VAL_UNITS = 300
TEST_UNITS = 300
EPOCHS = 80
PATIENCE = 15
LEARNING_RATE = 1.0e-4
WARMUP_EPOCHS = 1
DECAY_EPOCHS = 30
MIN_LEARNING_RATE = 1.0e-6
DATA_SEED = 1701
NORM_STAT_UNITS = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mother-store", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--reuse-k", type=int, nargs=2, default=(10, 42),
                        metavar=("K0", "K1"))
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--null-test", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=1400.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def scientific_config(args, epochs, patience, units, val_units, test_units,
                      norm_units):
    steps_per_epoch = 2 * units // BATCH_SIZE
    return {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "mother_store": os.path.abspath(args.mother_store),
        "label": args.label,
        "reuse_k": list(args.reuse_k),
        "mother_equivalents": MOTHER_EQUIVALENTS,
        "rotation_policy": "all-random",
        "source_split": SOURCE_SPLIT,
        "features": FEATURE_SET,
        "architecture": "energyflow-scaled-sum",
        "phi_sizes": PHI_SIZES,
        "f_sizes": F_SIZES,
        "batch_size": BATCH_SIZE,
        "units_per_epoch_per_class": units,
        "val_units_per_class": val_units,
        "test_units_per_class": test_units,
        "epochs": epochs,
        "patience": patience,
        "learning_rate": LEARNING_RATE,
        "warmup_epochs": WARMUP_EPOCHS,
        "warmup_steps": WARMUP_EPOCHS * steps_per_epoch,
        "decay_epochs": DECAY_EPOCHS,
        "decay_steps": DECAY_EPOCHS * steps_per_epoch,
        "min_learning_rate": MIN_LEARNING_RATE,
        "clipnorm": 0.0,
        "latent_dropout": 0.0,
        "f_dropout": 0.0,
        "phi_l2": 0.0,
        "f_l2": 0.0,
        "jit_compile": False,
        "data_seed": DATA_SEED,
        "normalization_seed": DATA_SEED,
        "validation_definition_seed": DATA_SEED + 999,
        "test_definition_seed": DATA_SEED + 2026,
        "epoch_seed_formula": "data_seed * 100003 + epoch",
        "model_seed": args.model_seed,
        "null_test": args.null_test,
        "null_definition": (
            "permutation of labels over the same k{}-vs-k{} units".format(
                args.reuse_k[0], args.reuse_k[1])),
        "selection_metric": "validation loss",
        "min_epochs": 0,
        # Ignored for loss selection, but retained at the successful N=420
        # value so the recorded optimizer/selection recipe matches exactly.
        "min_delta": 1.0e-4,
        "min_delta_sigma": 1.0,
        "norm_stat_units_per_class": norm_units,
        "norm_particle_weighting": "all particles, matching N=420 recipe",
        "test_events_overlap_sources": True,
        "max_minutes": args.max_minutes,
        "progress_every": args.progress_every,
    }


def load_or_create_seed_definitions(path, reuse_k, units_per_class, seed,
                                    null_test):
    """Persist compact unit definitions as (label, physical k, RNG seed)."""
    if os.path.isfile(path):
        with np.load(path) as payload:
            labels = payload["labels"].astype(np.int32)
            physical_k = payload["physical_k"].astype(np.int32)
            seeds = payload["seeds"].astype(np.uint64)
    else:
        rng = np.random.default_rng(seed)
        physical_k = np.repeat(np.asarray(reuse_k, np.int32), units_per_class)
        labels = np.repeat(np.arange(2, dtype=np.int32), units_per_class)
        seeds = rng.integers(
            0, np.iinfo(np.int64).max, size=len(labels), dtype=np.int64
        ).astype(np.uint64)
        if null_test:
            labels = rng.permutation(labels)
        np.savez(path, labels=labels, physical_k=physical_k, seeds=seeds)

    expected = 2 * units_per_class
    if not (labels.shape == physical_k.shape == seeds.shape == (expected,)):
        raise ValueError("saved unit definitions have the wrong shape")
    physical_counts = [
        int(np.count_nonzero(physical_k == k)) for k in reuse_k]
    if physical_counts != [units_per_class, units_per_class]:
        raise ValueError(
            "saved unit definitions do not contain equal reuse-factor counts")
    if np.bincount(labels, minlength=2).tolist() != [units_per_class] * 2:
        raise ValueError("saved unit definitions are not label balanced")
    if not null_test:
        expected_k = np.asarray(reuse_k, dtype=np.int32)[labels]
        if not np.array_equal(physical_k, expected_k):
            raise ValueError(
                "saved main-study labels do not match their reuse factors")
    return list(zip(labels.tolist(), physical_k.tolist(), seeds.tolist()))


def epoch_definitions(reuse_k, units_per_class, epoch, null_test, rng=None):
    if rng is None:
        rng = np.random.default_rng(DATA_SEED * 100003 + epoch)
    physical_k = np.repeat(np.asarray(reuse_k, np.int32), units_per_class)
    labels = np.repeat(np.arange(2, dtype=np.int32), units_per_class)
    seeds = rng.integers(
        0, np.iinfo(np.int64).max, size=len(labels), dtype=np.int64
    ).astype(np.uint64)
    if null_test:
        labels = rng.permutation(labels)
    return list(zip(labels.tolist(), physical_k.tolist(), seeds.tolist()))


def unit_features(store, mother_pool, physical_k, seed):
    definition = sample_definition(
        np.random.default_rng(seed), mother_pool, physical_k,
        MOTHER_EQUIVALENTS, "all-random")
    raw = store.rotated_mothers(
        definition["mothers"], definition["angles"])
    return lc.build_features(raw, feature_set=FEATURE_SET)


def padded_batch(chunk, store, mother_pool, mean, std):
    arrays = [
        (unit_features(store, mother_pool, physical_k, seed) - mean) / std
        for _, physical_k, seed in chunk
    ]
    labels = np.asarray([label for label, _, _ in chunk], dtype=np.int32)
    max_particles = max(len(array) for array in arrays)
    x = np.zeros(
        (len(arrays), max_particles, arrays[0].shape[1]), dtype=np.float32)
    for index, array in enumerate(arrays):
        x[index, :len(array)] = array
    y = np.zeros((len(arrays), 2), dtype=np.float32)
    y[np.arange(len(arrays)), labels] = 1.0
    return x, y, labels


def balanced_batches(definitions, store, mother_pool, mean, std, rng):
    by_label = [
        [definition for definition in definitions if definition[0] == label]
        for label in (0, 1)
    ]
    if len(by_label[0]) != len(by_label[1]):
        raise ValueError("balanced training requires equal class counts")
    half = BATCH_SIZE // 2
    if len(by_label[0]) % half:
        raise ValueError("units per class must be divisible by batch_size/2")
    orders = [rng.permutation(len(group)) for group in by_label]
    for start in range(0, len(by_label[0]), half):
        chunk = (
            [by_label[0][i] for i in orders[0][start:start + half]]
            + [by_label[1][i] for i in orders[1][start:start + half]]
        )
        rng.shuffle(chunk)
        yield padded_batch(chunk, store, mother_pool, mean, std)


def ordinary_batches(definitions, store, mother_pool, mean, std):
    for start in range(0, len(definitions), BATCH_SIZE):
        yield padded_batch(
            definitions[start:start + BATCH_SIZE],
            store, mother_pool, mean, std)


def predict(model, definitions, store, mother_pool, mean, std,
            progress_every=0, label="validation"):
    labels, scores = [], []
    n_batches = math.ceil(len(definitions) / BATCH_SIZE)
    for step, (x, _, batch_labels) in enumerate(
            ordinary_batches(definitions, store, mother_pool, mean, std), 1):
        probabilities = np.asarray(model.predict_on_batch(x))
        labels.extend(batch_labels.tolist())
        scores.extend(probabilities[:, 1].tolist())
        if progress_every and (
                step == 1 or step % progress_every == 0
                or step == n_batches):
            print("  {} batch {}/{}".format(label, step, n_batches),
                  flush=True)
    return np.asarray(labels, np.int32), np.asarray(scores, np.float64)


def compute_normalization(store, train_pool, reuse_k, norm_units):
    # Match the successful file-level N=420 normalization stream: a fresh
    # generator initialized directly from the fixed data seed.
    rng = np.random.default_rng(DATA_SEED)
    particle_counts = []

    def feature_stream():
        for physical_k in reuse_k:
            seeds = rng.integers(
                0, np.iinfo(np.int64).max, size=norm_units, dtype=np.int64)
            for seed in seeds:
                features = unit_features(
                    store, train_pool, physical_k, int(seed))
                particle_counts.append(len(features))
                yield features

    mean, std = lc.compute_norm_stats(feature_stream())
    latent_scale = 1.0 / float(np.median(particle_counts))
    return mean, std, latent_scale


def save_test_outputs(result_dir, definitions, labels, scores, reuse_k,
                      test_units, state, args):
    predictions = (scores >= 0.5).astype(np.int32)
    accuracy = float(np.mean(predictions == labels))
    auc = lc.auc_score(labels, scores)
    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)

    with open(os.path.join(result_dir, "test_scores.csv"), "w",
              newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", "physical_k", "score_k{}".format(
            reuse_k[1])])
        for definition, label, score in zip(definitions, labels, scores):
            writer.writerow([int(label), int(definition[1]), float(score)])

    summary = {
        "label": args.label,
        "reuse_k": list(reuse_k),
        "mother_equivalents": MOTHER_EQUIVALENTS,
        "unique_mothers_per_event": {
            str(k): MOTHER_EQUIVALENTS // k for k in reuse_k
        },
        "rotation_policy": "all-random",
        "null_test": args.null_test,
        "test_auc": auc,
        "test_accuracy": accuracy,
        "confusion_matrix": confusion.tolist(),
        "test_units_per_class": test_units,
        "test_mode": "overlapping held-out events; point estimate only",
        "best_val_auc": state["max_val_auc"],
        "min_val_loss": state["min_val_loss"],
        "best_epoch": state["best_epoch"],
        "epochs_run": state["epoch"],
    }
    with open(os.path.join(result_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print("TEST AUC {:.6f}; accuracy {:.4f}".format(auc, accuracy))


def main():
    args = parse_args()
    reuse_k = tuple(args.reuse_k)
    if reuse_k[0] == reuse_k[1] or any(k < 1 for k in reuse_k):
        raise SystemExit("--reuse-k requires two distinct positive integers")
    for k in reuse_k:
        if MOTHER_EQUIVALENTS % k:
            raise SystemExit(
                "{} mother-equivalents is not divisible by k={}".format(
                    MOTHER_EQUIVALENTS, k))

    if args.smoke:
        units, val_units, test_units = 4, 4, 4
        epochs, patience, norm_units = 2, 2, 2
    else:
        units, val_units, test_units, norm_units = (
            UNITS_PER_EPOCH, VAL_UNITS, TEST_UNITS, NORM_STAT_UNITS)
        epochs, patience = EPOCHS, PATIENCE

    start_time = time.time()
    result_dir = os.path.join("variable_k_results", args.label)
    os.makedirs(result_dir, exist_ok=True)
    summary_path = os.path.join(result_dir, "summary.json")
    if os.path.isfile(summary_path):
        print("complete -> {}".format(summary_path))
        return

    config = scientific_config(
        args, epochs, patience, units, val_units, test_units, norm_units)
    write_or_validate_config(
        os.path.join(result_dir, "config.json"), config)

    print("[{}] loading {}".format(args.label, args.mother_store), flush=True)
    store = MotherStore(args.mother_store)
    cycle_split = lc.load_or_create_cycle_split(
        os.path.join(result_dir, "source_split.npz"), store.cycle_ids,
        SOURCE_SPLIT, DATA_SEED)
    pools = {
        name: store.mothers_for_cycles(cycles)
        for name, cycles in cycle_split.items()
    }
    for split, pool in pools.items():
        required = max(MOTHER_EQUIVALENTS // k for k in reuse_k)
        if len(pool) < required:
            raise SystemExit(
                "{} has {} mothers but an event needs {}".format(
                    split, len(pool), required))
    print("  cycles train/val/test = {} | mothers = {}".format(
        "/".join(str(len(cycle_split[name]))
                 for name in ("train", "val", "test")),
        "/".join("{:,}".format(len(pools[name]))
                 for name in ("train", "val", "test"))))
    print("  fixed {:,} mother-equivalents: k={} -> {} unique mothers".format(
        MOTHER_EQUIVALENTS, reuse_k,
        tuple(MOTHER_EQUIVALENTS // k for k in reuse_k)))

    stats_path = os.path.join(result_dir, "norm_stats.json")
    feature_names = lc.feature_names(FEATURE_SET)
    if os.path.isfile(stats_path):
        mean, std, latent_scale = lc.load_norm_stats(stats_path)
        with open(stats_path) as handle:
            saved_names = json.load(handle).get("names")
        if saved_names != feature_names:
            raise SystemExit("cached normalization uses different features")
    else:
        mean, std, latent_scale = compute_normalization(
            store, pools["train"], reuse_k, norm_units)
        lc.save_norm_stats(
            stats_path, mean, std, feature_names, latent_scale)
    print("  {} features | latent scale 1/{:.0f}".format(
        len(mean), 1.0 / latent_scale))

    val_definitions = load_or_create_seed_definitions(
        os.path.join(result_dir, "validation_units.npz"),
        reuse_k, val_units, DATA_SEED + 999, args.null_test)
    test_definitions = load_or_create_seed_definitions(
        os.path.join(result_dir, "test_units.npz"),
        reuse_k, test_units, DATA_SEED + 2026, args.null_test)

    # The adapter ends here: it defines synthetic mother-reuse events and
    # materializes them. Model fitting and validation below use the exact
    # shared engine used by pfn_libtest_train.py.
    steps_per_epoch = 2 * units // BATCH_SIZE
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    decay_steps = DECAY_EPOCHS * steps_per_epoch
    print("  EnergyFlow scaled sum | batch {} | LR {} | "
          "{}-epoch warmup + {}-epoch cosine decay".format(
              BATCH_SIZE, LEARNING_RATE, WARMUP_EPOCHS, DECAY_EPOCHS))

    def train_batches_for_epoch(epoch):
        # Use one epoch RNG for unit definitions and batch shuffling, exactly
        # as the successful file-level trainer does.
        rng = np.random.default_rng(DATA_SEED * 100003 + epoch)
        definitions = epoch_definitions(
            reuse_k, units, epoch, args.null_test, rng=rng)
        return balanced_batches(
            definitions, store, pools["train"], mean, std, rng)

    def predict_validation(model):
        return predict(
            model, val_definitions, store, pools["val"], mean, std,
            args.progress_every)

    training_config = {
        "result_dir": result_dir,
        "n_features": len(mean),
        "latent_scale": latent_scale,
        "phi_sizes": PHI_SIZES,
        "f_sizes": F_SIZES,
        "arch": "energyflow",
        "jit": False,
        "lr": LEARNING_RATE,
        "warmup_steps": warmup_steps,
        "decay_steps": decay_steps,
        "min_lr": MIN_LEARNING_RATE,
        "clipnorm": 0.0,
        "latent_dropout": 0.0,
        "f_dropout": 0.0,
        "phi_l2": 0.0,
        "f_l2": 0.0,
        "model_seed": args.model_seed,
        "select_metric": "loss",
        "min_delta": 1.0e-4,
        "min_delta_sigma": 1.0,
        "epochs": epochs,
        "patience": patience,
        "min_epochs": 0,
        "units_per_epoch": units,
        "batch_size": BATCH_SIZE,
        "max_minutes": args.max_minutes,
        "progress_every": args.progress_every,
    }
    model, state, training_complete = run_binary_pfn_training(
        training_config, train_batches_for_epoch, predict_validation,
        start_time=start_time)
    if not training_complete:
        return

    labels, scores = predict(
        model, test_definitions, store, pools["test"], mean, std,
        args.progress_every, label="test")
    save_test_outputs(
        result_dir, test_definitions, labels, scores, reuse_k, test_units,
        state, args)
    print("outputs -> {}".format(result_dir))


if __name__ == "__main__":
    main()
