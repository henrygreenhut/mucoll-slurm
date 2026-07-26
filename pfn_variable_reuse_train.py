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
from pfn_libtest_train import (
    append_history,
    current_learning_rate,
    initial_state,
    per_unit_cross_entropy,
    save_state,
    update_validation_state,
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
NULL_EPOCHS = 40
NULL_PATIENCE = 8
LEARNING_RATE = 1.0e-4
WARMUP_EPOCHS = 1
DECAY_EPOCHS = 30
MIN_LEARNING_RATE = 1.0e-6
DATA_SEED = 1701
NORM_STAT_UNITS = 100
NORM_PARTICLES_PER_UNIT = 100000
# Shared immutable-config helper currently uses schema version 2.
CONFIG_SCHEMA_VERSION = 2


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
        "decay_epochs": DECAY_EPOCHS,
        "min_learning_rate": MIN_LEARNING_RATE,
        "clipnorm": 0.0,
        "jit_compile": False,
        "data_seed": DATA_SEED,
        "model_seed": args.model_seed,
        "null_test": args.null_test,
        "null_definition": "permutation of labels over the same k10/k42 units",
        "selection_metric": "validation loss",
        "min_delta_sigma": 1.0,
        "norm_stat_units_per_class": norm_units,
        "norm_particles_per_unit": NORM_PARTICLES_PER_UNIT,
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
    if sorted(np.unique(physical_k).tolist()) != sorted(reuse_k):
        raise ValueError("saved unit definitions use different reuse factors")
    if np.bincount(labels, minlength=2).tolist() != [units_per_class] * 2:
        raise ValueError("saved unit definitions are not label balanced")
    return list(zip(labels.tolist(), physical_k.tolist(), seeds.tolist()))


def epoch_definitions(reuse_k, units_per_class, epoch, null_test):
    path_seed = DATA_SEED * 100003 + epoch
    rng = np.random.default_rng(path_seed)
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
    rng = np.random.default_rng(DATA_SEED + 314159)
    count = 0
    sum1 = None
    sum2 = None
    particle_counts = []
    for physical_k in reuse_k:
        seeds = rng.integers(
            0, np.iinfo(np.int64).max, size=norm_units, dtype=np.int64)
        for seed in seeds:
            features = unit_features(
                store, train_pool, physical_k, int(seed))
            particle_counts.append(len(features))
            if len(features) > NORM_PARTICLES_PER_UNIT:
                keep = rng.choice(
                    len(features), NORM_PARTICLES_PER_UNIT, replace=False)
                features = features[keep]
            values = features.astype(np.float64)
            batch_sum = values.sum(axis=0)
            batch_sum2 = np.square(values).sum(axis=0)
            sum1 = batch_sum if sum1 is None else sum1 + batch_sum
            sum2 = batch_sum2 if sum2 is None else sum2 + batch_sum2
            count += len(values)
    mean = sum1 / count
    variance = np.maximum(sum2 / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1.0e-6] = 1.0
    latent_scale = 1.0 / float(np.median(particle_counts))
    return mean.astype(np.float32), std.astype(np.float32), latent_scale


def save_test_outputs(result_dir, definitions, labels, scores, reuse_k,
                      state, args):
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
        "test_units_per_class": TEST_UNITS,
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
        epochs = NULL_EPOCHS if args.null_test else EPOCHS
        patience = NULL_PATIENCE if args.null_test else PATIENCE

    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.model_seed)
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
    state_path = os.path.join(result_dir, "state.json")
    state = initial_state("loss")

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
        reuse_k, val_units, DATA_SEED + 9001, args.null_test)
    test_definitions = load_or_create_seed_definitions(
        os.path.join(result_dir, "test_units.npz"),
        reuse_k, test_units, DATA_SEED + 202607, args.null_test)

    steps_per_epoch = 2 * units // BATCH_SIZE
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    decay_steps = DECAY_EPOCHS * steps_per_epoch
    model = lc.build_pfn_energyflow_scaled(
        len(mean), latent_scale, phi_sizes=PHI_SIZES, f_sizes=F_SIZES,
        jit_compile=False, lr=LEARNING_RATE,
        warmup_steps=warmup_steps, clipnorm=0.0,
        decay_steps=decay_steps, min_lr=MIN_LEARNING_RATE)
    print("  EnergyFlow scaled sum | batch {} | LR {} | "
          "{}-epoch warmup + {}-epoch cosine decay".format(
              BATCH_SIZE, LEARNING_RATE, WARMUP_EPOCHS, DECAY_EPOCHS))

    if hasattr(model.optimizer, "build"):
        model.optimizer.build(model.trainable_variables)
    checkpoint_values = {
        "epoch": tf.Variable(0, dtype=tf.int64, trainable=False),
        "max_val_auc": tf.Variable(-1.0, dtype=tf.float64, trainable=False),
        "max_val_auc_epoch": tf.Variable(-1, dtype=tf.int64, trainable=False),
        "min_val_loss": tf.Variable(
            float("inf"), dtype=tf.float64, trainable=False),
        "min_val_loss_epoch": tf.Variable(
            -1, dtype=tf.int64, trainable=False),
        "best_metric_value": tf.Variable(
            float("inf"), dtype=tf.float64, trainable=False),
        "best_epoch": tf.Variable(-1, dtype=tf.int64, trainable=False),
    }
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=model.optimizer, **checkpoint_values)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, os.path.join(result_dir, "resume_checkpoint"),
        max_to_keep=1)
    if checkpoint_manager.latest_checkpoint:
        status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
        try:
            status.assert_consumed()
        except (AssertionError, ValueError) as exc:
            raise SystemExit(
                "checkpoint mismatch; use a new label\n{}".format(exc))
        checkpoint_state = {
            name: (int(value.numpy()) if "epoch" in name
                   else float(value.numpy()))
            for name, value in checkpoint_values.items()
        }
        if not os.path.isfile(state_path):
            raise SystemExit("checkpoint exists but state.json is missing")
        with open(state_path) as handle:
            saved_state = json.load(handle)
        state = checkpoint_state
        state["done"] = bool(saved_state["done"])
        print("  resumed from epoch {} (best epoch {})".format(
            state["epoch"], state["best_epoch"]))
    elif os.path.isfile(state_path):
        raise SystemExit(
            "state.json exists but the full model/Adam checkpoint is missing; "
            "use a new label")

    best_weights = os.path.join(result_dir, "best.weights.h5")
    last_weights = os.path.join(result_dir, "last.weights.h5")
    history_path = os.path.join(result_dir, "history.csv")
    progress_every = args.progress_every

    while not state["done"] and state["epoch"] < epochs:
        epoch = state["epoch"]
        definitions = epoch_definitions(
            reuse_k, units, epoch, args.null_test)
        rng = np.random.default_rng(DATA_SEED * 200003 + epoch)
        losses = []
        train_start = time.time()
        for step, (x, y, _) in enumerate(
                balanced_batches(
                    definitions, store, pools["train"], mean, std, rng), 1):
            output = model.train_on_batch(x, y)
            losses.append(float(
                output[0] if isinstance(output, (list, tuple)) else output))
            if progress_every and (
                    step == 1 or step % progress_every == 0
                    or step == steps_per_epoch):
                print("  train batch {}/{}: mean loss {:.4f}, {:.0f}s".format(
                    step, steps_per_epoch, np.mean(losses),
                    time.time() - train_start), flush=True)
        train_seconds = time.time() - train_start

        val_start = time.time()
        labels, scores = predict(
            model, val_definitions, store, pools["val"], mean, std,
            progress_every)
        val_seconds = time.time() - val_start
        val_auc = lc.auc_score(labels, scores)
        unit_losses = per_unit_cross_entropy(labels, scores)
        val_loss = float(np.mean(unit_losses))
        val_loss_sem = float(
            np.std(unit_losses, ddof=1) / np.sqrt(len(unit_losses)))
        state["epoch"] = epoch + 1
        improved = update_validation_state(
            state, val_auc, val_loss, val_loss_sem,
            "loss", 0.0, 1.0, epoch)
        if improved:
            model.save_weights(best_weights)
        model.save_weights(last_weights)
        for name, value in checkpoint_values.items():
            value.assign(state[name])
        checkpoint_manager.save(checkpoint_number=state["epoch"])
        append_history(history_path, {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_loss_sem": val_loss_sem,
            "val_auc": val_auc,
            "learning_rate": current_learning_rate(model),
            "train_seconds": round(train_seconds, 1),
            "val_seconds": round(val_seconds, 1),
            "seconds": round(train_seconds + val_seconds, 1),
        })
        save_state(state_path, state)
        print("epoch {}: loss {:.4f} | val loss {:.4f} (SEM {:.4f}) | "
              "val AUC {:.4f}{} | train {:.0f}s + val {:.0f}s".format(
                  epoch, np.mean(losses), val_loss, val_loss_sem, val_auc,
                  " *" if improved else "", train_seconds, val_seconds),
              flush=True)

        if lc.should_early_stop(state, patience, 0):
            state["done"] = True
            save_state(state_path, state)
            print("early stop: no validation-loss improvement for {} epochs"
                  .format(patience))
        if (args.max_minutes > 0
                and (time.time() - start_time) / 60.0 > args.max_minutes):
            print("wall-clock limit reached; checkpoint saved")
            return

    if state["epoch"] >= epochs:
        state["done"] = True
        save_state(state_path, state)
    if args.smoke:
        print("smoke training complete; test evaluation skipped")
        return
    if os.path.isfile(best_weights):
        model.load_weights(best_weights)

    labels, scores = predict(
        model, test_definitions, store, pools["test"], mean, std,
        progress_every, label="test")
    save_test_outputs(
        result_dir, test_definitions, labels, scores, reuse_k, state, args)
    print("outputs -> {}".format(result_dir))


if __name__ == "__main__":
    main()
