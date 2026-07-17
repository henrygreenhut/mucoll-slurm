#!/usr/bin/env python3
"""Train a multiclass PFN to identify mother-level BIB reuse multiplicity."""

import argparse
import csv
import json
import os
import time

import numpy as np

import libtest_common as lc
from variable_reuse_common import MotherStore, cycle_split_mothers, sample_definition


PHI_SIZES = (200, 200, 256)
F_SIZES = (200, 200, 200)
SOURCE_SPLIT = (0.50, 0.25, 0.25)
NORM_PARTICLES_PER_CLASS = 100000


def parse_args():
    scratch = os.environ.get("PSCRATCH", "")
    default_store = (scratch + "/mucoll/libtest/stores/gen_split_mothers_MUPLUS.h5"
                     if scratch else "gen_split_mothers_MUPLUS.h5")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mother-store", default=default_store)
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", default="variable_k_results")
    parser.add_argument("--reuse-k", type=int, nargs="+",
                        default=[1, 2, 3, 6, 10, 14, 21, 42])
    parser.add_argument("--mother-equivalents", type=int, default=29400,
                        help="fixed copies/event; 29400 is about 420 old cycle files")
    parser.add_argument("--rotation-policy",
                        choices=["all-random", "baseline-unrotated"],
                        default="all-random")
    parser.add_argument("--units-per-epoch", type=int, default=20,
                        help="training pseudo-events per k and epoch")
    parser.add_argument("--val-units", type=int, default=10,
                        help="fixed validation pseudo-events per k")
    parser.add_argument("--test-units", type=int, default=30,
                        help="overlapping held-out pseudo-events per k")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=0,
                        help="do not apply early stopping before this many "
                             "epochs have completed")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int,
                        help="TensorFlow initialization seed (default: --seed)")
    parser.add_argument("--latent-scale", default="auto",
                        help="auto, none, or a numeric constant")
    parser.add_argument("--null-test", action="store_true",
                        help="independently permute k labels in train/val/test")
    parser.add_argument("--max-minutes", type=float, default=25.0)
    args = parser.parse_args()
    args.split_fracs = SOURCE_SPLIT
    args.norm_stat_units = 1
    args.norm_particles_per_unit = NORM_PARTICLES_PER_CLASS
    return args


def save_json(path, value):
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2)


def load_state(path):
    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)
    return {"epoch": 0, "best_val_macro_auc": -1.0,
            "best_val_accuracy": 0.0, "best_epoch": -1, "done": False}


def append_history(path, row):
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "epoch", "train_loss", "val_accuracy", "val_macro_auc", "seconds"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_definitions(rng, samplers, split, units_per_class, null_test=False):
    definitions = []
    for class_id, sampler in enumerate(samplers):
        for _ in range(units_per_class):
            definition = sample_definition(
                rng, sampler["pools"][split], sampler["reuse_k"],
                sampler["mother_equivalents"], sampler["rotation_policy"])
            definitions.append([class_id, definition])
    if null_test:
        labels = rng.permutation([item[0] for item in definitions])
        for item, label in zip(definitions, labels):
            item[0] = int(label)
    return definitions


def unit_features(store, definition):
    raw = store.rotated_mothers(definition["mothers"], definition["angles"])
    return lc.build_features(raw)


def batches(definitions, store, args, mean, std, n_classes, rng=None):
    order = np.arange(len(definitions))
    if rng is not None:
        rng.shuffle(order)
    for first in range(0, len(order), args.batch_size):
        selected = [definitions[index] for index in order[first:first + args.batch_size]]
        arrays = [unit_features(store, definition)
                  for _, definition in selected]
        max_particles = max(len(array) for array in arrays)
        x = np.zeros((len(arrays), max_particles, len(mean)), dtype=np.float32)
        for index, array in enumerate(arrays):
            x[index, :len(array)] = (array - mean) / std
        y = np.zeros((len(arrays), n_classes), dtype=np.float32)
        for index, (class_id, _) in enumerate(selected):
            y[index, class_id] = 1.0
        yield x, y


def predict(model, definitions, store, args, mean, std, n_classes):
    labels = []
    probabilities = []
    for x, y in batches(definitions, store, args, mean, std, n_classes):
        probabilities.append(model.predict_on_batch(x))
        labels.extend(np.argmax(y, axis=1).tolist())
    return np.asarray(labels, dtype=np.int32), np.concatenate(probabilities)


def classification_metrics(labels, probabilities, n_classes):
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predictions == labels))
    aucs = []
    for class_id in range(n_classes):
        aucs.append(lc.auc_score(labels == class_id, probabilities[:, class_id]))
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    return accuracy, float(np.nanmean(aucs)), aucs, confusion


def compute_stats(store, samplers, args):
    rng = np.random.default_rng(args.seed + 314159)
    count = 0
    sum1 = None
    sum2 = None
    particle_counts = []
    for sampler in samplers:
        for _ in range(args.norm_stat_units):
            definition = sample_definition(
                rng, sampler["pools"]["train"], sampler["reuse_k"],
                sampler["mother_equivalents"], sampler["rotation_policy"])
            features = unit_features(store, definition)
            particle_counts.append(len(features))
            if len(features) > args.norm_particles_per_unit:
                positions = rng.choice(len(features), args.norm_particles_per_unit,
                                       replace=False)
                features = features[positions]
            values = features.astype(np.float64)
            sum1 = values.sum(axis=0) if sum1 is None else sum1 + values.sum(axis=0)
            sum2 = (values ** 2).sum(axis=0) if sum2 is None else sum2 + (values ** 2).sum(axis=0)
            count += len(values)
    mean = sum1 / count
    variance = np.maximum(sum2 / count - mean ** 2, 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    if args.latent_scale == "auto":
        latent_scale = 1.0 / float(np.median(particle_counts))
    elif args.latent_scale == "none":
        latent_scale = 1.0
    else:
        latent_scale = float(args.latent_scale)
    return mean.astype(np.float32), std.astype(np.float32), latent_scale


def main():
    args = parse_args()
    if args.model_seed is None:
        args.model_seed = args.seed
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.model_seed)
    reuse_values = sorted(set(args.reuse_k))
    if len(reuse_values) < 2:
        raise SystemExit("provide at least two distinct --reuse-k values")
    for reuse_k in reuse_values:
        if args.mother_equivalents % reuse_k:
            raise SystemExit("{} mother-equivalents is not divisible by k={}"
                             .format(args.mother_equivalents, reuse_k))
    if args.batch_size < 1 or args.units_per_epoch < 1:
        raise SystemExit("batch size and units per epoch must be positive")
    if args.min_epochs < 0 or args.min_epochs > args.epochs:
        raise SystemExit("--min-epochs must be between 0 and --epochs")

    result_dir = os.path.join(args.outdir, args.label)
    os.makedirs(result_dir, exist_ok=True)
    summary_path = os.path.join(result_dir, "summary.json")
    if os.path.isfile(summary_path):
        print("[{}] complete: {}".format(args.label, summary_path))
        return
    config_path = os.path.join(result_dir, "config.json")
    save_json(config_path, vars(args))
    state_path = os.path.join(result_dir, "state.json")
    history_path = os.path.join(result_dir, "history.csv")
    stats_path = os.path.join(result_dir, "norm_stats.json")
    best_weights = os.path.join(result_dir, "best.weights.h5")
    last_weights = os.path.join(result_dir, "last.weights.h5")
    state = load_state(state_path)
    start_time = time.time()

    print("[{}] loading mother bank {}".format(args.label, args.mother_store), flush=True)
    store = MotherStore(args.mother_store)
    pools = cycle_split_mothers(store, tuple(args.split_fracs), args.seed + 7001)
    print("  {:,} mothers in {} cycles; split mothers train/val/test = {}"
          .format(store.n_mothers, store.n_cycles,
                  "/".join("{:,}".format(len(pools[name]))
                           for name in ("train", "val", "test"))))
    samplers = [{
        "reuse_k": reuse_k,
        "mother_equivalents": args.mother_equivalents,
        "rotation_policy": args.rotation_policy,
        "pools": pools,
    } for reuse_k in reuse_values]
    for split, pool in pools.items():
        if len(pool) < args.mother_equivalents:
            raise SystemExit("{} has {:,} mothers but k=1 requires {:,}"
                             .format(split, len(pool), args.mother_equivalents))

    if os.path.isfile(stats_path):
        mean, std, latent_scale = lc.load_norm_stats(stats_path)
    else:
        mean, std, latent_scale = compute_stats(store, samplers, args)
        lc.save_norm_stats(stats_path, mean, std, lc.FEATURE_NAMES, latent_scale)
    print("  classes k={} | fixed {:,} mother-equivalents | latent scale {:.4g}"
          .format(reuse_values, args.mother_equivalents, latent_scale))

    n_classes = len(reuse_values)
    rng_val = np.random.default_rng(args.seed + 9001)
    val_definitions = make_definitions(
        rng_val, samplers, "val", args.val_units, args.null_test)

    model = lc.build_pfn(len(mean), latent_scale, PHI_SIZES, F_SIZES,
                         n_classes=n_classes)
    if hasattr(model.optimizer, "build"):
        model.optimizer.build(model.trainable_variables)
    checkpoint_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_best_auc = tf.Variable(-1.0, dtype=tf.float64, trainable=False)
    checkpoint_best_accuracy = tf.Variable(0.0, dtype=tf.float64, trainable=False)
    checkpoint_best_epoch = tf.Variable(-1, dtype=tf.int64, trainable=False)
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=model.optimizer, epoch=checkpoint_epoch,
        best_val_macro_auc=checkpoint_best_auc,
        best_val_accuracy=checkpoint_best_accuracy,
        best_epoch=checkpoint_best_epoch)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, os.path.join(result_dir, "resume_checkpoint"), max_to_keep=1)
    if checkpoint_manager.latest_checkpoint:
        status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
        status.assert_existing_objects_matched()
        state["epoch"] = int(checkpoint_epoch.numpy())
        state["best_val_macro_auc"] = float(checkpoint_best_auc.numpy())
        state["best_val_accuracy"] = float(checkpoint_best_accuracy.numpy())
        state["best_epoch"] = int(checkpoint_best_epoch.numpy())
        print("  resumed model + Adam from epoch {} (best macro AUC {:.4f})"
              .format(state["epoch"], state["best_val_macro_auc"]))
    elif state["epoch"] > 0 and os.path.isfile(last_weights):
        model.load_weights(last_weights)
        print("  resumed legacy weights from epoch {} (best macro AUC {:.4f}); "
              "optimizer state was unavailable".format(
                  state["epoch"], state["best_val_macro_auc"]))

    while not state["done"] and state["epoch"] < args.epochs:
        epoch = state["epoch"]
        rng = np.random.default_rng(args.seed * 100003 + epoch)
        train_definitions = make_definitions(
            rng, samplers, "train", args.units_per_epoch, args.null_test)
        losses = []
        epoch_start = time.time()
        for x, y in batches(train_definitions, store, args, mean, std,
                            n_classes, rng):
            output = model.train_on_batch(x, y)
            losses.append(float(output[0] if isinstance(output, (list, tuple)) else output))
        labels, probabilities = predict(
            model, val_definitions, store, args, mean, std, n_classes)
        accuracy, macro_auc, _, _ = classification_metrics(
            labels, probabilities, n_classes)
        seconds = time.time() - epoch_start

        improved = macro_auc > state["best_val_macro_auc"] + 1e-4
        state["epoch"] = epoch + 1
        if improved:
            state["best_val_macro_auc"] = macro_auc
            state["best_val_accuracy"] = accuracy
            state["best_epoch"] = epoch
            model.save_weights(best_weights)
        model.save_weights(last_weights)
        checkpoint_epoch.assign(state["epoch"])
        checkpoint_best_auc.assign(state["best_val_macro_auc"])
        checkpoint_best_accuracy.assign(state["best_val_accuracy"])
        checkpoint_best_epoch.assign(state["best_epoch"])
        checkpoint_manager.save(checkpoint_number=state["epoch"])
        append_history(history_path, {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "val_accuracy": accuracy, "val_macro_auc": macro_auc,
            "seconds": round(seconds, 1),
        })
        save_json(state_path, state)
        print("epoch {}: loss {:.4f} | val acc {:.4f} | macro AUC {:.4f}{} | {:.0f}s"
              .format(epoch, np.mean(losses), accuracy, macro_auc,
                      " *" if improved else "", seconds), flush=True)

        if lc.should_early_stop(state, args.patience, args.min_epochs):
            state["done"] = True
            save_json(state_path, state)
            print("early stop: no validation improvement for {} epochs"
                  .format(args.patience))
        if args.max_minutes > 0 and (time.time() - start_time) / 60.0 > args.max_minutes:
            print("wall-clock limit reached; checkpoint saved")
            return

    if state["epoch"] >= args.epochs:
        state["done"] = True
        save_json(state_path, state)
    if os.path.isfile(best_weights):
        model.load_weights(best_weights)

    rng_test = np.random.default_rng(args.seed + 202607)
    test_definitions = make_definitions(
        rng_test, samplers, "test", args.test_units, args.null_test)
    labels, probabilities = predict(
        model, test_definitions, store, args, mean, std, n_classes)
    accuracy, macro_auc, aucs, confusion = classification_metrics(
        labels, probabilities, n_classes)

    with open(os.path.join(result_dir, "test_scores.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class", "true_k"] +
                        ["p_k{}".format(value) for value in reuse_values])
        for label, row in zip(labels, probabilities):
            writer.writerow([int(label), reuse_values[int(label)]] + row.tolist())
    np.savetxt(os.path.join(result_dir, "confusion_matrix.csv"), confusion,
               delimiter=",", fmt="%d")
    save_json(summary_path, {
        "label": args.label,
        "reuse_k": reuse_values,
        "mother_equivalents": args.mother_equivalents,
        "rotation_policy": args.rotation_policy,
        "null_test": args.null_test,
        "test_accuracy": accuracy,
        "test_macro_auc": macro_auc,
        "test_one_vs_rest_auc": dict((str(k), auc) for k, auc in zip(reuse_values, aucs)),
        "confusion_matrix": confusion.tolist(),
        "test_units_per_class": args.test_units,
        "test_events_overlap_sources": True,
        "uncertainty_note": "point metrics only; source-cycle bootstrap not yet applied",
        "best_val_macro_auc": state["best_val_macro_auc"],
        "best_val_accuracy": state["best_val_accuracy"],
        "best_epoch": state["best_epoch"],
        "epochs_run": state["epoch"],
        "config": vars(args),
    })
    print("TEST accuracy {:.4f}; macro one-vs-rest AUC {:.4f}"
          .format(accuracy, macro_auc))
    print("outputs -> {}".format(result_dir))


if __name__ == "__main__":
    main()
