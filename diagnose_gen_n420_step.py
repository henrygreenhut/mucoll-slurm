#!/usr/bin/env python3
"""Time the first N=420 raw-PFN optimizer steps without training an epoch."""

import argparse
import gc
import json
import os
import resource
import time

import numpy as np

import libtest_common as lc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm1-store", required=True)
    parser.add_argument("--norm42-store", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--n-files", type=int, default=420)
    parser.add_argument("--clone-factor", type=int, default=42)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--phi-sizes", type=int, nargs="+", default=[100, 100, 128])
    parser.add_argument("--f-sizes", type=int, nargs="+", default=[200, 200, 200])
    args = parser.parse_args()
    if args.n_files % args.clone_factor:
        raise SystemExit("--n-files must be divisible by --clone-factor")
    if args.steps < 1 or min(args.batch_sizes) < 1:
        raise SystemExit("--steps and every batch size must be positive")
    return args


def rss_gib():
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def timed(label, function):
    print("ENTER {} | max RSS {:.2f} GiB".format(label, rss_gib()), flush=True)
    start = time.perf_counter()
    result = function()
    print("RETURN {} | {:.3f} s | max RSS {:.2f} GiB"
          .format(label, time.perf_counter() - start, rss_gib()), flush=True)
    return result


def pad_batch(features, labels):
    max_particles = max(len(event) for event in features)
    n_features = features[0].shape[1]
    particles = np.zeros(
        (len(features), max_particles, n_features), dtype=np.float32)
    for index, event in enumerate(features):
        particles[index, :len(event)] = event
    targets = np.zeros((len(features), 2), dtype=np.float32)
    targets[np.arange(len(features)), labels] = 1.0
    return particles, targets


def main():
    args = parse_args()

    with open(args.norm_stats) as handle:
        stats_payload = json.load(handle)
    if stats_payload.get("names") != lc.feature_names("paper"):
        raise SystemExit("{} does not contain paper-feature normalization"
                         .format(args.norm_stats))
    mean, std, _ = lc.load_norm_stats(args.norm_stats)

    norm1 = timed("load norm1 HDF5 store", lambda: lc.Store(args.norm1_store))
    norm42 = timed("load norm42 HDF5 store", lambda: lc.Store(args.norm42_store))

    common, pos1, pos42 = lc.common_positions(norm1, norm42)
    split = lc.split_indices(len(common))["train"]
    pools = (pos1[split], pos42[split])
    stores = (norm1, norm42)
    files_per_event = (args.n_files, args.n_files // args.clone_factor)
    print("paired cycles {} | train cycles {} | files/event U={} R={}"
          .format(len(common), len(split), *files_per_event), flush=True)

    tf, tf_keras = timed(
        "import TensorFlow/tf_keras",
        lambda: (__import__("tensorflow"), __import__("tf_keras")))
    print("TensorFlow {} | GPUs {}".format(
        tf.__version__, tf.config.list_physical_devices("GPU")), flush=True)

    rng = np.random.default_rng(args.seed)

    def build_event(class_id, batch_size, step, slot):
        tag = "bs{} step{} slot{} class{}".format(
            batch_size, step, slot, "U" if class_id == 0 else "R")
        positions = rng.choice(
            pools[class_id], size=files_per_event[class_id], replace=False)
        raw = timed(
            "{} concatenate source files".format(tag),
            lambda: stores[class_id].file_arrays(positions))
        features = timed(
            "{} build paper features".format(tag),
            lambda: lc.build_features(raw, feature_set="paper"))
        del raw
        normalized = timed(
            "{} normalize features".format(tag),
            lambda: ((features - mean) / std).astype(np.float32, copy=False))
        del features
        print("{} particles {:,}".format(tag, len(normalized)), flush=True)
        return normalized

    for batch_size in args.batch_sizes:
        print("\n===== BATCH SIZE {} =====".format(batch_size), flush=True)
        tf_keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        model = timed(
            "bs{} build and compile raw EnergyFlow PFN".format(batch_size),
            lambda: lc.build_pfn_energyflow(
                len(mean), phi_sizes=tuple(args.phi_sizes),
                f_sizes=tuple(args.f_sizes), jit_compile=False,
                lr=args.lr, warmup_steps=args.warmup_steps, clipnorm=0.0))
        print("bs{} JIT model={} optimizer={}".format(
            batch_size, getattr(model, "_jit_compile", None),
            getattr(model.optimizer, "jit_compile", None)), flush=True)
        if hasattr(model.optimizer, "build"):
            timed("bs{} materialize Adam slots".format(batch_size),
                  lambda: model.optimizer.build(model.trainable_variables))

        for step in range(args.steps):
            # Alternate labels for batch size 1; larger batches are balanced.
            if batch_size == 1:
                labels = np.asarray([step % 2], dtype=np.int32)
            else:
                labels = np.asarray(
                    [slot % 2 for slot in range(batch_size)], dtype=np.int32)
            events = [
                build_event(int(label), batch_size, step, slot)
                for slot, label in enumerate(labels)
            ]
            particles, targets = timed(
                "bs{} step{} pad batch".format(batch_size, step),
                lambda: pad_batch(events, labels))
            print("bs{} step{} padded shape {} | ENTER train_on_batch"
                  .format(batch_size, step, particles.shape), flush=True)
            start = time.perf_counter()
            output = model.train_on_batch(particles, targets)
            print("bs{} step{} RETURN train_on_batch | {:.3f} s | output {}"
                  .format(batch_size, step, time.perf_counter() - start, output),
                  flush=True)
            del events, particles, targets
            gc.collect()

        del model
        tf_keras.backend.clear_session()
        gc.collect()

    print("\nDIAGNOSTIC COMPLETE", flush=True)


if __name__ == "__main__":
    main()
