#!/usr/bin/env python3
"""Train event-level and single-track SIM-vs-COUNT PFNs on physics observables.

Loads the ``.npz`` track stores written by count_tracker_features.py for two
samples (default SIM vs COUNT) across train/val/test, applies the physical-track
transform, and trains the standard EnergyFlow PFN (reusing the RECO study's
``build_pfn_energyflow``). Reports held-out test AUC and a provenance summary.
For single-track training, each nonempty event has equal total weight within
its class. Labels are permuted by event group for the code-sanity null.

Runs in the PFN training env (EnergyFlow, TensorFlow, and tf_keras), on GPU.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for libtest_common
from count_tracker_track_features import (  # noqa: E402
    FEATURES, FEATURE_DEFINITIONS, RAW_FEATURES, pfn_features,
)
from count_tracker_metrics import paired_event_bootstrap, weighted_auc  # noqa: E402

TRAINING_SEED = 12345
PHI_SIZES = (64, 64, 64)
F_SIZES = (64, 64, 64)
SPLITS = ("train", "val", "test")
RECIPES = {
    "fixed": {"learning_rate": 1e-3, "warmup_epochs": 0,
              "decay_epochs": 0, "min_learning_rate": 0.0,
              "f_dropout": 0.0},
    "fixed_dropout": {"learning_rate": 1e-3, "warmup_epochs": 0,
                      "decay_epochs": 0, "min_learning_rate": 0.0,
                      "f_dropout": 0.1},
    "stabilized_dropout": {"learning_rate": 1e-4, "warmup_epochs": 1,
                           "decay_epochs": 30, "min_learning_rate": 1e-6,
                           "f_dropout": 0.1},
    "stabilized": {"learning_rate": 1e-4, "warmup_epochs": 1, "decay_epochs": 30,
                   "min_learning_rate": 1e-6, "f_dropout": 0.0},
}


def load_store(store_dir, construction, sample, split):
    prefix = Path(store_dir) / f"{construction}_{sample}_{split}"
    manifest = json.loads(prefix.with_suffix(".json").read_text())
    if manifest.get("schema_version") != 2:
        raise SystemExit(f"{prefix}.json must be a version-2 AtIP track store; rebuild old stores")
    expected = {"construction": construction, "sample": sample, "split": split,
                "track_collection": "SiTracks", "track_state": "AtIP"}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SystemExit(f"{prefix}.json has {key}={manifest.get(key)!r}, expected {value!r}")
    if tuple(manifest["features"]) != RAW_FEATURES:
        raise SystemExit(f"{prefix}.json has unexpected features {manifest['features']}")
    with np.load(prefix.with_suffix(".npz")) as data:
        tracks = data["tracks"].astype(np.float32)
        counts = data["n_tracks"].astype(np.int64)
    if tracks.ndim != 3 or tracks.shape[2] != len(RAW_FEATURES):
        raise SystemExit(f"{prefix}.npz has an unexpected track shape")
    if counts.shape != (len(tracks),) or np.any(counts < 0) or np.any(counts > tracks.shape[1]):
        raise SystemExit(f"{prefix}.npz has invalid n_tracks")
    if manifest["n_events"] != len(tracks) or manifest["total_tracks"] != int(counts.sum()):
        raise SystemExit(f"{prefix} manifest does not match its track counts")
    return tracks, counts, manifest


def pad_width(array, width):
    if array.shape[1] == width:
        return array
    out = np.zeros((len(array), width, array.shape[2]), dtype=np.float32)
    out[:, :array.shape[1]] = array
    return out


def one_hot(labels):
    out = np.zeros((len(labels), 2), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def combine(raw_a, counts_a, raw_b, counts_b, width):
    xa = pfn_features(pad_width(raw_a, width), counts_a)
    xb = pfn_features(pad_width(raw_b, width), counts_b)
    if len(xa) != len(xb):
        raise SystemExit(f"class counts differ: {len(xa)} vs {len(xb)}")
    x = np.concatenate([xa, xb])
    y = np.asarray([0] * len(xa) + [1] * len(xb), dtype=np.int32)
    return x, y


def single_track_examples(raw_a, counts_a, raw_b, counts_b):
    """One-track PFN inputs with equal total weight per nonempty source event."""
    if len(counts_a) != len(counts_b):
        raise SystemExit("single-track comparison requires the same number of events per class")
    samples = []
    for label, (raw, counts) in enumerate(((raw_a, counts_a), (raw_b, counts_b))):
        features = pfn_features(raw, counts)
        n_nonempty = int(np.count_nonzero(counts))
        if n_nonempty == 0:
            raise SystemExit(f"class {label} has no reconstructed tracks")
        mask = np.arange(raw.shape[1])[None, :] < counts[:, None]
        tracks = features[mask][:, None, :]
        groups = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        weights = 1.0 / (n_nonempty * counts[groups].astype(np.float64))
        samples.append((tracks, groups, weights))
    x = np.concatenate([part[0] for part in samples])
    y = np.concatenate([np.full(len(part[0]), label, dtype=np.int32)
                        for label, part in enumerate(samples)])
    groups = np.concatenate([part[1] + label * len(counts_a)
                             for label, part in enumerate(samples)])
    weights = np.concatenate([part[2] for part in samples])
    # Each class contributes half the objective; mean weight is one for Keras.
    weights *= len(weights) / 2.0
    return x, y, weights.astype(np.float32), groups


def grouped_permuted_labels(labels, groups, split):
    """Permute labels at event level, keeping all tracks from an event together."""
    _, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    event_labels = labels[first]
    return permuted_labels(event_labels, split)[inverse]


def validate_sim_count_pair(manifest_a, manifest_b, split):
    """A paired comparison must refer to the same prepared source events."""
    ids_a = [event["event_id"] for event in manifest_a["events"]]
    ids_b = [event["event_id"] for event in manifest_b["events"]]
    if ids_a != ids_b:
        raise SystemExit(f"{split}: SIM and COUNT stores have different event IDs or order")
    if manifest_a["conditions_manifest_sha256"] != manifest_b["conditions_manifest_sha256"]:
        raise SystemExit(f"{split}: SIM and COUNT stores use different conditions manifests")


def permuted_labels(labels, split):
    rng = np.random.default_rng(TRAINING_SEED + 50000 + SPLITS.index(split))
    return rng.permutation(labels)


def recipe_config(name, steps_per_epoch):
    config = dict(RECIPES[name])
    config["warmup_steps"] = config["warmup_epochs"] * steps_per_epoch
    config["decay_steps"] = config["decay_epochs"] * steps_per_epoch
    return config


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_provenance():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"], text=True).strip()
        return {"commit": commit, "dirty": bool(status),
                "tracked_changes": status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "tracked_changes": None}


def store_identity(store_dir, construction, sample_a, sample_b):
    identity = {}
    for split in SPLITS:
        for sample in (sample_a, sample_b):
            npz = Path(store_dir) / f"{construction}_{sample}_{split}.npz"
            identity[f"{sample}_{split}"] = sha256_file(npz)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--construction", required=True)
    parser.add_argument("--sample-a", default="SIM")
    parser.add_argument("--sample-b", default="COUNT")
    parser.add_argument("--unit", choices=("event", "track"), default="event",
                        help="classification example: all tracks in an event or one track")
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", default="count_tracker_pfn_results")
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="fixed")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--bootstrap-draws", type=int, default=1000,
                        help="paired-event resamples for the conditional test AUC interval")
    parser.add_argument("--permute-labels", action="store_true",
                        help="label-permutation null on the same two samples")
    args = parser.parse_args()
    if args.bootstrap_draws < 0:
        parser.error("--bootstrap-draws cannot be negative")

    result_dir = Path(args.outdir) / args.label
    if result_dir.exists() and any(result_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty result dir: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)

    store_dir = Path(args.store_dir).resolve()
    raw = {split: {sample: load_store(store_dir, args.construction, sample, split)
                   for sample in (args.sample_a, args.sample_b)}
           for split in SPLITS}
    width = max(raw[s][sample][0].shape[1] for s in SPLITS
                for sample in (args.sample_a, args.sample_b))
    if width == 0:
        raise SystemExit("all track stores are empty")
    data = {}
    split_sizes = {}
    for split in SPLITS:
        tracks_a, counts_a, _ = raw[split][args.sample_a]
        tracks_b, counts_b, _ = raw[split][args.sample_b]
        if {args.sample_a, args.sample_b} == {"SIM", "COUNT"}:
            validate_sim_count_pair(raw[split][args.sample_a][2],
                                    raw[split][args.sample_b][2], split)
        if args.unit == "event":
            x, y = combine(tracks_a, counts_a, tracks_b, counts_b, width)
            weights = None
            groups = np.arange(len(y), dtype=np.int64)
        else:
            x, y, weights, groups = single_track_examples(
                tracks_a, counts_a, tracks_b, counts_b)
        if args.permute_labels:
            y = (permuted_labels(y, split) if args.unit == "event"
                 else grouped_permuted_labels(y, groups, split))
        data[split] = (x, y, weights, groups)
        split_sizes[split] = {
            "events_per_sample": int(len(counts_a)),
            "nonempty_events": {args.sample_a: int(np.count_nonzero(counts_a)),
                                args.sample_b: int(np.count_nonzero(counts_b))},
            "tracks": {args.sample_a: int(counts_a.sum()),
                       args.sample_b: int(counts_b.sum())},
            "examples": int(len(y)),
        }
        print(f"{split}: {len(y)} {args.unit} examples, width {x.shape[1]}")

    np.random.seed(TRAINING_SEED)
    import tensorflow as tf
    tf.random.set_seed(TRAINING_SEED)
    from libtest_common import build_pfn_energyflow
    try:
        from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    x_train, y_train, w_train, _ = data["train"]
    x_val, y_val, w_val, _ = data["val"]
    steps_per_epoch = int(math.ceil(len(y_train) / args.batch_size))
    config = recipe_config(args.recipe, steps_per_epoch)
    model = build_pfn_energyflow(
        input_dim=len(FEATURES), phi_sizes=PHI_SIZES, f_sizes=F_SIZES,
        jit_compile=False, lr=config["learning_rate"],
        warmup_steps=config["warmup_steps"], decay_steps=config["decay_steps"],
        min_lr=config["min_learning_rate"], f_dropouts=config["f_dropout"])

    weights = result_dir / "best.weights.h5"
    rng = np.random.default_rng(TRAINING_SEED)
    order = rng.permutation(len(y_train))
    fit_args = {}
    if w_train is not None:
        fit_args["sample_weight"] = w_train[order]
    validation_data = ((x_val, one_hot(y_val), w_val) if w_val is not None
                       else (x_val, one_hot(y_val)))
    history = model.fit(
        x_train[order], one_hot(y_train[order]),
        validation_data=validation_data,
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        **fit_args,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args.patience,
                          min_delta=1e-4, restore_best_weights=True, verbose=1),
            ModelCheckpoint(str(weights), monitor="val_loss", save_best_only=True,
                            save_weights_only=True, verbose=1)])

    x_test, y_test, w_test, groups_test = data["test"]
    test_scores = model.predict(x_test, batch_size=args.batch_size)[:, 1]
    test_weights = (w_test if w_test is not None
                    else np.ones(len(y_test), np.float32))
    test_auc = weighted_auc(y_test, test_scores, test_weights)
    test_interval = (paired_event_bootstrap(
        y_test, test_scores, test_weights, groups_test,
        split_sizes["test"]["events_per_sample"], args.bootstrap_draws,
        TRAINING_SEED + 90000) if args.bootstrap_draws else None)
    np.savez(result_dir / "test_predictions.npz", scores=test_scores,
             labels=y_test, groups=groups_test,
             weights=test_weights)

    with open(result_dir / "history.csv", "w", newline="") as handle:
        keys = list(history.history)
        writer = csv.writer(handle)
        writer.writerow(["epoch"] + keys)
        for epoch in range(len(history.history[keys[0]])):
            writer.writerow([epoch + 1] + [history.history[k][epoch] for k in keys])

    summary = {
        "label": args.label,
        "construction": args.construction,
        "samples": [args.sample_a, args.sample_b],
        "unit": args.unit,
        "label_mode": "permuted null" if args.permute_labels else "physical",
        "features": list(FEATURES),
        "physical_observables": ["pt", "eta", "phi", "d0", "z0"],
        "track_collection": "SiTracks",
        "track_state": "AtIP",
        "feature_definitions": FEATURE_DEFINITIONS,
        "architecture": {"Phi": list(PHI_SIZES), "F": list(F_SIZES),
                         "aggregation": "sum", "F_dropout": config["f_dropout"]},
        "training": {"recipe": args.recipe, "epochs_requested": args.epochs,
                     "batch_size": args.batch_size, "patience": args.patience,
                     "optimizer": "Adam", "seed": TRAINING_SEED,
                     "learning_rate": config["learning_rate"],
                     "warmup_steps": config["warmup_steps"],
                     "decay_steps": config["decay_steps"],
                     "min_learning_rate": config["min_learning_rate"],
                     "epochs_run": len(history.history["loss"])},
        "store_dir": str(store_dir),
        "store_identity": store_identity(store_dir, args.construction,
                                         args.sample_a, args.sample_b),
        "split_sizes": split_sizes,
        "weighting": ("equal total weight per nonempty event within each class"
                      if args.unit == "track" else "one example per event"),
        "estimand": ("distinguish one fitted track drawn from a uniformly chosen "
                     "nonempty source event" if args.unit == "track" else
                     "distinguish the complete reconstructed track set of one source event"),
        "source_domain": raw["test"][args.sample_a][2].get("source_domain"),
        "generator_training_holdout": raw["test"][args.sample_a][2].get(
            "generator_training_holdout"),
        "physical_event_boundaries": raw["test"][args.sample_a][2].get(
            "physical_event_boundaries"),
        "code": git_provenance(),
        "implementation": {"class": "energyflow.archs.PFN",
                           "energyflow": __import__("energyflow").__version__,
                           "tensorflow": tf.__version__},
        "selection": {"best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                      "best_val_loss": float(np.min(history.history["val_loss"]))},
        "results": {"test": {"auc": test_auc, "examples": int(len(y_test)),
                             "auc_interval_95": test_interval}},
        "uncertainty_note": (
            "The paired-event interval is conditional on the source reservoir. "
            "For norm42_reservoir, SIM rows come from the one-event model training "
            "input; classifier splits are separate draws, not unseen physical BIB."
            if args.construction.startswith("norm42_reservoir") else
            "Held-out events may reuse source cycles and are correlated"),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"test AUC = {test_auc:.6f}")
    print(f"results -> {result_dir}")


if __name__ == "__main__":
    main()
