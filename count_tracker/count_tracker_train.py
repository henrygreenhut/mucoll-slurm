#!/usr/bin/env python3
"""Train a SIM-vs-COUNT track PFN on Kinematic-7 stores.

Loads the ``.npz`` track stores written by count_tracker_features.py for two
samples (default SIM vs COUNT) across train/val/test, applies the Kinematic-7
transform, and trains the standard EnergyFlow PFN (reusing the RECO study's
``build_pfn_energyflow``). Reports held-out test AUC and a provenance summary.
The label-permutation null (--permute-labels) reuses the same two samples with
deterministically shuffled labels.

Runs in the PFN training env (energyflow + tf_keras + sklearn), on GPU.
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

TRAINING_SEED = 12345
PHI_SIZES = (64, 64, 64)
F_SIZES = (64, 64, 64)
SPLITS = ("train", "val", "test")
RECIPES = {
    "stabilized_dropout": {"learning_rate": 1e-4, "warmup_epochs": 1,
                           "decay_epochs": 30, "min_learning_rate": 1e-6,
                           "f_dropout": 0.1},
    "stabilized": {"learning_rate": 1e-4, "warmup_epochs": 1, "decay_epochs": 30,
                   "min_learning_rate": 1e-6, "f_dropout": 0.0},
}


def load_store(store_dir, construction, sample, split):
    prefix = Path(store_dir) / f"{construction}_{sample}_{split}"
    manifest = json.loads(prefix.with_suffix(".json").read_text())
    if tuple(manifest["features"]) != RAW_FEATURES:
        raise SystemExit(f"{prefix}.json has unexpected features {manifest['features']}")
    tracks = np.load(prefix.with_suffix(".npz"))["tracks"].astype(np.float32)
    return tracks, manifest


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


def combine(raw_a, raw_b, width):
    xa = pfn_features(pad_width(raw_a, width))
    xb = pfn_features(pad_width(raw_b, width))
    if len(xa) != len(xb):
        raise SystemExit(f"class counts differ: {len(xa)} vs {len(xb)}")
    x = np.concatenate([xa, xb])
    y = np.asarray([0] * len(xa) + [1] * len(xb), dtype=np.int32)
    return x, y


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
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", default="count_tracker_pfn_results")
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="stabilized_dropout")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--permute-labels", action="store_true",
                        help="label-permutation null on the same two samples")
    args = parser.parse_args()

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
    data = {}
    for split in SPLITS:
        x, y = combine(raw[split][args.sample_a][0], raw[split][args.sample_b][0], width)
        if args.permute_labels:
            y = permuted_labels(y, split)
        data[split] = (x, y)
        print(f"{split}: {len(y)} events, width {width}")

    np.random.seed(TRAINING_SEED)
    import tensorflow as tf
    tf.random.set_seed(TRAINING_SEED)
    from libtest_common import build_pfn_energyflow
    from sklearn.metrics import roc_auc_score
    try:
        from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    x_train, y_train = data["train"]
    x_val, y_val = data["val"]
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
    history = model.fit(
        x_train[order], one_hot(y_train[order]),
        validation_data=(x_val, one_hot(y_val)),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args.patience,
                          min_delta=1e-4, restore_best_weights=True, verbose=1),
            ModelCheckpoint(str(weights), monitor="val_loss", save_best_only=True,
                            save_weights_only=True, verbose=1)])

    x_test, y_test = data["test"]
    test_scores = model.predict(x_test, batch_size=args.batch_size)[:, 1]
    test_auc = float(roc_auc_score(y_test, test_scores))

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
        "label_mode": "permuted null" if args.permute_labels else "physical",
        "features": list(FEATURES),
        "feature_definitions": FEATURE_DEFINITIONS,
        "architecture": {"Phi": list(PHI_SIZES), "F": list(F_SIZES),
                         "aggregation": "sum", "F_dropout": config["f_dropout"]},
        "training": {"recipe": args.recipe, "epochs_requested": args.epochs,
                     "batch_size": args.batch_size, "patience": args.patience,
                     "optimizer": "Adam", "seed": TRAINING_SEED,
                     "epochs_run": len(history.history["loss"])},
        "store_dir": str(store_dir),
        "store_identity": store_identity(store_dir, args.construction,
                                         args.sample_a, args.sample_b),
        "events_per_split": {s: int(len(data[s][1])) for s in SPLITS},
        "code": git_provenance(),
        "implementation": {"class": "energyflow.archs.PFN",
                           "energyflow": __import__("energyflow").__version__,
                           "tensorflow": tf.__version__},
        "selection": {"best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                      "best_val_loss": float(np.min(history.history["val_loss"]))},
        "results": {"test": {"auc": test_auc, "events": int(len(y_test))}},
        "uncertainty_note": "held-out events may reuse source cycles and are correlated",
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"test AUC = {test_auc:.6f}")
    print(f"results -> {result_dir}")


if __name__ == "__main__":
    main()
