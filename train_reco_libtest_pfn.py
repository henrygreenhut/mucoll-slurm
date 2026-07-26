#!/usr/bin/env python3
"""Train one split-safe PFN for the N=420 reconstructed-BIB study."""

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from libtest_common import build_pfn_energyflow


FEATURES = (
    "log_pt", "eta", "sin_phi", "cos_phi", "log_energy", "charge",
    "is_charged", "is_photon", "is_neutral",
)
RAW_FEATURES = (
    "pt", "eta", "phi", "energy", "mass", "charge", "pdg", "px", "py", "pz",
)
RAW = {name: i for i, name in enumerate(RAW_FEATURES)}
N_FILES = 420
EXPECTED_EVENTS = {"train": 2000, "val": 400, "test": 800}
TRAINING_SEED = 12345
PHI_SIZES = (64, 64, 64)
F_SIZES = (64, 64, 64)
RECIPES = {
    # Preserve the original Perlmutter EnergyFlow behavior for the OSCAR
    # reproduction: PFN's internal compile, fixed Adam LR=1e-3, and no
    # regularization.
    "baseline": {
        "learning_rate": 1e-3,
        "warmup_epochs": 0,
        "decay_epochs": 0,
        "min_learning_rate": 0.0,
        "f_dropout": 0.0,
        "explicit_compile": False,
    },
    "stabilized": {
        "learning_rate": 1e-4,
        "warmup_epochs": 1,
        "decay_epochs": 30,
        "min_learning_rate": 1e-6,
        "f_dropout": 0.0,
        "explicit_compile": True,
    },
    "stabilized_dropout": {
        "learning_rate": 1e-4,
        "warmup_epochs": 1,
        "decay_epochs": 30,
        "min_learning_rate": 1e-6,
        "f_dropout": 0.1,
        "explicit_compile": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--class-a", required=True)
    parser.add_argument("--class-b", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", default="reco_pfn_results")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="baseline")
    return parser.parse_args()


def load_store(path):
    with h5py.File(path, "r") as h5:
        feature_names = h5.attrs.get("features", "")
        if isinstance(feature_names, bytes):
            feature_names = feature_names.decode()
        if tuple(feature_names.split(",")) != RAW_FEATURES:
            raise ValueError(
                "{} has unexpected features {!r}; expected {!r}".format(
                    path, feature_names, ",".join(RAW_FEATURES)))
        particles = h5["particles"][:].astype(np.float32)
        source_file = h5["source_file"][:]
        source_event = h5["source_event"][:]
    return particles, source_file, source_event


def pad_width(array, width):
    if array.shape[1] == width:
        return array
    out = np.zeros((len(array), width, array.shape[2]), dtype=np.float32)
    out[:, :array.shape[1]] = array
    return out


def pfn_features(raw):
    mask = raw[:, :, RAW["pt"]] > 0
    out = np.zeros((len(raw), raw.shape[1], len(FEATURES)), dtype=np.float32)
    pt = np.maximum(raw[:, :, RAW["pt"]], 0)
    eta = raw[:, :, RAW["eta"]]
    phi = raw[:, :, RAW["phi"]]
    energy = np.maximum(raw[:, :, RAW["energy"]], 0)
    charge = raw[:, :, RAW["charge"]]
    pfo_type = np.abs(raw[:, :, RAW["pdg"]]).astype(np.int64)
    charged = np.abs(charge) > 0.1
    photon = (~charged) & (pfo_type == 22)
    neutral = (~charged) & (~photon)

    values = (
        np.log1p(pt) / 6.0,
        np.clip(eta / 5.0, -2.0, 2.0),
        np.sin(phi),
        np.cos(phi),
        np.log1p(energy) / 6.0,
        np.clip(charge, -3.0, 3.0) / 3.0,
        charged.astype(np.float32),
        photon.astype(np.float32),
        neutral.astype(np.float32),
    )
    for index, value in enumerate(values):
        out[:, :, index][mask] = value[mask]
    return out


def one_hot(labels):
    out = np.zeros((len(labels), 2), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def load_pair(store_dir, n_files, class_a, class_b, split, expected):
    paths = [store_dir / "n{}_{}_{}.h5".format(n_files, cls, split)
             for cls in (class_a, class_b)]
    loaded = [load_store(path) for path in paths]
    if len(loaded[0][0]) != len(loaded[1][0]):
        raise SystemExit("{} class counts differ: {} vs {}".format(
            split, len(loaded[0][0]), len(loaded[1][0])))
    if len(loaded[0][0]) != expected:
        raise SystemExit("{} has {} events/class; expected {}".format(
            split, len(loaded[0][0]), expected))
    return loaded


def combine_pair(pair, width):
    arrays = [pfn_features(pad_width(item[0], width)) for item in pair]
    x = np.concatenate(arrays)
    n = len(arrays[0])
    y = np.asarray([0] * n + [1] * n, dtype=np.int32)
    metadata = []
    for class_id, (_, files, events) in enumerate(pair):
        metadata.extend((class_id, f.decode() if isinstance(f, bytes) else str(f), int(e))
                        for f, e in zip(files, events))
    return x, y, metadata


def recipe_config(name, steps_per_epoch):
    config = dict(RECIPES[name])
    config["warmup_steps"] = config["warmup_epochs"] * steps_per_epoch
    config["decay_steps"] = config["decay_epochs"] * steps_per_epoch
    config["jit_compile"] = False if config["explicit_compile"] else None
    config["clipnorm"] = 0.0
    return config


def get_pfn(input_dim, recipe, steps_per_epoch):
    """Build the requested standard EnergyFlow PFN recipe."""
    config = recipe_config(recipe, steps_per_epoch)
    if config["explicit_compile"]:
        return build_pfn_energyflow(
            input_dim=input_dim,
            phi_sizes=PHI_SIZES,
            f_sizes=F_SIZES,
            jit_compile=False,
            lr=config["learning_rate"],
            warmup_steps=config["warmup_steps"],
            decay_steps=config["decay_steps"],
            min_lr=config["min_learning_rate"],
            clipnorm=0.0,
            f_dropouts=config["f_dropout"],
        )

    # Do not route the reproduction through the new optimizer builder:
    # leaving EnergyFlow's original internal compile untouched is part of
    # checking the Perlmutter setup on OSCAR.
    try:
        from energyflow.archs.efn import PFN
    except ImportError:
        from energyflow.archs import PFN
    return PFN(input_dim=input_dim, Phi_sizes=PHI_SIZES,
               F_sizes=F_SIZES)


def underlying_model(model):
    """Return Keras model while preserving EnergyFlow's baseline wrapper."""
    return getattr(model, "model", model)


def git_provenance():
    """Return the checked-out revision and whether tracked code is modified."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"],
            text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def callbacks(weights, patience):
    try:
        from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    return [
        EarlyStopping(monitor="val_loss", patience=patience, min_delta=1e-4,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(weights), monitor="val_loss", save_best_only=True,
                        save_weights_only=True, verbose=1),
    ]


def auc_and_scores(model, x, y, batch_size):
    scores = model.predict(x, batch_size=batch_size)[:, 1]
    return float(roc_auc_score(y, scores)), scores


def write_scores(path, cohort, y, scores, metadata):
    with open(path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(["cohort", "true_label", "score", "source_file", "source_event"])
        for label, score, (_, source_file, source_event) in zip(y, scores, metadata):
            writer.writerow([cohort, int(label), "{:.12g}".format(score),
                             source_file, source_event])


def save_roc(path, y, scores, auc):
    fpr, tpr, _ = roc_curve(y, scores)
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(fpr, tpr, label="PFN (AUC={:.3f})".format(auc))
    plt.plot([0, 1], [0, 1], "--", color="0.5")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()
    np.random.seed(TRAINING_SEED)
    import tensorflow as tf
    tf.random.set_seed(TRAINING_SEED)

    store_dir = Path(args.store_dir).resolve()
    pairs = {
        split: load_pair(
            store_dir, N_FILES, args.class_a, args.class_b, split,
            EXPECTED_EVENTS[split])
        for split in ("train", "val", "test")
    }
    width = max(item[0].shape[1] for pair in pairs.values() for item in pair)
    data = {split: combine_pair(pair, width) for split, pair in pairs.items()}
    for split, (x, y, _) in data.items():
        print("{}: {} events, width {}".format(split, len(y), x.shape[1]))

    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["val"]
    rng = np.random.default_rng(TRAINING_SEED)
    train_order = rng.permutation(len(y_train))
    val_order = rng.permutation(len(y_val))

    result_dir = Path(args.outdir) / args.label
    result_dir.mkdir(parents=True, exist_ok=True)
    weights = result_dir / "best.weights.h5"
    steps_per_epoch = int(math.ceil(len(y_train) / args.batch_size))
    training_config = recipe_config(args.recipe, steps_per_epoch)
    print("recipe {}: {}".format(args.recipe, training_config))
    model = get_pfn(len(FEATURES), args.recipe, steps_per_epoch)
    history = model.fit(
        x_train[train_order], one_hot(y_train[train_order]),
        validation_data=(x_val[val_order], one_hot(y_val[val_order])),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        callbacks=callbacks(weights, args.patience),
    )
    if weights.is_file():
        underlying_model(model).load_weights(weights)

    with open(result_dir / "history.csv", "w", newline="") as handle:
        keys = list(history.history)
        writer = csv.writer(handle)
        writer.writerow(["epoch"] + keys)
        for epoch in range(len(history.history[keys[0]])):
            writer.writerow([epoch + 1] + [history.history[key][epoch] for key in keys])

    scores_path = result_dir / "test_scores.csv"
    if scores_path.exists():
        scores_path.unlink()
    x_test, y_test, test_metadata = data["test"]
    test_auc, test_scores = auc_and_scores(
        model, x_test, y_test, args.batch_size)
    write_scores(scores_path, "test", y_test, test_scores, test_metadata)
    save_roc(result_dir / "roc.pdf", y_test, test_scores, test_auc)

    summary = {
        "label": args.label,
        "class_a": args.class_a,
        "class_b": args.class_b,
        "n_files": N_FILES,
        "features": list(FEATURES),
        "architecture": {"Phi": list(PHI_SIZES), "F": list(F_SIZES),
                         "aggregation": "sum",
                         "F_dropout": training_config["f_dropout"]},
        "implementation": {
            "class": "energyflow.archs.PFN",
            "energyflow": __import__("energyflow").__version__,
            "tensorflow": tf.__version__,
        },
        "code": git_provenance(),
        "training": {
            "recipe": args.recipe,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "early_stopping_monitor": "val_loss",
            "optimizer": "Adam",
            "learning_rate": training_config["learning_rate"],
            "warmup_epochs": training_config["warmup_epochs"],
            "warmup_steps": training_config["warmup_steps"],
            "decay": ("cosine" if training_config["decay_steps"] else "none"),
            "decay_epochs": training_config["decay_epochs"],
            "decay_steps": training_config["decay_steps"],
            "min_learning_rate": training_config["min_learning_rate"],
            "clipnorm": training_config["clipnorm"],
            "jit_compile": training_config["jit_compile"],
        },
        "seed": TRAINING_SEED,
        "epochs_run": len(history.history["loss"]),
        "results": {"test": {"auc": test_auc, "events": int(len(y_test))}},
        "uncertainty_note": (
            "held-out events may reuse source files and are therefore correlated"
        ),
    }
    with open(result_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print("test AUC = {:.6f}".format(test_auc))
    print("results -> {}".format(result_dir))


if __name__ == "__main__":
    main()
