#!/usr/bin/env python3
"""Train one split-safe PFN for the N=420 reconstructed-BIB study."""

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


FEATURES = (
    "log_pt", "eta", "sin_phi", "cos_phi", "log_energy", "charge",
    "is_charged", "is_photon", "is_neutral",
)
RAW = {name: i for i, name in enumerate(
    ("pt", "eta", "phi", "energy", "mass", "charge", "type", "px", "py", "pz"))}
N_FILES = 420
EXPECTED_EVENTS = {"train": 2000, "val": 400, "test_a": 400, "test_b": 400}


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
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def load_store(path):
    with h5py.File(path, "r") as h5:
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
    pfo_type = np.abs(raw[:, :, RAW["type"]]).astype(np.int64)
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


def get_pfn(input_dim):
    """Build the EnergyFlow PFN used for the original RECO study."""
    try:
        from energyflow.archs.efn import PFN
    except ImportError:
        from energyflow.archs import PFN
    return PFN(input_dim=input_dim, Phi_sizes=(64, 64, 64),
               F_sizes=(64, 64, 64))


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
    np.random.seed(args.seed)
    import tensorflow as tf
    tf.random.set_seed(args.seed)

    store_dir = Path(args.store_dir).resolve()
    pairs = {
        split: load_pair(
            store_dir, N_FILES, args.class_a, args.class_b, split,
            EXPECTED_EVENTS[split])
        for split in ("train", "val", "test_a", "test_b")
    }
    width = max(item[0].shape[1] for pair in pairs.values() for item in pair)
    data = {split: combine_pair(pair, width) for split, pair in pairs.items()}
    for split, (x, y, _) in data.items():
        print("{}: {} events, width {}".format(split, len(y), x.shape[1]))

    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["val"]
    rng = np.random.default_rng(args.seed)
    train_order = rng.permutation(len(y_train))
    val_order = rng.permutation(len(y_val))

    result_dir = Path(args.outdir) / args.label
    result_dir.mkdir(parents=True, exist_ok=True)
    weights = result_dir / "best.weights.h5"
    model = get_pfn(len(FEATURES))
    history = model.fit(
        x_train[train_order], one_hot(y_train[train_order]),
        validation_data=(x_val[val_order], one_hot(y_val[val_order])),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        callbacks=callbacks(weights, args.patience),
    )
    if weights.is_file():
        model.model.load_weights(weights)

    with open(result_dir / "history.csv", "w", newline="") as handle:
        keys = list(history.history)
        writer = csv.writer(handle)
        writer.writerow(["epoch"] + keys)
        for epoch in range(len(history.history[keys[0]])):
            writer.writerow([epoch + 1] + [history.history[key][epoch] for key in keys])

    scores_path = result_dir / "test_scores.csv"
    if scores_path.exists():
        scores_path.unlink()
    results = {}
    combined_y = []
    combined_scores = []
    for split in ("test_a", "test_b"):
        x, y, metadata = data[split]
        auc, scores = auc_and_scores(model, x, y, args.batch_size)
        results[split] = {"auc": auc, "events": int(len(y))}
        combined_y.append(y)
        combined_scores.append(scores)
        write_scores(scores_path, split, y, scores, metadata)
        print("{} AUC = {:.6f}".format(split, auc))

    combined_y = np.concatenate(combined_y)
    combined_scores = np.concatenate(combined_scores)
    combined_auc = float(roc_auc_score(combined_y, combined_scores))
    results["combined"] = {"auc": combined_auc, "events": int(len(combined_y))}
    save_roc(result_dir / "roc.pdf", combined_y, combined_scores, combined_auc)

    summary = {
        "label": args.label,
        "class_a": args.class_a,
        "class_b": args.class_b,
        "n_files": N_FILES,
        "features": list(FEATURES),
        "architecture": {"Phi": [64, 64, 64], "F": [64, 64, 64],
                         "aggregation": "sum"},
        "implementation": {
            "class": "energyflow.archs.PFN",
            "energyflow": __import__("energyflow").__version__,
            "tensorflow": tf.__version__,
        },
        "code": git_provenance(),
        "training": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "early_stopping_monitor": "val_loss",
        },
        "seed": args.seed,
        "epochs_run": len(history.history["loss"]),
        "results": results,
        "uncertainty_note": (
            "test_a and test_b use disjoint source-cycle pools; events within "
            "each cohort may reuse files and are therefore correlated"
        ),
    }
    with open(result_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print("combined AUC = {:.6f}".format(combined_auc))
    print("results -> {}".format(result_dir))


if __name__ == "__main__":
    main()
