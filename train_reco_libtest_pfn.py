#!/usr/bin/env python3
"""Train one split-safe PFN for a fixed-size reconstructed-BIB study."""

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from libtest_common import build_pfn_energyflow
from reco_libtest_features import (
    FEATURES,
    FEATURE_DEFINITIONS,
    RAW,
    RAW_FEATURES,
    pfn_features,
)


DEFAULT_N_FILES = 420
DEFAULT_EXPECTED_EVENTS = {"train": 2000, "val": 400, "test": 800}
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
    parser.add_argument("--n-files", type=int, default=DEFAULT_N_FILES)
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--class-a", required=True)
    parser.add_argument("--class-b", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--dataset-tag",
        required=True,
        help="short identity that must also occur in --label, e.g. trackfix",
    )
    parser.add_argument(
        "--require-clean-code",
        action="store_true",
        help="refuse training when tracked git files differ from HEAD",
    )
    parser.add_argument(
        "--require-pfo-track-links",
        action="store_true",
        help="require every input store to contain linked charged PFOs",
    )
    parser.add_argument("--outdir", default="reco_pfn_results")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--train-events", type=int, default=DEFAULT_EXPECTED_EVENTS["train"]
    )
    parser.add_argument(
        "--val-events", type=int, default=DEFAULT_EXPECTED_EVENTS["val"]
    )
    parser.add_argument(
        "--test-events", type=int, default=DEFAULT_EXPECTED_EVENTS["test"]
    )
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="baseline")
    parser.add_argument(
        "--permute-labels",
        action="store_true",
        help="label-permutation null using the same two physical samples",
    )
    return parser.parse_args()


def load_store(path, expected_n_files=None):
    with h5py.File(path, "r") as h5:
        stored_n_files = h5.attrs.get("n_files")
        if (
            expected_n_files is not None
            and stored_n_files is not None
            and int(stored_n_files) != expected_n_files
        ):
            raise ValueError(
                "{} stores N={}, expected N={}".format(
                    path, int(stored_n_files), expected_n_files
                )
            )
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


def one_hot(labels):
    out = np.zeros((len(labels), 2), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def load_pair(store_dir, n_files, class_a, class_b, split, expected):
    paths = [store_dir / "n{}_{}_{}.h5".format(n_files, cls, split)
             for cls in (class_a, class_b)]
    loaded = [load_store(path, n_files) for path in paths]
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


def permuted_labels(labels, split):
    split_number = {"train": 0, "val": 1, "test": 2}[split]
    rng = np.random.default_rng(TRAINING_SEED + 50000 + split_number)
    return rng.permutation(labels)


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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_provenance():
    """Return the exact tracked-code state used by the process."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"],
            text=True,
        ).strip()
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
        return {
            "commit": commit,
            "dirty": bool(status),
            "tracked_changes": status.splitlines(),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "dirty": None,
            "tracked_changes": None,
            "tracked_diff_sha256": None,
        }


def runtime_provenance():
    keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_NODELIST",
        "CUDA_VISIBLE_DEVICES",
    )
    return {
        "started_utc": utc_now(),
        "hostname": socket.gethostname(),
        "command": [sys.executable] + sys.argv,
        "slurm": {key: os.environ.get(key) for key in keys},
    }


def json_attr(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def store_provenance(path):
    """Fingerprint one immutable HDF5 input and its reconstruction content."""
    path = Path(path).resolve()
    with h5py.File(path, "r") as h5:
        source_files = sorted({
            item.decode() if isinstance(item, bytes) else str(item)
            for item in h5["source_file"][:]
        })
        source_digest = hashlib.sha256(
            "\n".join(source_files).encode()
        ).hexdigest()
        statistics = {}
        for name in (
            "n_particles",
            "n_tracks",
            "n_clusters",
            "pfo_track_links",
        ):
            if name in h5:
                values = h5[name][:]
                statistics[name] = {
                    "total": int(np.sum(values)),
                    "mean": float(np.mean(values)),
                }
        particles_shape = [int(value) for value in h5["particles"].shape]
        attrs = {
            str(key): json_attr(value)
            for key, value in h5.attrs.items()
        }
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "particles_shape": particles_shape,
        "source_root_files": len(source_files),
        "source_file_list_sha256": source_digest,
        "attributes": attrs,
        "collection_statistics": statistics,
    }


def dataset_provenance(store_dir, n_files, class_a, class_b):
    stores = {}
    for split in ("train", "val", "test"):
        stores[split] = {}
        for class_name in (class_a, class_b):
            path = (
                Path(store_dir)
                / "n{}_{}_{}.h5".format(n_files, class_name, split)
            )
            stores[split][class_name] = store_provenance(path)
    identity = {
        split: {
            class_name: item["sha256"]
            for class_name, item in classes.items()
        }
        for split, classes in stores.items()
    }
    return {
        "store_dir": str(Path(store_dir).resolve()),
        "stores": stores,
        "identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
    }


def require_pfo_track_links(dataset):
    missing = []
    for split, classes in dataset["stores"].items():
        for class_name, store in classes.items():
            total = (
                store["collection_statistics"]
                .get("pfo_track_links", {})
                .get("total", 0)
            )
            if total <= 0:
                missing.append("{}:{}".format(split, class_name))
    if missing:
        raise ValueError(
            "track-fixed dataset required, but no PFO-track links were found "
            "in {}".format(", ".join(missing))
        )


def write_json(path, value):
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


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
    if args.n_files <= 0 or args.n_files % 42:
        raise SystemExit("--n-files must be a positive multiple of 42")
    if args.dataset_tag.lower() not in args.label.lower():
        raise SystemExit(
            "dataset tag {!r} must occur in result label {!r}".format(
                args.dataset_tag, args.label
            )
        )

    code = git_provenance()
    if args.require_clean_code and code["dirty"]:
        raise SystemExit(
            "refusing non-reproducible training with tracked code changes:\n{}"
            .format("\n".join(code["tracked_changes"]))
        )

    result_dir = Path(args.outdir) / args.label
    if result_dir.exists() and any(result_dir.iterdir()):
        raise SystemExit(
            "refusing to overwrite nonempty result directory: {}".format(
                result_dir
            )
        )
    result_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(TRAINING_SEED)
    import tensorflow as tf
    tf.random.set_seed(TRAINING_SEED)

    store_dir = Path(args.store_dir).resolve()
    dataset = dataset_provenance(
        store_dir, args.n_files, args.class_a, args.class_b
    )
    if args.require_pfo_track_links:
        require_pfo_track_links(dataset)
    expected_events = {
        split: getattr(args, "{}_events".format(split))
        for split in ("train", "val", "test")
    }
    if any(value <= 0 for value in expected_events.values()):
        raise SystemExit("train, validation, and test event counts must be positive")
    runtime = runtime_provenance()
    run_context = {
        "status": "started",
        "label": args.label,
        "dataset_tag": args.dataset_tag,
        "classes": [args.class_a, args.class_b],
        "label_mode": (
            "deterministic split-wise permutation"
            if args.permute_labels else "physical class"
        ),
        "n_files": args.n_files,
        "dataset": dataset,
        "features": list(FEATURES),
        "feature_definitions": FEATURE_DEFINITIONS,
        "code": code,
        "runtime": runtime,
        "requested_training": {
            "recipe": args.recipe,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": TRAINING_SEED,
            "require_pfo_track_links": args.require_pfo_track_links,
            "events_per_class": expected_events,
        },
    }
    write_json(result_dir / "run_context.json", run_context)
    print(
        "dataset tag={} identity={} store_dir={}".format(
            args.dataset_tag, dataset["identity_sha256"], store_dir
        )
    )

    pairs = {
        split: load_pair(
            store_dir, args.n_files, args.class_a, args.class_b, split,
            expected_events[split])
        for split in ("train", "val", "test")
    }
    width = max(item[0].shape[1] for pair in pairs.values() for item in pair)
    data = {split: combine_pair(pair, width) for split, pair in pairs.items()}
    if args.permute_labels:
        data = {
            split: (values[0], permuted_labels(values[1], split), values[2])
            for split, values in data.items()
        }
    for split, (x, y, _) in data.items():
        print("{}: {} events, width {}".format(split, len(y), x.shape[1]))

    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["val"]
    rng = np.random.default_rng(TRAINING_SEED)
    train_order = rng.permutation(len(y_train))
    val_order = rng.permutation(len(y_val))

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
    roc_path = result_dir / "roc.pdf"
    save_roc(roc_path, y_test, test_scores, test_auc)

    history_path = result_dir / "history.csv"
    artifacts = {
        "best_weights": {
            "path": str(weights.resolve()),
            "sha256": sha256_file(weights),
            "bytes": weights.stat().st_size,
        },
        "history": {
            "path": str(history_path.resolve()),
            "sha256": sha256_file(history_path),
        },
        "test_scores": {
            "path": str(scores_path.resolve()),
            "sha256": sha256_file(scores_path),
        },
        "roc": {
            "path": str(roc_path.resolve()),
            "sha256": sha256_file(roc_path),
        },
    }

    summary = {
        "label": args.label,
        "dataset_tag": args.dataset_tag,
        "class_a": args.class_a,
        "class_b": args.class_b,
        "label_mode": (
            "deterministic split-wise permutation"
            if args.permute_labels else "physical class"
        ),
        "n_files": args.n_files,
        "dataset": dataset,
        "features": list(FEATURES),
        "feature_definitions": FEATURE_DEFINITIONS,
        "feature_preprocessing": {
            "clipping": False,
            "learned_normalization": False,
            "padding_value": 0.0,
        },
        "architecture": {"Phi": list(PHI_SIZES), "F": list(F_SIZES),
                         "aggregation": "sum",
                         "F_dropout": training_config["f_dropout"]},
        "implementation": {
            "class": "energyflow.archs.PFN",
            "energyflow": __import__("energyflow").__version__,
            "tensorflow": tf.__version__,
        },
        "code": code,
        "runtime": runtime,
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
            "events_per_class": expected_events,
        },
        "seed": TRAINING_SEED,
        "label_permutation_seed": (
            "training_seed + 50000 + split_index"
            if args.permute_labels else None
        ),
        "epochs_run": len(history.history["loss"]),
        "selection": {
            "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
            "best_val_loss": float(np.min(history.history["val_loss"])),
        },
        "results": {"test": {"auc": test_auc, "events": int(len(y_test))}},
        "artifacts": artifacts,
        "uncertainty_note": (
            "held-out events may reuse source files and are therefore correlated"
        ),
    }
    write_json(result_dir / "summary.json", summary)
    run_context["status"] = "complete"
    run_context["completed_utc"] = utc_now()
    run_context["summary"] = str((result_dir / "summary.json").resolve())
    run_context["artifacts"] = artifacts
    write_json(result_dir / "run_context.json", run_context)
    print("test AUC = {:.6f}".format(test_auc))
    print("results -> {}".format(result_dir))


if __name__ == "__main__":
    main()
