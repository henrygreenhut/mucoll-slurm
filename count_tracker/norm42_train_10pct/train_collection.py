#!/usr/bin/env python3
"""Train one 10%-corpus collection with the recorded Paper1 TabDDPM code."""

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys

import numpy as np


PAPER1_COMMIT = "90bd576c619416dbd7889eb1900ec80ac687ab92"
COLLECTIONS = ("VBC", "VEC", "ITBC", "ITEC", "OTBC", "OTEC")
STEPS = 200_000
NUM_TIMESTEPS = 1_000
BATCH_SIZE = 4_096
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0
LAYERS = (2_048, 2_048, 2_048, 2_048, 2_048)
DIM_T = 1_024
SEED = 42


def sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paper1_commit(root):
    marker = Path(root) / ".source_commit"
    if not marker.is_file():
        raise FileNotFoundError(f"missing Paper1 commit marker: {marker}")
    return marker.read_text().strip()


def model_name(collection):
    return (
        f"{collection}_TABDDPM_norm42_10pct_local_phi_"
        "cond-side-layer-module-sensor_"
        "t1000_s200000_h2048x5_dim1024_b4096"
    )


def validate_dataset(dataset_root, collection):
    collection_root = Path(dataset_root).resolve() / collection
    dataset = collection_root / "dataset"
    split_manifest_path = collection_root / "split_manifest.json"
    required = (
        "X_num_train.npy", "X_num_val.npy", "X_num_test.npy",
        "y_train.npy", "y_val.npy", "y_test.npy", "info.json",
    )
    for name in required:
        if not (dataset / name).is_file():
            raise FileNotFoundError(dataset / name)
    if not (collection_root / "y_lookup.npy").is_file():
        raise FileNotFoundError(collection_root / "y_lookup.npy")
    if not split_manifest_path.is_file():
        raise FileNotFoundError(split_manifest_path)

    info = json.loads((dataset / "info.json").read_text())
    split_manifest = json.loads(split_manifest_path.read_text())
    x_train = np.load(dataset / "X_num_train.npy", mmap_mode="r", allow_pickle=False)
    x_val = np.load(dataset / "X_num_val.npy", mmap_mode="r", allow_pickle=False)
    y_train = np.load(dataset / "y_train.npy", mmap_mode="r", allow_pickle=False)
    y_val = np.load(dataset / "y_val.npy", mmap_mode="r", allow_pickle=False)
    y_lookup = np.load(collection_root / "y_lookup.npy", mmap_mode="r", allow_pickle=False)
    expected = {
        "train_size": len(x_train),
        "val_size": len(x_val),
        "n_classes": len(y_lookup),
        "n_num_features": x_train.shape[1],
    }
    for key, value in expected.items():
        if int(info[key]) != int(value):
            raise ValueError(
                f"{collection} info {key}={info[key]} but arrays imply {value}"
            )
    if x_train.dtype != np.float32 or x_val.dtype != np.float32:
        raise ValueError(f"{collection}: numerical arrays must be float32")
    if y_train.dtype != np.int64 or y_val.dtype != np.int64:
        raise ValueError(f"{collection}: class arrays must be int64")
    if len(y_train) != len(x_train) or len(y_val) != len(x_val):
        raise ValueError(f"{collection}: numerical/class row mismatch")
    if y_train.min() != 0 or y_train.max() != len(y_lookup) - 1:
        raise ValueError(f"{collection}: training does not span all classes")
    if split_manifest["checks"]["training_covers_all_conditions"] is not True:
        raise ValueError(f"{collection}: split manifest failed condition coverage")
    return collection_root, dataset, info, split_manifest_path


def import_train(paper1_root):
    paper1_root = Path(paper1_root).resolve()
    commit = paper1_commit(paper1_root)
    if commit != PAPER1_COMMIT:
        raise ValueError(f"Paper1 commit {commit}; expected {PAPER1_COMMIT}")
    tabddpm = paper1_root / "diffusion" / "tabddpm_official"
    scripts = tabddpm / "scripts"
    if not scripts.is_dir():
        raise FileNotFoundError(scripts)
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(tabddpm))
    return importlib.import_module("train").train


def train_collection(collection, dataset_root, paper1_root, output_root):
    import torch

    if collection not in COLLECTIONS:
        raise ValueError(f"unknown collection {collection}")
    collection_root, dataset, info, split_manifest_path = validate_dataset(
        dataset_root, collection
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TabDDPM training")

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / model_name(collection)
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(destination if destination.exists() else partial)
    partial.mkdir()

    try:
        shutil.copytree(dataset, partial / "dataset")
        shutil.copy2(collection_root / "y_lookup.npy", partial / "y_lookup.npy")
        shutil.copy2(split_manifest_path, partial / "split_manifest.json")

        train = import_train(paper1_root)
        model_params = {
            "num_classes": int(info["n_classes"]),
            "is_y_cond": True,
            "rtdl_params": {
                "d_layers": list(LAYERS),
                "dropout": 0.0,
            },
            "dim_t": DIM_T,
        }
        transforms = {
            "seed": SEED,
            "normalization": "quantile",
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": None,
            "y_policy": "default",
        }
        train(
            parent_dir=str(partial),
            real_data_path=str(partial / "dataset"),
            steps=STEPS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            batch_size=BATCH_SIZE,
            model_type="mlp",
            model_params=model_params,
            num_timesteps=NUM_TIMESTEPS,
            gaussian_loss_type="mse",
            scheduler="cosine",
            T_dict=transforms,
            num_numerical_features=5,
            device=torch.device("cuda"),
            seed=SEED,
            change_val=False,
        )

        for name in ("model.pt", "model_ema.pt", "loss.csv"):
            path = partial / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"training did not produce {path}")
        loss_lines = (partial / "loss.csv").read_text().strip().splitlines()
        if not loss_lines[-1].startswith("200000"):
            raise RuntimeError("loss history does not end at step 200000")

        run_config = {
            "status": "training_completed",
            "collection": collection,
            "paper1_commit": PAPER1_COMMIT,
            "dataset_source": str(collection_root),
            "dataset_split_manifest_sha256": sha256(split_manifest_path),
            "features": ["logE", "t", "r", "local_phi", "z"],
            "conditions": ["side", "layer", "module", "sensor"],
            "steps": STEPS,
            "num_timesteps": NUM_TIMESTEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "layers": list(LAYERS),
            "dropout": 0.0,
            "dim_t": DIM_T,
            "normalization": "quantile",
            "scheduler": "cosine",
            "learning_rate_schedule": "linear decay implemented by recorded trainer",
            "seed": SEED,
            "checkpoint_resume": False,
            "posthoc_calibration": False,
            "runtime": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            },
            "outputs": {
                name: {"bytes": (partial / name).stat().st_size,
                       "sha256": sha256(partial / name)}
                for name in ("model.pt", "model_ema.pt", "loss.csv", "y_lookup.npy")
            },
        }
        (partial / "run_config.json").write_text(
            json.dumps(run_config, indent=2) + "\n"
        )
        os.replace(partial, destination)
        print(f"Training complete -> {destination}", flush=True)
    except Exception:
        (partial / "INCOMPLETE").write_text(
            "Training failed; do not use this directory.\n"
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, choices=COLLECTIONS)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--paper1-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    train_collection(
        args.collection, args.dataset_root, args.paper1_root, args.output_root
    )


if __name__ == "__main__":
    main()
