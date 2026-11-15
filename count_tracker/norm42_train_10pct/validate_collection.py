#!/usr/bin/env python3
"""Validate one trained collection against its held-out validation rows.

The generated arm uses ``CollectionSampler`` from the production COUNT
workflow.  Consequently it loads ``model.pt`` and applies the same inverse
geometry transform, endcap z snapping, material-map rejection, and retry
logic used when constructing reconstruction inputs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
COUNT_TRACKER = HERE.parent
if str(COUNT_TRACKER) not in sys.path:
    sys.path.insert(0, str(COUNT_TRACKER))

from count_tracker_conditions import COLLECTIONS  # noqa: E402
from count_tracker_sample import (  # noqa: E402
    FEATURE_ORDER,
    CollectionSampler,
    load_paper1,
    resolve_device,
    resolve_model_dir,
    run_rejection_loop,
    sha256_file,
)


FEATURE_NAMES = ("log_energy", "time", "r", "phi", "z")
CLASSIFIER_FEATURES = (
    "log_energy", "time", "r", "sin_phi", "cos_phi", "z",
    "side", "layer", "module", "sensor",
)


def collection_seed(seed, short):
    digest = hashlib.sha256(f"{seed}|{short}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "minimum": float(np.min(values)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "maximum": float(np.max(values)),
    }


def ks_distance(left, right):
    """Return the two-sample Kolmogorov--Smirnov distance without SciPy."""
    left = np.sort(np.asarray(left, dtype=np.float64))
    right = np.sort(np.asarray(right, dtype=np.float64))
    combined = np.concatenate((left, right))
    left_cdf = np.searchsorted(left, combined, side="right") / len(left)
    right_cdf = np.searchsorted(right, combined, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def feature_report(reference, generated):
    report = {}
    for index, name in enumerate(FEATURE_NAMES):
        entry = {
            "reference": summarize(reference[:, index]),
            "generated": summarize(generated[:, index]),
        }
        if name == "phi":
            entry["comparison"] = {
                "mean_sin_reference": float(np.mean(np.sin(reference[:, index]))),
                "mean_sin_generated": float(np.mean(np.sin(generated[:, index]))),
                "mean_cos_reference": float(np.mean(np.cos(reference[:, index]))),
                "mean_cos_generated": float(np.mean(np.cos(generated[:, index]))),
                "linear_ks_note": "reported for continuity; phi is periodic",
                "linear_ks_distance": ks_distance(
                    reference[:, index], generated[:, index]
                ),
            }
        else:
            reference_std = entry["reference"]["standard_deviation"]
            generated_std = entry["generated"]["standard_deviation"]
            pooled = np.sqrt((reference_std ** 2 + generated_std ** 2) / 2)
            entry["comparison"] = {
                "ks_distance": ks_distance(reference[:, index], generated[:, index]),
                "standardized_mean_difference": (
                    float((entry["generated"]["mean"] - entry["reference"]["mean"])
                          / pooled)
                    if pooled > 0 else None
                ),
            }
        report[name] = entry
    return report


def classifier_matrix(hits):
    phi = hits[:, 3]
    return np.column_stack((
        hits[:, 0], hits[:, 1], hits[:, 2], np.sin(phi), np.cos(phi),
        hits[:, 4], hits[:, 5:9],
    )).astype(np.float32)


def classifier_auc(reference, generated, seed):
    """Fixed diagnostic C2ST; this is not the reconstruction-level PFN."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    x = np.concatenate((classifier_matrix(reference), classifier_matrix(generated)))
    y = np.concatenate((
        np.zeros(len(reference), dtype=np.int8),
        np.ones(len(generated), dtype=np.int8),
    ))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=seed, stratify=y
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    classifier.fit(x_train, y_train)
    score = classifier.predict_proba(x_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, score)),
        "purpose": "fixed hit-level two-sample diagnostic; not the track PFN",
        "features": list(CLASSIFIER_FEATURES),
        "split": "independent stratified 50/50 train/test split",
        "model": "sklearn HistGradientBoostingClassifier",
        "parameters": {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "random_state": seed,
        },
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }


def validate(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"incomplete validation already exists: {partial}")
    partial.mkdir(parents=True)

    short = args.collection
    system_id, collection_name = COLLECTIONS[short]
    model_dir = resolve_model_dir(args.model_root, short)
    dataset = model_dir / "dataset"
    checkpoint = model_dir / "model.pt"
    validation_x = np.load(dataset / "X_num_val.npy", mmap_mode="r")
    validation_y = np.load(dataset / "y_val.npy", mmap_mode="r")
    y_lookup = np.load(model_dir / "y_lookup.npy")
    if len(validation_x) != len(validation_y):
        raise ValueError("validation feature and class arrays have different lengths")
    if args.hits > len(validation_x):
        raise ValueError(
            f"requested {args.hits:,} rows but {short} has only "
            f"{len(validation_x):,} validation rows"
        )

    seed = collection_seed(args.seed, short)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(validation_x), size=args.hits, replace=False))
    local_reference = np.column_stack((
        np.asarray(validation_x[selected], dtype=np.float32),
        y_lookup[np.asarray(validation_y[selected], dtype=np.int64)],
    )).astype(np.float32)
    conditions = np.column_stack((
        np.full(args.hits, system_id, dtype=np.int64),
        local_reference[:, 5:].astype(np.int64),
    ))

    started = time.time()
    paper1 = load_paper1(args.paper1_root)
    device = resolve_device(args.device)
    sampler = CollectionSampler(
        short, model_dir, device, paper1, inference_cache="data"
    )
    reference = paper1.inverse_geometry_transform(
        local_reference, "local_phi", collection_name, FEATURE_ORDER
    ).astype(np.float32)
    generated, sampling = run_rejection_loop(
        sampler,
        conditions,
        seed,
        max_rejection_rounds=args.max_rounds,
        unfilled_policy="drop",
        return_report=True,
        work_dir=partial,
        label=f"[{short} validation] ",
    )
    if not len(generated):
        raise RuntimeError("sampler did not produce any accepted hits")
    unfilled = sampling.pop("unfilled_conditions")

    np.save(partial / "reference_hits.npy", reference, allow_pickle=False)
    np.save(partial / "generated_hits.npy", generated, allow_pickle=False)
    np.save(partial / "requested_conditions.npy", conditions, allow_pickle=False)
    np.save(partial / "unfilled_conditions.npy", unfilled, allow_pickle=False)

    report = {
        "schema_version": 1,
        "status": "complete",
        "collection": short,
        "collection_name": collection_name,
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_filename": "model.pt",
        "checkpoint_sha256": sha256_file(checkpoint),
        "sampler": str((COUNT_TRACKER / "count_tracker_sample.py").resolve()),
        "paper1_root": str(Path(args.paper1_root).resolve()),
        "validation_source": "dataset/X_num_val.npy and dataset/y_val.npy",
        "selection": {
            "method": "uniform without replacement from held-out validation rows",
            "available_rows": int(len(validation_x)),
            "requested_rows": int(args.hits),
            "base_seed": int(args.seed),
            "collection_seed": int(seed),
        },
        "sampling": {
            **sampling,
            "max_rejection_rounds": int(args.max_rounds),
            "unfilled_policy": "drop and record",
            "inference_cache": "data",
            "acceptance_fraction": float(len(generated) / args.hits),
        },
        "features": feature_report(reference, generated),
        "classifier_two_sample_test": classifier_auc(reference, generated, seed),
        "runtime_seconds": float(time.time() - started),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "outputs": {},
    }
    for name in (
        "reference_hits.npy", "generated_hits.npy", "requested_conditions.npy",
        "unfilled_conditions.npy",
    ):
        path = partial / name
        report["outputs"][name] = {
            "rows": int(len(np.load(path, mmap_mode="r", allow_pickle=False))),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    (partial / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    os.replace(partial, output)
    print(json.dumps({
        "collection": short,
        "generated": int(len(generated)),
        "unfilled": int(len(unfilled)),
        "auc": report["classifier_two_sample_test"]["roc_auc"],
        "runtime_seconds": report["runtime_seconds"],
        "output": str(output),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, choices=tuple(COLLECTIONS))
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--paper1-root", required=True)
    parser.add_argument("--hits", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.hits <= 0:
        parser.error("--hits must be positive")
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    validate(args)


if __name__ == "__main__":
    main()
