#!/usr/bin/env python3
"""Evaluate a frozen RECO PFN on a separately produced confirmation cohort."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from reco_libtest_features import FEATURES, FEATURE_DEFINITIONS
from train_reco_libtest_pfn import (
    auc_and_scores,
    combine_pair,
    get_pfn,
    load_store,
    save_roc,
    underlying_model,
    write_scores,
)


N_FILES = 420
JOB_PATTERN = re.compile(r"(?:^|/)job_(\d+)(?:/|$)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--class-b", choices=("R", "null_b"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", default="reco_pfn_results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def job_key(source_file):
    match = JOB_PATTERN.search(str(source_file))
    if not match:
        raise ValueError(
            "cannot extract reconstruction job ID from {!r}".format(source_file)
        )
    return int(match.group(1))


def binary_crossentropy(labels, scores):
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return float(np.mean(
        -(labels * np.log(scores) + (1.0 - labels) * np.log(1.0 - scores))
    ))


def score_summary(labels, scores, class_b):
    result = {}
    for class_id, name in ((0, "U"), (1, class_b)):
        values = np.asarray(scores)[np.asarray(labels) == class_id]
        result[name] = {
            "events": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "fraction_above_0p5": float(np.mean(values > 0.5)),
        }
    return result


def paired_job_bootstrap(labels, scores, metadata, repetitions, seed):
    """Bootstrap paired 50-event reconstruction jobs, not individual events."""
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    groups = {0: {}, 1: {}}
    for index, (class_id, source_file, _) in enumerate(metadata):
        key = job_key(source_file)
        groups[int(class_id)].setdefault(key, []).append(index)

    keys_a = set(groups[0])
    keys_b = set(groups[1])
    if keys_a != keys_b:
        raise ValueError(
            "confirmation classes do not have paired job IDs: "
            "{} U-only, {} class-B-only".format(
                len(keys_a - keys_b), len(keys_b - keys_a)
            )
        )
    keys = np.asarray(sorted(keys_a), dtype=np.int64)
    if len(keys) < 2:
        raise ValueError("at least two paired reconstruction jobs are required")

    indices = {
        class_id: {
            key: np.asarray(values, dtype=np.int64)
            for key, values in by_key.items()
        }
        for class_id, by_key in groups.items()
    }
    rng = np.random.default_rng(seed)
    aucs = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        selected = np.concatenate(
            [
                np.concatenate([indices[class_id][int(key)] for key in sampled])
                for class_id in (0, 1)
            ]
        )
        aucs[repetition] = roc_auc_score(labels[selected], scores[selected])
    return aucs, int(len(keys))


def validate_checkpoint(summary, class_b):
    if summary.get("class_a") != "U" or summary.get("class_b") != class_b:
        raise ValueError(
            "checkpoint classes are {} vs {}, requested U vs {}".format(
                summary.get("class_a"), summary.get("class_b"), class_b
            )
        )
    if summary.get("n_files") != N_FILES:
        raise ValueError("checkpoint is not an N=420 model")
    if tuple(summary.get("features", ())) != FEATURES:
        raise ValueError("checkpoint feature list does not match current features")
    if summary.get("feature_definitions") != FEATURE_DEFINITIONS:
        raise ValueError(
            "checkpoint feature definitions do not match current preprocessing"
        )
    recipe = summary.get("training", {}).get("recipe")
    if recipe not in ("stabilized", "stabilized_dropout"):
        raise ValueError(
            "confirmation evaluator requires a stabilized checkpoint; got {!r}"
            .format(recipe)
        )
    return recipe


def steps_per_epoch(summary):
    training = summary["training"]
    if training.get("warmup_epochs", 0):
        return int(training["warmup_steps"] // training["warmup_epochs"])
    return max(int(training.get("decay_steps", 1)), 1)


def load_confirmation(store_dir, class_b):
    paths = [
        store_dir / "n{}_{}_confirmation.h5".format(N_FILES, class_name)
        for class_name in ("U", class_b)
    ]
    pair = [load_store(path) for path in paths]
    if len(pair[0][0]) != len(pair[1][0]):
        raise ValueError(
            "confirmation class counts differ: {} vs {}".format(
                len(pair[0][0]), len(pair[1][0])
            )
        )
    width = max(item[0].shape[1] for item in pair)
    return combine_pair(pair, width)


def main():
    args = parse_args()
    store_dir = Path(args.store_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_summary_path = checkpoint_dir / "summary.json"
    weights = checkpoint_dir / "best.weights.h5"
    if not checkpoint_summary_path.is_file() or not weights.is_file():
        raise SystemExit(
            "checkpoint directory must contain summary.json and best.weights.h5: {}"
            .format(checkpoint_dir)
        )

    with checkpoint_summary_path.open() as handle:
        checkpoint_summary = json.load(handle)
    recipe = validate_checkpoint(checkpoint_summary, args.class_b)

    x, labels, metadata = load_confirmation(store_dir, args.class_b)
    print(
        "confirmation: {} events/class, width {}, {} features".format(
            len(labels) // 2, x.shape[1], x.shape[2]
        )
    )

    model = get_pfn(
        len(FEATURES), recipe, steps_per_epoch(checkpoint_summary)
    )
    underlying_model(model).load_weights(str(weights))
    auc, scores = auc_and_scores(model, x, labels, args.batch_size)
    loss = binary_crossentropy(labels, scores)
    accuracy = float(np.mean((scores >= 0.5) == labels))

    bootstrap_aucs, n_jobs = paired_job_bootstrap(
        labels,
        scores,
        metadata,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )
    low, high = np.percentile(bootstrap_aucs, [2.5, 97.5])

    result_dir = Path(args.outdir) / args.label
    result_dir.mkdir(parents=True, exist_ok=True)
    scores_path = result_dir / "confirmation_scores.csv"
    if scores_path.exists():
        scores_path.unlink()
    write_scores(scores_path, "confirmation", labels, scores, metadata)
    save_roc(result_dir / "confirmation_roc.pdf", labels, scores, auc)
    with (result_dir / "job_bootstrap_auc.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["repetition", "auc"])
        writer.writerows(enumerate(bootstrap_aucs))

    summary = {
        "label": args.label,
        "evaluation": "frozen-checkpoint confirmation",
        "class_a": "U",
        "class_b": args.class_b,
        "n_files": N_FILES,
        "events_per_class": int(len(labels) // 2),
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "weights": str(weights),
            "weights_sha256": sha256(weights),
            "training_code": checkpoint_summary.get("code"),
            "recipe": recipe,
            "original_test": checkpoint_summary.get("results", {}).get("test"),
        },
        "results": {
            "auc": auc,
            "binary_crossentropy": loss,
            "accuracy_at_0p5": accuracy,
            "score_summary": score_summary(labels, scores, args.class_b),
        },
        "uncertainty": {
            "method": "paired reconstruction-job bootstrap",
            "paired_jobs": n_jobs,
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
            "auc_standard_deviation": float(np.std(bootstrap_aucs, ddof=1)),
            "auc_95_percentile_interval": [float(low), float(high)],
            "limitation": (
                "This captures reconstruction-job variation but not uncertainty "
                "from repeatedly drawing the same finite held-out BIB source pool."
            ),
        },
    }
    with (result_dir / "confirmation_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("confirmation AUC = {:.6f}".format(auc))
    print("confirmation loss = {:.6f} (ln2 = {:.6f})".format(
        loss, np.log(2.0)))
    print(
        "paired-job bootstrap 95% interval = [{:.6f}, {:.6f}]".format(
            low, high
        )
    )
    print("results -> {}".format(result_dir))


if __name__ == "__main__":
    main()
