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
    git_provenance,
    get_pfn,
    load_store,
    require_pfo_track_links,
    runtime_provenance,
    save_roc,
    sha256_file,
    store_provenance,
    underlying_model,
    utc_now,
    write_json,
    write_scores,
)


DEFAULT_N_FILES = 420
JOB_PATTERN = re.compile(r"(?:^|/)job_(\d+)(?:/|$)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=DEFAULT_N_FILES)
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--class-b", choices=("R", "null_b"), required=True)
    parser.add_argument("--expected-training-dataset-tag", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--require-clean-code", action="store_true")
    parser.add_argument("--outdir", default="reco_pfn_results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args()


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


def validate_checkpoint(
    summary, n_files, class_b, expected_dataset_tag, weights
):
    if summary.get("class_a") != "U" or summary.get("class_b") != class_b:
        raise ValueError(
            "checkpoint classes are {} vs {}, requested U vs {}".format(
                summary.get("class_a"), summary.get("class_b"), class_b
            )
        )
    if summary.get("n_files") != n_files:
        raise ValueError(
            "checkpoint N={} does not match requested N={}".format(
                summary.get("n_files"), n_files
            )
        )
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
    actual_tag = summary.get("dataset_tag")
    if actual_tag != expected_dataset_tag:
        raise ValueError(
            "checkpoint dataset tag is {!r}; expected {!r}".format(
                actual_tag, expected_dataset_tag
            )
        )
    require_pfo_track_links(summary["dataset"])
    expected_hash = (
        summary.get("artifacts", {})
        .get("best_weights", {})
        .get("sha256")
    )
    if not expected_hash:
        raise ValueError("checkpoint summary does not fingerprint best weights")
    actual_hash = sha256_file(weights)
    if actual_hash != expected_hash:
        raise ValueError(
            "checkpoint weight hash differs from its training summary: "
            "{} != {}".format(actual_hash, expected_hash)
        )
    return recipe


def steps_per_epoch(summary):
    training = summary["training"]
    if training.get("warmup_epochs", 0):
        return int(training["warmup_steps"] // training["warmup_epochs"])
    return max(int(training.get("decay_steps", 1)), 1)


def load_confirmation(store_dir, n_files, class_b):
    paths = [
        store_dir / "n{}_{}_confirmation.h5".format(n_files, class_name)
        for class_name in ("U", class_b)
    ]
    pair = [load_store(path, n_files) for path in paths]
    if len(pair[0][0]) != len(pair[1][0]):
        raise ValueError(
            "confirmation class counts differ: {} vs {}".format(
                len(pair[0][0]), len(pair[1][0])
            )
        )
    width = max(item[0].shape[1] for item in pair)
    return combine_pair(pair, width)


def confirmation_provenance(store_dir, n_files, class_b):
    stores = {}
    for class_name in ("U", class_b):
        path = (
            Path(store_dir)
            / "n{}_{}_confirmation.h5".format(n_files, class_name)
        )
        stores[class_name] = store_provenance(path)
    identity = {
        class_name: item["sha256"]
        for class_name, item in stores.items()
    }
    result = {
        "store_dir": str(Path(store_dir).resolve()),
        "stores": stores,
        "identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
    }
    missing = [
        class_name
        for class_name, store in stores.items()
        if (
            store["collection_statistics"]
            .get("pfo_track_links", {})
            .get("total", 0)
        ) <= 0
    ]
    if missing:
        raise ValueError(
            "confirmation stores have no PFO-track links: {}".format(
                ", ".join(missing)
            )
        )
    return result


def main():
    args = parse_args()
    if args.n_files <= 0 or args.n_files % 42:
        raise SystemExit("--n-files must be a positive multiple of 42")
    if args.expected_training_dataset_tag.lower() not in args.label.lower():
        raise SystemExit(
            "training dataset tag {!r} must occur in evaluation label {!r}"
            .format(args.expected_training_dataset_tag, args.label)
        )
    code = git_provenance()
    if args.require_clean_code and code["dirty"]:
        raise SystemExit(
            "refusing evaluation with tracked code changes:\n{}".format(
                "\n".join(code["tracked_changes"])
            )
        )

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
    training_commit = checkpoint_summary.get("code", {}).get("commit")
    if not training_commit or code.get("commit") != training_commit:
        raise ValueError(
            "evaluation code commit {!r} differs from training commit {!r}"
            .format(code.get("commit"), training_commit)
        )
    recipe = validate_checkpoint(
        checkpoint_summary,
        args.n_files,
        args.class_b,
        args.expected_training_dataset_tag,
        weights,
    )

    result_dir = Path(args.outdir) / args.label
    if result_dir.exists() and any(result_dir.iterdir()):
        raise SystemExit(
            "refusing to overwrite nonempty result directory: {}".format(
                result_dir
            )
        )
    result_dir.mkdir(parents=True, exist_ok=True)
    confirmation_dataset = confirmation_provenance(
        store_dir, args.n_files, args.class_b
    )
    context = {
        "status": "started",
        "label": args.label,
        "evaluation": "frozen-checkpoint confirmation",
        "training_dataset_tag": args.expected_training_dataset_tag,
        "checkpoint_summary": str(checkpoint_summary_path),
        "checkpoint_weights": str(weights),
        "checkpoint_weights_sha256": sha256_file(weights),
        "confirmation_dataset": confirmation_dataset,
        "code": code,
        "runtime": runtime_provenance(),
    }
    write_json(result_dir / "evaluation_context.json", context)

    x, labels, metadata = load_confirmation(
        store_dir, args.n_files, args.class_b
    )
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
        "n_files": args.n_files,
        "events_per_class": int(len(labels) // 2),
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "training_dataset_tag": checkpoint_summary["dataset_tag"],
            "training_dataset_identity_sha256": (
                checkpoint_summary["dataset"]["identity_sha256"]
            ),
            "training_code": checkpoint_summary.get("code"),
            "recipe": recipe,
            "original_test": checkpoint_summary.get("results", {}).get("test"),
        },
        "confirmation_dataset": confirmation_dataset,
        "code": context["code"],
        "runtime": context["runtime"],
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
    write_json(result_dir / "confirmation_summary.json", summary)
    context["status"] = "complete"
    context["completed_utc"] = utc_now()
    context["summary"] = str(
        (result_dir / "confirmation_summary.json").resolve()
    )
    write_json(result_dir / "evaluation_context.json", context)

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
