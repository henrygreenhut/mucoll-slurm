#!/usr/bin/env python3
"""Check cached COUNT inference against the native Paper1 sampler on a GPU."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from count_tracker_sample import (
    CollectionSampler,
    load_paper1,
    resolve_device,
    resolve_model_dir,
    run_rejection_loop,
)


def generate(sampler, conditions, seed, work_dir, max_rounds):
    return run_rejection_loop(
        sampler,
        conditions,
        seed,
        max_rejection_rounds=max_rounds,
        unfilled_policy="drop",
        return_report=True,
        work_dir=work_dir,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--paper1-root", required=True)
    parser.add_argument("--collection", default="VBC")
    parser.add_argument("--conditions", required=True,
                        help="One <SHORT>_conditions.npy file")
    parser.add_argument("--hits", type=int, default=256,
                        help="Use the first N conditions (default: 256)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.hits < 1:
        parser.error("--hits must be positive")

    conditions = np.load(args.conditions, allow_pickle=False)[:args.hits]
    runtime = load_paper1(args.paper1_root)
    device = resolve_device(args.device)
    model_dir = resolve_model_dir(args.model_root, args.collection)
    native = CollectionSampler(args.collection, model_dir, device, runtime, "off")
    cached = CollectionSampler(args.collection, model_dir, device, runtime, "data")

    comparisons = []
    with tempfile.TemporaryDirectory(prefix="count_cache_validation_") as work:
        for offset in (0, 1):
            seed = args.seed + offset
            native_rows, native_report = generate(
                native, conditions, seed, work, args.max_rounds)
            cached_rows, cached_report = generate(
                cached, conditions, seed, work, args.max_rounds)
            native_summary = {
                key: value for key, value in native_report.items()
                if key != "unfilled_conditions"
            }
            cached_summary = {
                key: value for key, value in cached_report.items()
                if key != "unfilled_conditions"
            }
            comparisons.append({
                "seed": seed,
                "rows_equal": bool(np.array_equal(native_rows, cached_rows)),
                "reports_equal": native_summary == cached_summary,
                "unfilled_conditions_equal": bool(np.array_equal(
                    native_report["unfilled_conditions"],
                    cached_report["unfilled_conditions"],
                )),
                "max_abs_difference": (
                    float(np.max(np.abs(native_rows - cached_rows)))
                    if native_rows.size else 0.0),
                "native_report": native_summary,
                "cached_report": cached_summary,
            })

    result = {
        "status": ("exact match" if all(
            item["rows_equal"] and item["reports_equal"]
            and item["unfilled_conditions_equal"] for item in comparisons)
                   else "mismatch"),
        "collection": args.collection,
        "conditions": str(Path(args.conditions).resolve()),
        "hits": int(len(conditions)),
        "cached_dataset_preparations": cached.tabddpm_sample.dataset_preparations,
        "cached_discrete_column_scans": cached.tabddpm_sample.discrete_column_scans,
        "comparisons": comparisons,
    }
    print(json.dumps(result, indent=2))
    if result["status"] != "exact match":
        raise SystemExit(1)
    if result["cached_dataset_preparations"] != 1:
        raise RuntimeError("Cached sampler prepared the dataset more than once")
    if result["cached_discrete_column_scans"] != 1:
        raise RuntimeError("Cached sampler scanned discrete columns more than once")


if __name__ == "__main__":
    main()
