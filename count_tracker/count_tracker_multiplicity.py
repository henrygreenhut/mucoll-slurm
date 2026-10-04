#!/usr/bin/env python3
"""Evaluate reconstructed-track multiplicity alone on a paired test split.

This is a fixed-score diagnostic, not a trained classifier: the score is the
number of duplicate-removed SiTracks in an event. COUNT is class 1, so an AUC
below 0.5 means COUNT has fewer tracks. The symmetric AUC reports separability
without hiding that direction.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from count_tracker_metrics import paired_event_bootstrap, weighted_auc
from count_tracker_train import load_store, sha256_file, validate_sim_count_pair


def evaluate(store_dir, construction, n_draws=1000):
    _, sim, sim_manifest = load_store(store_dir, construction, "SIM", "test")
    _, count, count_manifest = load_store(store_dir, construction, "COUNT", "test")
    validate_sim_count_pair(sim_manifest, count_manifest, "test")
    if len(sim) < 1:
        raise ValueError("the test split has no events")
    scores = np.concatenate([sim, count]).astype(np.float64)
    labels = np.r_[np.zeros(len(sim), np.int32), np.ones(len(count), np.int32)]
    groups = np.arange(len(scores), dtype=np.int64)
    weights = np.ones(len(scores), dtype=np.float64)
    auc = weighted_auc(labels, scores)
    interval = (paired_event_bootstrap(labels, scores, weights, groups, len(sim),
                                       n_draws=n_draws) if n_draws else None)
    return {
        "unit": "event", "score": "number of duplicate-removed SiTracks",
        "positive_class": "COUNT", "events_per_sample": len(sim),
        "mean_tracks": {"SIM": float(sim.mean()), "COUNT": float(count.mean())},
        "empty_events": {"SIM": int(np.count_nonzero(sim == 0)),
                         "COUNT": int(np.count_nonzero(count == 0))},
        "auc_signed": auc, "auc_symmetric": max(auc, 1.0 - auc),
        "auc_interval_95_signed": interval,
        "stores_sha256": {
            sample: sha256_file(Path(store_dir) / f"{construction}_{sample}_test.npz")
            for sample in ("SIM", "COUNT")
        },
        "paired_counts": [{"event_id": event["event_id"],
                           "SIM": int(sim[i]), "COUNT": int(count[i])}
                          for i, event in enumerate(sim_manifest["events"])],
        "uncertainty_note": "paired-event interval is conditional on the reused source pool",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--construction", required=True)
    parser.add_argument("--output", required=True, help="new JSON result file")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_draws < 0:
        parser.error("--bootstrap-draws cannot be negative")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    result = evaluate(args.store_dir, args.construction, args.bootstrap_draws)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"COUNT/SIM track count alone: signed AUC {result['auc_signed']:.6f} -> {output}")


if __name__ == "__main__":
    main()
