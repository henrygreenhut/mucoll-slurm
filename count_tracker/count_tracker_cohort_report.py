#!/usr/bin/env python3
"""Aggregate digitized-hit survival and SiTrack counts for a paired cohort."""

import argparse
import json
from pathlib import Path

import numpy as np

from count_tracker_checkpoint_report import COLLECTIONS, arm_report


def summarize(conditions, events_root, split, event_id=None):
    conditions = Path(conditions).resolve()
    events_root = Path(events_root).resolve()
    source = json.loads((conditions / "manifest.json").read_text())
    events = [event for event in source["events"] if event["split"] == split]
    if event_id is not None:
        events = [event for event in events if event["event_id"] == event_id]
    if not events:
        target = f" event {event_id}" if event_id is not None else " events"
        raise ValueError(f"no {split}{target} in conditions manifest")

    paired, arms = [], {sample: [] for sample in ("SIM", "COUNT")}
    for event in events:
        event_id = event["event_id"]
        row = {"event_id": event_id}
        for sample in arms:
            result = arm_report(events_root / event_id / sample)
            arms[sample].append(result)
            row[sample] = result["total"]
        paired.append(row)

    aggregate = {}
    for sample, records in arms.items():
        collections = {}
        for short in COLLECTIONS:
            before = sum(record["collections"][short]["entering_digitization"]
                         for record in records)
            after = sum(record["collections"][short]["digitized"] for record in records)
            collections[short] = {
                "entering_digitization": before, "digitized": after,
                "survival_fraction": after / before if before else None,
            }
        before = sum(row["entering_digitization"] for row in collections.values())
        after = sum(row["digitized"] for row in collections.values())
        tracks = np.asarray([record["total"]["si_tracks"] for record in records], dtype=float)
        aggregate[sample] = {
            "collections": collections,
            "total": {
                "entering_digitization": before, "digitized": after,
                "survival_fraction": after / before if before else None,
                "si_tracks": int(tracks.sum()), "mean_si_tracks": float(tracks.mean()),
                "std_si_tracks": float(tracks.std(ddof=1)) if len(tracks) > 1 else 0.0,
                "min_si_tracks": int(tracks.min()), "max_si_tracks": int(tracks.max()),
            },
        }
    sim_tracks = aggregate["SIM"]["total"]["si_tracks"]
    count_tracks = aggregate["COUNT"]["total"]["si_tracks"]
    return {
        "kind": "count_tracker_paired_cohort_report",
        "construction": source["manifest"]["construction"], "split": split,
        "events": len(events), "source_domain": source["manifest"].get("source_domain"),
        "generator_training_holdout": source["manifest"].get("generator_training_holdout"),
        "model_split_used": source["manifest"].get("model_split_used"),
        "analysis_split_used": source["manifest"].get("analysis_split_used"),
        "classifier_ready": source["manifest"].get("classifier_ready"),
        "hit_selection": source.get("hit_selection"),
        "source_cycle_pool": {
            key: source["manifest"].get("source_cycle_pool", {}).get(key)
            for key in ("kind", "count")
        },
        "raw_time_selection": source.get("raw_time_selection"),
        "aggregate": aggregate,
        "count_over_sim_tracks": count_tracks / sim_tracks if sim_tracks else None,
        "paired": paired,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--events-root", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--event-id", help="report only this manifest event (checkpoint mode)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    result = summarize(args.conditions, args.events_root, args.split, args.event_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"COUNT/SIM tracks = {result['count_over_sim_tracks']:.6g}")


if __name__ == "__main__":
    main()
