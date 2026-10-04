#!/usr/bin/env python3
"""Reallocate a test-only paired pilot into event-grouped PFN train/val/test.

This is for an explicitly preliminary single-track classifier. It never edits
the OSCAR reco files or their conditions manifest. The source export contains
fitted AtIP helix parameters; physical observables are derived here with the
shared track-feature definition. The entire output directory is new and is
published only after all six stores are complete.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from count_tracker_features import pack_store
from count_tracker_track_features import RAW_FEATURES, track_row


SPLITS = ("train", "val", "test")
SAMPLES = ("SIM", "COUNT")
STATE_COLUMNS = ("phi", "omega", "tanLambda", "D0", "Z0")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_events(export, counts, seed):
    events = export["events"]
    if not events or any(event["split"] != "test" for event in events):
        raise ValueError("pilot splitter requires an existing test-only event cohort")
    if sum(counts.values()) != len(events) or any(counts[split] < 1 for split in SPLITS):
        raise ValueError("positive train/val/test event counts must sum to source events")
    ids = [event["event_id"] for event in events]
    if len(set(ids)) != len(ids):
        raise ValueError("source export has duplicate event IDs")
    order = np.random.default_rng(seed).permutation(len(events))
    assignment = {}
    offset = 0
    for split in SPLITS:
        for position in order[offset:offset + counts[split]]:
            assignment[ids[int(position)]] = split
        offset += counts[split]
    return assignment


def physical_rows(helix):
    rows = [track_row(*state) for state in helix]
    return np.asarray(rows, dtype=np.float32).reshape(-1, len(RAW_FEATURES))


def prepare(source, output, counts, seed):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    export = json.loads(source.read_text())
    if (export.get("kind") != "count_tracker_read_only_helix_export"
            or export.get("construction") != "norm42_reservoir"
            or tuple(export.get("state_columns", ())) != STATE_COLUMNS
            or export.get("track_collection") != "SiTracks"
            or export.get("track_state") != "AtIP"):
        raise ValueError("source must be a norm42 reservoir AtIP helix export")
    if any(set(event["arms"]) != set(SAMPLES) for event in export["events"]):
        raise ValueError("every exported event must contain SIM and COUNT arms")
    assignment = assign_events(export, counts, seed)
    construction = "norm42_reservoir_preliminary"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".count_pilot_", dir=output.parent) as tmp:
        work = Path(tmp)
        shutil.copyfile(source, work / "source_export.json")
        summary = {"kind": "count_tracker_preliminary_track_split",
                   "construction": construction, "original_construction": export["construction"],
                   "statistical_unit": "individual reconstructed track",
                   "split_unit": "paired reconstructed event", "seed": seed,
                   "counts": counts, "assignment": assignment,
                   "source_export_sha256": sha256_file(source),
                   "conditions_manifest_sha256": export["conditions_manifest_sha256"],
                   "source_domain": export.get("source_domain"),
                   "generator_training_holdout": export.get("generator_training_holdout"),
                   "physical_event_boundaries": export.get("physical_event_boundaries"),
                   "stores": {}}
        for split in SPLITS:
            selected = [event for event in export["events"]
                        if assignment[event["event_id"]] == split]
            for sample in SAMPLES:
                per_event = [physical_rows(event["arms"][sample]["helix"])
                             for event in selected]
                tracks, n_tracks = pack_store(per_event)
                prefix = work / f"{construction}_{sample}_{split}"
                np.savez(prefix.with_suffix(".npz"), tracks=tracks, n_tracks=n_tracks)
                manifest = {
                    "kind": "count_tracker_track_store", "schema_version": 2,
                    "construction": construction, "original_construction": export["construction"],
                    "sample": sample, "split": split,
                    "track_collection": "SiTracks", "track_state": "AtIP",
                    "magnetic_field_T": 5.0,
                    "pt_conversion": "pT [GeV] = 0.0015 / |omega [1/mm]|",
                    "features": list(RAW_FEATURES),
                    "n_events": len(selected), "max_tracks": int(tracks.shape[1]),
                    "total_tracks": int(n_tracks.sum()),
                    "mean_tracks": float(n_tracks.mean()),
                    "conditions_manifest_sha256": export["conditions_manifest_sha256"],
                    "source_export_sha256": sha256_file(source),
                    "source_domain": export.get("source_domain"),
                    "generator_training_holdout": export.get("generator_training_holdout"),
                    "physical_event_boundaries": export.get("physical_event_boundaries"),
                    "events": [{"event_id": event["event_id"],
                                "path": event["arms"][sample]["path"],
                                "reco_sha256": event["arms"][sample]["sha256"],
                                "n_tracks": len(event["arms"][sample]["helix"])}
                               for event in selected],
                }
                prefix.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
                summary["stores"][f"{sample}_{split}"] = {
                    "events": len(selected), "tracks": int(n_tracks.sum()),
                    "npz_sha256": sha256_file(prefix.with_suffix(".npz")),
                }
        (work / "pilot_split.json").write_text(json.dumps(summary, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Output appeared during preparation: {output}")
        os.replace(work, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="read-only helix JSON export")
    parser.add_argument("--output", required=True, help="new six-store directory")
    for split in SPLITS:
        parser.add_argument(f"--{split}-events", required=True, type=int)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    counts = {split: getattr(args, f"{split}_events") for split in SPLITS}
    if args.seed < 0 or any(value < 1 for value in counts.values()):
        parser.error("seed must be nonnegative and every split must have an event")
    summary = prepare(args.source, args.output, counts, args.seed)
    print(json.dumps({"assignment": summary["assignment"],
                      "stores": summary["stores"]}, indent=2))


if __name__ == "__main__":
    main()
