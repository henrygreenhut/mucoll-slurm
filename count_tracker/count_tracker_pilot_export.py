#!/usr/bin/env python3
"""Read-only JSON export of fitted AtIP helix parameters from paired reco ROOTs.

The script writes only to stdout. It can be streamed over SSH into the pinned
detector container without installing or changing code on OSCAR. Physics
observables are derived locally from the exported fitted parameters.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


STATE_BRANCHES = ("phi", "omega", "tanLambda", "D0", "Z0")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reco_path(events_root, split, event_id, sample):
    root = Path(events_root)
    for path in (root / split / event_id / sample / "reco" / "reco_output.edm4hep.root",
                 root / event_id / sample / "reco" / "reco_output.edm4hep.root"):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Missing {sample} reco for {split}/{event_id}")


def helix_rows(path):
    import uproot

    with uproot.open(path) as root:
        if "events" not in root or "podio_metadata" not in root:
            raise ValueError(f"{path}: missing events or PODIO metadata")
        events = root["events"]
        if events.num_entries != 1:
            raise ValueError(f"{path}: expected one reconstructed event")
        selected = np.asarray(events["SiTracks_objIdx"]["SiTracks_objIdx.index"].array()[0],
                              dtype=np.int64)
        begin = np.asarray(events["AllTracks"]["AllTracks.trackStates_begin"].array()[0],
                           dtype=np.int64)
        end = np.asarray(events["AllTracks"]["AllTracks.trackStates_end"].array()[0],
                         dtype=np.int64)
        states = events["_AllTracks_trackStates"]
        locations = np.asarray(states["_AllTracks_trackStates.location"].array()[0],
                               dtype=np.int64)
        columns = [np.asarray(states[f"_AllTracks_trackStates.{name}"].array()[0],
                              dtype=np.float64) for name in STATE_BRANCHES]
        rows = []
        for track in selected:
            if track < 0 or track >= len(begin):
                raise ValueError(f"{path}: invalid SiTracks subset index {track}")
            lo, hi = int(begin[track]), int(end[track])
            if lo < 0 or hi > len(locations) or lo >= hi:
                raise ValueError(f"{path}: invalid track-state range for AllTracks {track}")
            at_ip = np.flatnonzero(locations[lo:hi] == 1)
            if len(at_ip) == 0:
                raise ValueError(f"{path}: AllTracks {track} has no AtIP state")
            state = lo + int(at_ip[0])
            row = [float(column[state]) for column in columns]
            if not np.all(np.isfinite(row)) or row[1] == 0:
                raise ValueError(f"{path}: AllTracks {track} has invalid AtIP parameters")
            rows.append(row)
        return rows


def export(conditions, events_root):
    manifest_path = Path(conditions) / "manifest.json"
    report = json.loads(manifest_path.read_text())
    records = []
    for event in report["events"]:
        event_id, split = event["event_id"], event["split"]
        arms = {}
        for sample in ("SIM", "COUNT"):
            path = reco_path(events_root, split, event_id, sample)
            arms[sample] = {"path": str(path), "sha256": sha256_file(path),
                            "helix": helix_rows(path)}
        records.append({"event_id": event_id, "split": split, "arms": arms})
    return {
        "kind": "count_tracker_read_only_helix_export",
        "construction": report["manifest"]["construction"],
        "conditions_manifest_sha256": sha256_file(manifest_path),
        "source_domain": report["manifest"].get("source_domain"),
        "generator_training_holdout": report["manifest"].get("generator_training_holdout"),
        "physical_event_boundaries": report["manifest"].get("physical_event_boundaries"),
        "state_columns": list(STATE_BRANCHES),
        "track_collection": "SiTracks", "track_state": "AtIP",
        "events": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--events-root", required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.conditions, args.events_root), separators=(",", ":")))


if __name__ == "__main__":
    main()
