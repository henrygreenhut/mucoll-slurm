#!/usr/bin/env python3
"""Prepare COUNT sensor conditions from explicit, uncut SIM source lists.

This step reads CellIDs only. It does not select sources, filter hits, generate
hits, reconstruct events, or submit jobs. See COUNT_TRACKER_STUDY.md for the
manifest format and the still-open production choices.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np


COLLECTIONS = {
    "VBC": (1, "VertexBarrelCollection"),
    "VEC": (2, "VertexEndcapCollection"),
    "ITBC": (3, "InnerTrackerBarrelCollection"),
    "ITEC": (4, "InnerTrackerEndcapCollection"),
    "OTBC": (5, "OuterTrackerBarrelCollection"),
    "OTEC": (6, "OuterTrackerEndcapCollection"),
}
CELL_ID_ENCODING = "system:0:5,side:5:-2,layer:7:6,module:13:11,sensor:24:8"
POLARITIES = ("MUPLUS", "MUMINUS")
CONSTRUCTIONS = {
    "norm1": {"n_files_per_polarity": 420, "file_normalization": 1},
    "norm42": {"n_files_per_polarity": 10, "file_normalization": 42},
}


def decode_cell_ids(cell_ids, system):
    """Decode the declared MAIA encoding, retaining signed endcap side."""
    ids = np.asarray(cell_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("CellIDs must be a one-dimensional integer array")
    if ids.dtype.kind == "i" and np.any(ids < 0):
        raise ValueError("Negative tracker CellID")
    ids = ids.astype(np.uint64)
    if np.any(ids >> np.uint64(32)):
        raise ValueError("CellID has bits outside the declared tracker encoding")
    sides = ((ids >> np.uint64(5)) & np.uint64(3)).astype(np.int64)
    sides = np.where(sides >= 2, sides - 4, sides)
    conditions = np.column_stack((
        ids & np.uint64(31), sides,
        (ids >> np.uint64(7)) & np.uint64(63),
        (ids >> np.uint64(13)) & np.uint64(2047),
        (ids >> np.uint64(24)) & np.uint64(255),
    )).astype(np.int64)
    if np.any(conditions[:, 0] != system):
        raise ValueError("CellID system does not match its tracker collection")
    return conditions


def sensor_counts(conditions):
    """Count integer condition tuples without changing their meaning."""
    rows = np.asarray(conditions)
    if rows.ndim != 2 or rows.shape[1] != 5 or rows.dtype.kind not in "iu":
        raise ValueError("Conditions must be an integer array of shape (N, 5)")
    unique, counts = np.unique(rows, axis=0, return_counts=True)
    return Counter({tuple(map(int, row)): int(n) for row, n in zip(unique, counts)})


def require_matching_counts(expected, actual):
    """Reject lost or reassigned hits; never silently repair COUNT output."""
    expected_counts = sensor_counts(expected)
    actual_counts = sensor_counts(actual)
    if expected_counts != actual_counts:
        missing = sum((expected_counts - actual_counts).values())
        extra = sum((actual_counts - expected_counts).values())
        raise ValueError(
            f"Sensor occupancy mismatch: {missing} missing and {extra} extra hits"
        )


def validate_manifest(manifest, base):
    """Resolve explicit sources and reject source leakage across splits."""
    if manifest.get("schema_version") != 2:
        raise ValueError("Expected manifest schema_version 2 with explicit construction")
    if manifest.get("cell_id_encoding") != CELL_ID_ENCODING:
        raise ValueError("Manifest must declare the supported MAIA CellID encoding")
    construction = manifest.get("construction")
    if construction not in CONSTRUCTIONS:
        raise ValueError("construction must be norm1 or norm42")
    definition = CONSTRUCTIONS[construction]
    n_files = manifest.get("n_files_per_polarity")
    for key, value in {**definition, "norm1_equivalents_per_polarity": 420}.items():
        if type(manifest.get(key)) is not int or manifest[key] != value:
            raise ValueError(f"{construction} requires {key}={value}")
    if not isinstance(manifest.get("events"), list) or not manifest["events"]:
        raise ValueError("Manifest must contain a nonempty events list")
    identities = set()
    cycle_splits = {}
    path_identity = {}
    cycle_paths = {}
    for event in manifest["events"]:
        event_id = event.get("event_id")
        split = event.get("split")
        if not isinstance(event_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", event_id):
            raise ValueError("event_id must contain only letters, digits, '_' or '-'")
        if split not in ("train", "val", "test"):
            raise ValueError("Each event needs an explicit train, val, or test split")
        if (split, event_id) in identities:
            raise ValueError(f"Duplicate event identity: {split}/{event_id}")
        identities.add((split, event_id))
        sources = event.get("sources", {})
        if set(sources) != set(POLARITIES):
            raise ValueError("Each event needs MUPLUS and MUMINUS source lists")
        for polarity in POLARITIES:
            records = sources[polarity]
            if len(records) != n_files:
                raise ValueError(f"{event_id}/{polarity}: expected {n_files} files")
            used = set()
            for source in records:
                cycle = source["cycle"]
                entry = source["entry"]
                if type(cycle) is not int or cycle < 0:
                    raise ValueError("cycle must be a nonnegative integer")
                if type(entry) is not int or entry < 0:
                    raise ValueError("entry must be a nonnegative integer")
                path = Path(source["path"]).expanduser()
                path = (base / path).resolve() if not path.is_absolute() else path.resolve()
                source["path"] = str(path)
                if cycle in used:
                    raise ValueError(f"Repeated cycle in {event_id}/{polarity}: {cycle}")
                used.add(cycle)
                if cycle_splits.setdefault(cycle, split) != split:
                    raise ValueError(f"Source cycle {cycle} crosses dataset splits")
                if path_identity.setdefault(str(path), (polarity, cycle)) != (polarity, cycle):
                    raise ValueError(f"Conflicting identities for source file {path}")
                if cycle_paths.setdefault((polarity, cycle), str(path)) != str(path):
                    raise ValueError(f"Multiple files for {polarity} cycle {cycle}")
    return manifest


def read_source_counts(path, entry):
    """Read all six CellID branches, with no time or energy selection."""
    import uproot

    result = {}
    with uproot.open(path) as root:
        if "events" not in root or "podio_metadata" not in root:
            raise ValueError(f"Missing events or PODIO metadata: {path}")
        events = root["events"]
        if entry >= events.num_entries:
            raise ValueError(f"Entry {entry} does not exist in {path}")
        for short, (system, name) in COLLECTIONS.items():
            values = events[name][f"{name}.cellID"].array(
                entry_start=entry, entry_stop=entry + 1, library="ak"
            )[0]
            ids = np.asarray(values)
            result[short] = sensor_counts(decode_cell_ids(ids, system))
    return result


def prepare(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    manifest = validate_manifest(json.loads(raw), manifest_path.parent)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep partial products separate; publish only when every event succeeds.
    with tempfile.TemporaryDirectory(prefix=".count_conditions_", dir=output.parent) as work:
        work = Path(work)
        summaries = []
        for event in manifest["events"]:
            totals = {short: Counter() for short in COLLECTIONS}
            for polarity in POLARITIES:
                for source in event["sources"][polarity]:
                    counts = read_source_counts(source["path"], source["entry"])
                    for short in COLLECTIONS:
                        totals[short].update(counts[short])
            destination = work / event["split"] / event["event_id"]
            destination.mkdir(parents=True)
            summary = {"event_id": event["event_id"], "split": event["split"], "collections": {}}
            for short, counts in totals.items():
                sensors = np.asarray(sorted(counts), dtype=np.int64).reshape(-1, 5)
                multiplicities = np.asarray([counts[tuple(row)] for row in sensors], dtype=np.int64)
                conditions = np.repeat(sensors, multiplicities, axis=0)
                path = destination / f"{short}_conditions.npy"
                np.save(path, conditions, allow_pickle=False)
                summary["collections"][short] = {
                    "hits": len(conditions), "occupied_sensors": len(sensors),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            summaries.append(summary)
            print(f"Prepared {event['split']}/{event['event_id']}", flush=True)
        report = {
            "manifest": manifest, "input_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "selection": "all stored SIM tracker hits; no additional cuts",
            "condition_columns": ["system", "side", "layer", "module", "sensor"],
            "events": summaries,
        }
        (work / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Output appeared during preparation: {output}")
        work.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.output)


if __name__ == "__main__":
    main()
