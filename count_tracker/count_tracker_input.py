#!/usr/bin/env python3
"""Write one SIM or COUNT tracker input, without overlay timing selection.

Use the same conditions event and neutrino SIM record for a SIM/COUNT pair.
COUNT input arrays must already have geometric CellIDs assigned by GenBIB.
Only the detector environment supplies the PODIO/EDM4hep runtime imports.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from count_tracker_conditions import (
    CELL_ID_ENCODING, COLLECTIONS, CONSTRUCTIONS, POLARITIES, decode_cell_ids, sensor_counts,
    validate_manifest,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_count_rows(rows, system, expected):
    """Check assigned hits without clipping, filtering, or resampling them."""
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 10 or rows.dtype.kind not in "fiu":
        raise ValueError("Assigned hits must be a numeric (N, 10) array")
    if not np.all(np.isfinite(rows)):
        raise ValueError("Assigned hits contain nonfinite values")
    if np.any(rows[:, 0] < 0):
        raise ValueError("Assigned hits contain negative energy deposition")
    labels = rows[:, 5:]
    if not np.all(labels == np.rint(labels)):
        raise ValueError("Assigned hit sensor identifiers must be integers")
    limits = ((0, 31), (-2, 1), (0, 63), (0, 2047), (0, 255))
    for column, (low, high) in enumerate(limits):
        if np.any((labels[:, column] < low) | (labels[:, column] > high)):
            raise ValueError("Assigned sensor identifier exceeds the CellID field")
    labels = labels.astype(np.int64)
    if np.any(labels[:, 0] != system):
        raise ValueError("Assigned hits belong to a different tracker system")
    actual = sensor_counts(labels)
    if actual != expected:
        missing = sum((expected - actual).values())
        extra = sum((actual - expected).values())
        raise ValueError(f"Sensor occupancy mismatch: {missing} missing, {extra} extra")
    return labels


def pack_cell_ids(labels):
    """Pack previously validated sensor labels; retain signed side bits."""
    values = np.asarray(labels, dtype=np.int64)
    return (values[:, 0] | ((values[:, 1] & 3) << 5) |
            (values[:, 2] << 7) | (values[:, 3] << 13) |
            (values[:, 4] << 24)).astype(np.uint64)


def load_event(directory, split, event_id, construction):
    """Load the exact source record and check its saved condition digests."""
    directory = Path(directory).resolve()
    report = json.loads((directory / "manifest.json").read_text())
    manifest = validate_manifest(report["manifest"], directory)
    if manifest["construction"] != construction:
        raise ValueError("Conditions belong to a different SIM construction")
    matches = [event for event in manifest["events"]
               if (event["split"], event["event_id"]) == (split, event_id)]
    summaries = [event for event in report["events"]
                 if (event["split"], event["event_id"]) == (split, event_id)]
    if len(matches) != 1 or len(summaries) != 1:
        raise ValueError("Conditions must contain exactly one matching event")
    expected = {}
    for short, (system, _) in COLLECTIONS.items():
        path = directory / split / event_id / f"{short}_conditions.npy"
        saved = summaries[0]["collections"][short]
        if sha256_file(path) != saved["sha256"]:
            raise ValueError(f"Conditions changed after preparation: {path}")
        rows = np.load(path, allow_pickle=False)
        expected[short] = sensor_counts(rows)
        if np.any(rows[:, 0] != system) or len(rows) != saved["hits"]:
            raise ValueError(f"Invalid conditions or recorded hit count: {path}")
    return matches[0], expected


def read_frame(path, entry):
    from podio.root_io import Reader

    reader = Reader(str(path))
    frames = reader.get("events")
    if entry < 0 or entry >= len(frames):
        raise ValueError(f"Entry {entry} does not exist in {path}")
    # Keep the reader alive while its frame is used.
    return reader, frames[entry]


def hit_counts(collection, system):
    ids = np.fromiter((hit.getCellID() for hit in collection), dtype=np.uint64)
    return sensor_counts(decode_cell_ids(ids, system))


def append_sim_hits(collections, event):
    """Copy every BIB tracker hit, retaining scalar fields and removing links.

    BIB MCParticles are not merged, consistent with the existing study.
    clone(False) avoids dangling particle relations into separate source files.
    """
    for polarity in POLARITIES:
        for source in event["sources"][polarity]:
            reader, frame = read_frame(source["path"], source["entry"])
            for short, (_, name) in COLLECTIONS.items():
                for hit in frame.get(name):
                    copied = hit.clone(False)
                    copied.setOverlay(True)
                    collections[short].push_back(copied)
            del frame, reader


def append_count_hits(collections, directory, expected):
    import edm4hep

    provenance = {}
    for short, (system, name) in COLLECTIONS.items():
        path = Path(directory) / f"{name}_SimTrackerHit_conditional_reco9_0.npy"
        rows = np.load(path, mmap_mode="r", allow_pickle=False)
        labels = validate_count_rows(rows, system, expected[short])
        cell_ids = pack_cell_ids(labels)
        for row, cell_id in zip(rows, cell_ids):
            hit = collections[short].create()
            hit.setEDep(float(row[0]))
            hit.setPosition(edm4hep.Vector3d(*map(float, row[1:4])))
            hit.setTime(float(row[4]))
            hit.setCellID(int(cell_id))
            hit.setOverlay(True)
        provenance[short] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return provenance


def write_input(args):
    import edm4hep
    import podio
    from podio.root_io import Writer

    event, expected = load_event(args.conditions, args.split, args.event_id, args.construction)
    destination = Path(args.output).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}")
    reader, signal = read_frame(args.signal, args.signal_entry)
    particles = signal.get("MCParticles")
    if len(particles) != 1 or particles[0].getPDG() != 14:
        raise ValueError("Expected one PDG-14 neutrino in the signal SIM record")
    if len(particles[0].getParents()) or len(particles[0].getDaughters()):
        raise ValueError("Unexpected neutrino particle relations; explicit handling required")
    for _, name in COLLECTIONS.values():
        if len(signal.get(name)):
            raise ValueError("Neutrino SIM record has tracker hits; refusing to omit them")
    headers = signal.get("EventHeader")
    if len(headers) != 1:
        raise ValueError("Expected one signal EventHeader")

    frame = podio.Frame()
    copied_particles = edm4hep.MCParticleCollection()
    copied_particles.push_back(particles[0].clone(False))
    copied_headers = edm4hep.EventHeaderCollection()
    copied_headers.push_back(headers[0].clone(False))
    frame.put(copied_particles, "MCParticles")
    frame.put(copied_headers, "EventHeader")
    del signal, reader

    collections = {short: edm4hep.SimTrackerHitCollection() for short in COLLECTIONS}
    count_provenance = None
    if args.sample == "SIM":
        append_sim_hits(collections, event)
    else:
        count_provenance = append_count_hits(collections, args.count_arrays, expected)
    for short, (system, name) in COLLECTIONS.items():
        if hit_counts(collections[short], system) != expected[short]:
            raise ValueError(f"Final {short} occupancy differs from the conditions")
        frame.put(collections[short], name)

    metadata = podio.Frame()
    for _, name in COLLECTIONS.values():
        metadata.put_parameter(f"{name}__CellIDEncoding", CELL_ID_ENCODING)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".count_input_", dir=destination.parent) as work:
        work = Path(work)
        root_path = work / "input.edm4hep.root"
        writer = Writer(str(root_path))
        writer.write_frame(metadata, "metadata")
        writer.write_frame(frame, "events")
        writer._writer.finish()
        del writer
        # Reopen the serialized data before publishing a completed input.
        check_reader, check = read_frame(root_path, 0)
        for short, (system, name) in COLLECTIONS.items():
            if hit_counts(check.get(name), system) != expected[short]:
                raise ValueError(f"Serialized {short} occupancy changed")
        del check, check_reader
        report = {
            "sample": args.sample, "event": event,
            "construction": args.construction,
            **CONSTRUCTIONS[args.construction],
            "norm1_equivalents_per_polarity": 420,
            "signal": {"path": str(Path(args.signal).resolve()), "entry": args.signal_entry},
            "conditions_manifest_sha256": sha256_file(Path(args.conditions) / "manifest.json"),
            "count_arrays": count_provenance,
            "overlay_time_selection": False,
            "hits": {short: sum(counts.values()) for short, counts in expected.items()},
            "input_sha256": sha256_file(root_path),
        }
        (work / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        if destination.exists():
            raise FileExistsError(f"Output appeared during writing: {destination}")
        work.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True, help="Prepared condition directory")
    parser.add_argument("--construction", choices=tuple(CONSTRUCTIONS), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--sample", choices=("SIM", "COUNT"), required=True)
    parser.add_argument("--signal", required=True, help="Neutrino SIM ROOT file")
    parser.add_argument("--signal-entry", required=True, type=int)
    parser.add_argument("--count-arrays", help="Assigned arrays for this single event")
    parser.add_argument("--output", required=True, help="New output directory")
    args = parser.parse_args()
    if (args.sample == "COUNT") != bool(args.count_arrays):
        parser.error("--count-arrays is required for COUNT and must be omitted for SIM")
    if args.signal_entry < 0:
        parser.error("--signal-entry must be nonnegative")
    write_input(args)


if __name__ == "__main__":
    main()
