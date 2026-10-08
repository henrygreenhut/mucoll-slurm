#!/usr/bin/env python3
"""Write one v2.11-native EDM4hep event containing only BIB tracker hits."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from count_tracker_conditions import CELL_ID_ENCODING, COLLECTIONS  # noqa: E402
from count_tracker_input import (  # noqa: E402
    INPUT_CONSTRUCTIONS,
    append_count_hits,
    hit_counts,
    load_event,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(args):
    import edm4hep
    import podio
    from podio.root_io import Reader, Writer

    event, expected = load_event(
        args.conditions, args.split, args.event_id, args.construction
    )
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}")

    frame = podio.Frame()
    event_headers = edm4hep.EventHeaderCollection()
    event_header = event_headers.create()
    event_header.setEventNumber(0)
    event_header.setRunNumber(0)
    event_header.setTimeStamp(0)
    frame.put(event_headers, "EventHeader")

    collections = {short: edm4hep.SimTrackerHitCollection() for short in COLLECTIONS}
    provenance, targets = append_count_hits(
        collections, args.arrays, expected, args.count_tolerance
    )

    for short, (system, name) in COLLECTIONS.items():
        if hit_counts(collections[short], system) != targets[short]:
            raise ValueError(f"Final {short} occupancy differs from input arrays")
        frame.put(collections[short], name)

    metadata = podio.Frame()
    for _, name in COLLECTIONS.values():
        metadata.put_parameter(f"{name}__CellIDEncoding", CELL_ID_ENCODING)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v2_bib_input_", dir=destination.parent) as work:
        work = Path(work)
        root_path = work / "input.edm4hep.root"
        writer = Writer(str(root_path))
        writer.write_frame(metadata, "metadata")
        writer.write_frame(frame, "events")
        writer._writer.finish()
        del writer

        reader = Reader(str(root_path))
        frames = reader.get("events")
        if len(frames) != 1:
            raise ValueError("BIB-only output must contain exactly one event")
        check = frames[0]
        headers = check.get("EventHeader")
        if len(headers) != 1:
            raise ValueError("BIB-only output must contain one EventHeader")
        header = headers[0]
        if header.getEventNumber() != 0 or header.getRunNumber() != 0:
            raise ValueError("Serialized EventHeader changed")
        for short, (system, name) in COLLECTIONS.items():
            if hit_counts(check.get(name), system) != targets[short]:
                raise ValueError(f"Serialized {short} occupancy changed")
        del check, frames, reader

        report = {
            "sample": args.sample,
            "event_id": args.event_id,
            "construction": args.construction,
            "bib_only": True,
            "signal": None,
            "event_header": {"run": 0, "event": 0, "timestamp": 0},
            "arrays": str(args.arrays.resolve()),
            "hits": {short: sum(counts.values()) for short, counts in targets.items()},
            "input_sha256": sha256_file(root_path),
            "assigned_arrays": provenance,
        }
        (work / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        work.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--construction", choices=INPUT_CONSTRUCTIONS, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--sample", choices=("SIM", "COUNT"), required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--count-tolerance", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.count_tolerance <= 1.0:
        parser.error("--count-tolerance must be in [0, 1]")
    write(args)


if __name__ == "__main__":
    main()
