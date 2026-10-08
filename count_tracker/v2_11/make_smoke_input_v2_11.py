#!/usr/bin/env python3
"""Create a small v2.11-native event from an existing BIB-only input."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from count_tracker_conditions import CELL_ID_ENCODING, COLLECTIONS  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced_indices(size, limit):
    """Return deterministic indices spanning a collection without repetition."""
    if size <= limit:
        return range(size)
    return [(index * size) // limit for index in range(limit)]


def make_smoke_input(args):
    import edm4hep
    import podio
    from podio.root_io import Reader, Writer

    source = args.input.resolve()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}")

    reader = Reader(str(source))
    frames = reader.get("events")
    if len(frames) != 1:
        raise ValueError(f"Expected one event in {source}, found {len(frames)}")
    source_frame = frames[0]

    source_headers = source_frame.get("EventHeader")
    if len(source_headers) != 1:
        raise ValueError("Expected one EventHeader in the source event")
    source_header = source_headers[0]
    header_values = (
        source_header.getEventNumber(),
        source_header.getRunNumber(),
        source_header.getTimeStamp(),
    )

    selected = {}
    records = {}
    for short, (_, name) in COLLECTIONS.items():
        source_hits = source_frame.get(name)
        indices = evenly_spaced_indices(len(source_hits), args.hits_per_collection)
        records[short] = []
        for index in indices:
            hit = source_hits[index]
            position = hit.getPosition()
            momentum = hit.getMomentum()
            records[short].append((
                hit.getCellID(),
                hit.getEDep(),
                hit.getTime(),
                hit.getPathLength(),
                hit.getQuality(),
                (position.x, position.y, position.z),
                (momentum.x, momentum.y, momentum.z),
            ))
        selected[short] = len(records[short])
        if not selected[short]:
            raise ValueError(f"Source collection {name} is empty")

    # Do not retain objects owned by the source Reader in the output Frame.
    # Old podio releases can otherwise leave the serialized event invalid.
    del source_header, source_headers, source_frame, frames, reader

    frame = podio.Frame()
    headers = edm4hep.EventHeaderCollection()
    header = headers.create()
    header.setEventNumber(header_values[0])
    header.setRunNumber(header_values[1])
    header.setTimeStamp(header_values[2])
    frame.put(headers, "EventHeader")

    for short, (_, name) in COLLECTIONS.items():
        output_hits = edm4hep.SimTrackerHitCollection()
        for cell_id, edep, time, path_length, quality, position, momentum in records[short]:
            hit = output_hits.create()
            hit.setCellID(cell_id)
            hit.setEDep(edep)
            hit.setTime(time)
            hit.setPathLength(path_length)
            hit.setQuality(quality)
            hit.setPosition(edm4hep.Vector3d(*position))
            hit.setMomentum(edm4hep.Vector3f(*momentum))
        frame.put(output_hits, name)

    metadata = podio.Frame()
    for _, name in COLLECTIONS.values():
        metadata.put_parameter(f"{name}__CellIDEncoding", CELL_ID_ENCODING)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v2_smoke_", dir=destination.parent) as work:
        work = Path(work)
        root_path = work / "input.edm4hep.root"
        writer = Writer(str(root_path))
        writer.write_frame(metadata, "metadata")
        writer.write_frame(frame, "events")
        writer._writer.finish()
        del writer

        report = {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "selection": "deterministic evenly spaced source rows",
            "maximum_hits_per_collection": args.hits_per_collection,
            "selected_hits": selected,
            "output_sha256": sha256_file(root_path),
        }
        (work / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        work.rename(destination)

    print(f"Prepared v2.11 smoke input -> {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--hits-per-collection", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hits_per_collection <= 0:
        parser.error("--hits-per-collection must be positive")
    make_smoke_input(args)


if __name__ == "__main__":
    main()
