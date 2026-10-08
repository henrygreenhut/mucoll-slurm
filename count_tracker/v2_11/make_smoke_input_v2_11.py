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

    frame = podio.Frame()
    headers = edm4hep.EventHeaderCollection()
    headers.push_back(source_headers[0].clone(False))
    frame.put(headers, "EventHeader")

    selected = {}
    for short, (_, name) in COLLECTIONS.items():
        source_hits = source_frame.get(name)
        output_hits = edm4hep.SimTrackerHitCollection()
        indices = evenly_spaced_indices(len(source_hits), args.hits_per_collection)
        for index in indices:
            output_hits.push_back(source_hits[index].clone(False))
        selected[short] = len(output_hits)
        if not selected[short]:
            raise ValueError(f"Source collection {name} is empty")
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

        check_reader = Reader(str(root_path))
        check_frames = check_reader.get("events")
        if len(check_frames) != 1 or len(check_frames[0].get("EventHeader")) != 1:
            raise ValueError("Serialized smoke event or EventHeader is invalid")
        for short, (_, name) in COLLECTIONS.items():
            if len(check_frames[0].get(name)) != selected[short]:
                raise ValueError(f"Serialized {short} hit count changed")
        del check_frames, check_reader

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

    del source_frame, frames, reader
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
