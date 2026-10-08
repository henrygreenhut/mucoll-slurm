#!/usr/bin/env python3
"""Validate one v2.11 input, digitization output, or reconstruction output."""

import argparse
import json
from pathlib import Path


EXPECTED_TYPES = {
    "input": {
        name: ("edm4hep::SimTrackerHitCollection",)
        for name in (
            "VertexBarrelCollection",
            "VertexEndcapCollection",
            "InnerTrackerBarrelCollection",
            "InnerTrackerEndcapCollection",
            "OuterTrackerBarrelCollection",
            "OuterTrackerEndcapCollection",
        )
    },
    "digi": {
        row[2]: (
            "edm4hep::TrackerHitPlaneCollection",
            "edm4hep::TrackerHit3DCollection",
        )
        for row in (
            ("VBC", "VertexBarrelCollection", "VXDBarrelHits"),
            ("VEC", "VertexEndcapCollection", "VXDEndcapHits"),
            ("ITBC", "InnerTrackerBarrelCollection", "ITBarrelHits"),
            ("ITEC", "InnerTrackerEndcapCollection", "ITEndcapHits"),
            ("OTBC", "OuterTrackerBarrelCollection", "OTBarrelHits"),
            ("OTEC", "OuterTrackerEndcapCollection", "OTEndcapHits"),
        )
    },
    "reco": {
        "SeedTracks": ("edm4hep::TrackCollection",),
        "AllTracks": ("edm4hep::TrackCollection",),
        "SiTracks": ("edm4hep::TrackCollection",),
    },
}


def validate(args):
    from podio.root_io import Reader

    path = args.input.resolve()
    reader = Reader(str(path))
    frames = reader.get("events")
    if len(frames) != 1:
        raise ValueError(f"Expected one event in {path}, found {len(frames)}")
    frame = frames[0]
    available = set(frame.getAvailableCollections())

    if "EventHeader" not in available or len(frame.get("EventHeader")) != 1:
        raise ValueError("Expected exactly one EventHeader")

    collections = {}
    for name, expected_types in EXPECTED_TYPES[args.stage].items():
        if name not in available:
            raise ValueError(f"Missing {args.stage} collection: {name}")
        collection = frame.get(name)
        actual_type = bytes(collection.getTypeName()).decode()
        if actual_type not in expected_types:
            raise ValueError(
                f"{name} has type {actual_type}, expected one of {expected_types}"
            )
        collections[name] = {"type": actual_type, "entries": len(collection)}

    report = {
        "status": "valid",
        "stage": args.stage,
        "path": str(path),
        "events": 1,
        "collections": collections,
    }
    print(json.dumps(report, indent=2))
    del frame, frames, reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(EXPECTED_TYPES), required=True)
    parser.add_argument("--input", type=Path, required=True)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
