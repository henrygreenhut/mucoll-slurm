#!/usr/bin/env python3
"""Validate one materialized tracker theta subset."""

import argparse
import json
from pathlib import Path


COLLECTIONS = (
    ("VXDBarrelHitsSplit", "VXDBarrelHitsRelationsSplit", "VertexBarrelCollectionSplit"),
    ("VXDEndcapHitsSplit", "VXDEndcapHitsRelationsSplit", "VertexEndcapCollectionSplit"),
    ("ITBarrelHitsSplit", "ITBarrelHitsRelationsSplit", "InnerTrackerBarrelCollectionSplit"),
    ("ITEndcapHitsSplit", "ITEndcapHitsRelationsSplit", "InnerTrackerEndcapCollectionSplit"),
    ("OTBarrelHitsSplit", "OTBarrelHitsRelationsSplit", "OuterTrackerBarrelCollectionSplit"),
    ("OTEndcapHitsSplit", "OTEndcapHitsRelationsSplit", "OuterTrackerEndcapCollectionSplit"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

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

    counts = {}
    for hits, relations, simulated_hits in COLLECTIONS:
        missing = [name for name in (hits, relations, simulated_hits) if name not in available]
        if missing:
            raise ValueError(f"Missing split collections: {', '.join(missing)}")
        counts[hits] = len(frame.get(hits))
        counts[relations] = len(frame.get(relations))
        counts[simulated_hits] = len(frame.get(simulated_hits))

    print(json.dumps({"status": "valid", "path": str(path), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
