#!/usr/bin/env python3
"""Validate one materialized tracker theta subset."""

import argparse
import json
import math
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
    parser.add_argument("--theta-min", type=float)
    parser.add_argument("--theta-max", type=float)
    args = parser.parse_args()
    if (args.theta_min is None) != (args.theta_max is None):
        parser.error("pass both --theta-min and --theta-max, or neither")

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
        if args.theta_min is not None:
            for hit in frame.get(hits):
                position = hit.getPosition()
                theta = math.degrees(math.atan2(
                    math.hypot(position.x, position.y), position.z))
                inside = (args.theta_min <= theta <= args.theta_max
                          if args.theta_max == 180.0 else
                          args.theta_min <= theta < args.theta_max)
                if not inside:
                    raise ValueError(
                        f"{hits} contains theta={theta} outside "
                        f"[{args.theta_min}, {args.theta_max})")

    print(json.dumps({"status": "valid", "path": str(path), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
