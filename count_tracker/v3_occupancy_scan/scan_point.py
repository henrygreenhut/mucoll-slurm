#!/usr/bin/env python3
"""Print one occupancy point from scan.json for shell-array wrappers."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument(
        "--field", choices=("tag", "files_per_polarity", "event_id"), required=True)
    parser.add_argument("--event-index", type=int)
    args = parser.parse_args()
    points = json.loads(args.scan.read_text())["points"]
    if not 0 <= args.index < len(points):
        raise IndexError(f"point index {args.index} outside [0, {len(points)})")
    point = points[args.index]
    if args.field == "event_id":
        if args.event_index is None:
            parser.error("--field event_id requires --event-index")
        manifest = json.loads(Path(point["source_manifest"]).read_text())
        events = manifest["events"]
        if not 0 <= args.event_index < len(events):
            raise IndexError(
                f"event index {args.event_index} outside [0, {len(events)})")
        print(events[args.event_index]["event_id"])
    else:
        print(point[args.field])


if __name__ == "__main__":
    main()
