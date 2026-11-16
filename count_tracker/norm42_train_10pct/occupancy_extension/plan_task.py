#!/usr/bin/env python3
"""Resolve a point, event, or five-region task from an extension plan."""

import argparse
import json
from pathlib import Path


REGIONS = ((0, 30), (30, 70), (70, 110), (110, 150), (150, 180))


def tasks(plan, stage):
    rows = []
    for point_index, point in enumerate(plan["points"]):
        base = {
            "point_index": point_index,
            "tag": point["tag"],
            "mode": point["reconstruction"],
            "sim_point": point["sim_source_point"],
        }
        if stage == "point":
            rows.append(base)
            continue
        for event_index, event in enumerate(point["events"]):
            event_row = {
                **base,
                "event_index": event_index,
                "event_id": event["event_id"],
                "split": event["split"],
            }
            if stage == "event":
                rows.append(event_row)
            elif point["reconstruction"] == "five-region":
                for region_index, (lower, upper) in enumerate(REGIONS):
                    rows.append({
                        **event_row,
                        "region_index": region_index,
                        "theta_min": lower,
                        "theta_max": upper,
                        "region_tag": f"theta_{lower:03d}_{upper:03d}",
                    })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=("point", "event", "region"), required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--field")
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    rows = tasks(plan, args.stage)
    if args.count:
        print(len(rows))
        return
    if not 0 <= args.index < len(rows):
        raise IndexError(f"task {args.index} outside [0, {len(rows)})")
    row = rows[args.index]
    if args.field:
        print(row[args.field])
    else:
        print(json.dumps(row, separators=(",", ":")))


if __name__ == "__main__":
    main()
