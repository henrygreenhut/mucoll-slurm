#!/usr/bin/env python3
"""Partition the prepared low-occupancy wave into fast and supplemental plans.

The first plan retains every lower-occupancy point but limits n0464 to its
first two events.  The second contains the other three n0464 events.  Both
reuse the immutable conditions and resumable sample directory from the
original wave; reconstruction outputs remain separate under the new roots.
"""

import argparse
import copy
import json
import os
from pathlib import Path


SHARED = ("sources.json", "conditions", "count_samples")


def link_point(source_root, output_root, point):
    tag = point["tag"]
    source = source_root / tag
    destination = output_root / tag
    destination.mkdir(parents=True)
    for name in SHARED:
        target = source / name
        if not target.exists():
            raise FileNotFoundError(target)
        os.symlink(target.resolve(), destination / name,
                   target_is_directory=target.is_dir())


def subset_point(point, indices):
    result = copy.deepcopy(point)
    result["events"] = [point["events"][index] for index in indices]
    if "training_overlap" in point:
        result["training_overlap"] = [
            point["training_overlap"][index] for index in indices
        ]
    return result


def write_plan(source_root, output, plan, partition, points):
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    output.mkdir(parents=True)
    for point in points:
        link_point(source_root, output, point)
    document = copy.deepcopy(plan)
    document["kind"] = "norm42_10pct_model_existing_sim_occupancy_partition"
    document["derived_from"] = str((source_root / "plan.json").resolve())
    document["event_partition"] = partition
    document["points"] = points
    (output / "plan.json").write_text(json.dumps(document, indent=2) + "\n")


def prepare(args):
    source_root = Path(args.source_root).resolve()
    fast_root = Path(args.fast_root).resolve()
    remaining_root = Path(args.remaining_root).resolve()
    plan = json.loads((source_root / "plan.json").read_text())
    points = {point["tag"]: point for point in plan["points"]}
    n0464 = points["n0464"]
    if len(n0464["events"]) != 5:
        raise ValueError("expected five original n0464 events")

    fast_points = [copy.deepcopy(point) for point in plan["points"]
                   if point["tag"] != "n0464"]
    fast_points.append(subset_point(n0464, (0, 1)))
    remaining_points = [subset_point(n0464, (2, 3, 4))]

    write_plan(source_root, fast_root, plan, "first two n0464 events", fast_points)
    write_plan(source_root, remaining_root, plan,
               "remaining three n0464 events", remaining_points)
    print(f"Prepared fast Priority 2 -> {fast_root}")
    print(f"Prepared supplemental n0464 wave -> {remaining_root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--fast-root", required=True)
    parser.add_argument("--remaining-root", required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
