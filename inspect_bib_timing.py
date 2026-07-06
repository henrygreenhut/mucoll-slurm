#!/usr/bin/env python3

import argparse
import csv
import glob
import os
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


TRACKER_COLLECTIONS = [
    "VertexBarrelCollection",
    "VertexEndcapCollection",
    "InnerTrackerBarrelCollection",
    "InnerTrackerEndcapCollection",
    "OuterTrackerBarrelCollection",
    "OuterTrackerEndcapCollection",
]

CALO_COLLECTIONS = [
    "ECalBarrelCollection",
    "ECalEndcapCollection",
    "HCalBarrelCollection",
    "HCalEndcapCollection",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--label", required=True)
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument("--time-min", type=float, default=-0.5)
    parser.add_argument("--time-max", type=float, default=15.0)
    parser.add_argument("--outdir", default="bib_timing_checks")
    return parser.parse_args()


def find_root_files(inputs, n_files):
    files = []
    for item in inputs:
        matches = glob.glob(item) or [item]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                files.extend(path.glob("*.root"))
                files.extend(path.glob("*.edm4hep.root"))
            elif path.is_file() and path.suffix == ".root":
                files.append(path)
    return sorted({path.resolve() for path in files})[:n_files]


def branch_name(events, collection, field):
    candidates = [
        f"{collection}/{collection}.{field}",
        f"{collection}.{field}",
    ]
    keys = set(events.keys())
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def contribution_time_branch(events, collection):
    contrib = f"{collection}Contributions"
    candidates = [
        f"{contrib}/{contrib}.time",
        f"{contrib}.time",
    ]
    keys = set(events.keys())
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def values(events, branch, event):
    if branch is None:
        return np.asarray([], dtype=np.float64)
    return ak.to_numpy(events[branch].array(entry_start=event, entry_stop=event + 1)[0])


def summarize(path, event, collection, value_field, vals, times, time_min, time_max):
    in_window = (times >= time_min) & (times <= time_max) if len(times) else []
    return {
        "file": str(path),
        "event": event,
        "collection": collection,
        "value_field": value_field,
        "n_hits": int(len(vals)),
        "sum_value": float(np.sum(vals)) if len(vals) else 0.0,
        "n_times": int(len(times)),
        "min_time": float(np.min(times)) if len(times) else "",
        "max_time": float(np.max(times)) if len(times) else "",
        "n_in_window": int(np.sum(in_window)) if len(times) else "",
        "time_min": time_min,
        "time_max": time_max,
    }


def inspect_file(path, time_min, time_max):
    rows = []
    with uproot.open(path) as root_file:
        events = root_file["events"]
        for event in range(events.num_entries):
            for collection in TRACKER_COLLECTIONS:
                vals = values(events, branch_name(events, collection, "eDep"), event)
                times = values(events, branch_name(events, collection, "time"), event)
                if len(vals) or len(times):
                    rows.append(summarize(
                        path, event, collection, "eDep", vals, times, time_min, time_max
                    ))
            for collection in CALO_COLLECTIONS:
                vals = values(events, branch_name(events, collection, "energy"), event)
                times = values(events, contribution_time_branch(events, collection), event)
                if len(vals) or len(times):
                    rows.append(summarize(
                        path, event, collection, "energy", vals, times, time_min, time_max
                    ))
    return rows


def write_rows(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = list(rows[0]) if rows else [
        "file",
        "event",
        "collection",
        "value_field",
        "n_hits",
        "sum_value",
        "n_times",
        "min_time",
        "max_time",
        "n_in_window",
        "time_min",
        "time_max",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    files = find_root_files(args.inputs, args.n_files)
    if not files:
        raise SystemExit("No ROOT files found")

    rows = []
    for path in files:
        rows.extend(inspect_file(path, args.time_min, args.time_max))

    outdir = os.path.join(args.outdir, args.label)
    outpath = os.path.join(outdir, f"bib_timing_{args.label}.csv")
    write_rows(outpath, rows)

    print(f"ROOT files sampled: {len(files)}")
    print(f"Output -> {outpath}")
    for collection in TRACKER_COLLECTIONS + CALO_COLLECTIONS:
        selected = [row for row in rows if row["collection"] == collection]
        if not selected:
            continue
        n_hits = sum(row["n_hits"] for row in selected)
        n_times = sum(row["n_times"] for row in selected)
        in_window = sum(row["n_in_window"] for row in selected if row["n_in_window"] != "")
        finite_min = [row["min_time"] for row in selected if row["min_time"] != ""]
        finite_max = [row["max_time"] for row in selected if row["max_time"] != ""]
        min_time = min(finite_min) if finite_min else ""
        max_time = max(finite_max) if finite_max else ""
        print(
            f"{collection}: n_hits={n_hits}, n_times={n_times}, "
            f"min_time={min_time}, max_time={max_time}, in_window={in_window}"
        )


if __name__ == "__main__":
    main()
