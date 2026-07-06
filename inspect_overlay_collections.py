#!/usr/bin/env python3

import argparse
import csv
import glob
import os
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


COLLECTIONS = {
    "OverlayVertexBarrelCollection": "eDep",
    "OverlayVertexEndcapCollection": "eDep",
    "OverlayInnerTrackerBarrelCollection": "eDep",
    "OverlayInnerTrackerEndcapCollection": "eDep",
    "OverlayOuterTrackerBarrelCollection": "eDep",
    "OverlayOuterTrackerEndcapCollection": "eDep",
    "OverlayECalBarrelCollection": "energy",
    "OverlayECalEndcapCollection": "energy",
    "OverlayHCalBarrelCollection": "energy",
    "OverlayHCalEndcapCollection": "energy",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", default="overlay_diagnostics")
    return parser.parse_args()


def find_digi_files(inputs):
    files = []
    for item in inputs:
        matches = glob.glob(item) or [item]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                files.extend(path.glob("digi_output_*.edm4hep.root"))
                files.extend(path.glob("job_*/digi_output_*.edm4hep.root"))
            elif path.is_file() and path.name.startswith("digi_output_"):
                files.append(path)
    return sorted({path.resolve() for path in files})


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


def values(events, branch, event):
    if branch is None:
        return np.asarray([], dtype=np.float64)
    return ak.to_numpy(events[branch].array(entry_start=event, entry_stop=event + 1)[0])


def write_rows(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = list(rows[0]) if rows else [
        "file",
        "event",
        "collection",
        "value_field",
        "n_hits",
        "sum_value",
        "min_time",
        "max_time",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inspect_file(path):
    rows = []
    with uproot.open(path) as root_file:
        events = root_file["events"]
        for event in range(events.num_entries):
            for collection, field in COLLECTIONS.items():
                value_branch = branch_name(events, collection, field)
                time_branch = branch_name(events, collection, "time")
                vals = values(events, value_branch, event)
                times = values(events, time_branch, event)
                row = {
                    "file": str(path),
                    "event": event,
                    "collection": collection,
                    "value_field": field,
                    "n_hits": int(len(vals)),
                    "sum_value": float(np.sum(vals)) if len(vals) else 0.0,
                    "min_time": float(np.min(times)) if len(times) else "",
                    "max_time": float(np.max(times)) if len(times) else "",
                }
                rows.append(row)
    return rows


def main():
    args = parse_args()
    files = find_digi_files(args.inputs)
    if not files:
        raise SystemExit("No digi ROOT files found")

    rows = []
    for path in files:
        rows.extend(inspect_file(path))

    outdir = os.path.join(args.outdir, args.label)
    outpath = os.path.join(outdir, f"overlay_collections_{args.label}.csv")
    write_rows(outpath, rows)

    print(f"DIGI files: {len(files)}")
    print(f"Output -> {outpath}")
    for collection in COLLECTIONS:
        selected = [row for row in rows if row["collection"] == collection]
        n_hits = sum(row["n_hits"] for row in selected)
        total = sum(row["sum_value"] for row in selected)
        print(f"{collection}: n={n_hits}, sum={total:.6g}")


if __name__ == "__main__":
    main()
