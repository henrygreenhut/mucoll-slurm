#!/usr/bin/env python3
"""Compare unsplit v3 CKF tracks with the summed five-region result."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from report_scan import event_arm
from count_tracker_checkpoint_report import collection_size


def unsplit_tracks(point, event_id, sample):
    import uproot

    path = (
        point / "events" / event_id / sample / "unsplit_reco" /
        "reco_output.edm4hep.root"
    )
    with uproot.open(path) as root:
        return collection_size(root["events"], "SiTracks_objIdx", "index")


def summarize(scan_root, point_indices):
    scan_root = Path(scan_root).resolve()
    scan = json.loads((scan_root / "scan.json").read_text())
    definitions = scan["points"]
    results = []
    for index in point_indices:
        if not 0 <= index < len(definitions):
            raise IndexError(f"point index {index} outside [0, {len(definitions)})")
        definition = definitions[index]
        point = scan_root / definition["tag"]
        conditions = json.loads(
            (point / "conditions" / "manifest.json").read_text())
        rows = []
        for event in conditions["events"]:
            row = {"event_id": event["event_id"]}
            for sample in ("SIM", "COUNT"):
                split = event_arm(point, event["event_id"], sample)
                row[sample] = {
                    "split_si_tracks": split["si_tracks"],
                    "unsplit_si_tracks": unsplit_tracks(
                        point, event["event_id"], sample),
                }
            rows.append(row)

        aggregate = {}
        for sample in ("SIM", "COUNT"):
            split = np.asarray(
                [row[sample]["split_si_tracks"] for row in rows], dtype=float)
            unsplit = np.asarray(
                [row[sample]["unsplit_si_tracks"] for row in rows], dtype=float)
            aggregate[sample] = {
                "split_si_tracks": int(split.sum()),
                "unsplit_si_tracks": int(unsplit.sum()),
                "unsplit_over_split": (
                    float(unsplit.sum() / split.sum()) if split.sum() else None),
                "split_mean": float(split.mean()),
                "unsplit_mean": float(unsplit.mean()),
            }
        results.append({
            **definition,
            "point_index": index,
            "events": len(rows),
            "aggregate": aggregate,
            "split_count_over_sim": (
                aggregate["COUNT"]["split_si_tracks"] /
                aggregate["SIM"]["split_si_tracks"]),
            "unsplit_count_over_sim": (
                aggregate["COUNT"]["unsplit_si_tracks"] /
                aggregate["SIM"]["unsplit_si_tracks"]),
            "paired": rows,
        })
    return {"scan": scan, "point_indices": point_indices, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument(
        "--point-indices", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.scan_root, args.point_indices)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(
        "files  split_SIM  split_COUNT  split_C/S  "
        "unsplit_SIM  unsplit_COUNT  unsplit_C/S  SIM_u/s  COUNT_u/s")
    for row in report["results"]:
        sim = row["aggregate"]["SIM"]
        count = row["aggregate"]["COUNT"]
        print(
            f"{row['files_per_polarity']:5d}  "
            f"{sim['split_si_tracks']:9d}  {count['split_si_tracks']:11d}  "
            f"{row['split_count_over_sim']:9.4f}  "
            f"{sim['unsplit_si_tracks']:11d}  {count['unsplit_si_tracks']:13d}  "
            f"{row['unsplit_count_over_sim']:11.4f}  "
            f"{sim['unsplit_over_split']:7.4f}  "
            f"{count['unsplit_over_split']:9.4f}")


if __name__ == "__main__":
    main()
