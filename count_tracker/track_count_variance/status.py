#!/usr/bin/env python3
"""Report durable per-event progress for the track-count variance ensemble."""

import argparse
import json
from pathlib import Path


COLLECTIONS = ("VBC", "VEC", "ITBC", "ITEC", "OTBC", "OTEC")


def event_status(point, event):
    event_id = event["event_id"]
    split = event["split"]
    conditions = point / "conditions" / split / event_id
    samples = point / "count_samples" / split / event_id
    required = {
        "conditions": [conditions / f"{short}_conditions.npy" for short in COLLECTIONS],
        "sampling": [samples / f"tabddpm_{short}_samples.npy" for short in COLLECTIONS],
        "inputs": [
            point / "inputs" / event_id / sample / "input" / "input.edm4hep.root"
            for sample in ("SIM", "COUNT")
        ],
        "digitization": [
            point / "digi" / event_id / sample / "digi" / "digi_output.edm4hep.root"
            for sample in ("SIM", "COUNT")
        ],
        "reconstruction": [
            point / "events" / event_id / sample / "unsplit_reco" /
            "reco_output.edm4hep.root"
            for sample in ("SIM", "COUNT")
        ],
    }
    complete = {stage: all(path.is_file() for path in paths)
                for stage, paths in required.items()}
    return {"event_id": event_id, "complete": complete}


def inspect(scan_root):
    scan_root = Path(scan_root).resolve()
    scan = json.loads((scan_root / "scan.json").read_text())
    rows = []
    for point_definition in scan["points"]:
        point = scan_root / point_definition["tag"]
        manifest = json.loads(Path(point_definition["source_manifest"]).read_text())
        for event in manifest["events"]:
            row = event_status(point, event)
            row["batch"] = point_definition["tag"]
            rows.append(row)
    stages = ("conditions", "sampling", "inputs", "digitization", "reconstruction")
    return {
        "scan_root": str(scan_root),
        "planned_events": len(rows),
        "complete_events": {
            stage: sum(row["complete"][stage] for row in rows) for stage in stages
        },
        "events": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = inspect(args.scan_root)
    print(f"planned events: {result['planned_events']}")
    for stage, count in result["complete_events"].items():
        print(f"{stage:14s} {count:4d}/{result['planned_events']}")
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
