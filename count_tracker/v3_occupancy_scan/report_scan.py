#!/usr/bin/env python3
"""Report digitization survival and summed five-region SiTracks versus occupancy."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from count_tracker_checkpoint_report import COLLECTIONS, collection_size


REGIONS = (
    "theta_000_030", "theta_030_070", "theta_070_110",
    "theta_110_150", "theta_150_180",
)
SPLIT_HIT_COLLECTIONS = (
    "VXDBarrelHitsSplit", "VXDEndcapHitsSplit", "ITBarrelHitsSplit",
    "ITEndcapHitsSplit", "OTBarrelHitsSplit", "OTEndcapHitsSplit",
)


def event_arm(point, event_id, sample):
    import uproot

    digi_path = point / "digi" / event_id / sample / "digi" / "digi_output.edm4hep.root"
    with uproot.open(digi_path) as root:
        events = root["events"]
        entering = sum(collection_size(events, sim_name) for sim_name, _ in COLLECTIONS.values())
        digitized = sum(collection_size(events, digi_name) for _, digi_name in COLLECTIONS.values())

    regional = {}
    regional_digitized = {}
    for region in REGIONS:
        split_path = point / "events" / event_id / sample / "theta_split" / region / "split_output.edm4hep.root"
        with uproot.open(split_path) as root:
            events = root["events"]
            regional_digitized[region] = sum(
                collection_size(events, name) for name in SPLIT_HIT_COLLECTIONS)
        path = point / "events" / event_id / sample / "theta_reco" / region / "reco_output.edm4hep.root"
        with uproot.open(path) as root:
            regional[region] = collection_size(root["events"], "SiTracks_objIdx", "index")
    if sum(regional_digitized.values()) != digitized:
        raise ValueError(
            f"{event_id}/{sample}: theta regions contain "
            f"{sum(regional_digitized.values())} of {digitized} digitized hits")
    return {
        "entering_digitization": entering,
        "digitized": digitized,
        "survival_fraction": digitized / entering if entering else None,
        "si_tracks": sum(regional.values()),
        "regional_digitized_hits": regional_digitized,
        "regional_si_tracks": regional,
    }


def summarize(scan_root):
    scan_root = Path(scan_root).resolve()
    scan = json.loads((scan_root / "scan.json").read_text())
    results = []
    for definition in scan["points"]:
        point = scan_root / definition["tag"]
        conditions = json.loads((point / "conditions" / "manifest.json").read_text())
        rows = []
        for event in conditions["events"]:
            row = {"event_id": event["event_id"]}
            for sample in ("SIM", "COUNT"):
                row[sample] = event_arm(point, event["event_id"], sample)
            rows.append(row)

        aggregate = {}
        for sample in ("SIM", "COUNT"):
            before = sum(row[sample]["entering_digitization"] for row in rows)
            after = sum(row[sample]["digitized"] for row in rows)
            tracks = np.asarray([row[sample]["si_tracks"] for row in rows], dtype=float)
            aggregate[sample] = {
                "entering_digitization": before,
                "digitized": after,
                "survival_fraction": after / before if before else None,
                "si_tracks": int(tracks.sum()),
                "mean_si_tracks": float(tracks.mean()),
                "std_si_tracks": float(tracks.std(ddof=1)) if len(tracks) > 1 else 0.0,
            }
        ratio = aggregate["COUNT"]["si_tracks"] / aggregate["SIM"]["si_tracks"]
        results.append({
            **definition,
            "events": len(rows),
            "aggregate": aggregate,
            "count_over_sim_tracks": ratio,
            "paired": rows,
        })
    return {"scan": scan, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.scan_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("files  BX_frac  SIM_tracks  COUNT_tracks  COUNT/SIM  SIM_surv  COUNT_surv")
    for row in report["results"]:
        sim = row["aggregate"]["SIM"]
        count = row["aggregate"]["COUNT"]
        print(
            f"{row['files_per_polarity']:5d}  "
            f"{row['fraction_of_full_bx_reference']:7.4f}  "
            f"{sim['si_tracks']:10d}  {count['si_tracks']:12d}  "
            f"{row['count_over_sim_tracks']:9.4f}  "
            f"{sim['survival_fraction']:8.4f}  {count['survival_fraction']:10.4f}"
        )


if __name__ == "__main__":
    main()
