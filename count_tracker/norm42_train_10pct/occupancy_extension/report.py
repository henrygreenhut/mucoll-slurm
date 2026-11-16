#!/usr/bin/env python3
"""Report paired existing-SIM versus 10%-trained COUNT track multiplicities."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from count_tracker_checkpoint_report import collection_size  # noqa: E402


REGIONS = (
    "theta_000_030", "theta_030_070", "theta_070_110",
    "theta_110_150", "theta_150_180",
)


def tracks(path):
    import uproot
    with uproot.open(path) as root:
        return int(collection_size(root["events"], "SiTracks_objIdx", "index"))


def event_tracks(point, event_id, arm, mode):
    if mode == "unsplit":
        return tracks(
            point / "events" / event_id / arm / "unsplit_reco" /
            "reco_output.edm4hep.root"
        )
    return sum(
        tracks(
            point / "events" / event_id / arm / "theta_reco" / region /
            "reco_output.edm4hep.root"
        )
        for region in REGIONS
    )


def summary(values):
    return {
        "events": len(values),
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def build(root):
    root = Path(root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    results = []
    for point in plan["points"]:
        extension_point = root / point["tag"]
        sim_point = Path(point["sim_source_point"])
        rows = []
        for event in point["events"]:
            event_id = event["event_id"]
            sim = event_tracks(sim_point, event_id, "SIM", point["reconstruction"])
            count = event_tracks(extension_point, event_id, "COUNT", point["reconstruction"])
            rows.append({
                "event_id": event_id,
                "sim_tracks": sim,
                "count_10pct_tracks": count,
                "count_10pct_over_sim": count / sim if sim else None,
            })
        sim_values = [row["sim_tracks"] for row in rows]
        count_values = [row["count_10pct_tracks"] for row in rows]
        results.append({
            "tag": point["tag"],
            "files_per_polarity": point["files_per_polarity"],
            "fraction_of_full_bx_reference": point["fraction_of_full_bx_reference"],
            "reconstruction": point["reconstruction"],
            "sim": summary(sim_values),
            "count_10pct": summary(count_values),
            "ratio_of_summed_tracks": sum(count_values) / sum(sim_values),
            "paired": rows,
            "training_overlap": point["training_overlap"],
        })
    return {"kind": "norm42_10pct_model_occupancy_extension_report", "plan": plan, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.root)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    with args.csv.open("w", newline="") as stream:
        fields = (
            "bx_fraction", "files_per_polarity", "events", "reconstruction",
            "sim_mean_tracks", "sim_std_tracks", "count_10pct_mean_tracks",
            "count_10pct_std_tracks", "count_10pct_over_sim",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["results"]:
            writer.writerow({
                "bx_fraction": row["fraction_of_full_bx_reference"],
                "files_per_polarity": row["files_per_polarity"],
                "events": row["sim"]["events"],
                "reconstruction": row["reconstruction"],
                "sim_mean_tracks": row["sim"]["mean"],
                "sim_std_tracks": row["sim"]["sample_standard_deviation"],
                "count_10pct_mean_tracks": row["count_10pct"]["mean"],
                "count_10pct_std_tracks": row["count_10pct"]["sample_standard_deviation"],
                "count_10pct_over_sim": row["ratio_of_summed_tracks"],
            })
    print("BX%      events  SIM mean    COUNT10 mean  COUNT10/SIM")
    for row in report["results"]:
        print(
            f"{100 * row['fraction_of_full_bx_reference']:7.3f}  "
            f"{row['sim']['events']:6d}  {row['sim']['mean']:10.2f}  "
            f"{row['count_10pct']['mean']:12.2f}  {row['ratio_of_summed_tracks']:11.4f}"
        )


if __name__ == "__main__":
    main()
