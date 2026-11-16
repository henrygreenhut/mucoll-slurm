#!/usr/bin/env python3
"""Merge four priority-wave reports into the plot's 10%-trained-model table."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def add_values(store, fraction, files, reconstruction, sim, count, source):
    row = store.setdefault(fraction, {
        "files": files, "reconstruction": reconstruction,
        "sim": [], "count": [], "sources": [],
    })
    if row["files"] != files or row["reconstruction"] != reconstruction:
        raise ValueError(f"inconsistent metadata at BX fraction {fraction}")
    row["sim"].extend(sim); row["count"].extend(count); row["sources"].append(source)


def merge(fresh_reports, reuse_reports, output):
    store = {}
    for path in fresh_reports:
        report = json.loads(path.read_text())
        for point in report["results"]:
            add_values(
                store, point["fraction_of_full_bx_reference"],
                point["files_per_polarity"], "five-region",
                [row["SIM"]["si_tracks"] for row in point["paired"]],
                [row["COUNT"]["si_tracks"] for row in point["paired"]],
                str(path.resolve()),
            )
    for path in reuse_reports:
        report = json.loads(path.read_text())
        for point in report["results"]:
            add_values(
                store, point["fraction_of_full_bx_reference"],
                point["files_per_polarity"], point["reconstruction"],
                point["sim"]["values"], point["count_10pct"]["values"],
                str(path.resolve()),
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "bx_fraction", "files_per_polarity", "events", "reconstruction",
        "paired_sim_mean_tracks", "paired_sim_std_tracks",
        "count_10pct_mean_tracks", "count_10pct_std_tracks",
        "count_10pct_over_paired_sim", "source",
    )
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for fraction, row in sorted(store.items()):
            if len(row["sim"]) != len(row["count"]):
                raise ValueError(f"unpaired samples at BX fraction {fraction}")
            writer.writerow({
                "bx_fraction": fraction,
                "files_per_polarity": row["files"],
                "events": len(row["sim"]),
                "reconstruction": row["reconstruction"],
                "paired_sim_mean_tracks": statistics.mean(row["sim"]),
                "paired_sim_std_tracks": statistics.stdev(row["sim"]) if len(row["sim"]) > 1 else 0.0,
                "count_10pct_mean_tracks": statistics.mean(row["count"]),
                "count_10pct_std_tracks": statistics.stdev(row["count"]) if len(row["count"]) > 1 else 0.0,
                "count_10pct_over_paired_sim": sum(row["count"]) / sum(row["sim"]),
                "source": ";".join(row["sources"]),
            })
    print(f"Wrote {len(store)} occupancy points -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-report", action="append", type=Path, required=True)
    parser.add_argument("--reuse-report", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(args.fresh_report, args.reuse_report, args.output)


if __name__ == "__main__":
    main()
