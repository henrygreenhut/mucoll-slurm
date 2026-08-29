#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-dir", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def read_values(path):
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        values[key] = float(value)
    return values


def completed_jobs(directory):
    jobs = {
        path.name: path
        for path in Path(directory).glob("job_*")
        if (path / "complete").is_file()
    }
    if not jobs:
        raise SystemExit("no completed jobs in {}".format(directory))
    return jobs


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(directory, job_names):
    totals = {
        "events": 0.0,
        "gen_seconds": 0.0,
        "sim_seconds": 0.0,
        "digi_seconds": 0.0,
        "reco_seconds": 0.0,
        "total_seconds": 0.0,
    }
    reco_rows = []
    digi_rows = []

    for name in job_names:
        job = directory[name]
        timing = read_values(job / "timing.txt")
        for key in totals:
            totals[key] += timing[key]
        reco_rows.extend(csv_rows(job / "reco_summary.csv"))
        digi_path = job / "digi_summary.csv"
        if digi_path.is_file():
            digi_rows.extend(csv_rows(digi_path))

    events = int(totals["events"])
    if len(reco_rows) != events:
        raise SystemExit(
            "{} has {} timed events but {} RECO rows".format(
                next(iter(directory.values())).parent, events, len(reco_rows)
            )
        )

    result = {"events": float(events)}
    for stage in ("gen", "sim", "digi", "reco", "total"):
        result[stage + "_seconds_per_event"] = totals[stage + "_seconds"] / events

    for key in ("pfos", "charged_pfos", "tracks", "pfo_track_links"):
        result[key + "_per_event"] = sum(float(row[key]) for row in reco_rows) / events

    if digi_rows:
        if len(digi_rows) != events:
            raise SystemExit(
                "{} has {} timed events but {} DIGI rows".format(
                    next(iter(directory.values())).parent, events, len(digi_rows)
                )
            )
        for key in (
            "overlay_tracker_hits",
            "overlay_tracker_energy_GeV",
            "overlay_calo_hits",
            "overlay_calo_energy_GeV",
            "digi_tracker_hits",
            "digi_tracker_energy_GeV",
            "digi_calo_hits",
            "digi_calo_energy_GeV",
        ):
            result[key + "_per_event"] = (
                sum(float(row[key]) for row in digi_rows) / events
            )

    return result


def main():
    args = arguments()
    directories = {
        "legacy": completed_jobs(args.legacy_dir),
        "split_bh": completed_jobs(args.split_dir),
    }
    legacy_jobs = set(directories["legacy"])
    split_jobs = set(directories["split_bh"])
    if legacy_jobs != split_jobs:
        raise SystemExit(
            "completed job sets differ: legacy={} split_bh={}".format(
                sorted(legacy_jobs), sorted(split_jobs)
            )
        )

    job_names = sorted(legacy_jobs)
    summaries = {
        name: summarize(directory, job_names)
        for name, directory in directories.items()
    }

    rows = []
    for metric in summaries["legacy"]:
        legacy = summaries["legacy"][metric]
        split = summaries["split_bh"].get(metric)
        if split is None:
            continue
        ratio = split / legacy if legacy else float("nan")
        rows.append((metric, legacy, split, split - legacy, ratio))

    print("{:<38} {:>14} {:>14} {:>14} {:>12}".format(
        "metric", "legacy", "split_bh", "difference", "ratio"
    ))
    for metric, legacy, split, difference, ratio in rows:
        print("{:<38} {:>14.4f} {:>14.4f} {:>14.4f} {:>12.4f}".format(
            metric, legacy, split, difference, ratio
        ))

    if args.output:
        with Path(args.output).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("metric", "legacy", "split_bh", "difference", "ratio"))
            writer.writerows(rows)
        print("comparison ->", args.output)


if __name__ == "__main__":
    main()
