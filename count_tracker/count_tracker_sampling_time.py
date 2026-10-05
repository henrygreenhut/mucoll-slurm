#!/usr/bin/env python3
"""Record elapsed sampling time for completed SLURM array jobs.

This reads SLURM accounting only.  It reports both aggregate allocated GPU time
and the wall-clock span, since an array running on several GPUs has different
values for those two quantities.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess


FIELDS = ("JobID", "JobName", "State", "ExitCode", "ElapsedRaw",
          "Start", "End", "AllocTRES")


def parse_job(value):
    try:
        label, job_id = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("job must be SPLIT=ARRAY_JOB_ID") from exc
    if not label or not job_id.isdigit():
        raise argparse.ArgumentTypeError("job must be SPLIT=ARRAY_JOB_ID")
    return label, job_id


def query_array(job_id):
    command = [
        "sacct", "-j", job_id, "-X", "-n", "-P",
        f"--format={','.join(FIELDS)}",
    ]
    completed = subprocess.run(command, check=True, text=True,
                               stdout=subprocess.PIPE)
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if len(values) != len(FIELDS):
            raise ValueError(f"unexpected sacct row: {line}")
        row = dict(zip(FIELDS, values))
        # -X normally suppresses job steps. Keep only array elements explicitly.
        if "_" not in row["JobID"] or "." in row["JobID"]:
            continue
        row["elapsed_seconds"] = int(row.pop("ElapsedRaw"))
        rows.append(row)
    if not rows:
        raise ValueError(f"no array tasks found for job {job_id}")
    return rows, command


def gpu_count(alloc_tres):
    for item in alloc_tres.split(","):
        if item.startswith("gres/gpu="):
            return int(item.split("=", 1)[1])
    return 0


def summary(rows, events_per_task):
    elapsed = [row["elapsed_seconds"] for row in rows]
    starts = [datetime.fromisoformat(row["Start"]) for row in rows]
    ends = [datetime.fromisoformat(row["End"]) for row in rows]
    gpu_seconds = sum(seconds * gpu_count(row["AllocTRES"])
                      for seconds, row in zip(elapsed, rows))
    return {
        "tasks": len(rows),
        "events": len(rows) * events_per_task,
        "events_per_task": events_per_task,
        "states": sorted({row["State"] for row in rows}),
        "elapsed_seconds": {
            "total": sum(elapsed),
            "mean_per_task": statistics.mean(elapsed),
            "median_per_task": statistics.median(elapsed),
            "min_per_task": min(elapsed),
            "max_per_task": max(elapsed),
            "population_std_per_task": statistics.pstdev(elapsed),
        },
        "allocated_gpu_hours": gpu_seconds / 3600.0,
        "mean_allocated_gpu_seconds_per_event": (
            gpu_seconds / (len(rows) * events_per_task)
        ),
        "first_start": min(starts).isoformat(),
        "last_end": max(ends).isoformat(),
        "wall_clock_span_seconds": (max(ends) - min(starts)).total_seconds(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="append", type=parse_job, required=True,
                        metavar="SPLIT=ARRAY_JOB_ID")
    parser.add_argument("--events-per-task", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.events_per_task <= 0:
        raise SystemExit("--events-per-task must be positive")

    all_rows, splits, commands = [], {}, []
    for label, job_id in args.job:
        rows, command = query_array(job_id)
        for row in rows:
            row["split"] = label
            row["array_job_id"] = job_id
        all_rows.extend(rows)
        splits[label] = summary(rows, args.events_per_task)
        commands.append(command)
    if any(row["State"] != "COMPLETED" or row["ExitCode"] != "0:0"
           for row in all_rows):
        raise SystemExit("not all sampling array tasks completed successfully")

    report = {
        "kind": "count_tracker_sampling_time",
        "source": "SLURM accounting (sacct)",
        "queries": commands,
        "splits": splits,
        "all": summary(all_rows, args.events_per_task),
        "tasks": all_rows,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["all"], indent=2))


if __name__ == "__main__":
    main()
