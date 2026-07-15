#!/usr/bin/env python3
"""Submit the complete N=420 RECO data set as one packed CPU allocation."""

import argparse
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path


SPLIT_EVENTS = {
    "train": 2000,
    "val": 400,
    "test_a": 400,
    "test_b": 400,
}
JOB_ID_BASE = {
    "train": 0,
    "val": 100_000,
    "test_a": 200_000,
    "test_b": 300_000,
}
LIBRARY = {"U": "norm1", "R": "norm42", "null_b": "norm1"}
DIGI_OFFSET = {"U": 0, "R": 0, "null_b": 1_000_000}


def parse_args():
    scratch = os.environ.get("PSCRATCH", "")
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=list(SPLIT_EVENTS),
                        default=list(SPLIT_EVENTS))
    parser.add_argument("--classes", nargs="+", choices=list(LIBRARY),
                        default=list(LIBRARY))
    parser.add_argument("--events-per-job", type=int, default=50)
    parser.add_argument("--n-files", type=int, default=420)
    parser.add_argument("--tasks-per-node", type=int, default=64)
    parser.add_argument("--qos", default="debug")
    parser.add_argument("--time", default="00:30:00")
    parser.add_argument("--account", default="m5197")
    parser.add_argument("--pools", default=(scratch + "/mucoll/libtest/bib_pools")
                        if scratch else None, required=not bool(scratch))
    parser.add_argument("--outdir", default=(scratch + "/mucoll/libtest/reco_n420_pfn")
                        if scratch else None, required=not bool(scratch))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.events_per_job < 1 or args.tasks_per_node < 1:
        raise SystemExit("event and task counts must be positive")
    if args.n_files < 42 or args.n_files % 42:
        raise SystemExit("--n-files must be a positive multiple of 42")

    repo = Path(__file__).resolve().parent
    pools = Path(args.pools).resolve()
    outdir = Path(args.outdir).resolve()
    logs = repo / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = logs / "reco_n{}_{}.tsv".format(args.n_files, stamp)

    rows = []
    skipped = 0
    for split in args.splits:
        n_jobs = math.ceil(SPLIT_EVENTS[split] / args.events_per_job)
        for sample in args.classes:
            library = LIBRARY[sample]
            plus = pools / library / split / "MUPLUS"
            minus = pools / library / split / "MUMINUS"
            for directory in (plus, minus):
                if not directory.is_dir() or not any(directory.glob("*.root")):
                    raise SystemExit("empty or missing pool: {}".format(directory))

            bib_number = args.n_files if sample != "R" else args.n_files // 42
            study = "reco_libtest_n{}_{}/{}".format(args.n_files, sample, split)
            for index in range(n_jobs):
                job_id = JOB_ID_BASE[split] + index
                first = index * args.events_per_job
                nevents = min(args.events_per_job,
                              SPLIT_EVENTS[split] - first)
                expected = (outdir / study / "job_{}".format(job_id) /
                            "reco_output_{}.edm4hep.root".format(job_id))
                if expected.is_file() and expected.stat().st_size > 0 and not args.force:
                    skipped += 1
                    continue
                rows.append([
                    sample, split, str(index), str(job_id), str(nevents),
                    study, str(outdir), str(plus) + "/", str(minus) + "/",
                    str(bib_number), str(DIGI_OFFSET[sample]),
                ])

    if not rows:
        print("All requested RECO outputs already exist; nothing to submit.")
        return

    with manifest.open("w") as handle:
        for row in rows:
            handle.write("\t".join(row) + "\n")

    nodes = math.ceil(len(rows) / args.tasks_per_node)
    if args.qos == "debug" and nodes > 8:
        raise SystemExit("debug permits at most 8 nodes; lower the requested data set")

    slurm = repo / "submit_reco_libtest_packed.slurm"
    command = [
        "sbatch", "--parsable",
        "--account={}".format(args.account),
        "--qos={}".format(args.qos),
        "--time={}".format(args.time),
        "--nodes={}".format(nodes),
        "--ntasks-per-node={}".format(args.tasks_per_node),
        "--cpus-per-task=4",
        "--export=ALL,TASKS_PER_NODE={}".format(args.tasks_per_node),
        str(slurm), str(manifest),
    ]
    print("manifest: {}".format(manifest))
    print("tasks: {} ({} existing outputs skipped)".format(len(rows), skipped))
    print("allocation: {} CPU node(s), up to {} chains/node".format(
        nodes, args.tasks_per_node))
    print(" ".join(command))
    if not args.dry_run:
        result = subprocess.run(command, check=True, text=True,
                                capture_output=True)
        print("submitted packed job {}".format(
            result.stdout.strip().split(";", 1)[0]))


if __name__ == "__main__":
    main()
