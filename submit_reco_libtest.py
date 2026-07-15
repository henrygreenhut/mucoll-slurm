#!/usr/bin/env python3
"""Submit the fixed N=420 neutrino+BIB RECO data set on Perlmutter."""

import argparse
import math
import os
import subprocess
from pathlib import Path


LIBRARY = {"U": "norm1", "R": "norm42", "null_b": "norm1"}
DIGI_OFFSET = {"U": 0, "R": 0, "null_b": 1_000_000}
JOB_ID_BASE = {"train": 0, "val": 100_000, "test_a": 200_000, "test_b": 300_000}


def parse_args():
    scratch = os.environ.get("PSCRATCH", "")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True,
                        choices=list(JOB_ID_BASE))
    parser.add_argument("--classes", nargs="+", default=["U", "R", "null_b"],
                        choices=list(LIBRARY))
    parser.add_argument("--events-per-class", type=int, required=True)
    parser.add_argument("--events-per-job", type=int, default=50)
    parser.add_argument("--n-files", type=int, default=420)
    parser.add_argument("--pools", default=(scratch + "/mucoll/libtest/bib_pools")
                        if scratch else None, required=not bool(scratch))
    parser.add_argument("--outdir", default=(scratch + "/mucoll/libtest/reco_n420_pfn")
                        if scratch else None, required=not bool(scratch))
    parser.add_argument("--qos", default="shared")
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--force", action="store_true",
                        help="submit jobs whose RECO output already exists")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.events_per_class < 1 or args.events_per_job < 1:
        raise SystemExit("event counts must be positive")
    if args.n_files < 42 or args.n_files % 42:
        raise SystemExit("--n-files must be a positive multiple of 42")

    repo = Path(__file__).resolve().parent
    slurm = repo / "submit_pgun_perlmutter.slurm"
    pools = Path(args.pools).resolve()
    outdir = Path(args.outdir).resolve()
    n_jobs = math.ceil(args.events_per_class / args.events_per_job)
    submitted = skipped = 0

    print("split={} N={} events/class={} jobs/class={}".format(
        args.split, args.n_files, args.events_per_class, n_jobs))

    for sample in args.classes:
        library = LIBRARY[sample]
        plus = pools / library / args.split / "MUPLUS"
        minus = pools / library / args.split / "MUMINUS"
        for directory in (plus, minus):
            if not directory.is_dir() or not any(directory.glob("*.root")):
                raise SystemExit("empty or missing pool: {}".format(directory))

        bib_number = args.n_files if sample != "R" else args.n_files // 42
        study = "reco_libtest_n{}_{}/{}".format(args.n_files, sample, args.split)
        for index in range(n_jobs):
            job_id = JOB_ID_BASE[args.split] + index
            first = index * args.events_per_job
            nevents = min(args.events_per_job, args.events_per_class - first)
            expected = (outdir / study / "job_{}".format(job_id) /
                        "reco_output_{}.edm4hep.root".format(job_id))
            if expected.is_file() and expected.stat().st_size > 0 and not args.force:
                print("skip {} (exists)".format(expected))
                skipped += 1
                continue

            exports = ",".join([
                "ALL", "JOB_ID={}".format(job_id), "NEVENTS={}".format(nevents),
                "PDG=14", "PT=100", "THETA_MIN=10", "THETA_MAX=170",
                "USE_BIB=1", "STUDY_NAME={}".format(study),
                "OUTPUT_BASE_DIR={}".format(outdir),
                "BIB_MUPLUS={}/".format(plus),
                "BIB_MUMINUS={}/".format(minus),
                "BIB_NUMBER={}".format(bib_number),
                "DIGI_SEED_OFFSET={}".format(DIGI_OFFSET[sample]),
            ])
            command = [
                "sbatch", "--parsable", "--qos={}".format(args.qos),
                "--time={}".format(args.time),
                "--job-name=reco420_{}_{}_{:03d}".format(sample, args.split, index),
                "--export={}".format(exports), str(slurm),
            ]
            print(" ".join(command))
            if not args.dry_run:
                result = subprocess.run(command, check=True, text=True,
                                        capture_output=True)
                print("  job {}".format(result.stdout.strip().split(";", 1)[0]))
            submitted += 1

    print("submitted={} skipped={}".format(submitted, skipped))


if __name__ == "__main__":
    main()
