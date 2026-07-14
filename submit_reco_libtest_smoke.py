#!/usr/bin/env python3
"""Submit paired neutrino+BIB reco smoke samples on Perlmutter."""

import argparse
import os
import subprocess
from pathlib import Path


SAMPLES = {
    "U": "norm1",
    "R": "norm42",
    # Null labels independently resample the same norm1 pool.  A distinct
    # DIGI seed for null_b prevents identical overlay choices while retaining
    # the same neutrino GEN seed in both labels.
    "null_a": "norm1",
    "null_b": "norm1",
}

DIGI_SEED_OFFSETS = {"U": 0, "R": 0, "null_a": 0, "null_b": 1000000}


def parse_args():
    scratch = os.environ.get("PSCRATCH", "")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", default=(scratch + "/mucoll/libtest/bib_pools")
                        if scratch else None, required=not bool(scratch))
    parser.add_argument("--outdir", default=(scratch + "/mucoll/libtest/reco")
                        if scratch else None, required=not bool(scratch))
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test", "test_a", "test_b"])
    parser.add_argument("--classes", nargs="+", default=["U", "R"],
                        choices=list(SAMPLES))
    parser.add_argument("--jobs-per-class", type=int, default=2)
    parser.add_argument("--events-per-job", type=int, default=10)
    parser.add_argument("--job-id-start", type=int, default=0)
    parser.add_argument("--n-files", type=int, default=42,
                        help="norm1 files per event; must be a multiple of 42")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.jobs_per_class < 1 or args.events_per_job < 1:
        raise SystemExit("job and event counts must be positive")
    if args.n_files < 42 or args.n_files % 42:
        raise SystemExit("--n-files must be a positive multiple of 42")
    pools = Path(args.pools).resolve()
    outdir = Path(args.outdir).resolve()
    slurm = Path(__file__).resolve().parent / "submit_pgun_perlmutter.slurm"
    if not slurm.is_file():
        raise SystemExit("missing {}".format(slurm))

    commands = []
    for sample in args.classes:
        library = SAMPLES[sample]
        bib_number = (args.n_files // 42 if sample == "R"
                      else args.n_files)
        plus = pools / library / args.split / "MUPLUS"
        minus = pools / library / args.split / "MUMINUS"
        for path in (plus, minus):
            if not path.is_dir() or not any(path.glob("*.root")):
                raise SystemExit("empty or missing pool: {}".format(path))
        for offset in range(args.jobs_per_class):
            job_id = args.job_id_start + offset
            study = "reco_libtest_n{}_{}/{}".format(
                args.n_files, sample, args.split)
            exports = ",".join([
                "ALL", "JOB_ID={}".format(job_id),
                "NEVENTS={}".format(args.events_per_job),
                "PDG=14", "PT=100", "THETA_MIN=10", "THETA_MAX=170",
                "USE_BIB=1", "STUDY_NAME={}".format(study),
                "OUTPUT_BASE_DIR={}".format(outdir),
                "BIB_MUPLUS={}/".format(plus),
                "BIB_MUMINUS={}/".format(minus),
                "BIB_NUMBER={}".format(bib_number),
                "DIGI_SEED_OFFSET={}".format(DIGI_SEED_OFFSETS[sample]),
            ])
            command = [
                "sbatch", "--parsable",
                "--job-name=reco_{}_{}_{}".format(sample, args.split, job_id),
                "--export={}".format(exports), str(slurm),
            ]
            commands.append(command)

    print("{} jobs, {} events/class, split={}, n_files={}".format(
        len(commands), args.jobs_per_class * args.events_per_job, args.split,
        args.n_files))
    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            result = subprocess.run(command, check=True, text=True,
                                    capture_output=True)
            print("  job {}".format(result.stdout.strip().split(";", 1)[0]))


if __name__ == "__main__":
    main()
