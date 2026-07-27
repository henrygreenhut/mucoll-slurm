#!/usr/bin/env python3
"""Submit the complete N=420 RECO data set as one packed CPU allocation.

Ported from Perlmutter (whole-node shifter/srun packing, --account/--qos)
to OSCAR: the batch partition caps this account at MaxTRESPU=cpu=64 total
regardless of node count, so submit_reco_libtest_packed.slurm is now a
fixed 64-way array job (see that script) instead of scaling --nodes with
the row count. No --account/--qos on OSCAR.
"""

import argparse
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path


SPLIT_EVENTS = {
    "train": 2000,
    "val": 400,
    "test": 800,
}
JOB_ID_BASE = {
    "train": 0,
    "val": 100_000,
    "test": 200_000,
}
LIBRARY = {"U": "norm1", "R": "norm42", "null_b": "norm1"}
DIGI_OFFSET = {"U": 0, "R": 0, "null_b": 1_000_000}
N_FILES = 420
EVENTS_PER_JOB = 50


def parse_args():
    scratch = f"/oscar/scratch/{os.environ.get('USER', '')}"
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default="08:00:00",
                        help="SBATCH -t override for the 64-way array job")
    parser.add_argument("--pools", default=scratch + "/mucoll/libtest/bib_pools_simple")
    parser.add_argument("--outdir", default=scratch + "/mucoll/libtest/reco_n420_pfn_simple")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    repo = Path(__file__).resolve().parent
    pools = Path(args.pools).resolve()
    outdir = Path(args.outdir).resolve()
    oscar_scratch = Path("/oscar/scratch") / os.environ.get("USER", "")
    try:
        outdir.relative_to(oscar_scratch)
    except ValueError:
        raise SystemExit(
            "refusing output outside OSCAR user scratch {}: {}"
            .format(oscar_scratch, outdir)
        )
    logs = repo / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = logs / "reco_n{}_{}.tsv".format(N_FILES, stamp)

    rows = []
    skipped = 0
    for split in SPLIT_EVENTS:
        n_jobs = math.ceil(SPLIT_EVENTS[split] / EVENTS_PER_JOB)
        for sample in LIBRARY:
            library = LIBRARY[sample]
            plus = pools / library / split / "MUPLUS"
            minus = pools / library / split / "MUMINUS"
            for directory in (plus, minus):
                if not directory.is_dir() or not any(directory.glob("*.root")):
                    raise SystemExit("empty or missing pool: {}".format(directory))

            bib_number = N_FILES if sample != "R" else N_FILES // 42
            study = "reco_libtest_n{}_{}/{}".format(N_FILES, sample, split)
            for index in range(n_jobs):
                job_id = JOB_ID_BASE[split] + index
                first = index * EVENTS_PER_JOB
                nevents = min(EVENTS_PER_JOB,
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

    slurm = repo / "submit_reco_libtest_packed.slurm"
    command = [
        "sbatch", "--parsable",
        "--time={}".format(args.time),
        str(slurm), str(manifest),
    ]
    print("manifest: {}".format(manifest))
    print("tasks: {} ({} existing outputs skipped)".format(len(rows), skipped))
    print("allocation: fixed 64-way array (OSCAR batch partition MaxTRESPU=cpu=64),"
          " each shard looping over its assigned rows")
    print(" ".join(command))
    if not args.dry_run:
        result = subprocess.run(command, check=True,
                                universal_newlines=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        print("submitted packed job {}".format(
            result.stdout.strip().split(";", 1)[0]))


if __name__ == "__main__":
    main()
