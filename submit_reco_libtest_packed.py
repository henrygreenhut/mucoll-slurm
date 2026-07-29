#!/usr/bin/env python3
"""Submit one fixed-size RECO BIB data set as a packed CPU allocation.

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


DEFAULT_SPLIT_EVENTS = {
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
DEFAULT_N_FILES = 420
EVENTS_PER_JOB = 50


def files_per_event(n_files, sample):
    """Return actual library files overlaid per polarity for one class."""
    return n_files if sample != "R" else n_files // 42


def parse_args():
    scratch = f"/oscar/scratch/{os.environ.get('USER', '')}"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-files", type=int, default=DEFAULT_N_FILES,
        help="norm1-file equivalents per reconstructed event",
    )
    parser.add_argument(
        "--time",
        help=(
            "SBATCH time override (default: 8h at N=420, 16h at N=840, "
            "24h above N=840)"
        ),
    )
    parser.add_argument(
        "--memory",
        help=(
            "memory per array task (default: 8G at N=420, 16G at N=840, "
            "40G above N=840)"
        ),
    )
    parser.add_argument(
        "--dependency",
        help="optional Slurm dependency, e.g. afterok:4358400",
    )
    parser.add_argument("--pools", default=scratch + "/mucoll/libtest/bib_pools_simple")
    parser.add_argument(
        "--outdir",
        help="output root (default includes --n-files)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
        help="partitions to produce (default: all)",
    )
    parser.add_argument(
        "--train-events", type=int, default=DEFAULT_SPLIT_EVENTS["train"]
    )
    parser.add_argument(
        "--val-events", type=int, default=DEFAULT_SPLIT_EVENTS["val"]
    )
    parser.add_argument(
        "--test-events", type=int, default=DEFAULT_SPLIT_EVENTS["test"]
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_files <= 0 or args.n_files % 42:
        raise SystemExit("--n-files must be a positive multiple of 42")

    repo = Path(__file__).resolve().parent
    pools = Path(args.pools).resolve()
    outdir = Path(
        args.outdir
        or (
            "/oscar/scratch/{}/mucoll/libtest/reco_n{}_pfn_trackfix"
            .format(os.environ.get("USER", ""), args.n_files)
        )
    ).resolve()
    if args.n_files <= DEFAULT_N_FILES:
        default_walltime, default_memory = "08:00:00", "8G"
    elif args.n_files <= 840:
        default_walltime, default_memory = "16:00:00", "16G"
    else:
        default_walltime, default_memory = "24:00:00", "40G"
    walltime = args.time or default_walltime
    memory = args.memory or default_memory
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
    manifest = logs / "reco_n{}_{}.tsv".format(args.n_files, stamp)

    split_events = {
        "train": args.train_events,
        "val": args.val_events,
        "test": args.test_events,
    }
    if any(split_events[split] <= 0 for split in args.splits):
        raise SystemExit("requested split event counts must be positive")

    rows = []
    skipped = 0
    for split in args.splits:
        n_events = split_events[split]
        n_jobs = math.ceil(n_events / EVENTS_PER_JOB)
        for sample in LIBRARY:
            library = LIBRARY[sample]
            plus = pools / library / split / "MUPLUS"
            minus = pools / library / split / "MUMINUS"
            bib_number = files_per_event(args.n_files, sample)
            for directory in (plus, minus):
                available = (
                    len(list(directory.glob("*.root")))
                    if directory.is_dir() else 0
                )
                if available < bib_number:
                    raise SystemExit(
                        "pool {} has {} files; need {} for sample {}".format(
                            directory, available, bib_number, sample
                        )
                    )

            study = "reco_libtest_n{}_{}/{}".format(
                args.n_files, sample, split
            )
            for index in range(n_jobs):
                job_id = JOB_ID_BASE[split] + index
                first = index * EVENTS_PER_JOB
                nevents = min(EVENTS_PER_JOB, n_events - first)
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
        "--time={}".format(walltime),
        "--mem={}".format(memory),
    ]
    if args.dependency:
        if any(character.isspace() for character in args.dependency):
            raise SystemExit("--dependency cannot contain whitespace")
        command.append("--dependency={}".format(args.dependency))
    command.extend([str(slurm), str(manifest)])
    print("manifest: {}".format(manifest))
    print("tasks: {} ({} existing outputs skipped)".format(len(rows), skipped))
    print("allocation: fixed 64-way array (OSCAR batch partition MaxTRESPU=cpu=64),"
          " each shard looping over its assigned rows")
    print(
        "construction: U/null={} norm1 files; R={} norm42 files".format(
            files_per_event(args.n_files, "U"),
            files_per_event(args.n_files, "R"),
        )
    )
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
