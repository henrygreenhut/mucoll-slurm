#!/usr/bin/env python3

import argparse
import csv
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path


SAMPLES = ("U", "R", "null_b")
SPLITS = ("train", "val", "test")
LIBRARY = {"U": "norm1", "R": "norm42", "null_b": "norm1"}
BIB_FILES = {"U": 420, "R": 10, "null_b": 420}
DIGI_OFFSET = {"U": 0, "R": 0, "null_b": 1_000_000}
JOB_START = {"train": 0, "val": 100_000, "test": 300_000}
EVENTS_PER_JOB = 50
EVENTS_PER_SPLIT = 2000


def arguments():
    user = os.environ.get("USER", "")
    base = Path("/oscar/scratch") / user / "mucoll/libtest"
    parser = argparse.ArgumentParser(
        description="Rerun N=420 DIGI and RECO with calorimeter coning disabled."
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--time", default="08:00:00")
    parser.add_argument("--memory", default="8G")
    parser.add_argument("--pools", type=Path, default=base / "bib_pools_val25")
    parser.add_argument(
        "--train-val-source",
        type=Path,
        default=base / "reco_n420_pfn_trackfix_val25",
    )
    parser.add_argument(
        "--test-source",
        type=Path,
        default=base / "reco_n420_confirmation",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=base / "reco_n420_calo_unconed",
    )
    return parser.parse_args()


def signal_sim(args, split, job_id):
    if split == "test":
        return (
            args.test_source
            / "reco_libtest_n420_U/confirmation"
            / "job_{}".format(job_id)
            / "sim_output_{}.edm4hep.root".format(job_id)
        )
    return (
        args.train_val_source
        / "reco_libtest_n420_U"
        / split
        / "job_{}".format(job_id)
        / "sim_output_{}.edm4hep.root".format(job_id)
    )


def output_directory(outdir, sample, split, job_id):
    return (
        outdir
        / "reco_libtest_n420_{}".format(sample)
        / split
        / "job_{}".format(job_id)
    )


def require_pool(directory, needed):
    available = sum(1 for path in directory.glob("*.root") if path.is_file())
    if available < needed:
        raise SystemExit(
            "{} has {} ROOT files; {} are required".format(
                directory, available, needed
            )
        )


def make_rows(args):
    samples = ("U", "R") if args.benchmark else SAMPLES
    splits = ("train",) if args.benchmark else SPLITS
    jobs_per_split = 1 if args.benchmark else math.ceil(
        EVENTS_PER_SPLIT / EVENTS_PER_JOB
    )
    outdir = (
        args.outdir.parent / (args.outdir.name + "_benchmark")
        if args.benchmark else args.outdir
    )

    rows = []
    skipped = 0
    for split in splits:
        for sample in samples:
            library = LIBRARY[sample]
            plus = args.pools / library / split / "MUPLUS"
            minus = args.pools / library / split / "MUMINUS"
            require_pool(plus, BIB_FILES[sample])
            require_pool(minus, BIB_FILES[sample])
            for index in range(jobs_per_split):
                job_id = JOB_START[split] + index
                source = signal_sim(args, split, job_id)
                if not source.is_file():
                    raise SystemExit("missing signal SIM: {}".format(source))
                output = output_directory(outdir, sample, split, job_id)
                complete = output / "complete"
                reco = output / "reco_output_{}.edm4hep.root".format(job_id)
                if (
                    complete.is_file()
                    and reco.is_file()
                    and reco.stat().st_size > 0
                    and not args.force
                ):
                    skipped += 1
                    continue
                digi_seed = 42 + job_id + DIGI_OFFSET[sample]
                rows.append(
                    {
                        "sample": sample,
                        "split": split,
                        "chunk": index,
                        "job_id": job_id,
                        "events": EVENTS_PER_JOB,
                        "signal_sim": source,
                        "output_dir": output,
                        "bib_muplus": str(plus) + "/",
                        "bib_muminus": str(minus) + "/",
                        "bib_files_per_polarity": BIB_FILES[sample],
                        "digi_seed": digi_seed,
                    }
                )
    return rows, skipped, outdir


def main():
    args = arguments()
    scratch = Path("/oscar/scratch") / os.environ.get("USER", "")
    try:
        args.outdir.resolve().relative_to(scratch)
    except ValueError:
        raise SystemExit(
            "refusing output outside OSCAR scratch {}: {}".format(
                scratch, args.outdir.resolve()
            )
        )

    rows, skipped, outdir = make_rows(args)
    if not rows:
        print("All requested outputs are complete; nothing to submit.")
        return

    repo = Path(__file__).resolve().parent
    logs = repo / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kind = "benchmark" if args.benchmark else "production"
    manifest = logs / "reco_calo_unconed_{}_{}.tsv".format(kind, stamp)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    shards = min(64, len(rows))
    command = [
        "sbatch",
        "--parsable",
        "--array=0-{}".format(shards - 1),
        "--export=ALL,NUM_SHARDS={}".format(shards),
        "--time={}".format(args.time),
        "--mem={}".format(args.memory),
        str(repo / "submit_reco_calo_unconed.slurm"),
        str(manifest),
    ]
    print("mode: {}".format(kind))
    print("manifest: {}".format(manifest))
    print("output: {}".format(outdir))
    print("logical chunks: {} ({} complete chunks skipped)".format(len(rows), skipped))
    print("array shards: {}".format(shards))
    print(" ".join(command))
    if args.dry_run:
        return
    result = subprocess.run(
        command,
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("submitted job {}".format(result.stdout.strip().split(";", 1)[0]))


if __name__ == "__main__":
    main()
