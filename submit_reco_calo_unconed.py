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
DIGI_OFFSET = {"U": 0, "R": 0, "null_b": 1_000_000}
JOB_START = {"train": 0, "val": 100_000, "test": 300_000}
EVENTS_PER_JOB = 50
EVENTS_PER_SPLIT = 2000


def arguments():
    user = os.environ.get("USER", "")
    base = Path("/oscar/scratch") / user / "mucoll/libtest"
    parser = argparse.ArgumentParser(
        description="Run DIGI and RECO with calorimeter coning disabled."
    )
    parser.add_argument("--n-files", type=int, default=420)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-digi", action="store_true")
    parser.add_argument("--time")
    parser.add_argument("--memory")
    parser.add_argument("--dependency")
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
        default=None,
    )
    args = parser.parse_args()
    if args.n_files <= 0 or args.n_files % 42:
        parser.error("--n-files must be a positive multiple of 42")
    if args.n_files <= 420:
        default_time, default_memory = "08:00:00", "8G"
    elif args.n_files <= 840:
        default_time, default_memory = "16:00:00", "24G"
    else:
        default_time, default_memory = "24:00:00", "48G"
    args.time = args.time or default_time
    args.memory = args.memory or default_memory
    args.outdir = args.outdir or base / "reco_n{}_calo_unconed".format(
        args.n_files
    )
    return args


def bib_files(n_files, sample):
    return n_files // 42 if sample == "R" else n_files


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


def output_directory(outdir, n_files, sample, split, job_id):
    return (
        outdir
        / "reco_libtest_n{}_{}".format(n_files, sample)
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
            number = bib_files(args.n_files, sample)
            require_pool(plus, number)
            require_pool(minus, number)
            for index in range(jobs_per_split):
                job_id = JOB_START[split] + index
                source = signal_sim(args, split, job_id)
                if not source.is_file():
                    raise SystemExit("missing signal SIM: {}".format(source))
                output = output_directory(
                    outdir, args.n_files, sample, split, job_id
                )
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
                        "n_files": args.n_files,
                        "sample": sample,
                        "split": split,
                        "chunk": index,
                        "job_id": job_id,
                        "events": EVENTS_PER_JOB,
                        "signal_sim": source,
                        "output_dir": output,
                        "bib_muplus": str(plus) + "/",
                        "bib_muminus": str(minus) + "/",
                        "bib_files_per_polarity": number,
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
    manifest = logs / "reco_n{}_calo_unconed_{}_{}.tsv".format(
        args.n_files, kind, stamp
    )
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    shards = min(64, len(rows))
    command = [
        "sbatch",
        "--parsable",
        "--array=0-{}".format(shards - 1),
        "--export=ALL,NUM_SHARDS={},KEEP_DIGI_OUTPUT={}".format(
            shards, int(args.keep_digi)
        ),
        "--time={}".format(args.time),
        "--mem={}".format(args.memory),
    ]
    if args.dependency:
        command.append("--dependency={}".format(args.dependency))
    command.extend([
        str(repo / "submit_reco_calo_unconed.slurm"),
        str(manifest),
    ])
    print("mode: {}".format(kind))
    print("N: {}".format(args.n_files))
    print("manifest: {}".format(manifest))
    print("output: {}".format(outdir))
    print("logical chunks: {} ({} complete chunks skipped)".format(len(rows), skipped))
    print("array shards: {}".format(shards))
    print(
        "construction: U/null={} norm1 files per polarity; "
        "R={} norm42 files per polarity".format(
            bib_files(args.n_files, "U"), bib_files(args.n_files, "R")
        )
    )
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
