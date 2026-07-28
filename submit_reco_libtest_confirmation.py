#!/usr/bin/env python3
"""Submit a fresh, test-only RECO confirmation cohort on OSCAR.

The original 800-event/class test outputs are deliberately left untouched.
This script uses new job IDs (and therefore new particle-gun and digitization
seeds) while drawing BIB only from the already frozen held-out test pools.
"""

import argparse
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

from submit_reco_libtest_packed import files_per_event


SAMPLES = ("U", "R", "null_b")
LIBRARY = {"U": "norm1", "R": "norm42", "null_b": "norm1"}
DIGI_OFFSET = {"U": 0, "R": 0, "null_b": 1_000_000}
DEFAULT_N_FILES = 420
EVENTS_PER_JOB = 50
DEFAULT_EVENTS_PER_CLASS = 5000
JOB_ID_START = 300_000


def parse_args():
    scratch = "/oscar/scratch/{}".format(os.environ.get("USER", ""))
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=DEFAULT_N_FILES)
    parser.add_argument(
        "--study", choices=("charged7", "val2000", "val25"),
        default="charged7",
    )
    parser.add_argument(
        "--events-per-class", type=int, default=DEFAULT_EVENTS_PER_CLASS
    )
    parser.add_argument(
        "--time",
        help="SBATCH time override (default: 8 hours at N=420, 16 otherwise)",
    )
    parser.add_argument(
        "--memory",
        help="memory per array task (default: 8G at N=420, 16G otherwise)",
    )
    parser.add_argument(
        "--pools", default=scratch + "/mucoll/libtest/bib_pools_simple"
    )
    parser.add_argument(
        "--outdir",
        help="output root; defaults from OSCAR scratch and --n-files",
    )
    parser.add_argument(
        "--with-followups",
        action="store_true",
        help=(
            "also submit the store builder after successful production and "
            "the frozen-model evaluator after successful store building"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def manifest_rows(pools, outdir, n_files, events_per_class):
    if events_per_class <= 0:
        raise ValueError("events_per_class must be positive")

    rows = []
    skipped = 0
    n_jobs = int(math.ceil(events_per_class / EVENTS_PER_JOB))
    for sample in SAMPLES:
        library = LIBRARY[sample]
        plus = pools / library / "test" / "MUPLUS"
        minus = pools / library / "test" / "MUMINUS"
        bib_number = files_per_event(n_files, sample)
        for directory in (plus, minus):
            available = (
                len(list(directory.glob("*.root")))
                if directory.is_dir() else 0
            )
            if available < bib_number:
                raise SystemExit(
                    "held-out pool {} has {} files; need {}".format(
                        directory, available, bib_number
                    )
                )

        study = "reco_libtest_n{}_{}/confirmation".format(n_files, sample)
        for index in range(n_jobs):
            job_id = JOB_ID_START + index
            first = index * EVENTS_PER_JOB
            nevents = min(EVENTS_PER_JOB, events_per_class - first)
            expected = (
                outdir
                / study
                / "job_{}".format(job_id)
                / "reco_output_{}.edm4hep.root".format(job_id)
            )
            if expected.is_file() and expected.stat().st_size > 0:
                skipped += 1
                continue
            rows.append(
                [
                    sample,
                    "confirmation",
                    str(index),
                    str(job_id),
                    str(nevents),
                    study,
                    str(outdir),
                    str(plus) + "/",
                    str(minus) + "/",
                    str(bib_number),
                    str(DIGI_OFFSET[sample]),
                ]
            )
    return rows, skipped


def submit_followups(repo, n_files, study, production_job=None):
    store_command = ["sbatch", "--parsable"]
    if production_job is not None:
        store_command.append(
            "--dependency=afterok:{}".format(production_job)
        )
    store_command.append(
        str(repo / "submit_reco_libtest_confirmation_stores.slurm")
    )
    store_command.append(str(n_files))
    store_result = subprocess.run(
        store_command,
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    store_job = store_result.stdout.strip().split(";", 1)[0]
    print("submitted confirmation-store job {}".format(store_job))

    evaluate_result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            "--dependency=afterok:{}".format(store_job),
            str(repo / "submit_reco_libtest_confirmation_evaluate.slurm"),
            study,
            str(n_files),
        ],
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    evaluate_job = evaluate_result.stdout.strip().split(";", 1)[0]
    print("submitted frozen-evaluation job {}".format(evaluate_job))


def checkpoint_prefix(n_files, study):
    if study == "charged7" and n_files == 420:
        return "reco_n420_trackfix_directlog_charged7_stabilized_dropout"
    if study == "val2000" and n_files == 420:
        return (
            "reco_n420_trackfix_val2000_directlog_charged7_"
            "stabilized_dropout"
        )
    if study == "val25":
        return (
            "reco_n{}_trackfix_val25_directlog_charged7_"
            "stabilized_dropout".format(n_files)
        )
    raise ValueError("unsupported study/N combination: {}/{}".format(
        study, n_files
    ))


def require_checkpoints(repo, n_files, study):
    result_dir = repo / "reco_pfn_results"
    missing = []
    prefix = checkpoint_prefix(n_files, study)
    for label in (prefix + "_U_vs_R", prefix + "_null"):
        directory = result_dir / label
        for name in ("summary.json", "best.weights.h5"):
            path = directory / name
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(path)
    if missing:
        raise SystemExit(
            "cannot schedule frozen evaluation; missing checkpoint files:\n{}".format(
                "\n".join(str(path) for path in missing)
            )
        )


def main():
    args = parse_args()
    if args.n_files <= 0 or args.n_files % 42:
        raise SystemExit("--n-files must be a positive multiple of 42")
    if args.events_per_class <= 0:
        raise SystemExit("--events-per-class must be positive")
    if args.study in ("charged7", "val2000") and args.n_files != 420:
        raise SystemExit("use --study val25 for N={}".format(args.n_files))
    repo = Path(__file__).resolve().parent
    if args.with_followups and not args.dry_run:
        require_checkpoints(repo, args.n_files, args.study)
    pools = Path(args.pools).resolve()
    outdir = Path(
        args.outdir
        or (
            "/oscar/scratch/{}/mucoll/libtest/reco_n{}_confirmation"
            .format(os.environ.get("USER", ""), args.n_files)
        )
    ).resolve()
    walltime = args.time or (
        "08:00:00" if args.n_files == DEFAULT_N_FILES else "16:00:00"
    )
    memory = args.memory or (
        "8G" if args.n_files == DEFAULT_N_FILES else "16G"
    )
    scratch = Path("/oscar/scratch") / os.environ.get("USER", "")
    try:
        outdir.relative_to(scratch)
    except ValueError:
        raise SystemExit(
            "refusing output outside OSCAR user scratch {}: {}".format(
                scratch, outdir
            )
        )

    rows, skipped = manifest_rows(
        pools, outdir, args.n_files, args.events_per_class
    )
    if not rows:
        print("All confirmation RECO outputs already exist; nothing to submit.")
        if args.with_followups:
            if args.dry_run:
                print(
                    "sbatch {} {}".format(
                        repo / "submit_reco_libtest_confirmation_stores.slurm",
                        args.n_files,
                    )
                )
                print(
                    "then submit {} {} {} after the store job succeeds".format(
                        repo / "submit_reco_libtest_confirmation_evaluate.slurm",
                        args.study,
                        args.n_files,
                    )
                )
            else:
                submit_followups(repo, args.n_files, args.study)
        return

    logs = repo / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = logs / "reco_n{}_confirmation_{}.tsv".format(
        args.n_files, stamp
    )
    with manifest.open("w") as handle:
        for row in rows:
            handle.write("\t".join(row) + "\n")

    command = [
        "sbatch",
        "--parsable",
        "--time={}".format(walltime),
        "--mem={}".format(memory),
        str(repo / "submit_reco_libtest_packed.slurm"),
        str(manifest),
    ]
    print("manifest: {}".format(manifest))
    print(
        "tasks: {} ({} existing outputs skipped)".format(len(rows), skipped)
    )
    print(
        "confirmation target: {} fresh events/class from held-out test pools"
        .format(args.events_per_class)
    )
    print(" ".join(command))
    if args.dry_run:
        if args.with_followups:
            print(
                "sbatch --dependency=afterok:<production_job_id> {}".format(
                    repo / "submit_reco_libtest_confirmation_stores.slurm"
                ) + " {}".format(args.n_files)
            )
            print(
                "sbatch --dependency=afterok:<store_job_id> {}".format(
                    repo / "submit_reco_libtest_confirmation_evaluate.slurm"
                ) + " {} {}".format(args.study, args.n_files)
            )
        return

    result = subprocess.run(
        command,
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    production_job = result.stdout.strip().split(";", 1)[0]
    print("submitted confirmation job {}".format(production_job))

    if args.with_followups:
        submit_followups(
            repo, args.n_files, args.study, production_job
        )


if __name__ == "__main__":
    main()
