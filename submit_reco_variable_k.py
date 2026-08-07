#!/usr/bin/env python3
"""Submit reconstructed N=420 datasets for synthetic mother-reuse factors."""

import argparse
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path


REUSE_FACTORS = (1, 5, 7, 10, 21)
MOTHER_EQUIVALENTS = 29400
CHUNK_MOTHERS = 140
EVENTS_PER_JOB = 50
DEFAULT_EVENTS = {"train": 2000, "val": 2000, "test": 2000}
JOB_ID_BASE = {"train": 0, "val": 100000, "test": 200000}


def files_per_event(reuse_k):
    if reuse_k not in REUSE_FACTORS:
        raise ValueError("unsupported reuse factor: {}".format(reuse_k))
    return MOTHER_EQUIVALENTS // (CHUNK_MOTHERS * reuse_k)


def parse_args():
    user = os.environ.get("USER", "")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--library",
        default="/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_variable_k",
    )
    parser.add_argument(
        "--outdir",
        default="/oscar/scratch/{}/mucoll/libtest/reco_variable_k_n420".format(
            user
        ),
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=REUSE_FACTORS
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    for split, count in DEFAULT_EVENTS.items():
        parser.add_argument("--{}-events".format(split), type=int, default=count)
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--memory", default="12G")
    parser.add_argument("--dependency")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def root_count(directory):
    return len(list(directory.glob("*.root"))) if directory.is_dir() else 0


def main():
    args = parse_args()
    unsupported = sorted(set(args.k_values) - set(REUSE_FACTORS))
    if unsupported:
        raise SystemExit("unsupported reuse factors: {}".format(unsupported))
    if len(set(args.k_values)) != len(args.k_values):
        raise SystemExit("--k-values contains duplicates")

    repo = Path(__file__).resolve().parent
    library = Path(args.library).resolve()
    outdir = Path(args.outdir).resolve()
    scratch = Path("/oscar/scratch") / os.environ.get("USER", "")
    try:
        outdir.relative_to(scratch)
    except ValueError:
        raise SystemExit("output must be inside {}".format(scratch))

    manifest_path = library / "manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "reco-variable-k-chunks-v1":
        raise SystemExit("unexpected variable-k manifest schema")
    if manifest["mothers_per_chunk"] != CHUNK_MOTHERS:
        raise SystemExit("manifest chunk size does not match this submitter")
    if (
        manifest["mother_equivalents_per_event_per_polarity"]
        != MOTHER_EQUIVALENTS
    ):
        raise SystemExit("manifest event size does not match this submitter")

    event_counts = {
        split: getattr(args, "{}_events".format(split))
        for split in ("train", "val", "test")
    }
    if any(event_counts[split] <= 0 for split in args.splits):
        raise SystemExit("event counts must be positive")

    rows = []
    skipped = 0
    for k in args.k_values:
        overlay_files = files_per_event(k)
        library_sample = "k1_native" if k == 1 else "k{}".format(k)
        if overlay_files != manifest["files_per_event"][str(k)]:
            raise SystemExit("manifest has the wrong overlay count for k={}".format(k))
        for split in args.splits:
            plus = library / "SIM" / library_sample / split / "MUPLUS"
            minus = library / "SIM" / library_sample / split / "MUMINUS"
            for directory in (plus, minus):
                available = root_count(directory)
                expected = manifest["splits"][split]["chunk_count"]
                if available != expected:
                    raise SystemExit(
                        "{} has {} SIM files; expected {} complete chunks"
                        .format(directory, available, expected)
                    )
                if available < overlay_files:
                    raise SystemExit(
                        "{} has {} files; one event needs {}".format(
                            directory, available, overlay_files
                        )
                    )

            jobs = int(math.ceil(event_counts[split] / float(EVENTS_PER_JOB)))
            study = "reco_variable_n420_k{}/{}".format(k, split)
            for index in range(jobs):
                job_id = JOB_ID_BASE[split] + index
                first = index * EVENTS_PER_JOB
                nevents = min(EVENTS_PER_JOB, event_counts[split] - first)
                expected_output = (
                    outdir / study / "job_{}".format(job_id)
                    / "reco_output_{}.edm4hep.root".format(job_id)
                )
                if (
                    expected_output.is_file()
                    and expected_output.stat().st_size > 0
                    and not args.force
                ):
                    skipped += 1
                    continue
                rows.append([
                    "k{}".format(k),
                    split,
                    str(index),
                    str(job_id),
                    str(nevents),
                    study,
                    str(outdir),
                    str(plus) + "/",
                    str(minus) + "/",
                    str(overlay_files),
                    "0",
                ])

    if not rows:
        print("All requested variable-k RECO outputs already exist.")
        return

    logs = repo / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_manifest = logs / "reco_variable_k_{}.tsv".format(stamp)
    with task_manifest.open("w") as handle:
        for row in rows:
            handle.write("\t".join(row) + "\n")

    command = [
        "sbatch",
        "--parsable",
        "--time={}".format(args.time),
        "--mem={}".format(args.memory),
    ]
    if args.dependency:
        if any(character.isspace() for character in args.dependency):
            raise SystemExit("--dependency cannot contain whitespace")
        command.append("--dependency={}".format(args.dependency))
    command.extend([
        str(repo / "submit_reco_libtest_packed.slurm"),
        str(task_manifest),
    ])

    print("manifest: {}".format(task_manifest))
    print("tasks: {} ({} existing outputs skipped)".format(len(rows), skipped))
    print(
        "K values: {}".format(
            ", ".join(
                "k{}={} files/event/polarity".format(
                    k, files_per_event(k)
                )
                for k in args.k_values
            )
        )
    )
    print(" ".join(command))
    if not args.dry_run:
        result = subprocess.run(
            command,
            check=True,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("submitted packed job {}".format(
            result.stdout.strip().split(";", 1)[0]
        ))


if __name__ == "__main__":
    main()
