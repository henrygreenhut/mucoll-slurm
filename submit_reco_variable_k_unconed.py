#!/usr/bin/env python3

import csv
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path


SAMPLES = {"k1_native": 1, "k7": 7, "k21": 21}
SPLITS = ("train", "val", "test")
EVENTS_PER_SPLIT = 2000
EVENTS_PER_JOB = 50
JOB_START = {"train": 0, "val": 100000, "test": 300000}
MOTHER_EQUIVALENTS = 29400
MOTHERS_PER_FILE = 140


def signal_sim(base, split, job_id):
    if split == "test":
        return (
            base / "reco_n420_confirmation"
            / "reco_libtest_n420_U" / "confirmation"
            / "job_{}".format(job_id)
            / "sim_output_{}.edm4hep.root".format(job_id)
        )
    return (
        base / "reco_n420_pfn_trackfix_val25"
        / "reco_libtest_n420_U" / split
        / "job_{}".format(job_id)
        / "sim_output_{}.edm4hep.root".format(job_id)
    )


def load_manifest(path):
    with path.open() as handle:
        return json.load(handle)


def validate_manifests(library):
    native = load_manifest(library / "manifest_native1" / "manifest.json")
    previous = load_manifest(library / "manifest" / "manifest.json")
    for key in ("source_identity_sha256", "chunk_arrays_sha256"):
        if native[key] != previous[key]:
            raise SystemExit("native and existing manifests disagree on {}".format(key))
    for split in SPLITS:
        if native["splits"][split]["cycles"] != previous["splits"][split]["cycles"]:
            raise SystemExit("native and existing {} cycle splits differ".format(split))
    return native


def validate_sim(library, manifest):
    for sample in SAMPLES:
        for split in SPLITS:
            expected = manifest["splits"][split]["chunk_count"]
            for polarity in ("MUPLUS", "MUMINUS"):
                directory = library / "SIM" / sample / split / polarity
                observed = len(list(directory.glob("*.root"))) if directory.is_dir() else 0
                if observed != expected:
                    raise SystemExit(
                        "{} has {} SIM files; expected {}".format(
                            directory, observed, expected
                        )
                    )


def make_rows(library, output, signal_base):
    rows = []
    skipped = 0
    jobs_per_split = EVENTS_PER_SPLIT // EVENTS_PER_JOB
    for sample, reuse_k in SAMPLES.items():
        files_per_event = MOTHER_EQUIVALENTS // (MOTHERS_PER_FILE * reuse_k)
        for split in SPLITS:
            plus = library / "SIM" / sample / split / "MUPLUS"
            minus = library / "SIM" / sample / split / "MUMINUS"
            for index in range(jobs_per_split):
                job_id = JOB_START[split] + index
                destination = (
                    output / "reco_variable_n420_{}".format(sample)
                    / split / "job_{}".format(job_id)
                )
                reco = destination / "reco_output_{}.edm4hep.root".format(job_id)
                if reco.is_file() and reco.stat().st_size > 0 and (destination / "complete").is_file():
                    skipped += 1
                    continue
                source = signal_sim(signal_base, split, job_id)
                if not source.is_file():
                    raise SystemExit("missing neutrino SIM input: {}".format(source))
                rows.append({
                    "n_files": 420,
                    "sample": sample,
                    "split": split,
                    "chunk": index,
                    "job_id": job_id,
                    "events": EVENTS_PER_JOB,
                    "signal_sim": source,
                    "output_dir": destination,
                    "bib_muplus": str(plus) + "/",
                    "bib_muminus": str(minus) + "/",
                    "bib_files_per_polarity": files_per_event,
                    "digi_seed": 42 + job_id,
                })
    return rows, skipped


def main():
    user = os.environ["USER"]
    repo = Path(__file__).resolve().parent
    library = Path(
        "/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_variable_k"
    )
    scratch = Path("/oscar/scratch") / user / "mucoll" / "libtest"
    output = scratch / "reco_variable_k_n420_unconed"

    manifest = validate_manifests(library)
    validate_sim(library, manifest)
    rows, skipped = make_rows(library, output, scratch)
    if not rows:
        print("All native-k1, k7, and k21 unconed RECO outputs are complete.")
        return

    logs = repo / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_manifest = logs / "reco_variable_k_unconed_{}.tsv".format(stamp)
    with task_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    shards = min(64, len(rows))
    command = [
        "sbatch",
        "--parsable",
        "--array=0-{}".format(shards - 1),
        "--export=ALL,NUM_SHARDS={},KEEP_DIGI_OUTPUT=0".format(shards),
        "--time=08:00:00",
        "--mem=8G",
        str(repo / "submit_reco_calo_unconed.slurm"),
        str(task_manifest),
    ]
    print("manifest: {}".format(task_manifest))
    print("logical chunks: {} ({} complete chunks skipped)".format(len(rows), skipped))
    print("samples: k1_native=210, k7=30, k21=10 SIM files/event/polarity")
    result = subprocess.run(
        command,
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("submitted unconed job {}".format(result.stdout.strip().split(";", 1)[0]))


if __name__ == "__main__":
    main()
