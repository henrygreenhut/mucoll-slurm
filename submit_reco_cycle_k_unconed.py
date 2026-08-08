#!/usr/bin/env python3

import csv
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path


SAMPLES = {"k7": 7, "k21": 21}
SPLITS = ("train", "val", "test")
RECORDS_PER_SPLIT = 2000
RECORDS_PER_JOB = 50
JOB_START = {"train": 0, "val": 100000, "test": 300000}


def signal_sim(base, split, job_id):
    if split == "test":
        return (
            base / "reco_n420_confirmation" / "reco_libtest_n420_U"
            / "confirmation" / "job_{}".format(job_id)
            / "sim_output_{}.edm4hep.root".format(job_id)
        )
    return (
        base / "reco_n420_pfn_trackfix_val25" / "reco_libtest_n420_U"
        / split / "job_{}".format(job_id)
        / "sim_output_{}.edm4hep.root".format(job_id)
    )


def cycle_ids(directory):
    expression = re.compile(r"bib_sim_cycle_(\d+)\.edm4hep\.root$")
    output = set()
    for path in directory.glob("*.root"):
        match = expression.match(path.name)
        if not match:
            raise SystemExit("unexpected SIM filename: {}".format(path))
        output.add(int(match.group(1)))
    return output


def validate_sim(library, manifest):
    for sample in SAMPLES:
        for split in SPLITS:
            expected = set(manifest["splits"][split]["cycles"])
            for polarity in ("MUPLUS", "MUMINUS"):
                directory = library / "pools" / sample / split / polarity
                observed = cycle_ids(directory) if directory.is_dir() else set()
                if observed != expected:
                    raise SystemExit(
                        "{} has {} cycles; expected {}; missing {} extra {}".format(
                            directory, len(observed), len(expected),
                            len(expected - observed), len(observed - expected)
                        )
                    )


def make_rows(library, output, signal_base):
    rows = []
    skipped = 0
    jobs_per_split = RECORDS_PER_SPLIT // RECORDS_PER_JOB
    for sample, reuse_k in SAMPLES.items():
        number = 420 // reuse_k
        for split in SPLITS:
            plus = library / "pools" / sample / split / "MUPLUS"
            minus = library / "pools" / sample / split / "MUMINUS"
            for index in range(jobs_per_split):
                job_id = JOB_START[split] + index
                destination = (
                    output / "reco_cycle_n420_{}".format(sample)
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
                    "events": RECORDS_PER_JOB,
                    "signal_sim": source,
                    "output_dir": destination,
                    "bib_muplus": str(plus) + "/",
                    "bib_muminus": str(minus) + "/",
                    "bib_files_per_polarity": number,
                    "digi_seed": 42 + job_id,
                })
    return rows, skipped


def main():
    user = os.environ["USER"]
    repo = Path(__file__).resolve().parent
    library = Path("/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_cycle_k")
    manifest = json.loads((library / "pools" / "manifest.json").read_text())
    if manifest.get("schema") != "reco-cycle-k-v1":
        raise SystemExit("unexpected cycle-K manifest")
    validate_sim(library, manifest)

    scratch = Path("/oscar/scratch") / user / "mucoll" / "libtest"
    output = scratch / "reco_cycle_k_n420_unconed"
    rows, skipped = make_rows(library, output, scratch)
    if not rows:
        print("All K=7 and K=21 cycle-preserving unconed RECO outputs are complete.")
        return

    logs = repo / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_manifest = logs / "reco_cycle_k_unconed_{}.tsv".format(stamp)
    with task_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    shards = min(64, len(rows))
    command = [
        "sbatch", "--parsable", "--array=0-{}".format(shards - 1),
        "--export=ALL,NUM_SHARDS={},KEEP_DIGI_OUTPUT=0".format(shards),
        "--time=08:00:00", "--mem=8G",
        str(repo / "submit_reco_calo_unconed.slurm"), str(task_manifest),
    ]
    print("manifest: {}".format(task_manifest))
    print("logical chunks: {} ({} complete chunks skipped)".format(len(rows), skipped))
    print("construction: k7=60 and k21=20 source-cycle SIM files per polarity")
    result = subprocess.run(
        command, check=True, universal_newlines=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    print("submitted unconed job {}".format(result.stdout.strip().split(";", 1)[0]))


if __name__ == "__main__":
    main()
