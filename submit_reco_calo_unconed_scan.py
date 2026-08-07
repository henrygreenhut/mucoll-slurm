#!/usr/bin/env python3

import re
import subprocess
import sys
from pathlib import Path


SIZES = (840, 1260)
STORE_MEMORY = {840: "64G", 1260: "96G"}


def run(command):
    result = subprocess.run(
        command,
        check=False,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode:
        raise RuntimeError(
            "command failed with status {}: {}".format(
                result.returncode, " ".join(command)
            )
        )
    return result.stdout


def submitted_job(output):
    match = re.search(r"submitted job (\d+)", output)
    if not match:
        raise RuntimeError("submission did not report a job ID")
    return match.group(1)


def sbatch(*arguments):
    output = run(["sbatch", "--parsable"] + list(arguments)).strip()
    return output.split(";", 1)[0]


def submit_size(repo, n_files, dependency=None):
    command = [
        sys.executable,
        str(repo / "submit_reco_calo_unconed.py"),
        "--n-files", str(n_files),
    ]
    if dependency:
        command.extend(["--dependency", dependency])
    cpu = submitted_job(run(command))

    store = sbatch(
        "--mem={}".format(STORE_MEMORY[n_files]),
        "--dependency=afterok:{}".format(cpu),
        str(repo / "submit_reco_calo_unconed_stores.slurm"),
        str(n_files),
    )
    train = sbatch(
        "--dependency=afterok:{}".format(store),
        str(repo / "submit_reco_libtest_recipe.slurm"),
        "stabilized_dropout",
        "calo_unconed",
        str(n_files),
    )
    return cpu, store, train


def main():
    repo = Path(__file__).resolve().parent
    previous_train = None
    jobs = {}
    for n_files in SIZES:
        dependency = (
            "afterok:{}".format(previous_train) if previous_train else None
        )
        jobs[n_files] = submit_size(repo, n_files, dependency)
        previous_train = jobs[n_files][2]

    print("\nQueued sequential unconed scan:")
    for n_files in SIZES:
        cpu, store, train = jobs[n_files]
        print(
            "N={}: reconstruction={} stores={} training={}".format(
                n_files, cpu, store, train
            )
        )


if __name__ == "__main__":
    main()
