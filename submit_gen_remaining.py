#!/usr/bin/env python3
"""Submit one packed debug window for the unfinished GEN-level work."""

import argparse
import os
import subprocess
from pathlib import Path


N42_MAIN = "A0_n42_scaled_clean"
N210_EVAL = "EVAL_n210_paired_overlap"
N42_EVAL = "EVAL_n42_scaled_clean_paired_overlap"


def finished(results, label):
    return (results / label / "auc_summary.json").is_file()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="pfn_results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    results = (repo / args.results).resolve()
    labels = []

    n42_done = finished(results, N42_MAIN)
    if not n42_done:
        labels.append(N42_MAIN)
    if not finished(results, N210_EVAL):
        labels.append(N210_EVAL)
    if n42_done and not finished(results, N42_EVAL):
        labels.append(N42_EVAL)

    if not labels:
        print("N=42 continuation and both overlapping evaluations are complete.")
        return

    # The bundle has four ranks/GPUs. Only scientifically unblocked work is
    # included; in particular N=42 evaluation waits for final N=42 weights.
    labels = labels[:4]
    exports = "ALL,BUNDLE_CONFIG={},BUNDLE_LABELS={}".format(
        repo / "libtest_gen_remaining.json", ",".join(labels))
    command = [
        "sbatch", "--parsable", "--job-name=lt_gen_remaining",
        "--export={}".format(exports),
        str(repo / "submit_libtest_bundle.slurm"),
    ]

    print("packed labels: {}".format(", ".join(labels)))
    if not n42_done:
        print("N=42 overlapping evaluation is gated until training completes.")
    print(" ".join(command))
    if not args.dry_run:
        active = subprocess.run(
            ["squeue", "--noheader", "--user", os.environ["USER"],
             "--name", "lt_gen_remaining", "--states", "PENDING,RUNNING",
             "--format", "%A"],
            check=True, universal_newlines=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if active.stdout.strip():
            raise SystemExit("lt_gen_remaining is already active as job(s) {}"
                             .format(active.stdout.strip().replace("\n", ", ")))
        result = subprocess.run(
            command, check=True, universal_newlines=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("submitted job {}".format(
            result.stdout.strip().split(";", 1)[0]))


if __name__ == "__main__":
    main()
