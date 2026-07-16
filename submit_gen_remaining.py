#!/usr/bin/env python3
"""Submit one packed debug window for the unfinished GEN-level work."""

import argparse
import os
import subprocess
from pathlib import Path


N42_MAIN = "A0_n42_scaled_clean"
N210_EVAL = "EVAL_n210_paired_overlap"
N42_EVAL = "EVAL_n42_scaled_clean_paired_overlap"
N420_MAIN = "A0_n420_scaled_disjoint"
N420_NULL = "A0_n420_null_shared"
N420_EVAL = "EVAL_n420_paired_overlap"
N420_RAW_MAIN = "A0_n420_rawsum_disjoint"
N420_RAW_NULL = "A0_n420_rawsum_null_shared"
N420_RAW_EVAL = "EVAL_n420_rawsum_paired_overlap"


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
    n420_done = finished(results, N420_MAIN)
    n420_raw_done = finished(results, N420_RAW_MAIN)
    if not n42_done:
        labels.append(N42_MAIN)
    if not finished(results, N210_EVAL):
        labels.append(N210_EVAL)
    if not n420_done:
        labels.append(N420_MAIN)
    if not n420_raw_done:
        labels.append(N420_RAW_MAIN)
    if n42_done and not finished(results, N42_EVAL):
        labels.append(N42_EVAL)
    if not finished(results, N420_NULL):
        labels.append(N420_NULL)
    if not finished(results, N420_RAW_NULL):
        labels.append(N420_RAW_NULL)
    if n420_done and not finished(results, N420_EVAL):
        labels.append(N420_EVAL)
    if n420_raw_done and not finished(results, N420_RAW_EVAL):
        labels.append(N420_RAW_EVAL)

    if not labels:
        print("All remaining N=42, N=210, and N=420 GEN work is complete.")
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
    if not n420_done:
        print("N=420 scaled-sum evaluation is gated until training completes.")
    if not n420_raw_done:
        print("N=420 raw-sum evaluation is gated until training completes.")
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
