#!/usr/bin/env python3
"""Launch one configured libtest run on this Slurm task's assigned GPU."""

import json
import os
import shlex
import subprocess
import sys


def main():
    config_path = os.environ["BUNDLE_CONFIG"]
    labels = [value for value in os.environ["BUNDLE_LABELS"].split(",")
              if value]
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    if rank >= len(labels):
        print("bundle rank {} idle ({} configured labels)".format(
            rank, len(labels)), flush=True)
        return 0

    with open(config_path) as handle:
        plan = json.load(handle)
    runs = {run["label"]: run for run in plan["runs"]}
    label = labels[rank]
    run = runs[label]
    extra = shlex.split(run.get("train_args", ""))
    script = run.get("script", "submit_libtest_train.slurm")

    if script == "submit_libtest_evaluate.slurm":
        command = [sys.executable, "pfn_libtest_evaluate.py",
                   "--label", label, "--max-minutes", "25"] + extra
    elif script == "submit_libtest_train.slurm":
        scratch = os.environ["PSCRATCH"]
        polarity = os.environ.get("POLARITY", "MUPLUS")
        store_dir = os.environ.get(
            "STORE_DIR", os.path.join(scratch, "mucoll/libtest/stores"))
        command = [
            sys.executable, "pfn_libtest_train.py",
            "--norm1-store", os.path.join(
                store_dir, "gen_norm1_{}.h5".format(polarity)),
            "--norm42-store", os.path.join(
                store_dir, "gen_norm42_{}.h5".format(polarity)),
            "--label", label, "--max-minutes", "25",
        ] + extra
    else:
        raise SystemExit("unsupported bundled script: {}".format(script))

    print("bundle rank {} label={} gpu={} command={}".format(
        rank, label, os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        shlex.join(command)), flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
