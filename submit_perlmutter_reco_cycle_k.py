#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def submit(script, dependency, dry_run):
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append("--dependency=afterok:{}".format(dependency))
    command.append(str(script))
    print(" ".join(command))
    if dry_run:
        return "DRY_RUN"
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return result.stdout.strip().split(";", 1)[0]


def main():
    args = arguments()
    if "PSCRATCH" not in os.environ:
        raise SystemExit("PSCRATCH is not set; run this on Perlmutter")

    repo = Path(__file__).resolve().parent
    (repo / "logs").mkdir(exist_ok=True)

    prepare = submit(
        repo / "submit_perlmutter_reco_cycle_k_prepare.slurm", "", args.dry_run
    )
    smoke = submit(
        repo / "submit_perlmutter_reco_cycle_k_smoke.slurm", prepare, args.dry_run
    )
    gen = submit(
        repo / "submit_perlmutter_reco_cycle_k_gen.slurm", smoke, args.dry_run
    )
    sim = submit(
        repo / "submit_perlmutter_reco_cycle_k_sim.slurm", gen, args.dry_run
    )

    print("prepare={} smoke={} gen={} sim={}".format(prepare, smoke, gen, sim))
    print("status: python3 reco_cycle_k_perlmutter.py status")


if __name__ == "__main__":
    main()
