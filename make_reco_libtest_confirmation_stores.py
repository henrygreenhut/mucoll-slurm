#!/usr/bin/env python3
"""Build separate HDF5 stores for the frozen-model confirmation cohort."""

import argparse
import os
from pathlib import Path

import h5py

from make_reco_libtest_stores import pool_provenance, write_store


SAMPLES = ("U", "R", "null_b")
DEFAULT_N_FILES = 420
EXPECTED_EVENTS = 5000


def parse_args():
    scratch = "/oscar/scratch/{}".format(os.environ.get("USER", ""))
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=DEFAULT_N_FILES)
    parser.add_argument(
        "--events", type=int, default=EXPECTED_EVENTS,
        help="required confirmation events per class",
    )
    parser.add_argument(
        "--reco-dir",
        help="input root; defaults from OSCAR scratch and --n-files",
    )
    parser.add_argument(
        "--outdir",
        help="store root; defaults from OSCAR scratch and --n-files",
    )
    parser.add_argument(
        "--pool-manifest",
        help="held-out source-pool manifest to fingerprint in each store",
    )
    args = parser.parse_args()
    if args.n_files <= 0 or args.n_files % 42:
        parser.error("--n-files must be a positive multiple of 42")
    if args.events <= 0:
        parser.error("--events must be positive")
    args.reco_dir = (
        args.reco_dir
        or scratch + "/mucoll/libtest/reco_n{}_confirmation".format(
            args.n_files
        )
    )
    args.outdir = (
        args.outdir
        or scratch + "/mucoll/libtest/reco_n{}_confirmation_stores".format(
            args.n_files
        )
    )
    return args


def event_count(path):
    with h5py.File(path, "r") as h5:
        return int(len(h5["particles"]))


def main():
    args = parse_args()
    reco_dir = Path(args.reco_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    provenance, test_cycles = pool_provenance(args.pool_manifest, "test")
    if test_cycles and len(test_cycles) < args.n_files:
        raise SystemExit(
            "held-out source pool has {} cycles; N={} requires at least {}"
            .format(len(test_cycles), args.n_files, args.n_files)
        )
    provenance.update({
        "n_files": args.n_files,
        "cohort": "confirmation",
    })

    for sample in SAMPLES:
        source = (
            reco_dir
            / "reco_libtest_n{}_{}".format(args.n_files, sample)
            / "confirmation"
        )
        output = outdir / "n{}_{}_confirmation.h5".format(
            args.n_files, sample
        )
        temporary = output.with_suffix(".h5.tmp")
        if temporary.exists():
            temporary.unlink()

        print("\nconfirmation / {}".format(sample))
        write_store(
            source, temporary, sample, args.events, provenance,
        )
        found = event_count(temporary)
        if found != args.events:
            temporary.unlink()
            raise SystemExit(
                "{} has {} events; expected {}. Finish/retry RECO production "
                "before building stores.".format(
                    sample, found, args.events
                )
            )
        temporary.replace(output)
        print("validated {} events -> {}".format(found, output))


if __name__ == "__main__":
    main()
