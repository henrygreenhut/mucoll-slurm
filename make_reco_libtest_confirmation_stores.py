#!/usr/bin/env python3
"""Build separate HDF5 stores for the frozen-model confirmation cohort."""

import argparse
import os
from pathlib import Path

import h5py

from make_reco_libtest_stores import write_store


SAMPLES = ("U", "R", "null_b")
N_FILES = 420
EXPECTED_EVENTS = 5000


def parse_args():
    scratch = "/oscar/scratch/{}".format(os.environ.get("USER", ""))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reco-dir",
        default=scratch + "/mucoll/libtest/reco_n420_confirmation",
    )
    parser.add_argument(
        "--outdir",
        default=scratch + "/mucoll/libtest/reco_n420_confirmation_stores",
    )
    return parser.parse_args()


def event_count(path):
    with h5py.File(path, "r") as h5:
        return int(len(h5["particles"]))


def main():
    args = parse_args()
    reco_dir = Path(args.reco_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for sample in SAMPLES:
        source = (
            reco_dir
            / "reco_libtest_n{}_{}".format(N_FILES, sample)
            / "confirmation"
        )
        output = outdir / "n{}_{}_confirmation.h5".format(N_FILES, sample)
        temporary = output.with_suffix(".h5.tmp")
        if temporary.exists():
            temporary.unlink()

        print("\nconfirmation / {}".format(sample))
        write_store(source, temporary, sample)
        found = event_count(temporary)
        if found != EXPECTED_EVENTS:
            temporary.unlink()
            raise SystemExit(
                "{} has {} events; expected {}. Finish/retry RECO production "
                "before building stores.".format(
                    sample, found, EXPECTED_EVENTS
                )
            )
        temporary.replace(output)
        print("validated {} events -> {}".format(found, output))


if __name__ == "__main__":
    main()
