#!/usr/bin/env python3
"""Convert the fixed RECO directory layout into one HDF5 store per sample/split."""

import argparse
import os
from pathlib import Path

from pfn_make_h5 import write_store


SAMPLES = ("U", "R", "null_b")
SPLITS = ("train", "val", "test_a", "test_b")


def parse_args():
    scratch = os.environ.get("PSCRATCH", "")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=420)
    parser.add_argument("--reco-dir", default=(scratch + "/mucoll/libtest/reco_n420_pfn")
                        if scratch else None, required=not bool(scratch))
    parser.add_argument("--outdir", default=(scratch + "/mucoll/libtest/reco_n420_pfn_stores")
                        if scratch else None, required=not bool(scratch))
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES),
                        choices=SAMPLES)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS),
                        choices=SPLITS)
    return parser.parse_args()


def main():
    args = parse_args()
    reco_dir = Path(args.reco_dir).resolve()
    outdir = Path(args.outdir).resolve()
    for sample in args.samples:
        for split in args.splits:
            source = reco_dir / "reco_libtest_n{}_{}".format(args.n_files, sample) / split
            output = outdir / "n{}_{}_{}.h5".format(args.n_files, sample, split)
            print("\n{} / {}".format(sample, split))
            write_store([str(source)], str(output), sample)


if __name__ == "__main__":
    main()
