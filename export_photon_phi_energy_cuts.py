#!/usr/bin/env python3

import argparse
import json

import h5py
import numpy as np


def histograms(path, thresholds, bins):
    counts = np.zeros((len(thresholds), len(bins) - 1), dtype=np.int64)

    with h5py.File(path, "r") as source:
        particles = source["particles"]
        for start in range(0, len(particles["pdg"]), 1_000_000):
            stop = min(start + 1_000_000, len(particles["pdg"]))
            photon = particles["pdg"][start:stop] == 22
            energy = particles["E"][start:stop][photon]
            phi = np.arctan2(
                particles["py"][start:stop][photon],
                particles["px"][start:stop][photon],
            )

            for row, threshold in enumerate(thresholds):
                counts[row] += np.histogram(phi[energy > threshold], bins=bins)[0]

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("unrotated")
    parser.add_argument("rotated")
    parser.add_argument("thresholds", nargs="+", type=float)
    parser.add_argument("--bins", type=int, default=128)
    args = parser.parse_args()

    bins = np.linspace(-np.pi, np.pi, args.bins + 1)
    unrotated = histograms(args.unrotated, args.thresholds, bins)
    rotated = histograms(args.rotated, args.thresholds, bins)

    print(json.dumps({
        "unrotated_store": args.unrotated,
        "rotated_store": args.rotated,
        "thresholds": args.thresholds,
        "bins": bins.tolist(),
        "unrotated": unrotated.tolist(),
        "rotated": rotated.tolist(),
    }))


if __name__ == "__main__":
    main()
