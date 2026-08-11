#!/usr/bin/env python3

import argparse

import h5py
import numpy as np


CATEGORIES = {
    "photons": lambda pdg: pdg == 22,
    "neutrons": lambda pdg: np.abs(pdg) == 2112,
    "electrons_positrons": lambda pdg: np.abs(pdg) == 11,
    "charged_pions": lambda pdg: np.abs(pdg) == 211,
    "protons": lambda pdg: np.abs(pdg) == 2212,
    "muons": lambda pdg: np.abs(pdg) == 13,
    "kaons": lambda pdg: np.isin(np.abs(pdg), [130, 311, 321]),
    "lambda_baryons": lambda pdg: np.abs(pdg) == 3122,
    "light_nuclei": lambda pdg: np.abs(pdg) > 1_000_000_000,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bank")
    parser.add_argument("output")
    args = parser.parse_args()

    bins = np.linspace(-np.pi, np.pi, 33)
    native = np.zeros((len(CATEGORIES), len(bins) - 1), dtype=np.int64)
    rotated = np.zeros_like(native)

    with h5py.File(args.bank, "r") as source:
        particles = source["particles"]
        mother_offsets = source["mother_offsets"][:]
        rng = np.random.default_rng(1701)
        angles = rng.uniform(0, 2 * np.pi, len(mother_offsets) - 1)

        chunk_size = 1_000_000
        for start in range(0, len(particles["pdg"]), chunk_size):
            stop = min(start + chunk_size, len(particles["pdg"]))
            pdg = particles["pdg"][start:stop]
            phi = np.arctan2(
                particles["py"][start:stop],
                particles["px"][start:stop],
            )
            indices = np.arange(start, stop)
            mothers = np.searchsorted(mother_offsets, indices, side="right") - 1
            rotated_phi = (phi + angles[mothers] + np.pi) % (2 * np.pi) - np.pi

            for row, select in enumerate(CATEGORIES.values()):
                mask = select(pdg)
                native[row] += np.histogram(phi[mask], bins=bins)[0]
                rotated[row] += np.histogram(rotated_phi[mask], bins=bins)[0]

    np.savez_compressed(
        args.output,
        bins=bins,
        categories=np.array(list(CATEGORIES)),
        native=native,
        rotated=rotated,
    )


if __name__ == "__main__":
    main()
