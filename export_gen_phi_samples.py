#!/usr/bin/env python3

import argparse

import h5py
import numpy as np


def rotated_phi(phi, particle_indices, mother_offsets):
    mother_index = np.searchsorted(mother_offsets, particle_indices, side="right") - 1
    rng = np.random.default_rng(1701)
    angles = rng.uniform(0, 2 * np.pi, len(mother_offsets) - 1)
    return (phi + angles[mother_index] + np.pi) % (2 * np.pi) - np.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bank")
    parser.add_argument("output")
    args = parser.parse_args()

    with h5py.File(args.bank, "r") as source:
        particles = source["particles"]
        px = particles["px"][:]
        py = particles["py"][:]
        energy = particles["E"][:]
        pdg = particles["pdg"][:]
        mother_offsets = source["mother_offsets"][:]

    phi = np.arctan2(py, px)
    above_3 = energy > 3.0
    muon = np.abs(pdg) == 13
    muon_below_1 = muon & (energy < 1.0)

    above_3_indices = np.flatnonzero(above_3)
    muon_indices = np.flatnonzero(muon)
    muon_below_1_indices = np.flatnonzero(muon_below_1)

    np.savez_compressed(
        args.output,
        phi_E3=phi[above_3],
        phi_E3_rotated=rotated_phi(phi[above_3], above_3_indices, mother_offsets),
        is_muon_E3=muon[above_3],
        phi_muons=phi[muon],
        phi_muons_rotated=rotated_phi(phi[muon], muon_indices, mother_offsets),
        phi_muons_below_1=phi[muon_below_1],
        phi_muons_below_1_rotated=rotated_phi(
            phi[muon_below_1], muon_below_1_indices, mother_offsets
        ),
    )


if __name__ == "__main__":
    main()
