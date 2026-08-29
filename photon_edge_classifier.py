#!/usr/bin/env python3

import h5py
import numpy as np


UNROTATED = "/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores/gen_split_mothers_MUPLUS.h5"
ROTATED = "/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores/gen_fixed_k1_MUPLUS.h5"

N_CYCLES = 420
N_TEST = 300
SEED = 1701
EDGE_HALF_WIDTH = np.pi / 4


def load_edge_counts(path):
    with h5py.File(path, "r") as source:
        cycle_ids = source["cycle_ids"][:]
        if "offsets" in source:
            offsets = source["offsets"][:]
        else:
            offsets = source["mother_offsets"][:][source["cycle_offsets"][:]]

        particles = source["particles"]
        photon = particles["pdg"][:] == 22
        phi = np.arctan2(particles["py"][:], particles["px"][:])
        near_edge = photon & (np.abs(phi) > np.pi - EDGE_HALF_WIDTH)

    cumulative = np.pad(np.cumsum(near_edge, dtype=np.int64), (1, 0))
    counts = cumulative[offsets[1:]] - cumulative[offsets[:-1]]
    return cycle_ids, counts


def test_pools(unrotated_ids, rotated_ids):
    common = np.intersect1d(unrotated_ids, rotated_ids)
    shuffled = np.random.default_rng(SEED).permutation(common)
    first_test = round(0.50 * len(common)) + round(0.25 * len(common))
    test_ids = shuffled[first_test:]
    return (
        np.searchsorted(unrotated_ids, test_ids),
        np.searchsorted(rotated_ids, test_ids),
    )


def make_scores(counts, pool, rng):
    return np.array([
        counts[rng.choice(pool, N_CYCLES, replace=False)].sum()
        for _ in range(N_TEST)
    ])


def auc(native_scores, rotated_scores):
    native = np.sort(native_scores)
    lower = np.searchsorted(native, rotated_scores, side="left")
    equal = np.searchsorted(native, rotated_scores, side="right") - lower
    return np.mean((lower + 0.5 * equal) / len(native))


def main():
    unrotated_ids, unrotated_counts = load_edge_counts(UNROTATED)
    rotated_ids, rotated_counts = load_edge_counts(ROTATED)
    unrotated_pool, rotated_pool = test_pools(unrotated_ids, rotated_ids)

    rng = np.random.default_rng(SEED + 2026)
    unrotated_scores = make_scores(unrotated_counts, unrotated_pool, rng)
    rotated_scores = make_scores(rotated_counts, rotated_pool, rng)

    result = auc(unrotated_scores, rotated_scores)
    print("Unrotated mean: {:.1f}".format(unrotated_scores.mean()))
    print("Rotated mean:   {:.1f}".format(rotated_scores.mean()))
    print("Test AUC:       {:.4f}".format(result))


if __name__ == "__main__":
    main()
