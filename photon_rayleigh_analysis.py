#!/usr/bin/env python3

import argparse
import json

import h5py
import numpy as np


def cycle_vectors(path, harmonics):
    with h5py.File(path, "r") as source:
        cycle_ids = source["cycle_ids"][:]
        if "offsets" in source:
            offsets = source["offsets"][:]
        else:
            offsets = source["mother_offsets"][:][source["cycle_offsets"][:]]

        particles = source["particles"]
        photon = particles["pdg"][:] == 22
        phi = np.arctan2(particles["py"][:], particles["px"][:])

    counts = cycle_sums(photon, offsets)
    cosine = np.array([
        cycle_sums(photon * np.cos(n * phi), offsets)
        for n in harmonics
    ])
    sine = np.array([
        cycle_sums(photon * np.sin(n * phi), offsets)
        for n in harmonics
    ])
    return cycle_ids, counts, cosine, sine


def cycle_sums(values, offsets):
    cumulative = np.pad(np.cumsum(values, dtype=np.float64), (1, 0))
    return cumulative[offsets[1:]] - cumulative[offsets[:-1]]


def vectors(counts, cosine, sine):
    total = counts.sum()
    return np.column_stack((cosine.sum(axis=1) / total,
                            sine.sum(axis=1) / total))


def resample(counts, cosine, sine, samples, seed):
    rng = np.random.default_rng(seed)
    result = np.empty((samples, len(cosine), 2))
    for sample in range(samples):
        chosen = rng.integers(0, len(counts), len(counts))
        result[sample] = vectors(
            counts[chosen], cosine[:, chosen], sine[:, chosen]
        )
    return result


def summarize(harmonics, observed, bootstrap):
    rows = []
    for index, harmonic in enumerate(harmonics):
        cosine, sine = observed[index]
        radius = np.hypot(cosine, sine)
        bootstrap_radius = np.hypot(
            bootstrap[:, index, 0], bootstrap[:, index, 1]
        )
        covariance = np.cov(bootstrap[:, index].T)
        statistic = observed[index] @ np.linalg.pinv(covariance) @ observed[index]
        rows.append({
            "harmonic": int(harmonic),
            "cosine": float(cosine),
            "sine": float(sine),
            "radius": float(radius),
            "modulation_percent": float(200 * radius),
            "phase_degrees": float(np.degrees(np.arctan2(sine, cosine) / harmonic)),
            "radius_bootstrap_std": float(bootstrap_radius.std(ddof=1)),
            "cycle_level_p_value": float(np.exp(-0.5 * statistic)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("unrotated")
    parser.add_argument("rotated")
    parser.add_argument("--maximum-harmonic", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    harmonics = np.arange(1, args.maximum_harmonic + 1)
    unrotated = cycle_vectors(args.unrotated, harmonics)
    rotated = cycle_vectors(args.rotated, harmonics)

    common = np.intersect1d(unrotated[0], rotated[0])
    unrotated_positions = np.searchsorted(unrotated[0], common)
    rotated_positions = np.searchsorted(rotated[0], common)
    unrotated_data = tuple(values[..., unrotated_positions] for values in unrotated[1:])
    rotated_data = tuple(values[..., rotated_positions] for values in rotated[1:])

    unrotated_vector = vectors(*unrotated_data)
    rotated_vector = vectors(*rotated_data)
    unrotated_bootstrap = resample(
        *unrotated_data, args.bootstrap_samples, args.seed
    )
    rotated_bootstrap = resample(
        *rotated_data, args.bootstrap_samples, args.seed + 1
    )

    print(json.dumps({
        "cycles": int(len(common)),
        "unrotated_photons": int(unrotated_data[0].sum()),
        "rotated_photons": int(rotated_data[0].sum()),
        "bootstrap_unit": "source cycle",
        "unrotated": summarize(harmonics, unrotated_vector, unrotated_bootstrap),
        "rotated": summarize(harmonics, rotated_vector, rotated_bootstrap),
    }))


if __name__ == "__main__":
    main()
