#!/usr/bin/env python3

import argparse
import json

import h5py
import numpy as np


CATEGORIES = {
    "photons": lambda pdg: pdg == 22,
    "neutrons": lambda pdg: np.abs(pdg) == 2112,
    "electrons_positrons": lambda pdg: np.abs(pdg) == 11,
    "muons": lambda pdg: np.abs(pdg) == 13,
    "charged_pions": lambda pdg: np.abs(pdg) == 211,
    "protons": lambda pdg: np.abs(pdg) == 2212,
    "kaons": lambda pdg: np.isin(np.abs(pdg), [130, 311, 321]),
}


def cycle_sums(values, offsets):
    cumulative = np.pad(np.cumsum(values, dtype=np.float64), (1, 0))
    return cumulative[offsets[1:]] - cumulative[offsets[:-1]]


def load(path, harmonics):
    with h5py.File(path, "r") as source:
        cycle_ids = source["cycle_ids"][:]
        if "offsets" in source:
            offsets = source["offsets"][:]
        else:
            offsets = source["mother_offsets"][:][source["cycle_offsets"][:]]

        particles = source["particles"]
        pdg = particles["pdg"][:]
        phi = np.arctan2(particles["py"][:], particles["px"][:])

    result = {}
    for name, select in CATEGORIES.items():
        selected = select(pdg)
        counts = cycle_sums(selected, offsets)
        cosine = np.array([
            cycle_sums(selected * np.cos(n * phi), offsets)
            for n in harmonics
        ])
        sine = np.array([
            cycle_sums(selected * np.sin(n * phi), offsets)
            for n in harmonics
        ])
        result[name] = (counts, cosine, sine)
    return cycle_ids, result


def vector(counts, cosine, sine):
    total = counts.sum()
    return np.column_stack((cosine.sum(axis=1) / total,
                            sine.sum(axis=1) / total))


def bootstrap(counts, cosine, sine, samples, rng):
    result = np.empty((samples, len(cosine), 2))
    for sample in range(samples):
        chosen = rng.integers(0, len(counts), len(counts))
        result[sample] = vector(
            counts[chosen], cosine[:, chosen], sine[:, chosen]
        )
    return result


def summarize(harmonics, counts, cosine, sine, samples, rng):
    observed = vector(counts, cosine, sine)
    resampled = bootstrap(counts, cosine, sine, samples, rng)
    rows = []
    for index, harmonic in enumerate(harmonics):
        c, s = observed[index]
        radius = np.hypot(c, s)
        radii = np.hypot(resampled[:, index, 0], resampled[:, index, 1])
        covariance = np.cov(resampled[:, index].T)
        statistic = observed[index] @ np.linalg.pinv(covariance) @ observed[index]
        rows.append({
            "harmonic": int(harmonic),
            "cosine": float(c),
            "sine": float(s),
            "modulation_percent": float(200 * radius),
            "phase_degrees": float(
                np.degrees(np.arctan2(s, c) / harmonic)
            ),
            "modulation_bootstrap_std_percent": float(
                200 * radii.std(ddof=1)
            ),
            "cycle_level_p_value": float(np.exp(-0.5 * statistic)),
        })
    return {
        "particles": int(counts.sum()),
        "harmonics": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("unrotated")
    parser.add_argument("rotated")
    parser.add_argument("--maximum-harmonic", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    harmonics = np.arange(1, args.maximum_harmonic + 1)
    unrotated_ids, unrotated = load(args.unrotated, harmonics)
    rotated_ids, rotated = load(args.rotated, harmonics)

    common = np.intersect1d(unrotated_ids, rotated_ids)
    unrotated_positions = np.searchsorted(unrotated_ids, common)
    rotated_positions = np.searchsorted(rotated_ids, common)
    rng = np.random.default_rng(args.seed)

    result = {
        "cycles": int(len(common)),
        "bootstrap_unit": "source cycle",
        "unrotated": {},
        "rotated": {},
    }
    for name in CATEGORIES:
        u = tuple(values[..., unrotated_positions] for values in unrotated[name])
        r = tuple(values[..., rotated_positions] for values in rotated[name])
        result["unrotated"][name] = summarize(
            harmonics, *u, args.bootstrap_samples, rng
        )
        result["rotated"][name] = summarize(
            harmonics, *r, args.bootstrap_samples, rng
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
