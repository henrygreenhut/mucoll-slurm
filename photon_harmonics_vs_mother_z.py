#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def parse_bank(value):
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected LABEL=PATH") from error
    if not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(path)


def load_mother_statistics(path):
    with h5py.File(path, "r") as source:
        if "mother_decay_positions" not in source:
            raise RuntimeError(
                f"{path} has no mother_decay_positions dataset; rebuild the "
                "mother store with the current generator"
            )

        offsets = source["mother_offsets"][:]
        cycle_ids = source["mother_cycle_ids"][:]
        positions = source["mother_decay_positions"][:]
        particles = source["particles"]
        pdg = particles["pdg"][:]
        photon_indices = np.flatnonzero(pdg == 22)
        phi = np.arctan2(
            particles["py"][:][photon_indices],
            particles["px"][:][photon_indices],
        )

    number_of_mothers = len(offsets) - 1
    if len(cycle_ids) != number_of_mothers or positions.shape != (
        number_of_mothers,
        3,
    ):
        raise RuntimeError(f"inconsistent mother metadata in {path}")

    photon_mothers = np.searchsorted(offsets, photon_indices, side="right") - 1
    counts = np.bincount(photon_mothers, minlength=number_of_mothers)
    cosine = np.bincount(
        photon_mothers, weights=np.cos(2.0 * phi), minlength=number_of_mothers
    )
    sine = np.bincount(
        photon_mothers, weights=np.sin(2.0 * phi), minlength=number_of_mothers
    )
    return {
        "path": str(path.resolve()),
        "z": positions[:, 2].astype(np.float64),
        "cycle_ids": cycle_ids,
        "counts": counts,
        "cosine": cosine,
        "sine": sine,
    }


def bin_edges(datasets, width, lower=None, upper=None):
    if width <= 0:
        raise ValueError("bin width must be positive")
    if (lower is None) != (upper is None):
        raise ValueError("z minimum and maximum must be given together")

    if lower is None:
        lower = width * np.floor(min(data["z"].min() for data in datasets) / width)
        upper = width * np.ceil(max(data["z"].max() for data in datasets) / width)
    if upper <= lower:
        raise ValueError("z maximum must be greater than z minimum")

    number_of_bins = int(np.ceil((upper - lower) / width))
    return lower + width * np.arange(number_of_bins + 1)


def ratios(counts, cosine, sine):
    c2 = np.divide(
        cosine,
        counts,
        out=np.full_like(cosine, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    s2 = np.divide(
        sine,
        counts,
        out=np.full_like(sine, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    return c2, s2, np.hypot(c2, s2)


def bootstrap_profiles(counts, cosine, sine, samples, rng, batch_size=32):
    number_of_cycles, number_of_bins = counts.shape
    result = np.empty((samples, 3, number_of_bins))
    probabilities = np.full(number_of_cycles, 1.0 / number_of_cycles)

    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        weights = rng.multinomial(
            number_of_cycles, probabilities, size=stop - start
        ).astype(np.float64)
        result[start:stop] = np.stack(
            ratios(weights @ counts, weights @ cosine, weights @ sine),
            axis=1,
        )
    return result


def summarize(data, edges, bootstrap_samples, rng):
    number_of_bins = len(edges) - 1
    selected = (data["z"] >= edges[0]) & (data["z"] <= edges[-1])
    bins = np.searchsorted(edges, data["z"][selected], side="right") - 1
    bins[bins == number_of_bins] = number_of_bins - 1

    _, cycle_index = np.unique(data["cycle_ids"], return_inverse=True)
    number_of_cycles = int(cycle_index.max() + 1)
    cycle_counts = np.zeros((number_of_cycles, number_of_bins), dtype=np.int64)
    cycle_cosine = np.zeros((number_of_cycles, number_of_bins))
    cycle_sine = np.zeros((number_of_cycles, number_of_bins))
    indices = (cycle_index[selected], bins)
    np.add.at(cycle_counts, indices, data["counts"][selected])
    np.add.at(cycle_cosine, indices, data["cosine"][selected])
    np.add.at(cycle_sine, indices, data["sine"][selected])

    mother_counts = np.bincount(bins, minlength=number_of_bins)
    photon_counts = cycle_counts.sum(axis=0)
    c2, s2, a2 = ratios(photon_counts, cycle_cosine.sum(axis=0), cycle_sine.sum(axis=0))

    bootstrap = bootstrap_profiles(
        cycle_counts,
        cycle_cosine,
        cycle_sine,
        bootstrap_samples,
        rng,
    )

    intervals = np.full((4, 3, number_of_bins), np.nan)
    populated = photon_counts > 0
    intervals[:, :, populated] = np.nanpercentile(
        bootstrap[:, :, populated], [2.5, 16.0, 84.0, 97.5], axis=0
    )
    return {
        "z_low_mm": edges[:-1],
        "z_high_mm": edges[1:],
        "z_center_mm": 0.5 * (edges[:-1] + edges[1:]),
        "mother_counts": mother_counts,
        "photon_counts": photon_counts,
        "c2": c2,
        "s2": s2,
        "a2": a2,
        "ci95_low": intervals[0],
        "ci68_low": intervals[1],
        "ci68_high": intervals[2],
        "ci95_high": intervals[3],
        "number_of_cycles": number_of_cycles,
    }


def write_csv(path, results):
    fields = [
        "polarity",
        "z_low_mm",
        "z_high_mm",
        "z_center_mm",
        "number_of_mothers",
        "number_of_photons",
        "c2",
        "c2_low",
        "c2_high",
        "s2",
        "s2_low",
        "s2_high",
        "a2",
        "a2_low",
        "a2_high",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for label, result in results.items():
            for index in range(len(result["c2"])):
                writer.writerow(
                    {
                        "polarity": label,
                        "z_low_mm": result["z_low_mm"][index],
                        "z_high_mm": result["z_high_mm"][index],
                        "z_center_mm": result["z_center_mm"][index],
                        "number_of_mothers": result["mother_counts"][index],
                        "number_of_photons": result["photon_counts"][index],
                        "c2": result["c2"][index],
                        "c2_low": result["ci68_low"][0, index],
                        "c2_high": result["ci68_high"][0, index],
                        "s2": result["s2"][index],
                        "s2_low": result["ci68_low"][1, index],
                        "s2_high": result["ci68_high"][1, index],
                        "a2": result["a2"][index],
                        "a2_low": result["ci68_low"][2, index],
                        "a2_high": result["ci68_high"][2, index],
                    }
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank",
        action="append",
        required=True,
        type=parse_bank,
        metavar="POLARITY=PATH",
    )
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--bin-width-mm", type=float, default=1000.0)
    parser.add_argument("--z-min-mm", type=float)
    parser.add_argument("--z-max-mm", type=float)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    if args.bootstrap_samples < 2:
        parser.error("--bootstrap-samples must be at least 2")

    labels = [label for label, _ in args.bank]
    if len(set(labels)) != len(labels):
        parser.error("bank labels must be unique")

    datasets = []
    for label, path in args.bank:
        print(f"Loading {label}: {path}", flush=True)
        data = load_mother_statistics(path)
        print(
            f"  {len(data['cycle_ids']):,} mothers, "
            f"{int(data['counts'].sum()):,} photons",
            flush=True,
        )
        datasets.append(data)
    try:
        edges = bin_edges(datasets, args.bin_width_mm, args.z_min_mm, args.z_max_mm)
    except ValueError as error:
        parser.error(str(error))

    results = {}
    for index, (label, _) in enumerate(args.bank):
        print(f"Calculating {label} profile", flush=True)
        results[label] = summarize(
            datasets[index],
            edges,
            args.bootstrap_samples,
            np.random.default_rng(args.seed + index),
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    stem = args.output_directory / "photon_harmonics_vs_mother_z"
    arrays = {"labels": np.asarray(labels), "bin_edges_mm": edges}
    for label, result in results.items():
        for name, values in result.items():
            if name != "number_of_cycles":
                arrays[f"{label}_{name}"] = values
    np.savez(stem.with_suffix(".npz"), **arrays)
    write_csv(stem.with_suffix(".csv"), results)

    metadata = {
        "banks": {label: data["path"] for (label, _), data in zip(args.bank, datasets)},
        "particle_selection": "pdg == 22",
        "phi_definition": "atan2(py, px)",
        "harmonics": {
            "c2": "sum(cos(2 phi)) / number of photons",
            "s2": "sum(sin(2 phi)) / number of photons",
            "a2": "sqrt(c2^2 + s2^2)",
        },
        "bin_width_mm": args.bin_width_mm,
        "bootstrap_unit": "source cycle",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "cycles": {
            label: result["number_of_cycles"] for label, result in results.items()
        },
    }
    with stem.with_suffix(".json").open("w") as output:
        json.dump(metadata, output, indent=2)
        output.write("\n")

    print(f"Wrote {stem.with_suffix('.npz')}")
    print(f"Wrote {stem.with_suffix('.csv')}")
    print(f"Wrote {stem.with_suffix('.json')}")


if __name__ == "__main__":
    main()
