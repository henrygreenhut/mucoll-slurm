#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def load_mother_statistics(path):
    with h5py.File(path, "r") as source:
        if "mother_decay_positions" not in source:
            raise RuntimeError(
                f"{path} has no mother_decay_positions dataset; rebuild the "
                "mother store with the current generator"
            )

        offsets = source["mother_offsets"][:]
        positions = source["mother_decay_positions"][:]
        particles = source["particles"]
        pdg = particles["pdg"][:]
        photon_indices = np.flatnonzero(pdg == 22)
        phi = np.arctan2(
            particles["py"][:][photon_indices],
            particles["px"][:][photon_indices],
        )

    number_of_mothers = len(offsets) - 1
    if positions.shape != (number_of_mothers, 3):
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
        "z": positions[:, 2].astype(np.float64),
        "counts": counts,
        "cosine": cosine,
        "sine": sine,
    }


def bin_edges(z, width, lower=None, upper=None):
    if width <= 0:
        raise ValueError("bin width must be positive")
    if (lower is None) != (upper is None):
        raise ValueError("z minimum and maximum must be given together")

    if lower is None:
        lower = width * np.floor(z.min() / width)
        upper = width * np.ceil(z.max() / width)
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


def summarize(data, edges):
    number_of_bins = len(edges) - 1
    selected = (data["z"] >= edges[0]) & (data["z"] <= edges[-1])
    bins = np.searchsorted(edges, data["z"][selected], side="right") - 1
    bins[bins == number_of_bins] = number_of_bins - 1

    mother_counts = np.bincount(bins, minlength=number_of_bins)
    photon_counts = np.bincount(
        bins, weights=data["counts"][selected], minlength=number_of_bins
    ).astype(np.int64)
    cosine = np.bincount(
        bins, weights=data["cosine"][selected], minlength=number_of_bins
    )
    sine = np.bincount(bins, weights=data["sine"][selected], minlength=number_of_bins)
    c2, s2, a2 = ratios(photon_counts, cosine, sine)
    return {
        "z_low_mm": edges[:-1],
        "z_high_mm": edges[1:],
        "z_center_mm": 0.5 * (edges[:-1] + edges[1:]),
        "mother_counts": mother_counts,
        "photon_counts": photon_counts,
        "c2": c2,
        "s2": s2,
        "a2": a2,
    }


def write_csv(path, result):
    fields = [
        "z_low_mm",
        "z_high_mm",
        "z_center_mm",
        "number_of_mothers",
        "number_of_photons",
        "c2",
        "s2",
        "a2",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for index in range(len(result["c2"])):
            writer.writerow(
                {
                    "z_low_mm": result["z_low_mm"][index],
                    "z_high_mm": result["z_high_mm"][index],
                    "z_center_mm": result["z_center_mm"][index],
                    "number_of_mothers": result["mother_counts"][index],
                    "number_of_photons": result["photon_counts"][index],
                    "c2": result["c2"][index],
                    "s2": result["s2"][index],
                    "a2": result["a2"][index],
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--bin-width-mm", type=float, default=5000.0)
    parser.add_argument("--z-min-mm", type=float)
    parser.add_argument("--z-max-mm", type=float)
    args = parser.parse_args()

    print(f"Loading {args.bank}", flush=True)
    data = load_mother_statistics(args.bank)
    print(
        f"  {len(data['z']):,} mothers, {int(data['counts'].sum()):,} photons",
        flush=True,
    )
    try:
        edges = bin_edges(data["z"], args.bin_width_mm, args.z_min_mm, args.z_max_mm)
    except ValueError as error:
        parser.error(str(error))
    result = summarize(data, edges)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    stem = args.output_directory / "photon_harmonics_vs_mother_z"
    np.savez(stem.with_suffix(".npz"), **result)
    write_csv(stem.with_suffix(".csv"), result)

    metadata = {
        "bank": str(args.bank.resolve()),
        "particle_selection": "pdg == 22",
        "phi_definition": "atan2(py, px)",
        "harmonics": {
            "c2": "sum(cos(2 phi)) / number of photons",
            "s2": "sum(sin(2 phi)) / number of photons",
            "a2": "sqrt(c2^2 + s2^2)",
        },
        "bin_width_mm": args.bin_width_mm,
    }
    with stem.with_suffix(".json").open("w") as output:
        json.dump(metadata, output, indent=2)
        output.write("\n")

    print(f"Wrote {stem.with_suffix('.npz')}")
    print(f"Wrote {stem.with_suffix('.csv')}")
    print(f"Wrote {stem.with_suffix('.json')}")


if __name__ == "__main__":
    main()
