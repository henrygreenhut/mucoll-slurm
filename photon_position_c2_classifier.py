#!/usr/bin/env python3

"""Compare momentum and position C2 on identical held-out BIB constructions."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from photon_second_harmonic_classifier import (
    N_CYCLES, N_TEST, ROTATED, SEED, UNROTATED, auc, cycle_sums,
    make_scores, test_pools,
)


def load_sums(path):
    with h5py.File(path, "r") as source:
        ids = source["cycle_ids"][:]
        if np.any(np.diff(ids) <= 0):
            raise ValueError(f"Cycle IDs must be sorted and unique: {path}")
        offsets = (
            source["offsets"][:] if "offsets" in source
            else source["mother_offsets"][:][source["cycle_offsets"][:]]
        )
        particles = source["particles"]
        photons = particles["pdg"][:] == 22
        result = {}
        for name, fields in (
            ("momentum", ("px", "py")),
            ("position", ("vx", "vy")),
        ):
            x = particles[fields[0]][:]
            y = particles[fields[1]][:]
            finite = np.isfinite(x) & np.isfinite(y)
            on_axis = finite & (x == 0) & (y == 0)
            valid = photons & finite & ~on_axis
            phi = np.arctan2(y[valid], x[valid])
            values = np.zeros(len(photons), dtype=np.float64)
            values[valid] = np.cos(2 * phi)
            result[name] = {
                "counts": cycle_sums(valid, offsets),
                "cosine": cycle_sums(values, offsets),
                "excluded_on_axis": int(np.count_nonzero(photons & on_axis)),
                "excluded_nonfinite": int(np.count_nonzero(photons & ~finite)),
            }
            print(f"{path}: {name}, {np.count_nonzero(valid):,} valid photons", flush=True)
    return ids, result


def evaluate(first, second, pools):
    rng = np.random.default_rng(SEED + 2026)
    unrotated = make_scores(first["counts"], first["cosine"], pools[0], rng)
    rotated = make_scores(second["counts"], second["cosine"], pools[1], rng)
    null_rng = np.random.default_rng(SEED + 4040)
    null_a = make_scores(first["counts"], first["cosine"], pools[0], null_rng)
    null_b = make_scores(first["counts"], first["cosine"], pools[0], null_rng)
    for scores in (unrotated, rotated, null_a, null_b):
        if not np.all(np.isfinite(scores)):
            raise ValueError("A construction has an undefined C2")
    summary = {
        "auc_rotated_positive_score_C2": float(auc(unrotated, rotated)),
        "auc_rotated_positive_score_minus_C2": float(auc(-unrotated, -rotated)),
        "null_auc_score_C2": float(auc(null_a, null_b)),
        "unrotated_mean_C2": float(unrotated.mean()),
        "rotated_mean_C2": float(rotated.mean()),
        "unrotated_std_C2": float(unrotated.std(ddof=1)),
        "rotated_std_C2": float(rotated.std(ddof=1)),
        "excluded_photons": {
            label: {key: bank[key] for key in ("excluded_on_axis", "excluded_nonfinite")}
            for label, bank in (("unrotated", first), ("rotated", second))
        },
    }
    return summary, zip(unrotated, rotated, null_a, null_b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unrotated", default=UNROTATED)
    parser.add_argument("--rotated", default=ROTATED)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    first_ids, first = load_sums(args.unrotated)
    second_ids, second = load_sums(args.rotated)
    pools = test_pools(first_ids, second_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "unrotated": args.unrotated,
        "rotated": args.rotated,
        "seed": SEED,
        "source_split": [0.5, 0.25, 0.25],
        "test_cycles": len(pools[0]),
        "cycles_per_construction": N_CYCLES,
        "constructions_per_class": N_TEST,
        "selection": "PDG == 22; exclude undefined azimuth (on axis or nonfinite)",
        "score": "mean cos(2 phi); rotated is the positive class",
        "uncertainty": "Point estimates only; constructions reuse held-out cycles",
    }
    for name in ("momentum", "position"):
        summary[name], rows = evaluate(first[name], second[name], pools)
        with (args.output_dir / f"{name}_scores.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("unrotated_C2", "rotated_C2", "null_a_C2", "null_b_C2"))
            writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
