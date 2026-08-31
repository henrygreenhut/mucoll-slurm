#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ENERGY_EDGES = np.array([
    0.0,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.010,
    np.inf,
])
ENERGY_LABELS = [
    "<0.2",
    "0.2-0.5",
    "0.5-1",
    "1-2",
    "2-3",
    "3-5",
    "5-10",
    ">=10",
]
COARSE_GROUPS = ((0, 1), (2, 3), (4, 5), (6, 7))
COARSE_LABELS = ("<0.5", "0.5-2", "2-5", ">=5")
PHI_EDGES = np.linspace(-np.pi, np.pi, 65)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("unrotated")
    parser.add_argument("rotated")
    parser.add_argument("output")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-samples", type=int, default=5000)
    parser.add_argument("--histogram-bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args()


def cycle_layout(source):
    cycle_ids = source["cycle_ids"][:]
    if "offsets" in source:
        offsets = source["offsets"][:]
    else:
        offsets = source["mother_offsets"][:][source["cycle_offsets"][:]]
    return cycle_ids, offsets.astype(np.int64)


def load_store(path):
    with h5py.File(path, "r") as source:
        cycle_ids, offsets = cycle_layout(source)
        particles = source["particles"]
        pdg = particles["pdg"][:]
        photon_indices = np.flatnonzero(pdg == 22)
        del pdg
        energy = particles["E"][:][photon_indices].astype(np.float64)
        px = particles["px"][:][photon_indices].astype(np.float64)
        py = particles["py"][:][photon_indices].astype(np.float64)

    phi = np.arctan2(py, px)
    cycle = np.searchsorted(offsets[1:], photon_indices, side="right")
    energy_bin = np.searchsorted(ENERGY_EDGES, energy, side="right") - 1
    phi_bin = np.searchsorted(PHI_EDGES, phi, side="right") - 1
    phi_bin = np.clip(phi_bin, 0, len(PHI_EDGES) - 2)

    number_of_cycles = len(cycle_ids)
    number_of_energy_bins = len(ENERGY_LABELS)
    number_of_phi_bins = len(PHI_EDGES) - 1
    counts = np.zeros((number_of_energy_bins, number_of_cycles), dtype=np.int64)
    cosine = np.zeros_like(counts, dtype=np.float64)
    sine = np.zeros_like(counts, dtype=np.float64)
    histograms = np.zeros(
        (number_of_energy_bins, number_of_cycles, number_of_phi_bins),
        dtype=np.int64,
    )

    cos2 = np.cos(2.0 * phi)
    sin2 = np.sin(2.0 * phi)
    for energy_index in range(number_of_energy_bins):
        selected = energy_bin == energy_index
        selected_cycles = cycle[selected]
        counts[energy_index] = np.bincount(
            selected_cycles,
            minlength=number_of_cycles,
        )
        cosine[energy_index] = np.bincount(
            selected_cycles,
            weights=cos2[selected],
            minlength=number_of_cycles,
        )
        sine[energy_index] = np.bincount(
            selected_cycles,
            weights=sin2[selected],
            minlength=number_of_cycles,
        )
        flat_bin = selected_cycles * number_of_phi_bins + phi_bin[selected]
        histograms[energy_index] = np.bincount(
            flat_bin,
            minlength=number_of_cycles * number_of_phi_bins,
        ).reshape(number_of_cycles, number_of_phi_bins)

    return cycle_ids, counts, cosine, sine, histograms


def align(first, second):
    common = np.intersect1d(first[0], second[0])
    first_positions = np.searchsorted(first[0], common)
    second_positions = np.searchsorted(second[0], common)
    first_data = tuple(values[:, first_positions] for values in first[1:4]) + (
        first[4][:, first_positions],
    )
    second_data = tuple(values[:, second_positions] for values in second[1:4]) + (
        second[4][:, second_positions],
    )
    return common, first_data, second_data


def observed(counts, cosine, sine):
    total = counts.sum(axis=1)
    c2 = cosine.sum(axis=1) / total
    s2 = sine.sum(axis=1) / total
    return c2, s2


def bootstrap(first, second, samples, seed):
    rng = np.random.default_rng(seed)
    number_of_cycles = first[0].shape[1]
    number_of_bins = first[0].shape[0]
    result = {
        "unrotated_c2": np.empty((samples, number_of_bins)),
        "unrotated_s2": np.empty((samples, number_of_bins)),
        "rotated_c2": np.empty((samples, number_of_bins)),
        "rotated_s2": np.empty((samples, number_of_bins)),
    }
    for sample in range(samples):
        selected = rng.integers(0, number_of_cycles, number_of_cycles)
        for name, data in (("unrotated", first), ("rotated", second)):
            total = data[0][:, selected].sum(axis=1)
            result[name + "_c2"][sample] = data[1][:, selected].sum(axis=1) / total
            result[name + "_s2"][sample] = data[2][:, selected].sum(axis=1) / total
    return result


def sign_flip_p_values(first, second, samples, seed):
    if not np.array_equal(first[0], second[0]):
        raise ValueError("paired sign-flip test requires matching photon counts per cycle")
    rng = np.random.default_rng(seed)
    difference = first[1] - second[1]
    observed_difference = np.abs(difference.sum(axis=1))
    exceed = np.zeros(len(difference), dtype=np.int64)
    completed = 0
    batch_size = 250
    while completed < samples:
        batch = min(batch_size, samples - completed)
        signs = rng.integers(0, 2, size=(batch, difference.shape[1]), dtype=np.int8)
        signs = 2.0 * signs - 1.0
        permuted = np.abs(signs @ difference.T)
        exceed += np.count_nonzero(permuted >= observed_difference, axis=0)
        completed += batch
    return (exceed + 1.0) / (samples + 1.0)


def holm(p_values):
    p_values = np.asarray(p_values)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def vector_p_value(vector, samples):
    covariance = np.cov(samples.T)
    statistic = vector @ np.linalg.pinv(covariance) @ vector
    return float(np.exp(-0.5 * statistic))


def intervals(values):
    return {
        "std": float(np.std(values, ddof=1)),
        "low68": float(np.quantile(values, 0.16)),
        "high68": float(np.quantile(values, 0.84)),
        "low95": float(np.quantile(values, 0.025)),
        "high95": float(np.quantile(values, 0.975)),
    }


def grouped_histograms(data, groups):
    return np.stack([data[list(group)].sum(axis=0) for group in groups])


def histogram_bootstrap(first, second, groups, samples, seed):
    rng = np.random.default_rng(seed)
    first_hist = grouped_histograms(first[3], groups)
    second_hist = grouped_histograms(second[3], groups)
    number_of_cycles = first_hist.shape[1]
    residuals = np.empty((samples, len(groups), first_hist.shape[2]))
    for sample in range(samples):
        selected = rng.integers(0, number_of_cycles, number_of_cycles)
        unrotated = first_hist[:, selected].sum(axis=1)
        rotated = second_hist[:, selected].sum(axis=1)
        unrotated = unrotated / unrotated.sum(axis=1, keepdims=True)
        rotated = rotated / rotated.sum(axis=1, keepdims=True)
        residuals[sample] = np.divide(
            100.0 * (unrotated - rotated),
            rotated,
            out=np.full_like(rotated, np.nan),
            where=rotated > 0,
        )
    return (
        first_hist.sum(axis=1),
        second_hist.sum(axis=1),
        np.nanquantile(residuals, 0.16, axis=0),
        np.nanquantile(residuals, 0.84, axis=0),
    )


def main():
    args = arguments()
    unrotated = load_store(args.unrotated)
    rotated = load_store(args.rotated)
    common, unrotated, rotated = align(unrotated, rotated)

    if not np.array_equal(unrotated[0], rotated[0]):
        raise ValueError("unrotated and rotated stores have different photon counts")

    unrotated_c2, unrotated_s2 = observed(*unrotated[:3])
    rotated_c2, rotated_s2 = observed(*rotated[:3])
    boot = bootstrap(
        unrotated,
        rotated,
        args.bootstrap_samples,
        args.seed,
    )

    signed_p = sign_flip_p_values(
        unrotated,
        rotated,
        args.permutation_samples,
        args.seed + 1,
    )
    signed_p_holm = holm(signed_p)

    paired_vector_p = []
    unrotated_vector_p = []
    rotated_vector_p = []
    for index in range(len(ENERGY_LABELS)):
        unrotated_vector = np.array([
            unrotated_c2[index],
            unrotated_s2[index],
        ])
        unrotated_samples = np.column_stack((
            boot["unrotated_c2"][:, index],
            boot["unrotated_s2"][:, index],
        ))
        unrotated_vector_p.append(vector_p_value(
            unrotated_vector,
            unrotated_samples,
        ))

        rotated_vector = np.array([
            rotated_c2[index],
            rotated_s2[index],
        ])
        rotated_samples = np.column_stack((
            boot["rotated_c2"][:, index],
            boot["rotated_s2"][:, index],
        ))
        rotated_vector_p.append(vector_p_value(
            rotated_vector,
            rotated_samples,
        ))

        vector = np.array([
            unrotated_c2[index] - rotated_c2[index],
            unrotated_s2[index] - rotated_s2[index],
        ])
        samples = np.column_stack((
            boot["unrotated_c2"][:, index] - boot["rotated_c2"][:, index],
            boot["unrotated_s2"][:, index] - boot["rotated_s2"][:, index],
        ))
        paired_vector_p.append(vector_p_value(vector, samples))
    unrotated_vector_p = np.asarray(unrotated_vector_p)
    rotated_vector_p = np.asarray(rotated_vector_p)
    paired_vector_p = np.asarray(paired_vector_p)
    unrotated_vector_p_holm = holm(unrotated_vector_p)
    rotated_vector_p_holm = holm(rotated_vector_p)
    paired_vector_p_holm = holm(paired_vector_p)

    rows = []
    for index, label in enumerate(ENERGY_LABELS):
        unrotated_amplitude = 200.0 * np.hypot(unrotated_c2[index], unrotated_s2[index])
        rotated_amplitude = 200.0 * np.hypot(rotated_c2[index], rotated_s2[index])
        unrotated_vertical = -200.0 * unrotated_c2[index]
        rotated_vertical = -200.0 * rotated_c2[index]
        boot_unrotated_vertical = -200.0 * boot["unrotated_c2"][:, index]
        boot_rotated_vertical = -200.0 * boot["rotated_c2"][:, index]
        boot_difference = boot_unrotated_vertical - boot_rotated_vertical
        rows.append({
            "energy_bin_MeV": label,
            "photons": int(unrotated[0][index].sum()),
            "unrotated": {
                "c2": float(unrotated_c2[index]),
                "s2": float(unrotated_s2[index]),
                "vertical_modulation_percent": float(unrotated_vertical),
                "vertical_modulation_intervals": intervals(boot_unrotated_vertical),
                "amplitude_percent": float(unrotated_amplitude),
                "phase_degrees": float(np.degrees(np.arctan2(
                    unrotated_s2[index], unrotated_c2[index]
                ) / 2.0)),
                "vector_p_value": float(unrotated_vector_p[index]),
                "vector_holm_p_value": float(unrotated_vector_p_holm[index]),
            },
            "rotated": {
                "c2": float(rotated_c2[index]),
                "s2": float(rotated_s2[index]),
                "vertical_modulation_percent": float(rotated_vertical),
                "vertical_modulation_intervals": intervals(boot_rotated_vertical),
                "amplitude_percent": float(rotated_amplitude),
                "phase_degrees": float(np.degrees(np.arctan2(
                    rotated_s2[index], rotated_c2[index]
                ) / 2.0)),
                "vector_p_value": float(rotated_vector_p[index]),
                "vector_holm_p_value": float(rotated_vector_p_holm[index]),
            },
            "paired_difference": {
                "vertical_modulation_percent": float(unrotated_vertical - rotated_vertical),
                "intervals": intervals(boot_difference),
                "signed_axis_p_value": float(signed_p[index]),
                "signed_axis_holm_p_value": float(signed_p_holm[index]),
                "vector_p_value": float(paired_vector_p[index]),
                "vector_holm_p_value": float(paired_vector_p_holm[index]),
            },
        })

    hist_unrotated, hist_rotated, residual_low, residual_high = histogram_bootstrap(
        unrotated,
        rotated,
        COARSE_GROUPS,
        args.histogram_bootstrap_samples,
        args.seed + 2,
    )

    coarse_c2_unrotated = []
    coarse_s2_unrotated = []
    coarse_c2_rotated = []
    coarse_s2_rotated = []
    for group in COARSE_GROUPS:
        group = list(group)
        count = unrotated[0][group].sum()
        coarse_c2_unrotated.append(unrotated[1][group].sum() / count)
        coarse_s2_unrotated.append(unrotated[2][group].sum() / count)
        count = rotated[0][group].sum()
        coarse_c2_rotated.append(rotated[1][group].sum() / count)
        coarse_s2_rotated.append(rotated[2][group].sum() / count)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output.with_suffix(".json"), "w") as destination:
        json.dump({
            "unrotated_store": str(Path(args.unrotated).resolve()),
            "rotated_store": str(Path(args.rotated).resolve()),
            "cycles": int(len(common)),
            "bootstrap_unit": "matched source cycle",
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_samples": args.permutation_samples,
            "energy_bins": rows,
        }, destination, indent=2)
        destination.write("\n")

    np.savez_compressed(
        output.with_suffix(".npz"),
        phi_edges=PHI_EDGES,
        coarse_labels=np.asarray(COARSE_LABELS),
        hist_unrotated=hist_unrotated,
        hist_rotated=hist_rotated,
        residual_low=residual_low,
        residual_high=residual_high,
        coarse_c2_unrotated=np.asarray(coarse_c2_unrotated),
        coarse_s2_unrotated=np.asarray(coarse_s2_unrotated),
        coarse_c2_rotated=np.asarray(coarse_c2_rotated),
        coarse_s2_rotated=np.asarray(coarse_s2_rotated),
        fine_c2_unrotated=unrotated_c2,
        fine_s2_unrotated=unrotated_s2,
        fine_c2_rotated=rotated_c2,
        fine_s2_rotated=rotated_s2,
    )

    print(output.with_suffix(".json"))
    print(output.with_suffix(".npz"))


if __name__ == "__main__":
    main()
