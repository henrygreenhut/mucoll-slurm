import argparse
import csv
import json
from pathlib import Path

import numpy as np


RECORD = np.dtype([
    ("fid", "i4"), ("fid_mo", "i4"),
    *[(name, "f8") for name in (
        "e_kin", "x", "y", "z", "cx", "cy", "cz", "time",
        "x_mu", "y_mu", "z_mu",
    )],
])
FIELDS = ("px", "py", "pz", "E", "t", "vx", "vy", "vz")
ENERGY = np.array([0, .2, .45, .475, .480, .49, .5, .510, .512,
                   .522, .6, 1, 2, 5, 10, 50, np.inf])
MOTHER_Z = np.array([-np.inf, -120, -60, -45, -33, -27, -13, -6,
                     0, 6, 13, 27, 33, 45, 60, 120, np.inf])
ENTRY_Z = np.array([-np.inf, -10, -6, -4, -2, -1, -.5,
                    0, .5, 1, 2, 4, 6, 10, np.inf])
TIME = np.array([-np.inf, -10, -1, 0, 1, 10, 100, 1000, 10000, 1e6, np.inf])
PHOTON = np.dtype([
    ("cycle", "i4"), ("fid_mo", "i4"),
    *[(name, "f8") for name in (
        "energy", "z_mu", "z_entry", "radius", "time", "phi_p", "phi_x",
    )],
])


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(value) for value in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.integer):
            return int(item)
        return item
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator,
                     out=np.full(numerator.shape, np.nan), where=denominator != 0)


def cycle_statistics(cycle, group, values, n_cycles, n_groups):
    index = cycle * n_groups + group
    return np.stack([
        np.bincount(index, weights=weights, minlength=n_cycles * n_groups)
        .reshape(n_cycles, n_groups) for weights in values
    ], axis=-1)


def summarize(statistics):
    total = statistics.sum(axis=0)
    count = total[:, 0]
    rows = [{"photons": int(number)} for number in count]
    n_cycles = len(statistics)
    for column, name in enumerate(("momentum_c2", "momentum_s2",
                                   "position_c2", "position_s2"), start=1):
        mean = ratio(total[:, column], count)
        residual = statistics[:, :, column] - statistics[:, :, 0] * mean
        se = ratio(np.sqrt(np.nansum(residual ** 2, axis=0)
                           * n_cycles / (n_cycles - 1)), count)
        for index, row in enumerate(rows):
            row[name] = mean[index]
            row[name + "_se"] = se[index]
            row[name + "_low"] = mean[index] - 1.96 * se[index]
            row[name + "_high"] = mean[index] + 1.96 * se[index]
            row[name + "_contribution"] = total[index, column] / count.sum()
    return rows


def binned(photons, values, arrays, edges, n_cycles):
    shape = tuple(len(bins) - 1 for bins in edges)
    indices = [np.clip(np.searchsorted(bins, array, side="right") - 1, 0, size - 1)
               for array, bins, size in zip(arrays, edges, shape)]
    group = np.ravel_multi_index(indices, shape)
    statistics = cycle_statistics(photons["cycle"], group, values,
                                  n_cycles, int(np.prod(shape)))
    return statistics, summarize(statistics), shape


def load_bank(path):
    import h5py
    with h5py.File(path, "r") as bank:
        cycles = bank["cycle_ids"][:]
        offsets = bank["mother_offsets"][:][bank["cycle_offsets"][:]]
        selected = np.flatnonzero(bank["particles/pdg"][:] == 22)
        counts = np.bincount(np.searchsorted(offsets[1:], selected, side="right"),
                             minlength=len(cycles))
        photons = np.column_stack([bank["particles/" + field][:][selected]
                                   for field in FIELDS])
        result = {
            "path": str(path), "cycles": cycles,
            "filenames": bank["filenames"].asstr()[:].tolist(),
            "offsets": np.r_[0, counts.cumsum()], "photons": photons,
            "attributes": {key: str(value) for key, value in bank.attrs.items()},
        }
        if "mother_decay_positions" in bank:
            mothers = np.searchsorted(bank["mother_offsets"][:], selected,
                                       side="right") - 1
            result["mother_z"] = bank["mother_decay_positions"][:, 2][mothers]
    return result


def targeted_checks(archives):
    result = {}
    total = archives["cycle_totals"]
    for name, table, lower, upper in (
        ("entry_z_10_to_15_cm", "entry_fine", .1, .15),
        ("entry_z_5_to_15_cm", "entry_fine", .05, .15),
        ("entry_z_10_to_20_cm", "entry_fine", .1, .2),
        ("energy_510_to_512_keV", "energy", .510, .512),
    ):
        edges = archives[table + "_edges0"]
        bins = (edges[:-1] >= lower - 1e-9) & (edges[1:] <= upper + 1e-9)
        selected = archives[table + "_cycles"][:, bins, :].sum(axis=1)
        rows = summarize(np.stack([selected, total - selected], axis=1))
        result[name] = {"inside": rows[0], "outside": rows[1]}
        middle = len(selected) // 2
        result[name]["first_half"] = summarize(selected[:middle, None, :])[0]
        result[name]["second_half"] = summarize(selected[middle:, None, :])[0]
    return result


def audit_and_read(bank_paths, source, output):
    banks = [load_bank(path) for path in bank_paths]
    reference = banks[0]
    for bank in banks[1:]:
        if bank["filenames"] != reference["filenames"]:
            raise ValueError("Banks use different source-file sequences")
        np.testing.assert_array_equal(bank["cycles"], reference["cycles"])
    audit = [{"bank": bank["path"], "attributes": bank["attributes"],
              "count_mismatch_cycles": [], "value_mismatch_cycles": [],
              "max_absolute_difference": np.zeros(len(FIELDS)),
              "mother_z_max_difference_mm": 0.} for bank in banks]
    records = []
    raw_harmonics = []
    invalid = {"momentum": 0, "position": 0, "nonfinite": 0}
    for index, filename in enumerate(reference["filenames"]):
        path = source / filename
        if path.stat().st_size % RECORD.itemsize:
            raise ValueError(f"Incomplete raw record in {path}")
        raw = np.fromfile(path, RECORD)
        raw = raw[raw["fid"] == 7]
        expected = np.column_stack([
            raw["cx"] * raw["e_kin"], raw["cy"] * raw["e_kin"],
            raw["cz"] * raw["e_kin"], raw["e_kin"], raw["time"] * 1e9,
            raw["x"] * 10, raw["y"] * 10, raw["z"] * 10,
        ]).astype(np.float32)
        expected_order = np.lexsort(expected.T)
        for bank, check in zip(banks, audit):
            first, last = bank["offsets"][index:index + 2]
            actual = bank["photons"][first:last]
            if len(actual) != len(expected):
                check["count_mismatch_cycles"].append(int(reference["cycles"][index]))
                continue
            order = np.lexsort(actual.T)
            difference = np.abs(actual[order].astype(float)
                                - expected[expected_order].astype(float))
            check["max_absolute_difference"] = np.maximum(
                check["max_absolute_difference"], difference.max(axis=0, initial=0))
            if not np.array_equal(actual[order], expected[expected_order]):
                check["value_mismatch_cycles"].append(int(reference["cycles"][index]))
            if "mother_z" in bank:
                delta = np.abs(bank["mother_z"][first:last][order].astype(float)
                               - (raw["z_mu"] * 10).astype(np.float32)[expected_order])
                check["mother_z_max_difference_mm"] = max(
                    check["mother_z_max_difference_mm"], float(delta.max(initial=0)))
        item = np.empty(len(raw), PHOTON)
        item["cycle"] = index
        item["fid_mo"] = raw["fid_mo"]
        item["energy"] = raw["e_kin"] * 1000
        item["z_mu"] = raw["z_mu"] / 100
        item["z_entry"] = raw["z"] / 100
        item["radius"] = np.hypot(raw["x"], raw["y"]) / 100
        item["time"] = raw["time"] * 1e9
        item["phi_p"] = np.arctan2(raw["cy"], raw["cx"])
        item["phi_x"] = np.arctan2(raw["y"], raw["x"])
        valid_p = (raw["cx"] != 0) | (raw["cy"] != 0)
        valid_x = (raw["x"] != 0) | (raw["y"] != 0)
        finite = np.isfinite(np.column_stack([item[name] for name in PHOTON.names])).all(axis=1)
        invalid["momentum"] += int(np.count_nonzero(~valid_p))
        invalid["position"] += int(np.count_nonzero(~valid_x))
        invalid["nonfinite"] += int(np.count_nonzero(~finite))
        if not np.all(valid_p & valid_x & finite):
            raise ValueError(f"Invalid photon coordinates in {filename}: {invalid}")
        records.append(item)
        raw_harmonics.append([len(item), np.cos(2 * item["phi_p"]).sum(),
                              np.sin(2 * item["phi_p"]).sum(),
                              np.cos(2 * item["phi_x"]).sum(),
                              np.sin(2 * item["phi_x"]).sum()])
        if (index + 1) % 500 == 0:
            print(f"Raw audit: {index + 1}/{len(reference['filenames'])} cycles", flush=True)
    for bank, check in zip(banks, audit):
        check["fields"] = FIELDS
        check["harmonics_from_bank"] = {
            name: float(function(2 * np.arctan2(bank["photons"][:, y].astype(float),
                                               bank["photons"][:, x].astype(float))).mean())
            for name, function, x, y in (
                ("momentum_c2", np.cos, 0, 1), ("momentum_s2", np.sin, 0, 1),
                ("position_c2", np.cos, 5, 6), ("position_s2", np.sin, 5, 6),
            )
        }
    write_json(output / "raw_bank_audit.json", {
        "source": str(source), "record_bytes": RECORD.itemsize,
        "source_cycles": reference["cycles"], "source_filenames": reference["filenames"],
        "invalid": invalid, "banks": audit,
    })
    return np.concatenate(records), np.asarray(raw_harmonics), reference["cycles"]


def analyze(photons, cycle_totals, cycle_ids, output):
    n_cycles = len(cycle_ids)
    values = [np.ones(len(photons)), np.cos(2 * photons["phi_p"]),
              np.sin(2 * photons["phi_p"]), np.cos(2 * photons["phi_x"]),
              np.sin(2 * photons["phi_x"])]
    archives = {}
    configurations = {
        "energy": ([photons["energy"]], [ENERGY]),
        "mother_z": ([photons["z_mu"]], [MOTHER_Z]),
        "entry_z": ([photons["z_entry"]], [ENTRY_Z]),
        "time": ([photons["time"]], [TIME]),
        "radius": ([photons["radius"]], [np.array([0, .01, .02, .05, .1, .2, .5, 1, 2, 5, np.inf])]),
        "mother_entry": ([photons["z_mu"], photons["z_entry"]], [MOTHER_Z, ENTRY_Z]),
        "energy_entry": ([photons["energy"], photons["z_entry"]], [ENERGY, ENTRY_Z]),
        "energy_mother": ([photons["energy"], photons["z_mu"]], [ENERGY, MOTHER_Z]),
        "energy_time": ([photons["energy"], photons["time"]], [ENERGY, TIME]),
        "entry_fine": ([photons["z_entry"]],
                       [np.r_[-np.inf, np.linspace(-.5, .5, 21), np.inf]]),
        "entry_radius": ([photons["z_entry"], photons["radius"]],
                         [np.r_[-np.inf, np.linspace(-.5, .5, 11), np.inf],
                          np.array([0, .0239999, .0240001, .025, .03, .04, .05,
                                    .1, .2, .5, 1, 2, 5, 7])]),
    }
    for name, (arrays, edges) in configurations.items():
        statistics, rows, shape = binned(photons, values, arrays, edges, n_cycles)
        for index, row in enumerate(rows):
            for axis, position in enumerate(np.unravel_index(index, shape)):
                row[f"axis{axis}_low"] = edges[axis][position]
                row[f"axis{axis}_high"] = edges[axis][position + 1]
        write_csv(output / f"{name}.csv", rows)
        for axis, bins in enumerate(edges):
            archives[f"{name}_edges{axis}"] = bins
        archives[name + "_totals"] = statistics.sum(axis=0).reshape(shape + (5,))
        if len(shape) == 1:
            archives[name + "_cycles"] = statistics
        del statistics
        print(f"Summarized {name}", flush=True)

    for label, mask in (("below_2MeV", photons["energy"] < 2),
                         ("above_2MeV", photons["energy"] >= 2)):
        statistics, rows, shape = binned(
            photons[mask], [value[mask] for value in values],
            [photons["z_mu"][mask], photons["z_entry"][mask]],
            [MOTHER_Z, ENTRY_Z], n_cycles)
        name = "mother_entry_" + label
        archives[name + "_totals"] = statistics.sum(axis=0).reshape(shape + (5,))
        archives[name + "_edges0"] = MOTHER_Z
        archives[name + "_edges1"] = ENTRY_Z
        for index, row in enumerate(rows):
            first, second = np.unravel_index(index, shape)
            row.update(axis0_low=MOTHER_Z[first], axis0_high=MOTHER_Z[first + 1],
                       axis1_low=ENTRY_Z[second], axis1_high=ENTRY_Z[second + 1])
        write_csv(output / f"{name}.csv", rows)
        del statistics

    central_z = (photons["z_entry"] >= 0) & (photons["z_entry"] < .5)
    small_radius = (photons["radius"] >= .02) & (photons["radius"] < .05)
    short_time = (photons["time"] >= 0) & (photons["time"] < 1)
    cohorts = []
    for name, mask in (("entry_z_0_to_0p5_m", central_z),
                        ("entry_radius_0p02_to_0p05_m", small_radius),
                        ("recorded_time_0_to_1_ns", short_time),
                        ("all_three", central_z & small_radius & short_time),
                        ("r_2p4cm_surface", np.isclose(photons["radius"], .024,
                                                       rtol=0, atol=1e-8))):
        group = (~mask).astype(int)
        statistics = cycle_statistics(photons["cycle"], group, values, n_cycles, 2)
        for label, row in zip((name, "outside_" + name), summarize(statistics)):
            row["selection"] = label
            cohorts.append(row)
    write_csv(output / "cohorts.csv", cohorts)

    overall = summarize(cycle_totals[:, None, :])[0]
    subsets = {}
    for name, selected in (
        ("first_half", np.arange(n_cycles) < n_cycles // 2),
        ("second_half", np.arange(n_cycles) >= n_cycles // 2),
        ("even_cycle_ids", cycle_ids % 2 == 0),
        ("odd_cycle_ids", cycle_ids % 2 == 1),
        *[(f"block_{block + 1}", indices) for block, indices in
          enumerate(np.array_split(np.arange(n_cycles), 10))],
    ):
        subsets[name] = summarize(cycle_totals[selected, None, :])[0]

    rng = np.random.default_rng(1701)
    bootstrap = []
    for _ in range(2000):
        total = cycle_totals[rng.integers(0, n_cycles, n_cycles)].sum(axis=0)
        bootstrap.append(total[1:] / total[0])
    bootstrap = np.asarray(bootstrap)
    bootstrap_limits = {
        name: np.percentile(bootstrap[:, index], [2.5, 97.5])
        for index, name in enumerate(("momentum_c2", "momentum_s2",
                                       "position_c2", "position_s2"))
    }

    delta = np.angle(np.exp(1j * (photons["phi_p"] - photons["phi_x"])))
    relative = np.array([np.ones(len(delta)), np.cos(delta), np.sin(delta),
                         np.cos(2 * delta), np.sin(2 * delta),
                         (np.abs(delta) < np.deg2rad(20)).astype(float),
                         (np.abs(delta) > np.deg2rad(160)).astype(float)])
    phi_edges = np.linspace(-np.pi, np.pi, 49)
    archives["phi_edges"] = phi_edges
    archives["delta_histogram"] = np.histogram(delta, phi_edges)[0]
    archives["position_momentum_histogram"] = np.histogram2d(
        photons["phi_x"], photons["phi_p"], (phi_edges, phi_edges))[0]
    archives["position_delta_histogram"] = np.histogram2d(
        photons["phi_x"], delta, (phi_edges, phi_edges))[0]
    archives["energy_spectrum_edges"] = np.linspace(.35, .65, 601)
    archives["energy_spectrum_counts"] = np.histogram(
        photons["energy"], archives["energy_spectrum_edges"])[0]

    energy_bin = np.clip(np.searchsorted(ENERGY, photons["energy"], side="right") - 1,
                         0, len(ENERGY) - 2)
    entry_bin = np.clip(np.searchsorted(ENTRY_Z, photons["z_entry"], side="right") - 1,
                        0, len(ENTRY_Z) - 2)
    group = energy_bin * (len(ENTRY_Z) - 1) + entry_bin
    size = (len(ENERGY) - 1) * (len(ENTRY_Z) - 1)
    angular = np.stack([np.bincount(group, weights=value, minlength=size)
                        for value in [*values, *relative[1:]]], axis=1)
    count = angular[:, 0]
    mean = ratio(angular, count[:, None])
    predicted = mean[:, 3] * mean[:, 7] - mean[:, 4] * mean[:, 8]
    independent_prediction = np.nansum(count * predicted) / count.sum()
    angular_rows = []
    for index in range(size):
        energy_index, entry_index = divmod(index, len(ENTRY_Z) - 1)
        angular_rows.append({
            "energy_low_MeV": ENERGY[energy_index], "energy_high_MeV": ENERGY[energy_index + 1],
            "entry_z_low_m": ENTRY_Z[entry_index], "entry_z_high_m": ENTRY_Z[entry_index + 1],
            "photons": int(count[index]), "momentum_c2": mean[index, 1],
            "position_c2": mean[index, 3], "position_s2": mean[index, 4],
            "delta_cos2": mean[index, 7], "delta_sin2": mean[index, 8],
            "factorized_prediction_c2": predicted[index],
            "outward_aligned_fraction": mean[index, 9],
            "inward_aligned_fraction": mean[index, 10],
        })
    write_csv(output / "position_direction.csv", angular_rows)
    fid_codes, fid_group = np.unique(photons["fid_mo"], return_inverse=True)
    fid_statistics = cycle_statistics(photons["cycle"], fid_group, values,
                                      n_cycles, len(fid_codes))
    fid_rows = summarize(fid_statistics)
    for code, row in zip(fid_codes, fid_rows):
        row["fid_mo"] = int(code)
    write_csv(output / "raw_fid_mo.csv", fid_rows)
    archives["cycle_ids"] = cycle_ids
    archives["cycle_totals"] = cycle_totals
    np.savez_compressed(output / "aggregates.npz", **archives)
    write_json(output / "targeted_checks.json", targeted_checks(archives))
    summary = {
        "overall": overall, "source_cycle_bootstrap_95": bootstrap_limits,
        "subsets": subsets, "source_cycles": n_cycles,
        "ranges": {name: [float(photons[name].min()), float(photons[name].max())]
                   for name in ("energy", "z_mu", "z_entry", "radius", "time")},
        "direction": {
            "mean_cos_delta": relative[1].mean(), "mean_sin_delta": relative[2].mean(),
            "mean_cos2_delta": relative[3].mean(), "mean_sin2_delta": relative[4].mean(),
            "outward_aligned_fraction": relative[5].mean(),
            "inward_aligned_fraction": relative[6].mean(),
            "factorized_c2_energy_entry_strata": independent_prediction,
            "actual_momentum_c2": overall["momentum_c2"],
            "interpretation": "Factorization assumes position angle and relative direction are independent within each energy/entry-z bin; it is a descriptive check, not a causal decomposition.",
        },
        "uncertainty": "Binned intervals are normal-approximation source-cycle cluster intervals; overall intervals additionally use 2000 source-cycle bootstrap replicates. Exploratory bins are not multiple-testing corrected.",
        "units": "Energy MeV, positions m, recorded FLUKA time ns; position is detector-entry/handoff, not photon creation.",
        "source": "Raw, unrotated format-2 FLUKA MUPLUS records; fid=7; cycle 6291 absent according to bank manifest.",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"overall": overall, "direction": summary["direction"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    banks = [args.bank_dir / name for name in (
        "gen_split_mothers_MUPLUS.h5", "gen_split_mothers_with_positions_MUPLUS.h5")]
    photons, cycle_totals, cycle_ids = audit_and_read(banks, args.source, args.output)
    analyze(photons, cycle_totals, cycle_ids, args.output)


if __name__ == "__main__":
    main()
