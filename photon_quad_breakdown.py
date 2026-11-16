#!/usr/bin/env python3
"""Photon harmonics by approximate v0.4 beam-muon decay region.

Run in an environment with h5py, numpy, and matplotlib:
  python photon_quad_breakdown.py --bank-dir /path/to/stores --output-dir plots/quad_study
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BOUNDARIES = np.array([6., 13., 27., 33.])
REFERENCE = "https://indico.cern.ch/event/1279264/contributions/5382158/attachments/2637929/4564312/IR_Design_of_the_10_TeV_Muon_Collider_v06.pdf"


def load_bank(path):
    import h5py

    with h5py.File(path, "r") as bank:
        cycle_ids = bank["cycle_ids"][:]
        mother_cycles = bank["mother_cycle_ids"][:]
        offsets = bank["mother_offsets"][:]
        z = bank["mother_decay_positions"][:, 2].astype(float) / 1000
        particles = bank["particles"]
        photon_indices = np.flatnonzero(particles["pdg"][:] == 22)
        mothers = np.searchsorted(offsets, photon_indices, side="right") - 1
        cycles = np.searchsorted(cycle_ids, mother_cycles)
        if not np.array_equal(cycle_ids[cycles], mother_cycles):
            raise ValueError("Mother cycle IDs do not match the cycle table")
        if not np.all(np.isfinite(z)):
            raise ValueError("Nonfinite mother decay coordinates")
        statistics = np.zeros((len(z), 7))
        statistics[:, 0] = np.bincount(mothers, minlength=len(z))
        for start, fields in ((1, ("px", "py")), (4, ("vx", "vy"))):
            x = particles[fields[0]][:][photon_indices]
            y = particles[fields[1]][:][photon_indices]
            valid = np.isfinite(x) & np.isfinite(y) & ((x != 0) | (y != 0))
            phi = np.arctan2(y[valid].astype(float), x[valid].astype(float))
            selected = mothers[valid]
            for column, weights in (
                (start, None),
                (start + 1, np.cos(2 * phi)),
                (start + 2, np.sin(2 * phi)),
            ):
                statistics[:, column] = np.bincount(
                    selected, weights=weights, minlength=len(z))
        filenames = [str(name) for name in bank["filenames"].asstr()[:]]
        attrs = {key: str(value) for key, value in bank.attrs.items()}
    print(f"{path.name}: {len(z):,} recorded mothers, {len(photon_indices):,} photons", flush=True)
    return z, cycles, statistics, len(cycle_ids), filenames, attrs


def edges(boundaries):
    return np.r_[-np.inf, -boundaries[::-1], 0., boundaries, np.inf]


def aggregate(z, cycles, statistics, n_cycles, bins):
    region = np.searchsorted(bins, z, side="right") - 1
    n_regions = len(bins) - 1
    index = cycles * n_regions + region
    result = np.zeros((n_cycles, n_regions, 8))
    result[:, :, 0] = np.bincount(index, minlength=n_cycles * n_regions).reshape(n_cycles, n_regions)
    for column in range(7):
        result[:, :, column + 1] = np.bincount(
            index, weights=statistics[:, column],
            minlength=n_cycles * n_regions).reshape(n_cycles, n_regions)
    return result


def divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def summarize(data, bootstrap_samples, seed):
    total = data.sum(axis=0)
    result = {"recorded_mothers": total[:, 0], "photons": total[:, 1]}
    rng = np.random.default_rng(seed)
    resampled = np.empty((bootstrap_samples,) + total.shape)
    for sample in range(bootstrap_samples):
        chosen = rng.integers(0, len(data), len(data))
        resampled[sample] = data[chosen].sum(axis=0)
    for coordinate, column in (("momentum", 2), ("position", 5)):
        count = total[:, column]
        for name, numerator in (("c2", column + 1), ("s2", column + 2)):
            key = f"{coordinate}_{name}"
            result[key] = divide(total[:, numerator], count)
            samples = divide(resampled[:, :, numerator], resampled[:, :, column])
            populated = count > 0
            limits = np.full((2, len(count)), np.nan)
            limits[:, populated] = np.nanpercentile(samples[:, populated], [2.5, 97.5], axis=0)
            result[key + "_low"], result[key + "_high"] = limits
        result[coordinate + "_photon_fraction"] = count / count.sum()
        contribution = total[:, column + 1] / count.sum()
        result[coordinate + "_contribution"] = contribution
        samples = resampled[:, :, column + 1] / resampled[:, :, column].sum(axis=1)[:, None]
        result[coordinate + "_contribution_low"], result[coordinate + "_contribution_high"] = np.nanpercentile(samples, [2.5, 97.5], axis=0)
        result[coordinate + "_valid_photons"] = count
    return result


def rows_for(result, bins):
    rows = []
    for index in range(len(bins) - 1):
        row = {"z_low_m": float(bins[index]), "z_high_m": float(bins[index + 1])}
        row.update({key: float(value[index]) for key, value in result.items()})
        rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(all_rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    labels = ["Far −", "Outer F −", "D −", "Inner F −", "Central −",
              "Central +", "Inner F +", "D +", "Outer F +", "Far +"]
    x = np.arange(10)
    for quantity, ylabel in (("c2", r"$C_2$"), ("contribution", r"Contribution to total $C_2$")):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for row, (beam, rows) in enumerate(all_rows.items()):
            for col, coordinate in enumerate(("momentum", "position")):
                ax = axes[row, col]
                key = f"{coordinate}_{quantity}"
                mean = np.array([r[key] for r in rows])
                low = np.array([r[key + "_low"] for r in rows])
                high = np.array([r[key + "_high"] for r in rows])
                ax.vlines(x, low, high, color="#0072B2", linewidth=1.3)
                ax.scatter(x, mean, color="#0072B2", s=24, zorder=3)
                ax.axhline(0, color="0.4", linewidth=.8, linestyle="--")
                ax.axvline(4.5, color="0.7", linewidth=.7)
                ax.set_title(f"{beam}: {coordinate} azimuth")
                ax.set_ylabel(ylabel)
                ax.grid(axis="y", alpha=.2)
                ax.set_xticks(x, labels, rotation=45, ha="right")
        fig.suptitle("Photons by beam-muon decay region (approximate v0.4 groups)")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"photon_quad_all_regions_{quantity}.{suffix}", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (beam, rows) in zip(axes, all_rows.items()):
        counts = np.array([r["photons"] for r in rows])
        ax.bar(x, counts / counts.sum(), color="#0072B2")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_title(beam)
        ax.set_ylabel("Fraction of photons")
    fig.suptitle("Photons by beam-muon decay region (approximate v0.4 groups)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"photon_quad_all_regions_yield.{suffix}", dpi=180)
    plt.close(fig)

    rows = [all_rows["MUPLUS"][i] for i in (3, 2, 1, 0)]
    labels = ["Inner focusing\n6–13 m", "Defocusing\n13–27 m",
              "Outer focusing\n27–33 m", "Farther out\n>33 m"]
    x = np.arange(4)
    for quantity, ylabel in (("c2", r"$C_2$"), ("contribution", r"Contribution to total $C_2$")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, coordinate in zip(axes, ("momentum", "position")):
            key = f"{coordinate}_{quantity}"
            ax.vlines(x, [r[key + "_low"] for r in rows],
                      [r[key + "_high"] for r in rows], color="#0072B2")
            ax.scatter(x, [r[key] for r in rows], color="#0072B2", s=30, zorder=3)
            ax.axhline(0, color="0.4", linewidth=.8, linestyle="--")
            ax.set_xticks(x, labels)
            ax.set_ylabel(ylabel)
            ax.set_title(f"Photon {coordinate} azimuth")
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("MUPLUS BIB by beam-muon decay region\nApproximate v0.4 groups; 95% source-cycle bootstrap intervals")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"photon_quad_{quantity}.{suffix}", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x, [100 * r["momentum_photon_fraction"] for r in rows], color="#0072B2")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Fraction of all photons [%]")
    ax.set_title("MUPLUS BIB by beam-muon decay region\nApproximate v0.4 groups")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"photon_quad_yield.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        with (args.output_dir / "summary.json").open() as handle:
            make_plots(json.load(handle)["regions"], args.output_dir)
        return
    if args.bank_dir is None:
        parser.error("--bank-dir is required unless --plot-only is set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, provenance, sensitivity = {}, {}, []
    for beam in ("MUPLUS", "MUMINUS"):
        path = args.bank_dir / f"gen_split_mothers_with_positions_{beam}.h5"
        z, cycles, stats, n_cycles, filenames, attrs = load_bank(path)
        bins = edges(BOUNDARIES)
        data = aggregate(z, cycles, stats, n_cycles, bins)
        result = summarize(data, args.bootstrap_samples, 1701)
        all_rows[beam] = rows_for(result, bins)
        write_csv(args.output_dir / f"{beam}_regions.csv", all_rows[beam])
        provenance[beam] = {"bank": str(path), "cycles": n_cycles, "attributes": attrs, "source_files": filenames}
        for boundary in range(4):
            for shift in (-0.5, 0.5):
                moved = BOUNDARIES.copy()
                moved[boundary] += shift
                sums = aggregate(z, cycles, stats, n_cycles, edges(moved)).sum(axis=0)
                for index in range(10):
                    sensitivity.append({
                        "beam": beam, "boundary_m": BOUNDARIES[boundary], "shift_m": shift,
                        "region": index, "photons": sums[index, 1],
                        "momentum_c2": divide(sums[:, 3], sums[:, 2])[index],
                        "position_c2": divide(sums[:, 6], sums[:, 5])[index],
                    })
        print(beam, json.dumps(all_rows[beam]), flush=True)
    write_csv(args.output_dir / "boundary_sensitivity.csv", sensitivity)
    metadata = {
        "boundaries_abs_m": BOUNDARIES.tolist(), "reference": REFERENCE,
        "reference_page": 5, "boundary_status": "Rounded optical-group ranges read from figure; includes gaps, not verified individual magnets",
        "assignment": "signed beam-muon decay z in metres, assuming straight final-focus s maps to abs(z)",
        "bootstrap_unit": "source cycle", "bootstrap_samples": args.bootstrap_samples,
        "confidence_interval": "95% percentile bootstrap", "coordinate": "momentum atan2(py,px); position atan2(vy,vx)",
        "selection": "all PDG 22 photons; undefined angles excluded independently",
        "mother_counts": "Only mothers represented in the bank, not all simulated decays",
        "provenance": provenance,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump({"metadata": metadata, "regions": all_rows}, handle, indent=2)
    if not args.no_plots:
        make_plots(all_rows, args.output_dir)


if __name__ == "__main__":
    main()
