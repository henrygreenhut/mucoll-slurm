import argparse
import csv
import json
from pathlib import Path

import numpy as np


WIDTHS_MM = (100, 10, 1)
LOWER_MM = 6000
UPPER_MM = 60000
STEM = "photon_harmonics_vs_mother_z_muplus_c2"


def load_mothers(path):
    import h5py

    with h5py.File(path, "r") as source:
        offsets = source["mother_offsets"][:]
        distance = np.abs(source["mother_decay_positions"][:, 2].astype(float))
        cycle_ids = source["cycle_ids"][:]
        cycle = np.searchsorted(cycle_ids, source["mother_cycle_ids"][:])
        selected = np.flatnonzero(source["particles/pdg"][:] == 22)
        mothers = np.searchsorted(offsets, selected, side="right") - 1
        x = source["particles/px"][:][selected].astype(float)
        y = source["particles/py"][:][selected].astype(float)
        if not np.all(np.isfinite(x) & np.isfinite(y) & ((x != 0) | (y != 0))):
            raise ValueError("Undefined photon momentum phi")
        phi = np.arctan2(y, x)
        count = np.bincount(mothers, minlength=len(distance))
        cosine = np.bincount(mothers, weights=np.cos(2 * phi), minlength=len(distance))
        filenames = source["filenames"].asstr()[:].tolist()
    return distance, cycle, count, cosine, len(cycle_ids), filenames


def profile(distance, cycle, count, cosine, n_cycles, width):
    edges = np.arange(LOWER_MM, UPPER_MM + width, width)
    n_bins = len(edges) - 1
    selected = (distance >= LOWER_MM) & (distance < UPPER_MM)
    bins = np.searchsorted(edges, distance[selected], side="right") - 1
    photons = count[selected]
    weights = cosine[selected]
    totals = np.bincount(bins, weights=photons, minlength=n_bins)
    sums = np.bincount(bins, weights=weights, minlength=n_bins)
    mean = np.divide(sums, totals, out=np.full(n_bins, np.nan), where=totals > 0)

    visible = photons > 0
    keys, inverse = np.unique(cycle[selected][visible] * n_bins + bins[visible],
                              return_inverse=True)
    group_bins = keys % n_bins
    group_counts = np.bincount(inverse, weights=photons[visible])
    group_cosine = np.bincount(inverse, weights=weights[visible])
    residual = group_cosine - mean[group_bins] * group_counts
    variance = np.bincount(group_bins, weights=residual ** 2, minlength=n_bins)
    cycles_per_bin = np.bincount(group_bins, minlength=n_bins)
    se = np.divide(np.sqrt(variance * n_cycles / (n_cycles - 1)), totals,
                   out=np.full(n_bins, np.nan), where=(totals > 0) & (cycles_per_bin > 1))
    return {
        "distance_low_m": edges[:-1] / 1000,
        "distance_high_m": edges[1:] / 1000,
        "distance_center_m": (edges[:-1] + width / 2) / 1000,
        "mother_counts": np.bincount(bins, minlength=n_bins),
        "photon_mother_counts": np.bincount(bins[visible], minlength=n_bins),
        "cycles_with_photons": cycles_per_bin,
        "photon_counts": totals.astype(np.int64),
        "cosine_sum": sums,
        "c2": mean,
        "c2_se": se,
        "c2_low": mean - 1.96 * se,
        "c2_high": mean + 1.96 * se,
    }


def write_profile(output, width, result):
    base = output / f"{STEM}_{width}mm"
    np.savez_compressed(base.with_suffix(".npz"), **result)
    with base.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(result)
        writer.writerows(zip(*result.values()))


def plot_profiles(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    for width, label in zip(WIDTHS_MM, ("10 cm", "1 cm", "1 mm")):
        base = output / f"{STEM}_{width}mm"
        with np.load(base.with_suffix(".npz")) as data:
            distance = data["distance_center_m"]
            values = 100 * data["c2"]
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(distance, values, color="#0072B2", linewidth=.8,
                  marker=".", markersize=1.5)
        axis.axhline(0, color=".35", linewidth=.8)
        axis.set_xlim(LOWER_MM / 1000, UPPER_MM / 1000)
        axis.set_xlabel("Mother-muon decay distance from the IP [m]")
        axis.set_ylabel(r"$C_2$ [%]")
        axis.set_title(r"BIB photon $C_2$ versus mother-muon decay position"
                       + f"\n{label} bins")
        axis.grid(alpha=.2, linewidth=.5)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        for extension in ("pdf", "png"):
            figure.savefig(base.with_suffix("." + extension), dpi=200, bbox_inches="tight")
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.no_plots and args.plot_only:
        parser.error("--no-plots and --plot-only cannot be combined")
    if not args.plot_only:
        if args.bank is None:
            parser.error("--bank is required for recalculation")
        args.output.mkdir(parents=True, exist_ok=False)
        distance, cycle, count, cosine, n_cycles, filenames = load_mothers(args.bank)
        metadata = {
            "bank": str(args.bank), "source_cycles": n_cycles,
            "source_filenames": filenames,
            "original_bin_width_mm": 1000,
            "distance_range_m": [LOWER_MM / 1000, UPPER_MM / 1000],
            "phi": "atan2(py, px), photon momentum at FLUKA handoff",
            "distance": "absolute z of original beam-muon decay, not photon entry z",
            "selection": "All populated bins; no minimum photon count, smoothing, or interpolation. Empty bins are gaps.",
            "uncertainty": "CSV intervals are approximate 95% source-cycle cluster intervals; undefined for fewer than two contributing cycles, unreliable with very few cycles, and not shown on the line plots.",
            "profiles": [],
        }
        selected = (distance >= LOWER_MM) & (distance < UPPER_MM)
        for width in WIDTHS_MM:
            result = profile(distance, cycle, count, cosine, n_cycles, width)
            assert result["photon_counts"].sum() == count[selected].sum()
            np.testing.assert_allclose(result["cosine_sum"].sum(), cosine[selected].sum())
            write_profile(args.output, width, result)
            populated = result["photon_counts"] > 0
            metadata["profiles"].append({
                "bin_width_mm": width, "bins": len(populated),
                "populated_bins": int(populated.sum()),
                "empty_bins": int((~populated).sum()),
                "photons": int(result["photon_counts"].sum()),
                "cosine_sum": float(result["cosine_sum"].sum()),
                "median_photons_per_populated_bin": float(np.median(result["photon_counts"][populated])),
                "median_photon_mothers_per_populated_bin": float(np.median(result["photon_mother_counts"][populated])),
                "median_cycles_per_populated_bin": float(np.median(result["cycles_with_photons"][populated])),
                "populated_bins_with_fewer_than_5_cycles": int(np.count_nonzero(
                    populated & (result["cycles_with_photons"] < 5))),
            })
            print(metadata["profiles"][-1], flush=True)
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.no_plots:
        plot_profiles(args.output)


if __name__ == "__main__":
    main()
