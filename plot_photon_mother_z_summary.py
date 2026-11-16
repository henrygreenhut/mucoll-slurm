import argparse
import csv
import json
from pathlib import Path

import numpy as np

from photon_mother_z_fine_binning import load_mothers


REGIONS = ((6, 13, "Inner F"), (13, 27, "D"), (27, 33, "Outer F"), (33, 60, ""))
QUADRUPOLES = ((6.0, 8.0, "Q1"), (8.3, 10.3, "Q2"), (10.6, 12.6, "Q3"),
               (12.9, 26.5, "Q4"), (26.8, 32.8, "Q5"))
MINIMUM_CYCLES = 1000
MINIMUM_PHOTONS = 10000
BOOTSTRAPS = 2000
STEM = "photon_c2_vs_mother_z_source_supported"


def supported_bins(counts, edges, minimum_cycles, minimum_photons):
    result = []
    for lower, upper, _ in REGIONS:
        first, last = np.searchsorted(edges, [lower, upper])
        region_bins = []
        start = first
        for stop in range(first + 1, last + 1):
            per_cycle = counts[:, start:stop].sum(axis=1)
            if (np.count_nonzero(per_cycle) >= minimum_cycles
                    and per_cycle.sum() >= minimum_photons):
                region_bins.append((start, stop))
                start = stop
        if start < last:
            if region_bins:
                region_bins[-1] = (region_bins[-1][0], last)
            else:
                region_bins.append((first, last))
        result.extend(region_bins)
    return result


def bootstrap(counts, cosine, full_counts, samples=BOOTSTRAPS):
    rng = np.random.default_rng(1701)
    n_cycles = len(counts)
    c2 = np.empty((samples, counts.shape[1]))
    contribution = np.empty_like(c2)
    for index in range(samples):
        weights = np.bincount(rng.integers(0, n_cycles, n_cycles), minlength=n_cycles)
        numerator = weights @ cosine
        denominator = weights @ counts
        c2[index] = np.divide(numerator, denominator,
                              out=np.full_like(numerator, np.nan), where=denominator > 0)
        contribution[index] = numerator / (weights @ full_counts)
    return np.nanpercentile(c2, [2.5, 97.5], axis=0), np.percentile(
        contribution, [2.5, 97.5], axis=0)


def calculate(bank, output):
    distance, cycle, photons, cosine, n_cycles, filenames = load_mothers(bank)
    edges = np.arange(6., 61.)
    selected = (distance >= 6000) & (distance < 60000)
    group = np.searchsorted(edges * 1000, distance[selected], side="right") - 1
    index = cycle[selected] * (len(edges) - 1) + group
    shape = (n_cycles, len(edges) - 1)
    fine_counts = np.bincount(index, weights=photons[selected],
                              minlength=np.prod(shape)).reshape(shape)
    fine_cosine = np.bincount(index, weights=cosine[selected],
                              minlength=np.prod(shape)).reshape(shape)
    fine_mothers = np.bincount(group, weights=(photons[selected] > 0),
                               minlength=shape[1]).astype(int)
    pairs = supported_bins(fine_counts, edges, MINIMUM_CYCLES, MINIMUM_PHOTONS)
    counts = np.column_stack([fine_counts[:, first:last].sum(axis=1) for first, last in pairs])
    sums = np.column_stack([fine_cosine[:, first:last].sum(axis=1) for first, last in pairs])
    full_counts = np.bincount(cycle, weights=photons, minlength=n_cycles)
    np.testing.assert_allclose(counts.sum(axis=1), fine_counts.sum(axis=1))
    np.testing.assert_allclose(sums.sum(axis=1), fine_cosine.sum(axis=1))
    intervals, contribution_intervals = bootstrap(counts, sums, full_counts)
    total_photons = counts.sum(axis=0)
    total_cosine = sums.sum(axis=0)
    rows = []
    for bin_index, (first, last) in enumerate(pairs):
        cycles_with_photons = int(np.count_nonzero(counts[:, bin_index]))
        rows.append({
            "distance_low_m": float(edges[first]),
            "distance_high_m": float(edges[last]),
            "photons": int(total_photons[bin_index]),
            "photon_producing_mothers": int(fine_mothers[first:last].sum()),
            "source_cycles_with_photons": cycles_with_photons,
            "support_target_met": bool(cycles_with_photons >= MINIMUM_CYCLES
                                       and total_photons[bin_index] >= MINIMUM_PHOTONS),
            "c2": float(total_cosine[bin_index] / total_photons[bin_index]),
            "c2_low": float(intervals[0, bin_index]),
            "c2_high": float(intervals[1, bin_index]),
            "contribution": float(total_cosine[bin_index] / full_counts.sum()),
            "contribution_low": float(contribution_intervals[0, bin_index]),
            "contribution_high": float(contribution_intervals[1, bin_index]),
        })
    with (output / f"{STEM}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "source_cycle_statistics.npz", counts=counts,
                         cosine=sums, full_counts=full_counts, base_counts=fine_counts,
                         base_cosine=fine_cosine, base_edges_m=edges)
    metadata = {
        "bank": str(bank), "source_filenames": filenames, "source_cycles": n_cycles,
        "full_bank_photons": int(full_counts.sum()),
        "displayed_photons": int(total_photons.sum()),
        "minimum_distinct_cycles_with_photons": MINIMUM_CYCLES,
        "minimum_photons": MINIMUM_PHOTONS,
        "bin_rule": "Start with 1 m bins; merge adjacent bins until both count targets are met. Merge a leftover tail backwards. Never cross the approximate optical-group boundaries. No C2 values enter the bin choice.",
        "optical_groups_m": REGIONS,
        "bootstrap": {"unit": "source cycle", "samples": BOOTSTRAPS,
                      "seed": 1701, "interval": "95% percentile, pointwise",
                      "edges": "Fixed at the original sample's count-selected edges"},
        "contribution_normalization": "All photons in the full MUPLUS bank, including those outside the displayed 6–60 m range. Contributions are per bin, not per metre.",
        "coordinate": "Absolute original beam-muon decay z; NOT photon entry z",
        "phi": "Photon momentum atan2(py, px)",
        "bins": rows,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in metadata.items()
                      if key not in ("source_filenames", "bins")}, indent=2), flush=True)
    print("Final bins:", [(row["distance_low_m"], row["distance_high_m"],
                            row["source_cycles_with_photons"]) for row in rows], flush=True)


def plot(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = json.loads((output / "metadata.json").read_text())
    rows = metadata["bins"]
    lower = np.array([row["distance_low_m"] for row in rows])
    upper = np.array([row["distance_high_m"] for row in rows])
    center = (lower + upper) / 2
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, name, ylabel in zip(axes, ("c2", "contribution"),
                                 (r"$C_2$ [%]", "Contribution to full-bank\n" + r"$C_2$ [%]")):
        for group_index, (start, stop, label) in enumerate(REGIONS[:3]):
            axis.axvspan(start, stop, color=".5", alpha=.08 if group_index != 1 else .035,
                          linewidth=0, zorder=0)
        for boundary in (13, 27, 33):
            axis.axvline(boundary, color=".75", lw=.6, zorder=0)
        mean = 100 * np.array([row[name] for row in rows])
        low = 100 * np.array([row[name + "_low"] for row in rows])
        high = 100 * np.array([row[name + "_high"] for row in rows])
        axis.errorbar(center, mean, xerr=(upper - lower) / 2,
                       fmt="o", color="#0072B2", markersize=3.5,
                       elinewidth=.8, capsize=2, zorder=3)
        axis.vlines(center, low, high, color="#0072B2", linewidth=1, zorder=2)
        axis.axhline(0, color=".4", linewidth=.8)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.2, linewidth=.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(y=.18)
    for start, stop, label in REGIONS[:3]:
        axes[0].text((start + stop) / 2, .97, label,
                      transform=axes[0].get_xaxis_transform(), ha="center", va="top", color=".35")
    axes[0].text(.99, .97, "Approximate optical groups", transform=axes[0].transAxes,
                  ha="right", va="top", fontsize=9, color=".4")
    axes[0].set_title("BIB photon anisotropy by mother-muon decay position")
    axes[1].set_xlabel("Mother-muon decay distance from the IP [m]")
    axes[1].set_xlim(6, 60)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"{STEM}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def calculate_simple_statistics(bank, output):
    distance, cycle, photons, cosine, n_cycles, filenames = load_mothers(bank)
    if not np.all(np.isfinite(distance)):
        raise ValueError("Invalid mother-muon decay position")
    edges = np.arange(0., 61.)
    selected = (distance >= 0) & (distance < 60000)
    group = np.searchsorted(edges * 1000, distance[selected], side="right") - 1
    index = cycle[selected] * (len(edges) - 1) + group
    shape = (n_cycles, len(edges) - 1)
    counts = np.bincount(index, weights=photons[selected],
                         minlength=np.prod(shape)).reshape(shape)
    sums = np.bincount(index, weights=cosine[selected],
                       minlength=np.prod(shape)).reshape(shape)
    full_counts = np.bincount(cycle, weights=photons, minlength=n_cycles)
    np.testing.assert_allclose(counts.sum(), photons[selected].sum())
    np.testing.assert_allclose(sums.sum(), cosine[selected].sum())
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "simple_cycle_statistics.npz", base_counts=counts,
                         base_cosine=sums, base_edges_m=edges,
                         full_counts=full_counts, bank=str(bank),
                         source_filenames=filenames)
    print(f"0–6 m: {int(counts[:, :6].sum())} photons; "
          f"0–60 m: {int(counts.sum())} photons", flush=True)


def plot_simple(output):
    archive = output / "simple_cycle_statistics.npz"
    if not archive.exists():
        archive = output / "source_cycle_statistics.npz"
    with np.load(archive) as source:
        edges = source["base_edges_m"]
        per_cycle = source["base_counts"]
        counts = per_cycle.sum(axis=0)
        cycles = np.count_nonzero(per_cycle, axis=0)
        cosine = source["base_cosine"].sum(axis=0)
        full_count = source["full_counts"].sum()
    np.testing.assert_allclose(np.diff(edges), 1)
    c2 = np.divide(cosine, counts, out=np.full_like(cosine, np.nan), where=counts > 0)
    contribution = cosine / full_count
    stem = "photon_c2_vs_mother_z_simple_1m"
    with (output / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["distance_low_m", "distance_high_m", "photons",
                         "source_cycles_with_photons", "c2", "contribution"])
        writer.writerows(zip(edges[:-1], edges[1:], counts.astype(int),
                             cycles, c2, contribution))
    settings = {
        "source": f"{archive.name}: base_counts, base_cosine, base_edges_m, full_counts",
        "bin_width_m": 1, "distance_range_m": [float(edges[0]), float(edges[-1])],
        "axis_range_m": [0, float(edges[-1])],
        "unplotted_range_m": [0, float(edges[0])] if edges[0] > 0 else None,
        "bin_merging": False, "bootstrap": False, "optical_overlay": True,
        "quadrupole_intervals_m": QUADRUPOLES,
        "quadrupole_boundaries": "Visual estimates from the v0.4 final-focus diagram, not verified lattice coordinates; shading only, no bin reassignment",
        "minimum_count_cut": None, "displayed_photons": int(counts.sum()),
        "full_bank_photons": int(full_count),
        "contribution_normalization": "Full-bank photon count, not displayed-range count",
        "coordinate": "Absolute original beam-muon decay z; photon momentum phi",
        "panels": ["Photons per 1 m bin", "Local C2 (dimensionless)", "Contribution to full-bank C2 in percent"],
        "standalone_c2_plot": f"{stem}_c2",
    }
    (output / f"{stem}.json").write_text(json.dumps(settings, indent=2) + "\n")
    draw_simple_panels(edges, counts, c2, contribution,
                       "BIB photon anisotropy by mother-muon decay position",
                       "Mother-muon decay distance from the IP [m]", output / stem,
                       regions=QUADRUPOLES, panels=(0, 1, 2), xlim=(0, edges[-1]),
                       c2_percent=False)
    draw_simple_panels(edges, counts, c2, contribution,
                       "BIB photon anisotropy by mother-muon decay position",
                       "Mother-muon decay distance from the IP [m]",
                       output / f"{stem}_c2", regions=QUADRUPOLES, panels=(1,),
                       xlim=(0, edges[-1]), c2_percent=False)
    plot_simple_zoom(output)


def plot_simple_zoom(output):
    source_stem = "photon_c2_vs_mother_z_simple_1m"
    stem = f"{source_stem}_6_to_60m"
    rows = np.genfromtxt(output / f"{source_stem}.csv", delimiter=",", names=True)
    rows = rows[(rows["distance_low_m"] >= 6) & (rows["distance_high_m"] <= 60)]
    edges = np.r_[rows["distance_low_m"][0], rows["distance_high_m"]]
    np.testing.assert_allclose(np.diff(edges), 1)
    draw_simple_panels(edges, rows["photons"], rows["c2"], rows["contribution"],
                       "BIB photon anisotropy by mother-muon decay position",
                       "Mother-muon decay distance from the IP [m]", output / stem,
                       regions=QUADRUPOLES, xlim=(6, 60), c2_percent=False)
    settings = json.loads((output / f"{source_stem}.json").read_text())
    settings.update({"source": f"{source_stem}.csv", "distance_range_m": [6, 60],
                     "axis_range_m": [6, 60], "unplotted_range_m": None,
                     "displayed_photons": int(rows["photons"].sum()),
                     "view": "6–60 m crop with y axes autoscaled to visible bins; full-bank normalization unchanged"})
    settings.pop("standalone_c2_plot", None)
    (output / f"{stem}.json").write_text(json.dumps(settings, indent=2) + "\n")


def draw_simple_panels(edges, counts, c2, contribution, title, xlabel, output,
                       regions=(), panels=(0, 1, 2), xlim=None, c2_percent=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    height = 3 * len(panels) if len(panels) > 1 else 3.6
    figure, axes = plt.subplots(len(panels), 1, figsize=(10, height),
                                sharex=True, squeeze=False)
    axes = axes[:, 0]
    centers = (edges[:-1] + edges[1:]) / 2
    values = (counts, 100 * c2 if c2_percent else c2, 100 * contribution)
    labels = ("Photons / bin", r"$C_2$ [%]" if c2_percent else r"$C_2$",
              "Contribution to full-bank\n" + r"$C_2$ [%]")
    for axis, panel in zip(axes, panels):
        for start, stop, label in regions:
            axis.axvspan(start, stop, color="#9999ee", alpha=.16,
                          linewidth=0, zorder=0)
        axis.plot(centers, values[panel], "o-", color="#0072B2", markersize=3.5,
                  linewidth=1.2)
        axis.axhline(0, color=".4", linewidth=.8)
        axis.set_ylabel(labels[panel])
        axis.grid(axis="y", alpha=.2, linewidth=.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(y=.18)
        if panel == 0:
            axis.set_ylim(bottom=0)
            axis.yaxis.set_major_formatter(lambda value, position: f"{value:,.0f}")
    for start, stop, label in regions:
        axes[0].text((start + stop) / 2, .97, label, fontsize=9,
                      transform=axes[0].get_xaxis_transform(),
                      ha="center", va="top", color=".4")
    axes[0].set_title(title)
    axes[-1].set_xlabel(xlabel)
    axes[-1].set_xlim(*(xlim if xlim is not None else (edges[0], edges[-1])))
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output.with_suffix("." + suffix), dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--simple", action="store_true",
                        help="Plot saved 1 m statistics with estimated quadrupole shading, without merging or uncertainty intervals")
    args = parser.parse_args()
    if args.simple:
        if args.bank is not None:
            calculate_simple_statistics(args.bank, args.output)
        if not args.no_plots:
            plot_simple(args.output)
        return
    if args.plot_only and args.no_plots:
        parser.error("--plot-only and --no-plots cannot be combined")
    if not args.plot_only:
        if args.bank is None:
            parser.error("--bank is required for recalculation")
        args.output.mkdir(parents=True, exist_ok=False)
        calculate(args.bank, args.output)
    if not args.no_plots:
        plot(args.output)


if __name__ == "__main__":
    main()
