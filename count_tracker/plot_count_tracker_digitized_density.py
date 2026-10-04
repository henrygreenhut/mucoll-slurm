#!/usr/bin/env python3
"""Plot binned SIM/COUNT digitized tracker-hit position densities."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from plot_count_tracker_digitized_positions import load_hits, reco_files


PROJECTIONS = {
    "xy": {
        "columns": (0, 1),
        "labels": (r"$x$ [mm]", r"$y$ [mm]"),
        "ranges": ((-1500.0, 1500.0), (-1500.0, 1500.0)),
    },
    "xz": {
        "columns": (0, 2),
        "labels": (r"$x$ [mm]", r"$z$ [mm]"),
        "ranges": ((-1500.0, 1500.0), (-2250.0, 2250.0)),
    },
    "yz": {
        "columns": (1, 2),
        "labels": (r"$y$ [mm]", r"$z$ [mm]"),
        "ranges": ((-1500.0, 1500.0), (-2250.0, 2250.0)),
    },
}


def histogram(hits, specification, bins):
    horizontal, vertical = specification["columns"]
    return np.histogram2d(
        hits[:, horizontal],
        hits[:, vertical],
        bins=(bins, bins),
        range=specification["ranges"],
    )


def plot_projection(all_hits, projection, bins, output):
    specification = PROJECTIONS[projection]
    histograms = {
        sample: histogram(hits, specification, bins)
        for sample, hits in all_hits.items()
    }
    maximum = max(float(np.max(values[0])) for values in histograms.values())
    norm = LogNorm(vmin=1.0, vmax=maximum)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 14,
        "axes.titlesize": 19,
        "axes.labelsize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.0,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.7), constrained_layout=True)
    image = None
    for axis, sample in zip(axes, ("SIM", "COUNT")):
        counts, horizontal_edges, vertical_edges = histograms[sample]
        image = axis.pcolormesh(
            horizontal_edges,
            vertical_edges,
            np.ma.masked_equal(counts.T, 0),
            cmap="viridis",
            norm=norm,
            shading="flat",
            rasterized=True,
        )
        axis.set_title(f"{sample} ({len(all_hits[sample]):,} hits)", pad=8)
        axis.set_xlabel(specification["labels"][0])
        axis.set_ylabel(specification["labels"][1])
        axis.set_xlim(*specification["ranges"][0])
        axis.set_ylim(*specification["ranges"][1])
        axis.set_aspect("equal", adjustable="box")
        axis.tick_params(direction="out", length=4)

    colorbar = fig.colorbar(image, ax=axes, pad=0.035)
    colorbar.set_label("Hits / bin")
    fig.suptitle(
        rf"Digitized tracker hits: ${projection}$ projection ({bins}$\times${bins} bins)",
        fontsize=22,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    horizontal_range, vertical_range = specification["ranges"]
    return {
        "bins": [bins, bins],
        "horizontal_range_mm": list(horizontal_range),
        "vertical_range_mm": list(vertical_range),
        "bin_size_mm": [
            (horizontal_range[1] - horizontal_range[0]) / bins,
            (vertical_range[1] - vertical_range[0]) / bins,
        ],
        "shared_color_range_hits_per_bin": [1.0, maximum],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=400)
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be positive")

    all_hits, counts, files = {}, {}, {}
    for sample in ("SIM", "COUNT"):
        paths = reco_files(args.events_root, sample)
        all_hits[sample], counts[sample] = load_hits(paths)
        files[sample] = len(paths)

    output = Path(args.output_dir)
    plot_settings = {}
    for projection in PROJECTIONS:
        plot_settings[projection] = plot_projection(
            all_hits, projection, args.bins, output / projection
        )

    report = {
        "events_root": str(Path(args.events_root).resolve()),
        "files": files,
        "total_hits": {
            sample: int(len(values)) for sample, values in all_hits.items()
        },
        "collection_hits": counts,
        "value": "raw hits per spatial bin",
        "normalization": "shared logarithmic scale within each projection",
        "projections": plot_settings,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
