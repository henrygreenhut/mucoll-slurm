#!/usr/bin/env python3
"""Plot matched SIM/COUNT digitized tracker-hit position projections.

The plots use a fixed-size, deterministic uniform sample from each arm.  A
uniform sample preserves the relative populations of the six tracker
collections while keeping dense event displays legible.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLLECTIONS = {
    "VXDBarrelHits": ("Vertex barrel", "#0066cc"),
    "VXDEndcapHits": ("Vertex endcap", "#00a6d6"),
    "ITBarrelHits": ("Inner tracker barrel", "#008f5a"),
    "ITEndcapHits": ("Inner tracker endcap", "#9a8700"),
    "OTBarrelHits": ("Outer tracker barrel", "#6f3fb5"),
    "OTEndcapHits": ("Outer tracker endcap", "#8f4b2e"),
}


def reco_files(events_root, sample):
    paths = sorted(Path(events_root).glob(
        f"*/{sample}/reco/reco_output.edm4hep.root"
    ))
    if not paths:
        raise FileNotFoundError(
            f"No {sample} reco_output.edm4hep.root files below {events_root}"
        )
    return paths


def load_hits(paths):
    import uproot

    parts = []
    collection_counts = {name: 0 for name in COLLECTIONS}
    for path in paths:
        with uproot.open(path) as handle:
            events = handle["events"]
            for collection_index, collection in enumerate(COLLECTIONS):
                coordinates = []
                for axis in "xyz":
                    branch = events[collection][f"{collection}.position.{axis}"]
                    arrays = branch.array(library="np")
                    coordinates.append(
                        np.concatenate(arrays) if len(arrays) else np.empty(0)
                    )
                size = len(coordinates[0])
                if not all(len(values) == size for values in coordinates):
                    raise ValueError(f"{path}: inconsistent {collection} coordinates")
                collection_counts[collection] += size
                if size:
                    label = np.full(size, collection_index, dtype=np.int8)
                    parts.append(np.column_stack((*coordinates, label)))
    if not parts:
        raise ValueError("No digitized tracker hits found")
    return np.concatenate(parts), collection_counts


def uniform_sample(hits, maximum, seed):
    if len(hits) <= maximum:
        return hits
    indices = np.random.default_rng(seed).choice(len(hits), maximum, replace=False)
    return hits[np.sort(indices)]


def projected(sampled, projection):
    x, y, z = sampled[:, 0] / 10.0, sampled[:, 1] / 10.0, sampled[:, 2] / 10.0
    if projection == "xy":
        return x, y
    if projection == "xz":
        return x, z
    if projection == "yz":
        return y, z
    if projection == "rz":
        return z, np.hypot(x, y)
    raise ValueError(f"Unknown projection: {projection}")


def plot_projection(samples, totals, projection, output):
    labels = {
        "xy": (r"$x$ [cm]", r"$y$ [cm]"),
        "xz": (r"$x$ [cm]", r"$z$ [cm]"),
        "yz": (r"$y$ [cm]", r"$z$ [cm]"),
        "rz": (r"$z$ [cm]", r"$r$ [cm]"),
    }
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
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), constrained_layout=True)
    for axis, sample in zip(axes, ("SIM", "COUNT")):
        hits = samples[sample]
        horizontal, vertical = projected(hits, projection)
        collection_ids = hits[:, 3].astype(np.int8)
        for index, (collection, (_label, color)) in enumerate(COLLECTIONS.items()):
            selected = collection_ids == index
            axis.scatter(
                horizontal[selected], vertical[selected], s=3.0, alpha=0.48,
                linewidths=0, color=color, rasterized=True,
            )
        axis.set_title(
            f"{sample} ({len(hits):,} of {totals[sample]:,} hits)", pad=8
        )
        axis.set_xlabel(labels[projection][0])
        axis.set_ylabel(labels[projection][1])
        axis.grid(True, alpha=0.20, linewidth=0.8)
        axis.tick_params(direction="out", length=4)
        if projection == "xy":
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlim(-155, 155)
            axis.set_ylim(-155, 155)
        elif projection in ("xz", "yz"):
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlim(-155, 155)
            axis.set_ylim(-225, 225)
        else:
            axis.set_xlim(-225, 225)
            axis.set_ylim(0, 155)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor=color, markeredgewidth=0, label=label)
        for label, color in COLLECTIONS.values()
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               frameon=False, fontsize=13, columnspacing=1.8)
    fig.suptitle(rf"Digitized tracker hits: ${projection}$ projection", fontsize=22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-points", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if args.max_points < 1:
        parser.error("--max-points must be positive")

    all_hits, sampled, counts, files = {}, {}, {}, {}
    for offset, sample in enumerate(("SIM", "COUNT")):
        paths = reco_files(args.events_root, sample)
        all_hits[sample], counts[sample] = load_hits(paths)
        sampled[sample] = uniform_sample(
            all_hits[sample], args.max_points, args.seed + offset
        )
        files[sample] = len(paths)

    output = Path(args.output_dir)
    for projection in ("xy", "xz", "yz", "rz"):
        plot_projection(
            sampled,
            {sample: len(values) for sample, values in all_hits.items()},
            projection,
            output / projection,
        )

    report = {
        "events_root": str(Path(args.events_root).resolve()),
        "files": files,
        "seed": args.seed,
        "sampling": "uniform without replacement across all six collections",
        "maximum_points_per_arm": args.max_points,
        "total_hits": {sample: int(len(values)) for sample, values in all_hits.items()},
        "plotted_hits": {sample: int(len(values)) for sample, values in sampled.items()},
        "collection_hits": counts,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
