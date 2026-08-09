#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"native": "#0072B2", "synthetic": "#D55E00"}


class PhiStore:
    def __init__(self, path):
        with h5py.File(path, "r") as source:
            if "offsets" in source:
                self.offsets = source["offsets"][:]
            else:
                mother_offsets = source["mother_offsets"][:]
                self.offsets = mother_offsets[source["cycle_offsets"][:]]
            self.cycle_ids = source["cycle_ids"][:]
            self.px = source["particles/px"][:]
            self.py = source["particles/py"][:]

    def positions(self, cycle_ids):
        positions = np.searchsorted(self.cycle_ids, cycle_ids)
        if not np.array_equal(self.cycle_ids[positions], cycle_ids):
            raise ValueError("a requested cycle is absent from the store")
        return positions

    def phi_histogram(self, positions, bins):
        px = np.concatenate([
            self.px[self.offsets[position]:self.offsets[position + 1]]
            for position in positions
        ])
        py = np.concatenate([
            self.py[self.offsets[position]:self.offsets[position + 1]]
            for position in positions
        ])
        counts, _ = np.histogram(np.arctan2(py, px), bins=bins)
        density = counts / (counts.sum() * np.diff(bins))
        return density, len(px)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        default=(
            "pfn_results/"
            "pm_n420_k1_vs_k1_synthetic_scaled_lr1e-4_decay80_mseed1_v1"
        ),
    )
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--line-events", type=int, default=25)
    parser.add_argument("--bins", type=int, default=48)
    parser.add_argument(
        "--output-dir", default="plots/gen_n420_native_vs_synthetic_k1_phi"
    )
    return parser.parse_args()


def load_test_rows(result_dir, config, stores):
    with np.load(result_dir / "source_split.npz") as payload:
        test_cycles = payload["test"]

    pools = [store.positions(test_cycles) for store in stores]
    files_per_test = [
        config["n_files"],
        config["n_files"] // config["clone_factor"],
    ]
    count = config["eval_point_units"]
    rng = np.random.default_rng(config["data_seed"] + 2026)

    rows = []
    for pool, files in zip(pools, files_per_test):
        rows.append(np.stack([
            rng.choice(pool, size=files, replace=False)
            for _ in range(count)
        ]))
    return rows


def calculate_histograms(stores, rows, bins):
    densities = []
    multiplicities = []
    for label, store, class_rows in zip(
        ("native", "synthetic"), stores, rows
    ):
        print("{}: {} test constructions".format(label, len(class_rows)))
        class_densities = []
        class_multiplicities = []
        for index, positions in enumerate(class_rows):
            density, multiplicity = store.phi_histogram(positions, bins)
            class_densities.append(density)
            class_multiplicities.append(multiplicity)
            if (index + 1) % 25 == 0 or index + 1 == len(class_rows):
                print("  {}/{}".format(index + 1, len(class_rows)), flush=True)
        densities.append(np.asarray(class_densities))
        multiplicities.append(np.asarray(class_multiplicities))
    return densities, multiplicities


def save(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote {}".format(output.with_suffix(".pdf")))
    print("wrote {}".format(output.with_suffix(".png")))


def plot_examples(densities, bins, count, output):
    indices = np.linspace(0, len(densities[0]) - 1, count, dtype=int)
    columns = 2
    rows = int(np.ceil(count / columns))
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(rows, columns, figsize=(10, 2.8 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for axis, index in zip(axes, indices):
        axis.step(
            centers, densities[0][index], where="mid", linewidth=1.6,
            color=COLORS["native"], label="Native K=1"
        )
        axis.step(
            centers, densities[1][index], where="mid", linewidth=1.6,
            color=COLORS["synthetic"], label="Synthetic K=1"
        )
        axis.axhline(1.0 / (2.0 * np.pi), color="0.6", linestyle=":", linewidth=1)
        axis.set_title("Test construction {}".format(index + 1))
        axis.set_ylabel("Particle density")
        axis.grid(alpha=0.18, linewidth=0.5)

    for axis in axes[len(indices):]:
        axis.set_visible(False)
    for axis in axes[-columns:]:
        axis.set_xlabel(r"Particle $\phi$ [rad]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("GEN N=420 held-out azimuthal distributions", y=0.995)
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.965)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, output / "individual_examples")


def plot_lines(densities, bins, count, output):
    indices = np.linspace(0, len(densities[0]) - 1, count, dtype=int)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)

    for axis, label, values in zip(
        axes, ("Native K=1", "Synthetic K=1"), densities
    ):
        color = COLORS["native" if label.startswith("Native") else "synthetic"]
        for index in indices:
            axis.step(
                centers, values[index], where="mid", color=color,
                alpha=0.22, linewidth=0.8
            )
        axis.step(
            centers, np.mean(values, axis=0), where="mid", color=color,
            linewidth=2.3, label="Mean of all 300"
        )
        axis.axhline(1.0 / (2.0 * np.pi), color="0.45", linestyle=":", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel(r"Particle $\phi$ [rad]")
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.legend(frameon=False)

    axes[0].set_ylabel("Particle density")
    fig.suptitle("GEN N=420 held-out azimuthal distributions")
    fig.tight_layout()
    save(fig, output / "individual_lines")


def plot_heatmaps(densities, bins, output):
    extent = (bins[0], bins[-1], len(densities[0]), 1)
    maximum = max(np.quantile(values, 0.995) for values in densities)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharex=True, sharey=True)

    for axis, label, values in zip(
        axes, ("Native K=1", "Synthetic K=1"), densities
    ):
        image = axis.imshow(
            values, aspect="auto", interpolation="nearest", origin="upper",
            extent=extent, cmap="viridis", vmin=0.0, vmax=maximum
        )
        axis.set_title(label)
        axis.set_xlabel(r"Particle $\phi$ [rad]")

    axes[0].set_ylabel("Held-out test construction")
    colorbar = fig.colorbar(image, ax=axes, pad=0.02)
    colorbar.set_label("Particle density")
    fig.suptitle("GEN N=420 azimuthal distributions for all test constructions")
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.1, top=0.9, wspace=0.08)
    save(fig, output / "all_test_heatmap")


def main():
    args = arguments()
    result_dir = Path(args.result_dir)
    with (result_dir / "config.json").open() as handle:
        config = json.load(handle)

    if config["clone_factor"] != 1:
        raise SystemExit("this plot requires a native-vs-synthetic K=1 result")
    if args.examples < 1 or args.line_events < 1 or args.bins < 3:
        raise SystemExit("examples, line-events, and bins must be positive")

    stores = [
        PhiStore(config["norm1_store"]),
        PhiStore(config["norm42_store"]),
    ]
    rows = load_test_rows(result_dir, config, stores)
    bins = np.linspace(-np.pi, np.pi, args.bins + 1)
    densities, multiplicities = calculate_histograms(stores, rows, bins)

    output = Path(args.output_dir)
    plot_examples(densities, bins, args.examples, output)
    plot_lines(densities, bins, args.line_events, output)
    plot_heatmaps(densities, bins, output)

    for label, values in zip(("native", "synthetic"), multiplicities):
        print(
            "{} particles/construction: min={} median={} mean={:.1f} max={}".format(
                label, values.min(), int(np.median(values)), values.mean(),
                values.max()
            )
        )


if __name__ == "__main__":
    main()
