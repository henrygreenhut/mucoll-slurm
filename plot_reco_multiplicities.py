#!/usr/bin/env python3

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"U": "#0072B2", "R": "#D55E00", "null_b": "#009E73"}
LABELS = {
    "U": "Unique mothers",
    "R": "42× within-event mother reuse",
    "null_b": "Unique-mother null sample",
}
SPLITS = ("train", "val", "test")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-files", type=int, default=420)
    return parser.parse_args()


def feature_names(value):
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).split(",")


def read_store(path):
    with h5py.File(path, "r") as store:
        n_pfos = store["n_particles"][:].astype(int)
        n_clusters = store["n_clusters"][:].astype(int)
        charge_index = feature_names(store.attrs["features"]).index("charge")
        charges = store["particles"][:, :, charge_index]

    valid = np.arange(charges.shape[1])[None, :] < n_pfos[:, None]
    n_charged = np.sum(valid & (np.abs(charges) > 0.1), axis=1)
    return n_pfos, n_clusters, n_charged


def load_sample(store_dir, n_files, sample):
    pieces = [
        read_store(store_dir / f"n{n_files}_{sample}_{split}.h5")
        for split in SPLITS
    ]
    return tuple(
        np.concatenate([piece[index] for piece in pieces])
        for index in range(3)
    )


def bins_for(arrays):
    high = max(int(np.max(values)) for values in arrays)
    return np.arange(-0.5, high + 1.5)


def make_figure(data, samples, output):
    names = ("PFOs per event", "Clusters per event", "Charged PFOs per event")
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))

    for index, (axis, name) in enumerate(zip(axes, names)):
        arrays = [data[sample][index] for sample in samples]
        bins = bins_for(arrays)
        for sample, values in zip(samples, arrays):
            axis.hist(
                values,
                bins=bins,
                weights=np.full(len(values), 1.0 / len(values)),
                histtype="step",
                linewidth=1.8,
                color=COLORS[sample],
                label=LABELS[sample],
            )
        axis.set_xlabel(name)
        axis.set_yscale("log")
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Fraction of events")
    axes[0].legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    args = arguments()
    store_dir = Path(args.store_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = {
        sample: load_sample(store_dir, args.n_files, sample)
        for sample in ("U", "R", "null_b")
    }
    make_figure(data, ("U", "R"), outdir / "reco_n420_unconed_multiplicities")
    make_figure(data, ("U", "null_b"), outdir / "reco_n420_unconed_multiplicities_null")

    for sample, values in data.items():
        print(
            sample,
            "events=", len(values[0]),
            "mean PFOs=", round(float(np.mean(values[0])), 3),
            "mean clusters=", round(float(np.mean(values[1])), 3),
            "mean charged PFOs=", round(float(np.mean(values[2])), 3),
        )
    print("plots ->", outdir)


if __name__ == "__main__":
    main()
