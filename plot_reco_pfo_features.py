#!/usr/bin/env python3
"""Plot the PFO inputs used by the N=420 track-fixed RECO PFN."""

import argparse
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reco_libtest_features import FEATURES, RAW, RAW_FEATURES, pfn_features


SAMPLES = ("U", "R")
SPLITS = ("train", "val", "test")
LABELS = {
    "U": "unique mothers",
    "R": "42x within-event reuse",
}
COLORS = {"U": "#0072B2", "R": "#D55E00"}
LINESTYLES = {"U": "-", "R": "--"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=420)
    parser.add_argument("--store-dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    scratch = os.environ.get("PSCRATCH", "")
    if args.store_dir is None:
        if not scratch:
            parser.error("--store-dir is required when PSCRATCH is unset")
        args.store_dir = (
            f"{scratch}/mucoll/libtest/"
            f"reco_n{args.n_files}_pfn_stores_trackfix"
        )
    if args.output is None:
        args.output = (
            f"plots/reco_n{args.n_files}_trackfix_whole_distributions/"
            "pfo_feature_distributions.png"
        )
    return args


def feature_names(h5):
    value = h5.attrs["features"]
    if isinstance(value, bytes):
        value = value.decode()
    return tuple(value.split(","))


def load_sample(store_dir, n_files, sample):
    features = []
    multiplicities = []

    for split in SPLITS:
        path = store_dir / f"n{n_files}_{sample}_{split}.h5"
        if not path.is_file():
            raise SystemExit(f"missing store: {path}")

        with h5py.File(path, "r") as h5:
            if feature_names(h5) != RAW_FEATURES:
                raise SystemExit(f"unexpected PFO features in {path}")
            raw = h5["particles"][:].astype(np.float64)
            counts = h5["n_particles"][:].astype(np.int64)

        valid = np.arange(raw.shape[1])[None, :] < counts[:, None]
        valid &= raw[:, :, RAW["pt"]] > 0
        transformed = pfn_features(raw)
        features.append(transformed[valid])
        multiplicities.append(np.sum(valid, axis=1))

    return np.concatenate(features), np.concatenate(multiplicities)


def continuous_bins(first, second):
    values = np.concatenate((first, second))
    return np.linspace(np.min(values), np.max(values), 61)


def draw_continuous(ax, first, second, xlabel, log_y=False):
    bins = continuous_bins(first, second)
    for sample, values in zip(SAMPLES, (first, second)):
        ax.hist(
            values, bins=bins, density=True, histtype="step", linewidth=1.8,
            linestyle=LINESTYLES[sample], color=COLORS[sample],
            label=LABELS[sample])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.2, linewidth=0.5)


def draw_discrete(ax, first, second, xlabel, object_level=True):
    low = int(np.floor(min(np.min(first), np.min(second))))
    high = int(np.ceil(max(np.max(first), np.max(second))))
    bins = np.arange(low - 0.5, high + 1.5)
    for sample, values in zip(SAMPLES, (first, second)):
        weights = np.full(len(values), 1.0 / len(values))
        ax.hist(
            values, bins=bins, weights=weights, histtype="step",
            linewidth=1.8, linestyle=LINESTYLES[sample],
            color=COLORS[sample], label=LABELS[sample])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of PFOs" if object_level else "fraction of events")
    ax.grid(alpha=0.2, linewidth=0.5)


def draw_multiplicity(ax, first, second):
    values = np.concatenate((first, second))
    bins = np.linspace(np.min(values), np.max(values), 41)
    for sample, sample_values in zip(SAMPLES, (first, second)):
        weights = np.full(len(sample_values), 1.0 / len(sample_values))
        ax.hist(
            sample_values, bins=bins, weights=weights, histtype="step",
            linewidth=1.8, linestyle=LINESTYLES[sample],
            color=COLORS[sample], label=LABELS[sample])
    ax.set_xlabel("PFOs per reconstructed event")
    ax.set_ylabel("fraction of events")
    ax.set_yscale("log")
    ax.grid(alpha=0.2, linewidth=0.5)


def main():
    args = parse_args()
    store_dir = Path(args.store_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    data = {}
    for sample in SAMPLES:
        data[sample] = load_sample(store_dir, args.n_files, sample)

    columns = {name: index for index, name in enumerate(FEATURES)}
    features0, multiplicities0 = data["U"]
    features1, multiplicities1 = data["R"]
    phi0 = np.arctan2(
        features0[:, columns["sin_phi"]],
        features0[:, columns["cos_phi"]])
    phi1 = np.arctan2(
        features1[:, columns["sin_phi"]],
        features1[:, columns["cos_phi"]])

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    draw_continuous(
        axes[0, 0], features0[:, columns["log_pt"]],
        features1[:, columns["log_pt"]], r"$\ln(p_T/\mathrm{GeV})$", True)
    draw_continuous(
        axes[0, 1], features0[:, columns["eta"]],
        features1[:, columns["eta"]], r"PFO $\eta$")
    draw_continuous(axes[0, 2], phi0, phi1, r"PFO $\phi$ [rad]")
    draw_continuous(
        axes[1, 0], features0[:, columns["sin_phi"]],
        features1[:, columns["sin_phi"]], r"PFO $\sin\phi$")
    draw_continuous(
        axes[1, 1], features0[:, columns["cos_phi"]],
        features1[:, columns["cos_phi"]], r"PFO $\cos\phi$")
    draw_continuous(
        axes[1, 2], features0[:, columns["log_energy"]],
        features1[:, columns["log_energy"]], r"$\ln(E/\mathrm{GeV})$", True)
    draw_discrete(
        axes[2, 0], features0[:, columns["charge"]],
        features1[:, columns["charge"]], r"PFO charge$/e$")
    draw_discrete(
        axes[2, 1], features0[:, columns["is_charged"]],
        features1[:, columns["is_charged"]], "PFO charged indicator")
    draw_multiplicity(axes[2, 2], multiplicities0, multiplicities1)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        f"RECO-level PFO distributions at equal N={args.n_files}", fontsize=16)
    fig.text(
        0.5, 0.01,
        f"Track-fixed neutrino-gun events; train, validation, and test "
        f"combined for description; unique: {args.n_files} source files; "
        f"reused: {args.n_files // 42} source files x 42 rotations",
        ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    for sample in SAMPLES:
        features, multiplicities = data[sample]
        print(
            f"{LABELS[sample]}: {len(multiplicities):,} events, "
            f"{len(features):,} PFOs")
    print(f"chart -> {output}")


if __name__ == "__main__":
    main()
