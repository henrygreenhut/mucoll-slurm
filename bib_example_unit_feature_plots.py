#!/usr/bin/env python3
"""Plot every expanded GEN PFN input for representative N=420 units."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import libtest_common as lc
from pfn_libtest_train import UnitSampler


NORM1_STORE = os.path.expanduser(
    "~/mucoll/stores/gen_norm1_reconstructed_MUPLUS.h5")
NORM42_STORE = (
    f"/oscar/scratch/{os.environ.get('USER', '')}/mucoll/stores/"
    "gen_norm42_MUPLUS.h5")

FEATURE_SET = "expanded"
N_FILES = 420
CLONE_FACTOR = 42
SPLIT_FRACS = (0.50, 0.25, 0.25)
DATA_SEED = 1701
N_EXAMPLES = 25
OUT_PNG = "plots/bib_example_unit_feature_plots.png"

LABEL0 = "unique mothers"
LABEL1 = "42x within-event reuse"
COLORS = {LABEL0: "#0072B2", LABEL1: "#D55E00"}

CONTINUOUS_FEATURES = [
    "logpt",
    "theta",
    "cosphi",
    "sinphi",
    "loge",
    "asinh_t",
    "asinh_vz",
    "asinh_vr",
]

FEATURE_LABELS = {
    "logpt": r"$\log_{10}(p_T/\mathrm{GeV})$",
    "theta": r"$\theta$ [rad]",
    "cosphi": r"$\cos\phi$",
    "sinphi": r"$\sin\phi$",
    "loge": r"$\log_{10}(E/\mathrm{GeV})$",
    "asinh_t": r"$\operatorname{asinh}(t/\mathrm{ns})$",
    "asinh_vz": r"$\operatorname{asinh}(v_z/\mathrm{mm})$",
    "asinh_vr": r"$\operatorname{asinh}(r_{\mathrm{vertex}}/\mathrm{mm})$",
}


def build_samplers():
    norm1 = lc.Store(NORM1_STORE)
    norm42 = lc.Store(NORM42_STORE)
    common, norm1_positions, norm42_positions = lc.common_positions(
        norm1, norm42)
    cycle_split = lc.create_cycle_split(common, SPLIT_FRACS, DATA_SEED)
    train = lc.cycle_split_positions(common, cycle_split)["train"]

    unique = UnitSampler(
        norm1, {"train": norm1_positions[train]}, N_FILES, FEATURE_SET)
    reused = UnitSampler(
        norm42, {"train": norm42_positions[train]},
        N_FILES // CLONE_FACTOR, FEATURE_SET)
    return norm1, norm42, unique, reused


def sample_features(store, sampler, rng):
    units = []
    for _ in range(N_EXAMPLES):
        positions = sampler.random_unit(rng, "train")
        raw = store.file_arrays(positions)
        units.append(lc.build_features(raw, feature_set=FEATURE_SET))
    return np.concatenate(units)


def plot_histogram(ax, first, second, name):
    both = np.concatenate((first, second))
    bins = np.linspace(np.min(both), np.max(both), 80)
    ax.hist(
        first, bins=bins, histtype="step", density=True, linewidth=1.8,
        color=COLORS[LABEL0], label=LABEL0)
    ax.hist(
        second, bins=bins, histtype="step", density=True, linewidth=1.8,
        color=COLORS[LABEL1], label=LABEL1)
    ax.set_xlabel(FEATURE_LABELS[name])
    ax.set_ylabel("density")
    ax.grid(alpha=0.2, linewidth=0.5)


def plot_particle_categories(ax, first, second, columns):
    categories = lc.PDG_ONEHOT
    fractions0 = [np.mean(first[:, columns[name]]) for name in categories]
    fractions1 = [np.mean(second[:, columns[name]]) for name in categories]
    labels = [r"$\gamma$", "neutron", r"$e^\pm$", r"$\mu^\pm$", "other"]
    positions = np.arange(len(categories))
    width = 0.36

    ax.bar(
        positions - width / 2, fractions0, width,
        color=COLORS[LABEL0], label=LABEL0)
    ax.bar(
        positions + width / 2, fractions1, width,
        color=COLORS[LABEL1], label=LABEL1)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("fraction of particles")
    ax.set_xlabel("particle category indicators")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)


def main():
    norm1, norm42, unique, reused = build_samplers()
    rng = np.random.default_rng(DATA_SEED)
    features0 = sample_features(norm1, unique, rng)
    features1 = sample_features(norm42, reused, rng)

    names = lc.feature_names(FEATURE_SET)
    columns = {name: index for index, name in enumerate(names)}

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, name in zip(axes.flat, CONTINUOUS_FEATURES):
        plot_histogram(
            ax, features0[:, columns[name]], features1[:, columns[name]],
            name)
    plot_particle_categories(axes.flat[-1], features0, features1, columns)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("GEN PFN inputs at equal N=420 BIB", fontsize=16)
    fig.text(
        0.5, 0.01,
        f"{N_EXAMPLES} pseudo-events per class; "
        f"unique: {unique.files_per_unit} source cycles; "
        f"reused: {reused.files_per_unit} source cycles x "
        f"{CLONE_FACTOR} rotations",
        ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)
    print(f"{LABEL0}: {len(features0):,} particles")
    print(f"{LABEL1}: {len(features1):,} particles")
    print(f"chart -> {OUT_PNG}")


if __name__ == "__main__":
    main()
