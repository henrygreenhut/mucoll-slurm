#!/usr/bin/env python3
"""Plot particle features and event-level reuse effects for N=420 GEN BIB."""

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
N_BINS = 64
OUT_PNG = "plots/bib_example_unit_feature_plots.png"

LABEL0 = "unique mothers"
LABEL1 = "42x within-event reuse"
COLORS = {LABEL0: "#0072B2", LABEL1: "#D55E00"}
LINESTYLES = {LABEL0: "-", LABEL1: "--"}

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
    "phi": r"$\phi$ [rad]",
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


def invariant_signature_fraction(raw):
    names = ("pz", "E", "t", "vz", "pdg")
    dtype = [(name, raw[name].dtype) for name in names]
    signatures = np.empty(len(raw["E"]), dtype=dtype)
    for name in names:
        signatures[name] = raw[name]
    return np.unique(signatures).size / len(signatures)


def sample_events(store, sampler, rng):
    events = []
    multiplicities = []
    unique_fractions = []
    for _ in range(N_EXAMPLES):
        positions = sampler.random_unit(rng, "train")
        raw = store.file_arrays(positions)
        events.append(lc.build_features(raw, feature_set=FEATURE_SET))
        multiplicities.append(len(raw["E"]))
        unique_fractions.append(invariant_signature_fraction(raw))
    return events, np.asarray(multiplicities), np.asarray(unique_fractions)


def histogram_range(events0, events1, column):
    low = min(np.min(event[:, column]) for event in events0 + events1)
    high = max(np.max(event[:, column]) for event in events0 + events1)
    return np.linspace(low, high, N_BINS + 1)


def event_histograms(events, column, bins):
    widths = np.diff(bins)
    return np.asarray([
        np.histogram(event[:, column], bins=bins)[0] / (len(event) * widths)
        for event in events
    ])


def phi_histograms(events, columns, bins):
    widths = np.diff(bins)
    histograms = []
    for event in events:
        phi = np.arctan2(
            event[:, columns["sinphi"]], event[:, columns["cosphi"]])
        histograms.append(
            np.histogram(phi, bins=bins)[0] / (len(phi) * widths))
    return np.asarray(histograms)


def plot_histogram_band(ax, histograms, bins, label):
    centers = 0.5 * (bins[:-1] + bins[1:])
    low, mean, high = np.percentile(histograms, [16, 50, 84], axis=0)
    ax.fill_between(
        centers, low, high, color=COLORS[label], alpha=0.18, linewidth=0)
    ax.plot(
        centers, mean, color=COLORS[label], linestyle=LINESTYLES[label],
        linewidth=1.8, label=label)


def plot_feature(ax, events0, events1, column, name):
    bins = histogram_range(events0, events1, column)
    plot_histogram_band(ax, event_histograms(events0, column, bins), bins, LABEL0)
    plot_histogram_band(ax, event_histograms(events1, column, bins), bins, LABEL1)
    ax.set_xlabel(FEATURE_LABELS[name])
    ax.set_ylabel("per-event density")
    ax.grid(alpha=0.2, linewidth=0.5)


def plot_phi(ax, events0, events1, columns):
    bins = np.linspace(-np.pi, np.pi, N_BINS + 1)
    plot_histogram_band(ax, phi_histograms(events0, columns, bins), bins, LABEL0)
    plot_histogram_band(ax, phi_histograms(events1, columns, bins), bins, LABEL1)
    ax.axhline(1 / (2 * np.pi), color="black", linestyle=":", linewidth=1)
    ax.set_xlabel(FEATURE_LABELS["phi"])
    ax.set_ylabel("per-event density")
    ax.grid(alpha=0.2, linewidth=0.5)


def plot_particle_categories(ax, events0, events1, columns):
    categories = lc.PDG_ONEHOT
    labels = [r"$\gamma$", "neutron", r"$e^\pm$", r"$\mu^\pm$", "other"]
    positions = np.arange(len(categories))
    offsets = {LABEL0: -0.18, LABEL1: 0.18}

    for event_label, events in ((LABEL0, events0), (LABEL1, events1)):
        fractions = np.asarray([
            [np.mean(event[:, columns[name]]) for name in categories]
            for event in events
        ])
        low, middle, high = np.percentile(fractions, [16, 50, 84], axis=0)
        ax.errorbar(
            positions + offsets[event_label], middle,
            yerr=np.vstack((middle - low, high - middle)), fmt="o",
            color=COLORS[event_label], capsize=3, label=event_label)

    ax.set_xticks(positions, labels)
    ax.set_ylabel("fraction per event")
    ax.set_xlabel("particle category indicators")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)


def plot_event_values(ax, first, second, ylabel):
    values = [first, second]
    boxes = ax.boxplot(values, positions=[0, 1], widths=0.45, patch_artist=True,
                       showfliers=False)
    for patch, label in zip(boxes["boxes"], (LABEL0, LABEL1)):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.20)
        patch.set_edgecolor(COLORS[label])
    rng = np.random.default_rng(DATA_SEED)
    for position, event_values, label in zip(
            (0, 1), values, (LABEL0, LABEL1)):
        jitter = rng.uniform(-0.10, 0.10, len(event_values))
        ax.scatter(
            position + jitter, event_values, s=18, alpha=0.75,
            color=COLORS[label])
    ax.set_xticks([0, 1], [LABEL0, LABEL1], rotation=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)


def main():
    norm1, norm42, unique, reused = build_samplers()
    rng = np.random.default_rng(DATA_SEED)
    events0, multiplicities0, unique_fractions0 = sample_events(
        norm1, unique, rng)
    events1, multiplicities1, unique_fractions1 = sample_events(
        norm42, reused, rng)

    names = lc.feature_names(FEATURE_SET)
    columns = {name: index for index, name in enumerate(names)}

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    for ax, name in zip(axes.flat, CONTINUOUS_FEATURES):
        plot_feature(ax, events0, events1, columns[name], name)

    plot_phi(axes.flat[8], events0, events1, columns)
    plot_particle_categories(axes.flat[9], events0, events1, columns)
    plot_event_values(
        axes.flat[10], multiplicities0, multiplicities1,
        "particles per pseudo-event")
    plot_event_values(
        axes.flat[11], unique_fractions0, unique_fractions1,
        "unique invariant signatures / particles")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("GEN PFN inputs and reuse fluctuations at equal N=420 BIB",
                 fontsize=16)
    fig.text(
        0.5, 0.01,
        f"Median and 16-84% range over {N_EXAMPLES} pseudo-events per class; "
        f"unique: {unique.files_per_unit} source cycles; "
        f"reused: {reused.files_per_unit} source cycles x "
        f"{CLONE_FACTOR} rotations",
        ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.945))

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)

    print(f"{LABEL0}: {multiplicities0.sum():,} particles")
    print(f"{LABEL1}: {multiplicities1.sum():,} particles")
    print(
        f"median unique-signature fraction: "
        f"{LABEL0}={np.median(unique_fractions0):.6f}, "
        f"{LABEL1}={np.median(unique_fractions1):.6f}")
    print(f"chart -> {OUT_PNG}")


if __name__ == "__main__":
    main()
