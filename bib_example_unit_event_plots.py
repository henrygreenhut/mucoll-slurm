#!/usr/bin/env python3

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import libtest_common as lc


NORM1_STORE = os.path.expanduser(
    "~/mucoll/stores/gen_norm1_reconstructed_MUPLUS.h5")
NORM42_STORE = (
    f"/oscar/scratch/{os.environ.get('USER', '')}/mucoll/stores/"
    "gen_norm42_MUPLUS.h5")

N_FILES = 420
CLONE_FACTOR = 42
SPLIT_FRACS = (0.50, 0.25, 0.25)
DATA_SEED = 1701
N_EVENTS = 100
ENERGY_THRESHOLD = 10.0
OUT = "plots/bib_example_unit_event_plots"

UNIQUE = "unique mothers"
REUSED = "42x within-event reuse"
COLORS = {UNIQUE: "#0072B2", REUSED: "#D55E00"}


def training_positions(unique_store, reused_store):
    common, unique_positions, reused_positions = lc.common_positions(
        unique_store, reused_store)
    split = lc.create_cycle_split(common, SPLIT_FRACS, DATA_SEED)
    train = lc.cycle_split_positions(common, split)["train"]
    return unique_positions[train], reused_positions[train]


def sample_events(store, positions, files_per_event, rng):
    values = {
        "muons": [],
        "high_energy": [],
        "high_energy_sum": [],
        "multiplicity": [],
    }

    for _ in range(N_EVENTS):
        selected = lc.sample_unit_positions(rng, positions, files_per_event)
        raw = store.file_arrays(selected)
        energy = raw["E"]
        high_energy = energy > ENERGY_THRESHOLD

        values["muons"].append(np.count_nonzero(np.abs(raw["pdg"]) == 13))
        values["high_energy"].append(np.count_nonzero(high_energy))
        values["high_energy_sum"].append(np.sum(energy[high_energy]))
        values["multiplicity"].append(len(energy))

    return {name: np.asarray(array) for name, array in values.items()}


def count_bins(first, second):
    maximum = max(np.max(first), np.max(second))
    width = max(1, int(np.ceil(maximum / 36)))
    return np.arange(-0.5 * width, maximum + 1.5 * width, width)


def draw(ax, first, second, bins, xlabel):
    for values, label in ((first, UNIQUE), (second, REUSED)):
        weights = np.ones(len(values)) / len(values)
        ax.hist(
            values,
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=2,
            color=COLORS[label],
            label=label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of pseudo-events")
    ax.grid(alpha=0.2, linewidth=0.5)


def report(label, values):
    print(label)
    for name, array in values.items():
        print(
            f"  {name:16s} mean={np.mean(array):.3f} "
            f"std={np.std(array):.3f} median={np.median(array):.3f} "
            f"min={np.min(array):.3f} max={np.max(array):.3f}"
        )
    print(
        f"  fraction with no E > {ENERGY_THRESHOLD:g} GeV: "
        f"{np.mean(values['high_energy'] == 0):.3f}"
    )


def main():
    unique_store = lc.Store(NORM1_STORE)
    reused_store = lc.Store(NORM42_STORE)
    unique_positions, reused_positions = training_positions(
        unique_store, reused_store)

    unique = sample_events(
        unique_store, unique_positions, N_FILES,
        np.random.default_rng(DATA_SEED))
    reused = sample_events(
        reused_store, reused_positions, N_FILES // CLONE_FACTOR,
        np.random.default_rng(DATA_SEED + 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    draw(
        axes[0, 0], unique["muons"], reused["muons"],
        count_bins(unique["muons"], reused["muons"]),
        r"generator muons per pseudo-event, $|\mathrm{PDG}|=13$")
    draw(
        axes[0, 1], unique["high_energy"], reused["high_energy"],
        count_bins(unique["high_energy"], reused["high_energy"]),
        rf"particles with $E>{ENERGY_THRESHOLD:g}$ GeV per pseudo-event")

    unique_tail = np.log10(1 + unique["high_energy_sum"])
    reused_tail = np.log10(1 + reused["high_energy_sum"])
    tail_bins = np.linspace(
        min(np.min(unique_tail), np.min(reused_tail)),
        max(np.max(unique_tail), np.max(reused_tail)), 32)
    draw(
        axes[1, 0], unique_tail, reused_tail, tail_bins,
        rf"$\log_{{10}}[1+\sum_{{E>{ENERGY_THRESHOLD:g}\,\mathrm{{GeV}}}} "
        r"E/\mathrm{GeV}]$")

    unique_multiplicity = unique["multiplicity"] / 1e6
    reused_multiplicity = reused["multiplicity"] / 1e6
    multiplicity_bins = np.linspace(
        min(np.min(unique_multiplicity), np.min(reused_multiplicity)),
        max(np.max(unique_multiplicity), np.max(reused_multiplicity)), 32)
    draw(
        axes[1, 1], unique_multiplicity, reused_multiplicity,
        multiplicity_bins, "particles per pseudo-event [millions]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Event-level GEN observables at equal N=420", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=180)
    fig.savefig(OUT + ".pdf")
    plt.close(fig)

    report(UNIQUE, unique)
    report(REUSED, reused)
    print(f"plots -> {OUT}.png and {OUT}.pdf")


if __name__ == "__main__":
    main()
