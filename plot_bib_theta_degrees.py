#!/usr/bin/env python3

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

N_FILES = 420
CLONE_FACTOR = 42
SPLIT_FRACS = (0.50, 0.25, 0.25)
DATA_SEED = 1701
N_EXAMPLES = 25
BINS = np.linspace(0, 180, 81)
OUTPUT = "plots/bib_example_unit_theta_degrees"


def build_samplers():
    norm1 = lc.Store(NORM1_STORE)
    norm42 = lc.Store(NORM42_STORE)
    common, norm1_positions, norm42_positions = lc.common_positions(
        norm1, norm42)
    cycle_split = lc.create_cycle_split(common, SPLIT_FRACS, DATA_SEED)
    train = lc.cycle_split_positions(common, cycle_split)["train"]

    unique = UnitSampler(
        norm1, {"train": norm1_positions[train]}, N_FILES, "paper")
    reused = UnitSampler(
        norm42, {"train": norm42_positions[train]},
        N_FILES // CLONE_FACTOR, "paper")
    return norm1, norm42, unique, reused


def theta_histogram(store, sampler, rng, label):
    counts = np.zeros(len(BINS) - 1, dtype=np.int64)
    particles = 0
    for number in range(1, N_EXAMPLES + 1):
        positions = sampler.random_unit(rng, "train")
        raw = store.file_arrays(positions)
        pt = np.hypot(raw["px"], raw["py"])
        theta = np.degrees(np.arctan2(pt, raw["pz"]))
        counts += np.histogram(theta, bins=BINS)[0]
        particles += len(theta)
        print(f"{label}: {number}/{N_EXAMPLES}", flush=True)
    density = counts / (particles * np.diff(BINS))
    return density, particles


def main():
    norm1, norm42, unique, reused = build_samplers()
    rng = np.random.default_rng(DATA_SEED)
    unique_density, unique_particles = theta_histogram(
        norm1, unique, rng, "unique")
    reused_density, reused_particles = theta_histogram(
        norm42, reused, rng, "reused")

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.stairs(
        unique_density, BINS, linewidth=1.8, color="#0072B2",
        label="unique mothers")
    axis.stairs(
        reused_density, BINS, linewidth=1.8, color="#D55E00",
        linestyle="--", label="42x within-event reuse")
    axis.set_xlabel(r"Polar angle $\theta$ [deg]")
    axis.set_ylabel("density")
    axis.set_xlim(0, 180)
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.legend(frameon=False)
    figure.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    figure.savefig(f"{OUTPUT}.pdf")
    figure.savefig(f"{OUTPUT}.png", dpi=180)
    plt.close(figure)

    print(f"unique particles: {unique_particles:,}")
    print(f"reused particles: {reused_particles:,}")
    print(f"charts -> {OUTPUT}.pdf, {OUTPUT}.png")


if __name__ == "__main__":
    main()
