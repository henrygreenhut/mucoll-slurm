import argparse
import glob
import os
from pathlib import Path

import awkward as ak
import matplotlib
import numpy as np
import uproot

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLITS = ("train", "val", "test")

PFO_BRANCHES = (
    "PandoraPFOs.momentum.x",
    "PandoraPFOs.momentum.y",
    "PandoraPFOs.momentum.z",
    "PandoraPFOs.energy",
    "PandoraPFOs.mass",
)


def get_arguments():
    scratch = os.environ.get("PSCRATCH")
    default_reco_directory = None
    if scratch:
        default_reco_directory = (
            scratch + "/mucoll/libtest/reco_n420_pfn_trackfix"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reco-dir",
        default=default_reco_directory,
        required=default_reco_directory is None,
    )
    parser.add_argument(
        "--output",
        default="plots/reco_n420_trackfix_direct_root_phi.pdf",
    )
    return parser.parse_args()


def find_files(reco_directory, sample):
    sample_directory = reco_directory / ("reco_libtest_n420_" + sample)
    files = []

    for split in SPLITS:
        pattern = str(
            sample_directory
            / split
            / "job_*"
            / "reco_output_*.edm4hep.root"
        )
        split_files = sorted(glob.glob(pattern))
        if not split_files:
            raise SystemExit("No files found: " + pattern)
        files.extend(Path(path) for path in split_files)

    return files


def calculate_phi(px, py, pz, energy, mass):
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    pz = np.asarray(pz, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    pt = np.hypot(px, py)

    valid = (
        np.isfinite(pt)
        & np.isfinite(px)
        & np.isfinite(py)
        & np.isfinite(pz)
        & np.isfinite(energy)
        & np.isfinite(mass)
        & (pt > 0)
    )

    return np.arctan2(py[valid], px[valid])


def read_phi(files):
    all_phi = []
    event_count = 0

    for path in files:
        with uproot.open(path) as root_file:
            events = root_file["events"]
            pfos = events["PandoraPFOs"]
            arrays = pfos.arrays(PFO_BRANCHES, library="ak")

            values = []
            for branch in PFO_BRANCHES:
                flattened = ak.flatten(arrays[branch], axis=None)
                values.append(ak.to_numpy(flattened))

            all_phi.append(calculate_phi(*values))
            event_count += events.num_entries

    return np.concatenate(all_phi), event_count


def make_plot(unique_phi, reused_phi, output):
    lowest_phi = min(unique_phi.min(), reused_phi.min())
    highest_phi = max(unique_phi.max(), reused_phi.max())
    bins = np.linspace(lowest_phi, highest_phi, 51)

    figure, axis = plt.subplots(figsize=(6.2, 4.2))

    axis.hist(
        unique_phi,
        bins=bins,
        weights=np.full(len(unique_phi), 1.0 / len(unique_phi)),
        histtype="step",
        linewidth=1.9,
        color="#0072B2",
        label="Unique mothers (420 unrotated files per polarity)",
    )

    axis.hist(
        reused_phi,
        bins=bins,
        weights=np.full(len(reused_phi), 1.0 / len(reused_phi)),
        histtype="step",
        linewidth=1.9,
        color="#D55E00",
        label=r"Reused mothers (10 files per polarity $\times$ 42)",
    )

    axis.set_xlabel(r"PFO $\phi$")
    axis.set_ylabel("Fraction of PFOs")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)

    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main():
    arguments = get_arguments()
    reco_directory = Path(arguments.reco_dir).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    unique_files = find_files(reco_directory, "U")
    reused_files = find_files(reco_directory, "R")

    unique_phi, unique_events = read_phi(unique_files)
    reused_phi, reused_events = read_phi(reused_files)

    print(
        "Unique: {} files, {} events, {} PFOs".format(
            len(unique_files), unique_events, len(unique_phi)
        )
    )
    print(
        "Reused: {} files, {} events, {} PFOs".format(
            len(reused_files), reused_events, len(reused_phi)
        )
    )

    make_plot(unique_phi, reused_phi, output)
    print("Plot: " + str(output))


if __name__ == "__main__":
    main()
