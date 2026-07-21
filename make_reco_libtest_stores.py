#!/usr/bin/env python3
"""Convert reconstructed BIB events into one PFN store per sample and split."""

import argparse
import glob
import os
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot


SAMPLES = ("U", "R", "null_b")
SPLITS = ("train", "val", "test")
N_FILES = 420
PFO_FEATURES = (
    "pt", "eta", "phi", "energy", "mass", "charge", "pdg", "px", "py", "pz",
)


def parse_args():
    scratch = os.environ.get("PSCRATCH", "")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reco-dir", default=(scratch + "/mucoll/libtest/reco_n420_pfn_simple")
                        if scratch else None, required=not bool(scratch))
    parser.add_argument("--outdir", default=(scratch + "/mucoll/libtest/reco_n420_pfn_stores_simple")
                        if scratch else None, required=not bool(scratch))
    return parser.parse_args()


def find_root_files(directory):
    pattern = str(Path(directory) / "job_*" / "reco_output_*.edm4hep.root")
    return [Path(path).resolve() for path in sorted(glob.glob(pattern))]


def read_root_file(path):
    events = uproot.open(path)["events"]
    pfos = events["PandoraPFOs"]
    px = pfos["PandoraPFOs.momentum.x"].array()
    py = pfos["PandoraPFOs.momentum.y"].array()
    pz = pfos["PandoraPFOs.momentum.z"].array()
    pt = np.sqrt(px * px + py * py)
    energy = pfos["PandoraPFOs.energy"].array()
    mass = pfos["PandoraPFOs.mass"].array()
    charge = pfos["PandoraPFOs.charge"].array()
    pdg = pfos["PandoraPFOs.PDG"].array()

    vectors = []
    for i in range(events.num_entries):
        columns = [pt[i], px[i], py[i], pz[i], energy[i], mass[i], charge[i], pdg[i]]
        pt_i, px_i, py_i, pz_i, energy_i, mass_i, charge_i, pdg_i = [
            ak.to_numpy(column).astype(np.float32) for column in columns
        ]
        good = (
            np.isfinite(pt_i) & np.isfinite(px_i) & np.isfinite(py_i)
            & np.isfinite(pz_i) & np.isfinite(energy_i) & np.isfinite(mass_i)
            & (pt_i > 0)
        )
        if not np.any(good):
            vectors.append(np.zeros((0, len(PFO_FEATURES)), dtype=np.float32))
            continue
        pt_i, px_i, py_i, pz_i, energy_i, mass_i, charge_i, pdg_i = [
            column[good] for column in
            (pt_i, px_i, py_i, pz_i, energy_i, mass_i, charge_i, pdg_i)
        ]
        event = np.stack(
            [
                pt_i,
                np.arcsinh(pz_i / np.maximum(pt_i, 1e-12)),
                np.arctan2(py_i, px_i),
                energy_i,
                mass_i,
                charge_i,
                pdg_i,
                px_i,
                py_i,
                pz_i,
            ],
            axis=1,
        ).astype(np.float32)
        vectors.append(event[np.argsort(event[:, 0])[::-1]])
    return vectors


def write_store(directory, output, class_name):
    root_files = find_root_files(directory)
    if not root_files:
        raise SystemExit("No RECO ROOT files found in {}".format(directory))

    events = []
    source_files = []
    source_events = []
    for path in root_files:
        file_events = read_root_file(path)
        events.extend(file_events)
        source_files.extend([str(path)] * len(file_events))
        source_events.extend(range(len(file_events)))

    width = max(max((len(event) for event in events), default=0), 1)
    particles = np.zeros((len(events), width, len(PFO_FEATURES)), dtype=np.float32)
    n_particles = np.asarray([len(event) for event in events], dtype=np.int32)
    for i, event in enumerate(events):
        particles[i, :len(event)] = event

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h5:
        h5.create_dataset("particles", data=particles, compression="gzip")
        h5.create_dataset("n_particles", data=n_particles)
        h5.create_dataset("source_event", data=np.asarray(source_events, dtype=np.int32))
        h5.create_dataset(
            "source_file",
            data=np.asarray(source_files, dtype=h5py.string_dtype("utf-8")),
        )
        h5.attrs["class_name"] = class_name
        h5.attrs["features"] = ",".join(PFO_FEATURES)
        h5.attrs["collection"] = "PandoraPFOs"
    print("{}: {} events, mean {:.2f} PFOs/event -> {}".format(
        class_name, len(events), np.mean(n_particles), output
    ))


def main():
    args = parse_args()
    reco_dir = Path(args.reco_dir).resolve()
    outdir = Path(args.outdir).resolve()
    for sample in SAMPLES:
        for split in SPLITS:
            source = reco_dir / "reco_libtest_n{}_{}".format(N_FILES, sample) / split
            output = outdir / "n{}_{}_{}.h5".format(N_FILES, sample, split)
            print("\n{} / {}".format(sample, split))
            write_store(source, output, sample)


if __name__ == "__main__":
    main()
