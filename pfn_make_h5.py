#!/usr/bin/env python3

import argparse
import glob
import os
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot

from ml_common import PFO_FEATURES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--class-name", required=True)
    return parser.parse_args()


def find_root_files(inputs):
    files = []
    for item in inputs:
        matches = glob.glob(item) or [item]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                files.extend(path.glob("job_*/reco_output_*.edm4hep.root"))
            elif path.is_file():
                files.append(path)
    return sorted({path.resolve() for path in files})


def pfo_branch(pfos, name, like, default=0):
    key = f"PandoraPFOs.{name}"
    try:
        return pfos[key].array()
    except Exception:
        pass
    return ak.zeros_like(like) + default


def pfo_vectors(pt, px, py, pz, energy, mass, charge, pfo_type):
    pt = ak.to_numpy(pt).astype(np.float32)
    px = ak.to_numpy(px).astype(np.float32)
    py = ak.to_numpy(py).astype(np.float32)
    pz = ak.to_numpy(pz).astype(np.float32)
    energy = ak.to_numpy(energy).astype(np.float32)
    mass = ak.to_numpy(mass).astype(np.float32)
    charge = ak.to_numpy(charge).astype(np.float32)
    pfo_type = ak.to_numpy(pfo_type).astype(np.float32)

    good = np.isfinite(pt) & np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)
    good &= np.isfinite(energy) & np.isfinite(mass)
    good &= pt > 0
    pt = pt[good]
    px = px[good]
    py = py[good]
    pz = pz[good]
    energy = energy[good]
    mass = mass[good]
    charge = charge[good]
    pfo_type = pfo_type[good]

    if len(pt) == 0:
        return np.zeros((0, len(PFO_FEATURES)), dtype=np.float32)

    eta = np.arcsinh(pz / np.maximum(pt, 1e-12))
    phi = np.arctan2(py, px)
    vectors = np.stack(
        [pt, eta, phi, energy, mass, charge, pfo_type, px, py, pz],
        axis=1,
    ).astype(np.float32)
    return vectors[np.argsort(vectors[:, 0])[::-1]]


def read_root_file(path):
    events = uproot.open(path)["events"]
    pfos = events["PandoraPFOs"]

    px = pfos["PandoraPFOs.momentum.x"].array()
    py = pfos["PandoraPFOs.momentum.y"].array()
    pz = pfos["PandoraPFOs.momentum.z"].array()
    pt = np.sqrt(px * px + py * py)
    energy = pfo_branch(pfos, "energy", pt)
    mass = pfo_branch(pfos, "mass", pt)
    charge = pfo_branch(pfos, "charge", pt)
    pfo_type = pfo_branch(pfos, "type", pt)

    vectors = []
    source_files = []
    source_events = []
    for i in range(events.num_entries):
        vectors.append(
            pfo_vectors(
                pt[i],
                px[i],
                py[i],
                pz[i],
                energy[i],
                mass[i],
                charge[i],
                pfo_type[i],
            )
        )
        source_files.append(str(path))
        source_events.append(i)

    return vectors, source_files, source_events


def write_store(inputs, output, class_name):
    root_files = find_root_files(inputs)
    if not root_files:
        raise SystemExit("No reco ROOT files found")

    all_events = []
    source_files = []
    source_events = []
    for path in root_files:
        events, files, event_numbers = read_root_file(path)
        all_events.extend(events)
        source_files.extend(files)
        source_events.extend(event_numbers)
        print(f"{path}: {len(events)} events")

    width = max(max((len(event) for event in all_events), default=0), 1)
    particles = np.zeros((len(all_events), width, len(PFO_FEATURES)), dtype=np.float32)
    n_particles = np.zeros(len(all_events), dtype=np.int32)

    for i, event in enumerate(all_events):
        n_particles[i] = len(event)
        particles[i, :len(event), :] = event

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
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

    print(f"Wrote {output}")
    print(f"Events: {len(all_events)}")
    print(f"PFO slots: {width}")
    print(f"Mean PFOs/event: {np.mean(n_particles):.2f}")


def main():
    args = parse_args()
    write_store(args.inputs, args.out, args.class_name)


if __name__ == "__main__":
    main()
