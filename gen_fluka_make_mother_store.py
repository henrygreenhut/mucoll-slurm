#!/usr/bin/env python3
"""Build the variable-reuse mother bank directly from FLUKA format-2 files.

Every record in a format-2 file is a detector-bound particle. Records with
the same (x_mu, y_mu, z_mu) decay position came from the same beam-muon
decay, so that exact triple defines one mother event.
"""

import argparse
import os
import shlex
import sys

import h5py
import numpy as np


FLOAT_KEYS = ("px", "py", "pz", "E", "t", "vx", "vy", "vz")
FLUKA_FORMAT_2 = np.dtype([
    ("fid", np.int32),
    ("fid_mo", np.int32),
    ("e_kin", np.float64),
    ("x", np.float64),
    ("y", np.float64),
    ("z", np.float64),
    ("cx", np.float64),
    ("cy", np.float64),
    ("cz", np.float64),
    ("time", np.float64),
    ("x_mu", np.float64),
    ("y_mu", np.float64),
    ("z_mu", np.float64),
])


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--fluka-dir", required=True)
    parser.add_argument("--tasklist", required=True,
                        help="original GEN tasklist, used to preserve cycle IDs")
    parser.add_argument("--benchmarks-dir",
                        default=os.path.join(here, "..", "mucoll-benchmarks"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--polarity", choices=("MUPLUS", "MUMINUS"),
                        default="MUPLUS")
    parser.add_argument("--exclude-cycle", type=int, action="append", default=[])
    return parser.parse_args()


def particle_tables(benchmarks_dir):
    bib_dir = os.path.join(
        os.path.abspath(benchmarks_dir), "generation", "bib")
    if not os.path.isdir(bib_dir):
        raise RuntimeError("missing mucoll-benchmarks BIB directory: {}".format(
            bib_dir))
    sys.path.insert(0, bib_dir)
    from bib_pdgs import FLUKA_PIDS, PDG_PROPS
    return FLUKA_PIDS, PDG_PROPS


def cycle_sources(tasklist, fluka_dir, polarity):
    """Return [(cycle_id, local_FLUKA_path)] from the production tasklist."""
    found = {}
    marker = "/GEN/{}/bib_gen_".format(polarity)
    with open(tasklist) as handle:
        for line in handle:
            if marker not in line:
                continue
            tokens = shlex.split(line)
            inputs = [token for token in tokens if token.endswith(".dat")]
            outputs = [token for token in tokens
                       if marker in token and token.endswith(".root")]
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError("cannot parse tasklist line: {}".format(line))
            name = os.path.basename(outputs[0])
            prefix = "bib_gen_"
            suffix = ".edm4hep.root"
            if not (name.startswith(prefix) and name.endswith(suffix)):
                raise RuntimeError("unexpected GEN filename: {}".format(name))
            cycle = int(name[len(prefix):-len(suffix)])
            source = os.path.join(fluka_dir, os.path.basename(inputs[0]))
            if cycle in found:
                raise RuntimeError("duplicate cycle {} in tasklist".format(cycle))
            if not os.path.isfile(source):
                raise RuntimeError("missing FLUKA source: {}".format(source))
            found[cycle] = source
    if not found:
        raise RuntimeError("no {} cycles found in {}".format(polarity, tasklist))
    cycles = sorted(found)
    if cycles != list(range(cycles[-1] + 1)):
        raise RuntimeError("tasklist cycle IDs are not contiguous from zero")
    return [(cycle, found[cycle]) for cycle in cycles]


def mother_groups(records):
    """Return stable particle order and counts for exact decay-position groups."""
    decay_positions = records[["x_mu", "y_mu", "z_mu"]]
    _, first, inverse = np.unique(
        decay_positions, return_index=True, return_inverse=True)

    # np.unique sorts keys. Relabel groups by first appearance so the bank
    # retains the natural ordering of the FLUKA stream.
    first_order = np.argsort(first)
    stable_labels = np.empty(len(first_order), dtype=np.int64)
    stable_labels[first_order] = np.arange(len(first_order))
    labels = stable_labels[inverse]
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=len(first_order)).astype(np.int64)
    return order, counts


def convert_records(records, fluka_pdgs, pdg_props, invert_z=False):
    pdg = np.asarray(
        [fluka_pdgs.get(int(fid), 0) for fid in records["fid"]],
        dtype=np.int32)
    known = np.asarray([int(value) in pdg_props for value in pdg])
    if not np.all(known):
        records = records[known]
        pdg = pdg[known]

    order, counts = mother_groups(records)
    records = records[order]
    pdg = pdg[order]
    mass = np.asarray([pdg_props[int(value)][1] for value in pdg],
                      dtype=np.float64)
    e_kin = records["e_kin"]
    momentum = np.sqrt(e_kin * e_kin + 2.0 * e_kin * mass)
    z_sign = -1.0 if invert_z else 1.0

    output = {
        "pdg": pdg,
        "px": (records["cx"] * momentum).astype(np.float32),
        "py": (records["cy"] * momentum).astype(np.float32),
        "pz": (z_sign * records["cz"] * momentum).astype(np.float32),
        "E": (e_kin + mass).astype(np.float32),
        "t": (records["time"] * 1.0e9).astype(np.float32),
        "vx": (records["x"] * 10.0).astype(np.float32),
        "vy": (records["y"] * 10.0).astype(np.float32),
        "vz": (z_sign * records["z"] * 10.0).astype(np.float32),
    }
    return counts, output, int((~known).sum())


def main():
    args = parse_args()
    fluka_pdgs, pdg_props = particle_tables(args.benchmarks_dir)
    sources = cycle_sources(args.tasklist, args.fluka_dir, args.polarity)
    excluded = set(args.exclude_cycle)
    sources = [(cycle, path) for cycle, path in sources
               if cycle not in excluded]
    invert_z = args.polarity == "MUMINUS"

    output_parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_parent, exist_ok=True)
    mother_offsets = [0]
    cycle_offsets = [0]
    mother_cycle_ids = []
    mother_local_ids = []
    skipped_particles = 0

    with h5py.File(args.output, "w") as output:
        particle_group = output.create_group("particles")
        datasets = {}
        for key in FLOAT_KEYS + ("pdg",):
            dtype = np.int32 if key == "pdg" else np.float32
            datasets[key] = particle_group.create_dataset(
                key, shape=(0,), maxshape=(None,), dtype=dtype,
                chunks=(1 << 20))

        for position, (cycle, path) in enumerate(sources):
            size = os.path.getsize(path)
            if size % FLUKA_FORMAT_2.itemsize:
                raise RuntimeError(
                    "{} has {} bytes, not a whole number of {}-byte "
                    "FLUKA format-2 records".format(
                        path, size, FLUKA_FORMAT_2.itemsize))
            records = np.fromfile(path, dtype=FLUKA_FORMAT_2)
            counts, raw, skipped = convert_records(
                records, fluka_pdgs, pdg_props, invert_z)
            skipped_particles += skipped

            particle_start = mother_offsets[-1]
            n_particles = len(raw["pdg"])
            for key, dataset in datasets.items():
                dataset.resize((particle_start + n_particles,))
                dataset[particle_start:] = raw[key]
            mother_offsets.extend(
                (particle_start + np.cumsum(counts)).tolist())
            n_mothers = len(counts)
            mother_cycle_ids.extend([cycle] * n_mothers)
            mother_local_ids.extend(range(n_mothers))
            cycle_offsets.append(cycle_offsets[-1] + n_mothers)

            if (position + 1) % 250 == 0 or position + 1 == len(sources):
                print("  {}/{} cycles, {:,} mothers, {:,} particles".format(
                    position + 1, len(sources), cycle_offsets[-1],
                    mother_offsets[-1]), flush=True)

        cycles = [cycle for cycle, _ in sources]
        output.create_dataset(
            "mother_offsets", data=np.asarray(mother_offsets, np.int64))
        output.create_dataset(
            "mother_cycle_ids", data=np.asarray(mother_cycle_ids, np.int64))
        output.create_dataset(
            "mother_local_ids", data=np.asarray(mother_local_ids, np.int32))
        output.create_dataset("cycle_ids", data=np.asarray(cycles, np.int64))
        output.create_dataset(
            "cycle_offsets", data=np.asarray(cycle_offsets, np.int64))
        output.create_dataset(
            "filenames",
            data=np.asarray([os.path.basename(path) for _, path in sources],
                            dtype=object),
            dtype=h5py.string_dtype())
        output.attrs["input_dir"] = os.path.abspath(args.fluka_dir)
        output.attrs["tasklist"] = os.path.abspath(args.tasklist)
        output.attrs["polarity"] = args.polarity
        output.attrs["schema"] = "split-mother-gen-v1"
        output.attrs["source_schema"] = "fluka-format-2-direct-v1"

    counts = np.diff(np.asarray(mother_offsets))
    print("done: {} cycles, {:,} mothers, {:,} particles; "
          "mother particles min/median/max = {}/{:.0f}/{}; skipped {} -> {}"
          .format(len(sources), len(counts), mother_offsets[-1],
                  counts.min(), np.median(counts), counts.max(),
                  skipped_particles, args.output))


if __name__ == "__main__":
    main()
