#!/usr/bin/env python3
"""Convert split-by-mother GEN EDM4hep files into a mother-indexed HDF5 bank."""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import h5py
import numpy as np

import libtest_common as lc


FLOAT_KEYS = ("px", "py", "pz", "E", "t", "vx", "vy", "vz")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--exclude-cycle", type=int, action="append", default=[])
    return parser.parse_args()


def detect_collection(tree):
    candidates = [key[:-len(".PDG")] for key in tree.keys()
                  if key.endswith(".PDG")]
    if not candidates:
        raise RuntimeError("no MCParticle PDG branch found")
    return min(candidates, key=len)


def read_file(task):
    path, collection = task
    import awkward as ak
    import uproot

    with uproot.open(path) as handle:
        tree = handle["events"]
        collection = collection or detect_collection(tree)
        branches = {
            "pdg": collection + ".PDG",
            "px": collection + ".momentum.x",
            "py": collection + ".momentum.y",
            "pz": collection + ".momentum.z",
            "t": collection + ".time",
            "vx": collection + ".vertex.x",
            "vy": collection + ".vertex.y",
            "vz": collection + ".vertex.z",
        }
        mass_branch = collection + ".mass"
        wanted = list(branches.values())
        if mass_branch in tree.keys():
            wanted.append(mass_branch)
        arrays = tree.arrays(wanted, library="ak")

    counts = ak.to_numpy(ak.num(arrays[branches["pdg"]])).astype(np.int64)
    output = {}
    for key, branch in branches.items():
        values = ak.to_numpy(ak.flatten(arrays[branch], axis=None))
        output[key] = values.astype(np.int32 if key == "pdg" else np.float32)
    momentum2 = (output["px"].astype(np.float64) ** 2 +
                 output["py"].astype(np.float64) ** 2 +
                 output["pz"].astype(np.float64) ** 2)
    if mass_branch in wanted:
        mass = ak.to_numpy(ak.flatten(arrays[mass_branch], axis=None))
        output["E"] = np.sqrt(momentum2 + mass.astype(np.float64) ** 2).astype(np.float32)
    else:
        output["E"] = np.sqrt(momentum2).astype(np.float32)
    if int(counts.sum()) != len(output["pdg"]):
        raise RuntimeError("event counts do not match flattened particles in {}".format(path))
    return collection, counts, output


def main():
    args = parse_args()
    paths = glob.glob(os.path.join(args.input_dir, "*.root"))
    if not paths:
        sys.exit("no ROOT files in {}".format(args.input_dir))
    cycle_ids, token_position, distinct = lc.assign_cycle_ids(paths)
    if distinct != len(paths):
        raise SystemExit("could not assign a unique cycle ID to every file")
    pairs = sorted(zip(cycle_ids, paths))
    excluded = set(args.exclude_cycle)
    pairs = [(cycle, path) for cycle, path in pairs if cycle not in excluded]
    cycles = [pair[0] for pair in pairs]
    paths = [pair[1] for pair in pairs]
    print("{} cycles; cycle token #{} from filename end; excluded {}".format(
        len(paths), token_position, sorted(excluded)))

    output_parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_parent, exist_ok=True)
    tasks = [(path, "") for path in paths]
    start = time.time()

    with h5py.File(args.output, "w") as output:
        particle_group = output.create_group("particles")
        datasets = {}
        for key in FLOAT_KEYS + ("pdg",):
            dtype = np.int32 if key == "pdg" else np.float32
            datasets[key] = particle_group.create_dataset(
                key, shape=(0,), maxshape=(None,), dtype=dtype,
                chunks=(1 << 20))

        mother_offsets = [0]
        cycle_offsets = [0]
        mother_cycle_ids = []
        mother_local_ids = []
        collection_seen = None

        with mp.Pool(args.workers) as pool:
            iterator = pool.imap(read_file, tasks)
            for position, (collection, counts, raw) in enumerate(iterator):
                collection_seen = collection_seen or collection
                particle_start = mother_offsets[-1]
                n_particles = len(raw["pdg"])
                for key, dataset in datasets.items():
                    dataset.resize((particle_start + n_particles,))
                    dataset[particle_start:] = raw[key]
                mother_offsets.extend((particle_start + np.cumsum(counts)).tolist())
                n_mothers = len(counts)
                mother_cycle_ids.extend([cycles[position]] * n_mothers)
                mother_local_ids.extend(range(n_mothers))
                cycle_offsets.append(cycle_offsets[-1] + n_mothers)
                if ((position + 1) % 250 == 0 or position + 1 == len(paths)):
                    rate = (position + 1) / (time.time() - start)
                    print("  {}/{} cycles, {:,} mothers, {:,} particles, {:.1f} files/s"
                          .format(position + 1, len(paths), cycle_offsets[-1],
                                  mother_offsets[-1], rate), flush=True)

        output.create_dataset("mother_offsets", data=np.asarray(mother_offsets, np.int64))
        output.create_dataset("mother_cycle_ids", data=np.asarray(mother_cycle_ids, np.int64))
        output.create_dataset("mother_local_ids", data=np.asarray(mother_local_ids, np.int32))
        output.create_dataset("cycle_ids", data=np.asarray(cycles, np.int64))
        output.create_dataset("cycle_offsets", data=np.asarray(cycle_offsets, np.int64))
        output.create_dataset(
            "filenames", data=np.asarray([os.path.basename(path) for path in paths], dtype=object),
            dtype=h5py.string_dtype())
        output.attrs["input_dir"] = args.input_dir
        output.attrs["collection"] = collection_seen or ""
        output.attrs["schema"] = "split-mother-gen-v1"

    counts = np.diff(np.asarray(mother_offsets))
    print("done: {} cycles, {:,} mothers, {:,} particles; median {:.0f} particles/mother -> {}"
          .format(len(paths), len(counts), mother_offsets[-1],
                  np.median(counts), args.output))


if __name__ == "__main__":
    main()
