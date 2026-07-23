#!/usr/bin/env python3
"""Convert a directory of GEN edm4hep ROOT files into one particle store.

One store = one (library, polarity). Reads every *.root file, flattens all
MCParticles across events, and writes flat float32 arrays + per-file offsets
so units can be assembled by file index at training time.

Run inside the mucoll-inspect env (needs uproot, awkward, h5py, numpy):

    python gen_libtest_make_store.py \
        --input-dir  $DATA_GROUP_DIR/bib-v3p0-fmt2-norm1/GEN/MUPLUS \
        --output     $PSCRATCH/mucoll/libtest/stores/gen_norm1_MUPLUS.h5 \
        --workers 64

Cycle id = last integer in the file basename (pairs norm1/norm42 files).
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import h5py
import numpy as np

FLOAT_KEYS = ["px", "py", "pz", "E", "t", "vx", "vy", "vz", "charge"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def sorted_by_cycle(paths):
    """Sort paths by cycle id (the varying integer token in the basename)."""
    import libtest_common as lc
    ids, pos, distinct = lc.assign_cycle_ids(paths)
    print(f"cycle id = integer token #{pos} from filename end"
          f" ({distinct} distinct over {len(paths)} files)")
    print(f"  e.g. {os.path.basename(paths[0])} -> {ids[0]}")
    if distinct != len(paths):
        raise SystemExit("cycle ids are not unique across files -- naming"
                         " convention ambiguous; inspect filenames and fix"
                         " assign_cycle_ids selection")
    order = np.argsort(ids)
    return [paths[i] for i in order], [ids[i] for i in order]


def detect_collection(tree):
    candidates = [k[:-len(".PDG")] for k in tree.keys() if k.endswith(".PDG")]
    if not candidates:
        raise RuntimeError(f"no branch ending in .PDG among: {list(tree.keys())[:20]}")
    return min(candidates, key=len)


def read_file(task):
    """Worker: (path, collection) -> (collection, dict of flat arrays)."""
    path, collection = task
    import awkward as ak
    import uproot

    with uproot.open(path) as f:
        tree_names = [k.split(";")[0] for k in f.keys()
                      if hasattr(f[k], "keys") and f[k].classname.startswith("TTree")]
        if "events" in tree_names:
            tree = f["events"]
        elif tree_names:
            tree = f[tree_names[0]]
        else:
            raise RuntimeError(f"no TTree in {path}")

        if not collection:
            collection = detect_collection(tree)

        branches = {
            "pdg": f"{collection}.PDG",
            "px": f"{collection}.momentum.x",
            "py": f"{collection}.momentum.y",
            "pz": f"{collection}.momentum.z",
            "t": f"{collection}.time",
            "vx": f"{collection}.vertex.x",
            "vy": f"{collection}.vertex.y",
            "vz": f"{collection}.vertex.z",
            "charge": f"{collection}.charge",
        }
        mass_branch = f"{collection}.mass"
        want = list(branches.values())
        if mass_branch in tree.keys():
            want.append(mass_branch)
        arrays = tree.arrays(want, library="ak")

    out = {}
    for key, br in branches.items():
        flat = ak.to_numpy(ak.flatten(arrays[br], axis=None))
        out[key] = flat.astype(np.int32 if key == "pdg" else np.float32)
    p2 = out["px"].astype(np.float64) ** 2 + out["py"].astype(np.float64) ** 2 \
        + out["pz"].astype(np.float64) ** 2
    if mass_branch in want:
        m = ak.to_numpy(ak.flatten(arrays[mass_branch], axis=None)).astype(np.float64)
        out["E"] = np.sqrt(p2 + m ** 2).astype(np.float32)
    else:
        out["E"] = np.sqrt(p2).astype(np.float32)
    return collection, out


def main():
    args = parse_args()
    files = glob.glob(os.path.join(args.input_dir, "*.root"))
    if not files:
        sys.exit(f"no *.root files in {args.input_dir}")
    files, cycles = sorted_by_cycle(files)
    print(f"{len(files)} files from {args.input_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tasks = [(p, "") for p in files]

    start = time.time()
    with h5py.File(args.output, "w") as out:
        grp = out.create_group("particles")
        dsets = {}
        for key in FLOAT_KEYS + ["pdg"]:
            dtype = np.int32 if key == "pdg" else np.float32
            dsets[key] = grp.create_dataset(
                key, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(1 << 20)
            )
        offsets = [0]
        collection_seen = None

        with mp.Pool(args.workers) as pool:
            for i, (coll, raw) in enumerate(pool.imap(read_file, tasks)):
                collection_seen = collection_seen or coll
                n = len(raw["pdg"])
                for key, dset in dsets.items():
                    dset.resize((offsets[-1] + n,))
                    dset[offsets[-1]:] = raw[key]
                offsets.append(offsets[-1] + n)
                if (i + 1) % 250 == 0 or i + 1 == len(files):
                    rate = (i + 1) / (time.time() - start)
                    print(f"  {i + 1}/{len(files)} files, {offsets[-1]:,} particles,"
                          f" {rate:.1f} files/s", flush=True)

        out.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
        out.create_dataset("cycle_ids", data=np.asarray(cycles, dtype=np.int64))
        out.create_dataset(
            "filenames",
            data=np.asarray([os.path.basename(p) for p in files], dtype=object),
            dtype=h5py.string_dtype(),
        )
        out.attrs["input_dir"] = args.input_dir
        out.attrs["collection"] = collection_seen or ""

    counts = np.diff(np.asarray(offsets))
    print(f"done: {len(files)} files, {offsets[-1]:,} particles "
          f"(median {np.median(counts):.0f}/file) -> {args.output}")


if __name__ == "__main__":
    main()
