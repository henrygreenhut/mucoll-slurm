#!/usr/bin/env python3
"""Reconstruct an unrotated (norm1-equivalent) particle store from a
42x-cloned/rotated GEN library, by deduplicating each cycle's particles
back to one representative per rotation group.

Background: as of 2026-07-22, both the plain top-level GEN/ directory and
the labeled bib-v3p0-fmt2-norm42-RandomRot/GEN/ directory under
/oscar/data/mleblan6/mucoll/bib/ were confirmed (via inspect_gen_rotation.py,
run against real files) to contain IDENTICAL, already-cloned content --
every particle belongs to a group of exactly 42 sharing (|p|, theta),
spread across the full phi range. No separately-provided unrotated GEN
library was found anywhere in that tree. This script recovers one.

Dedup key: pdg (exact) + rotation-invariant kinematics, rounded to absorb
floating-point noise from the rotation matrix. A random rotation about the
z-axis (or beam axis) only mixes (px,py) and (vx,vy) -- |p|, theta, vz, t,
and pdg are all exactly preserved, so using all five (not just |p| and
theta) makes accidental cross-particle collisions between GENUINELY
different mothers astronomically unlikely, at the cost of nothing (these
fields are already being read for the store itself).

One representative (the first one encountered, in file order) is kept per
group; the other ~41 rotated copies are discarded. Reports per-batch
group-size statistics throughout, so an anomaly (unexpected group sizes,
suggesting this file's structure doesn't match the validated pattern) is
visible immediately rather than silently producing a corrupted store.

Run inside the mucoll env (needs uproot, awkward, h5py, numpy):

    python gen_libtest_reconstruct_unrotated.py \
        --input-dir /oscar/data/mleblan6/mucoll/bib/bib-v3p0-fmt2-norm42-RandomRot/GEN/MUPLUS \
        --output    $HOME/mucoll/stores/gen_norm1_reconstructed_MUPLUS.h5 \
        --max-files 20 \
        --workers 8

Drop --max-files (or set to 0) for the full run once the sample looks
clean. Output format matches gen_libtest_make_store.py exactly, so it's a
drop-in replacement for gen_norm1_*.h5 in the existing training pipeline.
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import h5py
import numpy as np

FLOAT_KEYS = ["px", "py", "pz", "E", "t", "vx", "vy", "vz"]
DECIMALS = 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-files", type=int, default=0,
                        help="process only the first N files (by cycle "
                             "order) -- for validating the dedup pattern "
                             "on a sample before committing to a full run "
                             "(0 = all files)")
    parser.add_argument("--expected-group-size", type=int, default=42,
                        help="warn if a file's median group size deviates "
                             "from this (default 42, the confirmed value)")
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


def read_and_dedup_file(task):
    """Worker: (path, collection) -> (collection, dict of deduplicated flat
    arrays, group_size_array). group_size_array is one entry per KEPT
    representative, giving the size of the rotation group it came from --
    used for validation reporting, not written to the store."""
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
        }
        mass_branch = f"{collection}.mass"
        want = list(branches.values())
        if mass_branch in tree.keys():
            want.append(mass_branch)
        arrays = tree.arrays(want, library="ak")

    # Read at full (float64) precision for the dedup key -- casting to
    # float32 BEFORE computing p/theta (as the stored-output path does)
    # loses precision right before an arctan2, which can amplify that
    # error enough to push a genuine clone's rounded key across a
    # boundary and falsely split a clean 42-group. Confirmed empirically:
    # diagnose_group_anomalies.py (full float64 throughout) found zero
    # anomalies on files where this function's original float32-first
    # version reported singletons.
    raw64 = {}
    for key, br in branches.items():
        flat = ak.to_numpy(ak.flatten(arrays[br], axis=None))
        raw64[key] = flat.astype(np.int64 if key == "pdg" else np.float64)

    p2 = raw64["px"] ** 2 + raw64["py"] ** 2 + raw64["pz"] ** 2
    p = np.sqrt(p2)
    pt = np.hypot(raw64["px"], raw64["py"])
    theta = np.arctan2(pt, raw64["pz"])

    # Float32 copies for the actual stored output (matches
    # gen_libtest_make_store.py's dtypes) -- built AFTER the key, from the
    # same float64 source, not from a separately-truncated intermediate.
    raw = {key: (val.astype(np.int32) if key == "pdg" else val.astype(np.float32))
          for key, val in raw64.items()}
    if mass_branch in want:
        m = ak.to_numpy(ak.flatten(arrays[mass_branch], axis=None)).astype(np.float64)
        raw["E"] = np.sqrt(p2 + m ** 2).astype(np.float32)
    else:
        raw["E"] = np.sqrt(p2).astype(np.float32)

    # Rotation-invariant dedup key: pdg (exact) + |p|, theta, vz, t
    # (rounded). A pure rotation about the beam axis only touches
    # (px,py)/(vx,vy) -- everything else here is exactly preserved.
    key_arr = np.stack([
        raw64["pdg"].astype(np.float64),
        np.round(p, DECIMALS),
        np.round(theta, DECIMALS),
        np.round(raw64["vz"], DECIMALS),
        np.round(raw64["t"], DECIMALS),
    ], axis=1)
    _, first_idx, group_sizes = np.unique(
        key_arr, axis=0, return_index=True, return_counts=True)
    keep = np.sort(first_idx)  # preserve original file order

    out = {k: v[keep] for k, v in raw.items()}
    out["E"] = raw["E"][keep]
    # group_sizes is aligned to the SORTED unique rows, not to `keep`'s
    # order -- fine, we only need the distribution, not per-row alignment.
    return collection, out, group_sizes


def main():
    args = parse_args()
    files = glob.glob(os.path.join(args.input_dir, "*.root"))
    if not files:
        sys.exit(f"no *.root files in {args.input_dir}")
    files, cycles = sorted_by_cycle(files)
    if args.max_files:
        files = files[:args.max_files]
        cycles = cycles[:args.max_files]
    print(f"{len(files)} files from {args.input_dir}"
          f"{' (sampled)' if args.max_files else ''}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tasks = [(p, "") for p in files]

    start = time.time()
    all_group_sizes = []
    n_particles_before = 0
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
            for i, (coll, dedup, group_sizes) in enumerate(
                    pool.imap(read_and_dedup_file, tasks)):
                collection_seen = collection_seen or coll
                n = len(dedup["pdg"])
                for key, dset in dsets.items():
                    dset.resize((offsets[-1] + n,))
                    dset[offsets[-1]:] = dedup[key]
                offsets.append(offsets[-1] + n)
                n_particles_before += int(group_sizes.sum())
                all_group_sizes.append(group_sizes)

                median_gs = np.median(group_sizes)
                if abs(median_gs - args.expected_group_size) > 2:
                    print(f"  WARNING: {os.path.basename(files[i])} median "
                          f"group size {median_gs:.1f} deviates from "
                          f"expected {args.expected_group_size} -- inspect "
                          f"this file directly before trusting its output",
                          flush=True)

                if (i + 1) % 250 == 0 or i + 1 == len(files):
                    rate = (i + 1) / (time.time() - start)
                    print(f"  {i + 1}/{len(files)} files, {offsets[-1]:,} "
                          f"particles kept (of {n_particles_before:,} seen), "
                          f"{rate:.1f} files/s", flush=True)

        out.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
        out.create_dataset("cycle_ids", data=np.asarray(cycles, dtype=np.int64))
        out.create_dataset(
            "filenames",
            data=np.asarray([os.path.basename(p) for p in files], dtype=object),
            dtype=h5py.string_dtype(),
        )
        out.attrs["input_dir"] = args.input_dir
        out.attrs["collection"] = collection_seen or ""
        out.attrs["reconstruction_method"] = (
            "deduplicated from a 42x-cloned/rotated GEN library by "
            "grouping on (pdg, |p|, theta, vz, t) rounded to "
            f"{DECIMALS} decimals -- one representative kept per "
            "rotation group. See gen_libtest_reconstruct_unrotated.py."
        )

    all_group_sizes = np.concatenate(all_group_sizes) if all_group_sizes else np.array([])
    counts = np.diff(np.asarray(offsets))
    print(f"\ndone: {len(files)} files, {offsets[-1]:,} particles kept "
          f"(median {np.median(counts):.0f}/file) of {n_particles_before:,} "
          f"seen ({100*offsets[-1]/max(n_particles_before,1):.2f}% kept, "
          f"vs 1/42={100/42:.2f}% expected) -> {args.output}")
    if len(all_group_sizes):
        print(f"group size distribution across all files: "
              f"min={all_group_sizes.min()} median={np.median(all_group_sizes):.1f} "
              f"mean={all_group_sizes.mean():.2f} max={all_group_sizes.max()}")


if __name__ == "__main__":
    main()
