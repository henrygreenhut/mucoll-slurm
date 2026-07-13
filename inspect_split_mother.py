#!/usr/bin/env python3
"""Investigate the structure of a split-by-mother BIB GEN library.

Hypothesis to verify: each file holds one FLUKA cycle, subdivided into one
EVENT per source mother muon (~70 events/file, ~40 particles each), with
the same total particle content as the unsplit norm1 file of the same
cycle. If true, this library gives mother-level sampling granularity
without parsing parent links.

Run in the mucoll-inspect env (login node, reads a few files):

    source config.sh
    python inspect_split_mother.py \
        --split-dir $DATA_GROUP_DIR/bib-v3p0-fmt2-norm1-split-mother-norot/GEN/MUPLUS \
        --ref-dir   $DATA_GROUP_DIR/bib-v3p0-fmt2-norm1/GEN/MUPLUS
"""

import argparse
import glob
import os
import sys

import numpy as np

import libtest_common as lc


def parse_args():
    data_dir = os.environ.get("DATA_GROUP_DIR", "")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default=os.path.join(
        data_dir, "bib-v3p0-fmt2-norm1-split-mother-norot/GEN/MUPLUS"))
    parser.add_argument("--ref-dir", default=os.path.join(
        data_dir, "bib-v3p0-fmt2-norm1/GEN/MUPLUS"))
    parser.add_argument("--files", type=int, default=4,
                        help="number of files to inspect in depth")
    return parser.parse_args()


def library_index(label, directory):
    files = glob.glob(os.path.join(directory, "*.root"))
    if not files:
        sys.exit(f"ERROR: no *.root in {directory}")
    ids, pos, distinct = lc.assign_cycle_ids(files)
    print(f"{label}: {len(files)} files | e.g. {os.path.basename(files[0])}")
    print(f"  cycle id = token #{pos} from end ({distinct} distinct)")
    if distinct != len(files):
        print("  !! WARNING: ids not unique")
    return dict(zip(ids, files))


def open_events(path):
    import uproot
    f = uproot.open(path)
    keys = [k.split(";")[0] for k in f.keys()]
    tree = f["events"] if "events" in keys else f[keys[0]]
    return f, tree


def per_event_layout(path):
    """(n_events, particles-per-event array, flat rows, branches)."""
    import awkward as ak
    f, tree = open_events(path)
    coll = min((k[:-len(".PDG")] for k in tree.keys() if k.endswith(".PDG")),
               key=len)
    arr = tree.arrays([f"{coll}.PDG", f"{coll}.momentum.z", f"{coll}.time",
                       f"{coll}.vertex.z"], library="ak")
    counts = ak.to_numpy(ak.num(arr[f"{coll}.PDG"]))
    rows = np.column_stack([
        ak.to_numpy(ak.flatten(arr[f"{coll}.momentum.z"], axis=None)),
        ak.to_numpy(ak.flatten(arr[f"{coll}.time"], axis=None)),
        ak.to_numpy(ak.flatten(arr[f"{coll}.vertex.z"], axis=None)),
    ])
    branches = sorted(k.split(";")[0] for k in tree.keys())
    f.close()
    return len(counts), counts, rows, branches


def main():
    args = parse_args()
    split = library_index("split ", args.split_dir)
    ref = library_index("ref   ", args.ref_dir) if os.path.isdir(args.ref_dir) else {}

    common = sorted(set(split) & set(ref)) if ref else sorted(split)
    print(f"paired cycles with reference: {len(common)}\n" if ref else "")
    picks = [common[int(i)] for i in np.linspace(0, len(common) - 1, args.files)]

    ev_counts = []
    for cyc in picks:
        n_ev, counts, rows, branches = per_event_layout(split[cyc])
        ev_counts.append(n_ev)
        print(f"cycle {cyc}: {n_ev} events | particles/event"
              f" min/med/max = {counts.min()}/{int(np.median(counts))}/{counts.max()}"
              f" | total {counts.sum():,}")
        if ref:
            n_ev_r, counts_r, rows_r, branches_r = per_event_layout(ref[cyc])
            same_total = counts.sum() == counts_r.sum()
            # content check: identical multiset of (pz, t, vz) rows?
            a = rows[np.lexsort(rows.T)]
            b = rows_r[np.lexsort(rows_r.T)]
            same_content = a.shape == b.shape and np.array_equal(a, b)
            print(f"  vs unsplit cycle {cyc}: ref {n_ev_r} events,"
                  f" {counts_r.sum():,} particles | totals match: {same_total}"
                  f" | identical particle content (pz,t,vz): {same_content}")
            only_s = set(branches) - set(branches_r)
            only_r = set(branches_r) - set(branches)
            if only_s or only_r:
                print(f"  schema differs: split-only {sorted(only_s)},"
                      f" ref-only {sorted(only_r)}")

    print(f"\nevents/file over {len(picks)} files:"
          f" min {min(ev_counts)} max {max(ev_counts)}"
          f" (expect ~70 if one event per visible mother)")
    print("If events/file ~ mothers/file and content matches the unsplit"
          "\nlibrary exactly, this library provides mother-level sampling:"
          "\nunits can be defined by mother count, and custom reuse factors"
          "\n(e.g. the true 10.66x point) become constructible.")


if __name__ == "__main__":
    main()
