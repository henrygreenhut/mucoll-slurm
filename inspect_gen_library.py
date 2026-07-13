#!/usr/bin/env python3
"""Phase-0 audit of the norm1 vs norm42-RandomRot GEN libraries.

Verifies, on a few paired cycles, everything the library-discrimination test
assumes before any dataset is built:

  1. schema: same trees/branches in both libraries (no metadata leak)
  2. multiplicity: norm42 file has ~42x the particles of its norm1 partner
  3. clone structure: norm42 particles group into ~42-fold copies identical
     in (E, pz, t, vz) with phi spread across the group
  4. pairing: norm1 cycle k's particles all appear (42x) in norm42 cycle k
  5. phi marginals: norm1 retains azimuthal structure, norm42 flattened
  6. cut curves: survival vs energy threshold and |t| window (for choosing
     physics cuts at large unit sizes)
  7. provenance: whether mother/parent links exist (enables mother-level
     sampling later)

Run in the mucoll-inspect env (login node is fine, it reads a few files):

    source config.sh
    python inspect_gen_library.py --cycles 5
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

E_THRESHOLDS = [0.0, 0.001, 0.01, 0.1, 1.0]          # GeV
T_WINDOWS = [0.0, 1.0, 5.0, 10.0, 25.0, 100.0]       # |t| <, 0 = no cut


def parse_args():
    data_dir = os.environ.get("DATA_GROUP_DIR", "")
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm1-dir",
                        default=os.path.join(data_dir, "bib-v3p0-fmt2-norm1/GEN/MUPLUS"))
    parser.add_argument("--norm42-dir",
                        default=os.path.join(data_dir, "bib-v3p0-fmt2-norm42-RandomRot/GEN/MUPLUS"))
    parser.add_argument("--cycles", type=int, default=5,
                        help="number of paired cycles to inspect in depth")
    parser.add_argument("--clone-factor", type=int, default=42)
    return parser.parse_args()


def cycle_id(path):
    matches = re.findall(r"(\d+)", os.path.basename(path))
    return int(matches[-1]) if matches else -1


def list_library(label, directory):
    files = sorted(glob.glob(os.path.join(directory, "*.root")), key=cycle_id)
    print(f"{label}: {len(files)} files in {directory}")
    if not files:
        sys.exit(f"ERROR: no files for {label} -- check the directory")
    return {cycle_id(p): p for p in files}


def open_events(path):
    import uproot
    f = uproot.open(path)
    keys = [k.split(";")[0] for k in f.keys()]
    tree = f["events"] if "events" in keys else f[keys[0]]
    return f, tree


def collection_of(tree):
    cands = [k[:-len(".PDG")] for k in tree.keys() if k.endswith(".PDG")]
    if not cands:
        sys.exit(f"ERROR: no .PDG branch; branches: {list(tree.keys())[:30]}")
    return min(cands, key=len)


def load_particles(path):
    import awkward as ak
    f, tree = open_events(path)
    coll = collection_of(tree)
    names = {
        "pdg": f"{coll}.PDG", "px": f"{coll}.momentum.x",
        "py": f"{coll}.momentum.y", "pz": f"{coll}.momentum.z",
        "t": f"{coll}.time", "vx": f"{coll}.vertex.x",
        "vy": f"{coll}.vertex.y", "vz": f"{coll}.vertex.z",
    }
    mass = f"{coll}.mass"
    want = list(names.values()) + ([mass] if mass in tree.keys() else [])
    arrays = tree.arrays(want, library="ak")
    out = {k: ak.to_numpy(ak.flatten(arrays[b], axis=None)) for k, b in names.items()}
    p2 = out["px"].astype(np.float64)**2 + out["py"].astype(np.float64)**2 \
        + out["pz"].astype(np.float64)**2
    m = (ak.to_numpy(ak.flatten(arrays[mass], axis=None)).astype(np.float64)
         if mass in want else 0.0)
    out["E"] = np.sqrt(p2 + m**2)
    out["_branches"] = sorted(k.split(";")[0] for k in tree.keys())
    out["_toplevel"] = sorted(k.split(";")[0] for k in f.keys())
    out["_nevents"] = tree.num_entries
    out["_collection"] = coll
    f.close()
    return out


def clone_key(raw, decimals=None):
    """Rows keyed on rotation-invariant quantities (identical across clones)."""
    cols = np.column_stack([raw["E"], raw["pz"], raw["t"], raw["vz"]])
    if decimals is not None:
        with np.errstate(divide="ignore"):
            mags = np.where(cols != 0, np.floor(np.log10(np.abs(cols) + 1e-300)), 0)
        cols = np.round(cols / 10.0**mags, decimals) * 10.0**mags
    return cols


def group_sizes(keys):
    _, counts = np.unique(keys, axis=0, return_counts=True)
    return counts


def describe_sizes(label, counts, clone_factor):
    frac_cf = np.mean(counts == clone_factor)
    print(f"  {label}: {len(counts)} groups | sizes min/med/max ="
          f" {counts.min()}/{int(np.median(counts))}/{counts.max()}"
          f" | exactly {clone_factor}: {100*frac_cf:.1f}%")


def phi_hist(raw, bins=12):
    phi = np.arctan2(raw["py"], raw["px"])
    hist, _ = np.histogram(phi, bins=bins, range=(-np.pi, np.pi))
    return hist / hist.sum()


def survival_table(label, raw):
    e = raw["E"]
    t = np.abs(raw["t"])
    print(f"  {label} survival fractions (rows: E >= thr [GeV]; cols: |t| < win):")
    header = "    E\\t      " + "".join(f"{('inf' if w == 0 else w):>9}" for w in T_WINDOWS)
    print(header)
    for thr in E_THRESHOLDS:
        row = f"    {thr:<10g}"
        for win in T_WINDOWS:
            mask = e >= thr
            if win > 0:
                mask &= t < win
            row += f"{mask.mean():9.4f}"
        print(row)


def main():
    args = parse_args()
    lib1 = list_library("norm1 ", args.norm1_dir)
    lib42 = list_library("norm42", args.norm42_dir)

    common = sorted(set(lib1) & set(lib42))
    print(f"paired cycles present in both: {len(common)}"
          f" (norm1-only: {len(set(lib1)-set(lib42))},"
          f" norm42-only: {len(set(lib42)-set(lib1))})\n")
    if not common:
        sys.exit("ERROR: no paired cycles -- filename convention mismatch?")

    picks = [common[int(i)] for i in np.linspace(0, len(common) - 1, args.cycles)]
    print(f"inspecting cycles: {picks}\n")

    schema_diff_shown = False
    ratios, phi1_all, phi42_all = [], [], []
    for cyc in picks:
        r1 = load_particles(lib1[cyc])
        r42 = load_particles(lib42[cyc])

        if not schema_diff_shown:
            print(f"collection: '{r1['_collection']}' | events/file:"
                  f" norm1={r1['_nevents']} norm42={r42['_nevents']}")
            only1 = set(r1["_branches"]) - set(r42["_branches"])
            only42 = set(r42["_branches"]) - set(r1["_branches"])
            if only1 or only42:
                print(f"  !! SCHEMA DIFFERS (potential class leak)\n"
                      f"     norm1-only:  {sorted(only1)}\n"
                      f"     norm42-only: {sorted(only42)}")
            else:
                print("  schema: identical branch lists in both libraries")
            has_parents = any("parent" in b.lower() for b in r1["_branches"])
            print(f"  mother/parent provenance branches: "
                  f"{'PRESENT' if has_parents else 'absent'}\n")
            schema_diff_shown = True

        n1, n42 = len(r1["pdg"]), len(r42["pdg"])
        ratios.append(n42 / max(n1, 1))
        print(f"cycle {cyc}: norm1 {n1:,} particles | norm42 {n42:,}"
              f" | ratio {n42 / max(n1, 1):.2f}")

        # clone grouping in norm42 (exact float match, then rounded fallback)
        keys = clone_key(r42)
        counts = group_sizes(keys)
        if counts.max() == 1:
            keys = clone_key(r42, decimals=5)
            counts = group_sizes(keys)
            print("  (exact float match found no groups; using 6-sig-fig rounding)")
        describe_sizes("norm42 clone groups", counts, args.clone_factor)

        # pairing: norm1 rows should appear among norm42 rows
        k1 = clone_key(r1)
        k42 = keys
        view1 = {tuple(row) for row in k1}
        view42 = {tuple(row) for row in k42}
        matched = sum(1 for row in view1 if row in view42)
        print(f"  pairing: {matched}/{len(view1)} norm1 rows found in norm42"
              f" ({100 * matched / max(len(view1), 1):.1f}%)")

        # phi spread within the largest clone groups
        uniq, inv, cnts = np.unique(k42, axis=0, return_inverse=True,
                                    return_counts=True)
        big = np.argsort(cnts)[-3:]
        phi42 = np.arctan2(r42["py"], r42["px"])
        spreads = [np.ptp(phi42[inv == g]) for g in big]
        print(f"  phi range within 3 largest groups: "
              f"{', '.join(f'{s:.2f}' for s in spreads)} rad (uniform => ~2pi)")

        # composition
        for tag, r in (("norm1 ", r1), ("norm42", r42)):
            apdg = np.abs(r["pdg"])
            frac = lambda sel: 100 * np.mean(sel)
            print(f"  {tag} composition: gamma {frac(apdg==22):.1f}%"
                  f" | n {frac(apdg==2112):.1f}% | e {frac(apdg==11):.1f}%"
                  f" | mu {frac(apdg==13):.1f}%"
                  f" | other {frac(~np.isin(apdg,[22,2112,11,13])):.1f}%")

        phi1_all.append(phi_hist(r1))
        phi42_all.append(phi_hist(r42))
        survival_table("norm1 ", r1)
        print()

    print(f"multiplicity ratio over {len(picks)} paired cycles:"
          f" mean {np.mean(ratios):.2f} (expect ~{args.clone_factor})")
    phi1 = np.mean(phi1_all, axis=0)
    phi42 = np.mean(phi42_all, axis=0)
    print("\nphi marginals (12 bins, fraction per bin; uniform = 0.083):")
    print("  norm1 :", " ".join(f"{v:.3f}" for v in phi1))
    print("  norm42:", " ".join(f"{v:.3f}" for v in phi42))
    print(f"  max |bin - uniform|: norm1 {np.max(np.abs(phi1 - 1/12)):.4f},"
          f" norm42 {np.max(np.abs(phi42 - 1/12)):.4f}")
    print("\nGATE: proceed to conversion only if the ratio is ~42, clone groups"
          "\nare ~42-fold, pairing is ~100%, and the schema shows no leak.")


if __name__ == "__main__":
    main()
