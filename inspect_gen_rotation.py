#!/usr/bin/env python3
"""Check whether a GEN-level BIB ROOT file contains duplicate/rotated
particles -- i.e. whether the norm42-RandomRot cloning is already baked
into GEN, or whether GEN is shared/common between norm1 and norm42 (as
the identical file sizes on OSCAR suggested).

Signature of a rotated clone (established this session from the
Perlmutter norm1/norm42 stores): same E and theta (polar angle,
rotation-invariant), different phi (azimuthal angle, what RandomRot
changes) -- and clones should come in groups of ~42 (mean 42.64) per
unique underlying particle.

Usage (needs uproot, awkward -- same env as gen_libtest_make_store.py):
    python inspect_gen_rotation.py /path/to/bib_gen_0.edm4hep.root
"""

import sys

import awkward as ak
import numpy as np
import uproot


def detect_collection(tree):
    candidates = [k[:-len(".PDG")] for k in tree.keys() if k.endswith(".PDG")]
    if not candidates:
        raise RuntimeError(f"no branch ending in .PDG among: {list(tree.keys())[:20]}")
    return min(candidates, key=len)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python inspect_gen_rotation.py /path/to/file.edm4hep.root")
    path = sys.argv[1]

    with uproot.open(path) as f:
        tree_names = [k.split(";")[0] for k in f.keys()
                      if hasattr(f[k], "keys") and f[k].classname.startswith("TTree")]
        tree = f["events"] if "events" in tree_names else f[tree_names[0]]
        collection = detect_collection(tree)
        print(f"file: {path}")
        print(f"collection: {collection}")

        branches = {
            "px": f"{collection}.momentum.x",
            "py": f"{collection}.momentum.y",
            "pz": f"{collection}.momentum.z",
        }
        arrays = tree.arrays(list(branches.values()), library="ak")
        px = ak.to_numpy(ak.flatten(arrays[branches["px"]], axis=None)).astype(np.float64)
        py = ak.to_numpy(ak.flatten(arrays[branches["py"]], axis=None)).astype(np.float64)
        pz = ak.to_numpy(ak.flatten(arrays[branches["pz"]], axis=None)).astype(np.float64)

    n = len(px)
    pt = np.hypot(px, py)
    p = np.sqrt(px**2 + py**2 + pz**2)
    theta = np.arctan2(pt, pz)
    phi = np.arctan2(py, px)

    print(f"total particles in file: {n}")
    print(f"expected if NOT cloned (matches norm1 per-file scale, ~2985 mean): "
          f"{'plausible' if n < 20000 else 'too large -- looks cloned'}")
    print(f"expected if cloned ~42x (matches norm42 per-file scale, ~125369 mean): "
          f"{'plausible' if n > 50000 else 'too small -- does not look cloned'}")

    # Round (p, theta) to detect near-exact duplicates (rotation preserves
    # magnitude and polar angle exactly in principle, but floats may carry
    # tiny numerical noise from the rotation matrix -- round to a fine but
    # forgiving precision).
    key = np.round(p, decimals=6).astype(str)
    key = np.char.add(key, "|")
    key = np.char.add(key, np.round(theta, decimals=6).astype(str))

    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    n_unique = len(uniq)
    dup_group_sizes = counts[counts > 1]

    print(f"\nunique (|p|, theta) groups: {n_unique} (vs {n} total particles)")
    print(f"particles in a group size > 1 (i.e. sharing |p| and theta): "
          f"{dup_group_sizes.sum()} ({100*dup_group_sizes.sum()/n:.1f}% of file)")
    if len(dup_group_sizes) > 0:
        print(f"group size distribution: min={dup_group_sizes.min()} "
              f"median={np.median(dup_group_sizes):.0f} "
              f"mean={dup_group_sizes.mean():.2f} max={dup_group_sizes.max()}")
        print(f"groups with size in [38,46] (near the ~42.64 cloning factor): "
              f"{np.sum((dup_group_sizes >= 38) & (dup_group_sizes <= 46))}")

        # For a sample duplicate group, check phi actually varies (confirming
        # ROTATION specifically, not just coincidentally-identical particles).
        biggest_group_key = uniq[np.argmax(counts)]
        members = np.flatnonzero(key == biggest_group_key)
        print(f"\nlargest duplicate group: {len(members)} members")
        print(f"  phi values (should be spread out if rotated, identical if not): "
              f"min={phi[members].min():.4f} max={phi[members].max():.4f} "
              f"std={phi[members].std():.4f}")
        print(f"  p values (should be ~identical, confirming these are 'the same' "
              f"particle physically): {p[members][:5]}")
        print(f"  theta values (should be ~identical): {theta[members][:5]}")
    else:
        print("\nNO duplicate (|p|, theta) groups found -- every particle's "
              "momentum magnitude + polar angle is unique in this file. "
              "This file does NOT contain rotated clones.")

    print("\n=== verdict ===")
    if len(dup_group_sizes) > 0 and dup_group_sizes.mean() > 5:
        print("Rotated clones ARE present in this GEN file -- cloning is baked "
              "in at the GEN level.")
    else:
        print("Rotated clones are NOT present in this GEN file -- this file is "
              "the shared/common (unrotated) data. Confirms the earlier "
              "file-size-identity finding via actual content, not just size.")


if __name__ == "__main__":
    main()
