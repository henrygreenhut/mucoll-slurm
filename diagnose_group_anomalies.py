#!/usr/bin/env python3
"""Diagnose why some rotation groups in a GEN file aren't exactly size 42:
genuine un-cloned particles, or false splits from overly-strict dedup-key
rounding in gen_libtest_reconstruct_unrotated.py?

For every group that isn't size 42, reports its members' exact (unrounded)
kinematics and searches for a "near-miss" sibling elsewhere in the file --
a particle whose UNROUNDED (|p|, theta, vz, t) is extremely close but got
assigned a different rounded key. If near-misses are common, the rounding
is too strict (fix: coarsen DECIMALS or round in log-space). If not, these
are likely genuine singletons.

Usage: python diagnose_group_anomalies.py /path/to/bib_gen_N.edm4hep.root
"""

import sys

import awkward as ak
import numpy as np
import uproot

DECIMALS = 6


def detect_collection(tree):
    candidates = [k[:-len(".PDG")] for k in tree.keys() if k.endswith(".PDG")]
    return min(candidates, key=len)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python diagnose_group_anomalies.py /path/to/file.root")
    path = sys.argv[1]

    with uproot.open(path) as f:
        tree_names = [k.split(";")[0] for k in f.keys()
                      if hasattr(f[k], "keys") and f[k].classname.startswith("TTree")]
        tree = f["events"] if "events" in tree_names else f[tree_names[0]]
        collection = detect_collection(tree)
        branches = {
            "pdg": f"{collection}.PDG",
            "px": f"{collection}.momentum.x",
            "py": f"{collection}.momentum.y",
            "pz": f"{collection}.momentum.z",
            "t": f"{collection}.time",
            "vz": f"{collection}.vertex.z",
        }
        arrays = tree.arrays(list(branches.values()), library="ak")

    pdg = ak.to_numpy(ak.flatten(arrays[branches["pdg"]], axis=None)).astype(np.int32)
    px = ak.to_numpy(ak.flatten(arrays[branches["px"]], axis=None)).astype(np.float64)
    py = ak.to_numpy(ak.flatten(arrays[branches["py"]], axis=None)).astype(np.float64)
    pz = ak.to_numpy(ak.flatten(arrays[branches["pz"]], axis=None)).astype(np.float64)
    t = ak.to_numpy(ak.flatten(arrays[branches["t"]], axis=None)).astype(np.float64)
    vz = ak.to_numpy(ak.flatten(arrays[branches["vz"]], axis=None)).astype(np.float64)

    p = np.sqrt(px**2 + py**2 + pz**2)
    pt = np.hypot(px, py)
    theta = np.arctan2(pt, pz)

    key_arr = np.stack([pdg.astype(np.float64), np.round(p, DECIMALS),
                        np.round(theta, DECIMALS), np.round(vz, DECIMALS),
                        np.round(t, DECIMALS)], axis=1)
    uniq, inv, group_sizes = np.unique(key_arr, axis=0, return_inverse=True,
                                       return_counts=True)

    n = len(pdg)
    hist = np.bincount(group_sizes)
    print(f"total particles: {n}, total groups: {len(uniq)}")
    print(f"group size histogram (size: count of groups that size):")
    for size in range(1, min(len(hist), 45)):
        if hist[size] > 0:
            print(f"  {size:3d}: {hist[size]}")

    anomalous_mask = group_sizes[inv] != 42
    n_anomalous_particles = anomalous_mask.sum()
    print(f"\nparticles in non-42 groups: {n_anomalous_particles} "
          f"({100*n_anomalous_particles/n:.2f}% of file)")

    # For singleton groups specifically, check for a near-miss sibling.
    singleton_groups = np.flatnonzero(group_sizes == 1)
    print(f"\nsingleton groups: {len(singleton_groups)}")
    n_checked = 0
    n_near_miss = 0
    for g in singleton_groups[:30]:  # cap for readability
        idx = np.flatnonzero(inv == g)[0]
        # nearest neighbor in UNROUNDED (p, theta) among same-pdg particles
        same_pdg = np.flatnonzero((pdg == pdg[idx]) & (np.arange(n) != idx))
        if len(same_pdg) == 0:
            continue
        dp = np.abs(p[same_pdg] - p[idx])
        dtheta = np.abs(theta[same_pdg] - theta[idx])
        dvz = np.abs(vz[same_pdg] - vz[idx])
        dt = np.abs(t[same_pdg] - t[idx])
        combined = dp + dtheta + dvz + dt  # crude combined distance
        nearest = np.argmin(combined)
        n_checked += 1
        is_near = (dp[nearest] < 1e-3 and dtheta[nearest] < 1e-3
                  and dvz[nearest] < 1e-2 and dt[nearest] < 1e-2)
        n_near_miss += is_near
        print(f"  singleton idx={idx} pdg={pdg[idx]} p={p[idx]:.8f} "
              f"theta={theta[idx]:.8f} vz={vz[idx]:.6f} t={t[idx]:.6f}")
        print(f"    nearest same-pdg other particle: dp={dp[nearest]:.2e} "
              f"dtheta={dtheta[nearest]:.2e} dvz={dvz[nearest]:.2e} "
              f"dt={dt[nearest]:.2e} {'<- NEAR MISS' if is_near else ''}")

    print(f"\n=== verdict ===")
    if n_checked:
        print(f"{n_near_miss}/{n_checked} checked singletons have a very "
              f"close (likely-true-sibling) match elsewhere in the file.")
        if n_near_miss / n_checked > 0.3:
            print("Rounding is probably too strict -- these are likely FALSE "
                  "splits, not genuine singletons. Consider coarsening "
                  "DECIMALS or using a relative/log-space tolerance.")
        else:
            print("Most singletons have no close match -- likely GENUINE "
                  "uncloned particles, not a rounding artifact.")


if __name__ == "__main__":
    main()
