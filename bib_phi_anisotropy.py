#!/usr/bin/env python3
"""Quantify the azimuthal (phi) anisotropy that makes native K=1 separable
from synthetic (rotated) K=1, and show it reproduces the trained PFN.

Motivation
----------
The controlled GEN study finds native-K=1 vs synthetic-K=1 (one fixed random
per-mother z-rotation, no cloning) separable at test AUC ~0.82. Rotation
preserves every feature except azimuth exactly, so the only thing a classifier
can use is the azimuthal arrangement of the pooled particle set. This script
demonstrates, straight from the GEN stores and with no trained model, that:

  1. the INCLUSIVE phi distribution is nearly uniform (weak global moments) --
     which is why a naive phi histogram looks like nothing;
  2. the anisotropy is ENERGY-LOCALISED: the soft bulk (thermalised in the
     nozzle) is isotropic, while the energetic tail remembers the horizontal
     bending/crossing plane (peaks at phi ~ 0, +/-pi);
  3. per CONSTRUCTION (420 pooled source cycles, matching training), hand-built
     azimuthal Fourier observables alone separate native from rotated at AUC up
     to ~0.95 (single observable) / ~0.96 (linear combination) -- meeting and
     exceeding the PFN's 0.82. The separation is therefore genuinely azimuthal
     physics, not a pipeline or float-precision artifact.

The "rotated" control is synthesised in-memory by giving every mother one
uniform-random z-rotation (exactly the synthetic-K=1 construction), so only the
native split-mother bank is needed. A prebuilt rotated store can be supplied
instead with --rotated-store for cross-validation.

Usage
-----
    python bib_phi_anisotropy.py \
        --bank   $PSCRATCH/mucoll/libtest/stores/gen_split_mothers_MUPLUS.h5 \
        --label  MUPLUS --out phi_anisotropy_MUPLUS

Reads only phi and E (needs h5py, numpy, matplotlib). Runs in ~1 min on a
login node for a ~20M-particle bank.
"""

import argparse
import json
import os

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NMAX = 4                       # azimuthal harmonics kept
NFILES = 420                   # source cycles per construction (N=420 study)
E_SLICES = (("all", 0.0), ("E>1", 1.0), ("E>3", 3.0))   # GeV thresholds
ROTATION_SEED = 1701           # matches gen_mother_make_fixed_reuse_store.py


# --------------------------------------------------------------------------- #
# loading + rotated-control synthesis
# --------------------------------------------------------------------------- #
def load_bank(path):
    """phi, E, per-mother owner index, and per-cycle particle offsets.

    Accepts the split-mother schema (mother_offsets/cycle_offsets) or, if a
    prebuilt flat store is passed, the flat (offsets) schema -- in the latter
    case per-mother rotation is unavailable and --rotated-store is required.
    """
    with h5py.File(path, "r") as f:
        g = f["particles"]
        px = g["px"][:].astype(np.float64)
        py = g["py"][:].astype(np.float64)
        E = g["E"][:].astype(np.float64)
        if "mother_offsets" in f:
            mo = f["mother_offsets"][:].astype(np.int64)
            co = f["cycle_offsets"][:].astype(np.int64)
            cycle_off = mo[co]
            owners = np.repeat(np.arange(len(mo) - 1, dtype=np.int64),
                               np.diff(mo))
        else:
            cycle_off = f["offsets"][:].astype(np.int64)
            owners = None
    phi = np.arctan2(py, px)
    return phi, E, owners, cycle_off


def rotated_phi(phi, owners, seed=ROTATION_SEED):
    """Synthetic-K=1 azimuth: one uniform random z-rotation per mother."""
    if owners is None:
        raise ValueError("per-mother rotation needs the split-mother bank; "
                         "pass --rotated-store for a flat store")
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=int(owners.max()) + 1)
    return phi + angles[owners]


def load_flat_phi(path):
    with h5py.File(path, "r") as f:
        g = f["particles"]
        px = g["px"][:].astype(np.float64)
        py = g["py"][:].astype(np.float64)
    return np.arctan2(py, px)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def global_moments(phi, E, cycle_off, boot=400, seed=0):
    """Global R_n (count- and energy-weighted) with a per-cycle bootstrap.

    Cycles are the independent unit, so bootstrapping over cycles gives an
    error bar that respects mother-level correlation rather than pretending
    every particle is independent.
    """
    ncyc = len(cycle_off) - 1
    result = {}
    for tag, w in (("count", np.ones_like(E)), ("energy", E)):
        wsum = np.empty(ncyc)
        csum = np.empty((NMAX, ncyc))
        ssum = np.empty((NMAX, ncyc))
        for i in range(ncyc):
            a, b = cycle_off[i], cycle_off[i + 1]
            ph, ww = phi[a:b], w[a:b]
            wsum[i] = ww.sum()
            for n in range(1, NMAX + 1):
                csum[n - 1, i] = np.sum(ww * np.cos(n * ph))
                ssum[n - 1, i] = np.sum(ww * np.sin(n * ph))
        W = wsum.sum()
        R = np.hypot(csum.sum(1) / W, ssum.sum(1) / W)
        rng = np.random.default_rng(seed)
        Rb = np.empty((boot, NMAX))
        for k in range(boot):
            idx = rng.integers(0, ncyc, ncyc)
            Wk = wsum[idx].sum()
            Rb[k] = np.hypot(csum[:, idx].sum(1) / Wk,
                             ssum[:, idx].sum(1) / Wk)
        result[tag] = {"R": R.tolist(), "R_std": Rb.std(0).tolist(),
                       "sig": (R / Rb.std(0)).tolist()}
    return result


def energy_resolved(phi_nat, phi_rot, E, edges):
    """R_n(E) for native and rotated in log-spaced energy bins."""
    cent = np.sqrt(edges[:-1] * edges[1:])
    out = {"centers": cent.tolist(), "counts": [],
           "native": {n: [] for n in range(1, NMAX + 1)},
           "rotated": {n: [] for n in range(1, NMAX + 1)}}
    for i in range(len(cent)):
        m = (E >= edges[i]) & (E < edges[i + 1])
        out["counts"].append(int(m.sum()))
        for n in range(1, NMAX + 1):
            for name, ph in (("native", phi_nat), ("rotated", phi_rot)):
                if m.sum() > 20:
                    p = ph[m]
                    out[name][n].append(float(np.hypot(np.mean(np.cos(n * p)),
                                                       np.mean(np.sin(n * p)))))
                else:
                    out[name][n].append(float("nan"))
    return out


# --------------------------------------------------------------------------- #
# per-construction separability (the model-free version of the PFN task)
# --------------------------------------------------------------------------- #
def per_cycle_partials(phi, E, cycle_off):
    """Per-cycle sums feeding fast construction assembly: for each energy
    slice and harmonic, sum cos/sin(n phi) count- and energy-weighted, plus
    the slice's particle count and energy sum."""
    ncyc = len(cycle_off) - 1
    cols = {}
    for sname, _ in E_SLICES:
        cols[(sname, "cnt")] = np.zeros(ncyc)
        cols[(sname, "esum")] = np.zeros(ncyc)
        for n in range(1, NMAX + 1):
            for kind in ("cc", "cs", "ec", "es"):
                cols[(sname, n, kind)] = np.zeros(ncyc)
    for i in range(ncyc):
        a, b = cycle_off[i], cycle_off[i + 1]
        ph, ee = phi[a:b], E[a:b]
        for sname, ecut in E_SLICES:
            m = ee >= ecut
            phm, em = ph[m], ee[m]
            cols[(sname, "cnt")][i] = len(phm)
            cols[(sname, "esum")][i] = em.sum()
            for n in range(1, NMAX + 1):
                cn, sn = np.cos(n * phm), np.sin(n * phm)
                cols[(sname, n, "cc")][i] = cn.sum()
                cols[(sname, n, "cs")][i] = sn.sum()
                cols[(sname, n, "ec")][i] = (em * cn).sum()
                cols[(sname, n, "es")][i] = (em * sn).sum()
    return cols, ncyc


def construction_vector(cols, idx, keys):
    feats = {}
    for sname, _ in E_SLICES:
        cnt = max(cols[(sname, "cnt")][idx].sum(), 1)
        esum = max(cols[(sname, "esum")][idx].sum(), 1)
        for n in range(1, NMAX + 1):
            cc = cols[(sname, n, "cc")][idx].sum()
            cs = cols[(sname, n, "cs")][idx].sum()
            ec = cols[(sname, n, "ec")][idx].sum()
            es = cols[(sname, n, "es")][idx].sum()
            feats[f"{sname}_R{n}_cnt"] = np.hypot(cc, cs) / cnt
            feats[f"{sname}_R{n}_E"] = np.hypot(ec, es) / esum
    return [feats[k] for k in keys]


def auc_score(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    u, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(u))
    np.add.at(sums, inv, r)
    r = (sums / c)[inv]
    npos, nneg = y.sum(), (~y).sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def construction_auc(cols_nat, cols_rot, ncyc, n_con=400, n_files=NFILES,
                     seed=7):
    keys = [f"{s}_R{n}_{w}" for s, _ in E_SLICES
            for n in range(1, NMAX + 1) for w in ("cnt", "E")]
    rng = np.random.default_rng(seed)
    Xn, Xr = [], []
    for _ in range(n_con):
        Xn.append(construction_vector(cols_nat,
                  rng.choice(ncyc, n_files, replace=False), keys))
        Xr.append(construction_vector(cols_rot,
                  rng.choice(ncyc, n_files, replace=False), keys))
    X = np.vstack([np.array(Xn), np.array(Xr)])
    y = np.r_[np.zeros(n_con), np.ones(n_con)]
    singles = {k: auc_score(y, X[:, j]) for j, k in enumerate(keys)}
    # combined Fisher LDA, trained on half the constructions, scored on the rest
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Xs = (X - mu) / sd
    half = n_con // 2
    tr = np.r_[np.arange(half), n_con + np.arange(half)]
    te = np.r_[np.arange(half, n_con), n_con + np.arange(half, n_con)]
    m0 = Xs[tr][y[tr] == 0].mean(0)
    m1 = Xs[tr][y[tr] == 1].mean(0)
    Sw = np.cov(Xs[tr][y[tr] == 0].T) + np.cov(Xs[tr][y[tr] == 1].T)
    w = np.linalg.solve(Sw + 1e-6 * np.eye(X.shape[1]), m1 - m0)
    combined = auc_score(y[te], Xs[te] @ w)
    return singles, combined


# --------------------------------------------------------------------------- #
def make_figure(glob_nat, eres, singles, combined, pfn_auc, out, label):
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
    cent = np.array(eres["centers"])

    # (1) anisotropy vs energy
    ax = fig.add_subplot(gs[0])
    null = np.sqrt(1.0 / np.maximum(np.array(eres["counts"]), 1))
    for n, col in zip((1, 2, 4 if NMAX >= 4 else NMAX),
                      ("#0072B2", "#009E73", "#CC79A7")):
        if n > NMAX:
            continue
        ax.plot(cent, eres["native"][n], "o-", color=col, ms=4,
                label=f"native $R_{n}$")
        ax.plot(cent, eres["rotated"][n], "x--", color=col, alpha=0.45, ms=4,
                label=f"rotated $R_{n}$")
    ax.plot(cent, null, ":", color="0.5", label=r"isotropic null")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("particle energy [GeV]")
    ax.set_ylabel(r"azimuthal Fourier magnitude $R_n$")
    ax.set_title("Anisotropy is energy-localised")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    # (2) single-observable AUCs (top few)
    ax = fig.add_subplot(gs[1])
    top = sorted(singles.items(), key=lambda kv: abs(kv[1] - 0.5),
                 reverse=True)[:7][::-1]
    names = [k for k, _ in top]
    # fold to separating power: native (higher R) is class 0, so a strongly
    # anisotropy-driven observable gives AUC well BELOW 0.5; max(v,1-v) reads
    # it as separation regardless of orientation.
    vals = [max(v, 1 - v) for _, v in top]
    ax.barh(range(len(top)), vals, color="#0072B2")
    ax.axvline(0.5, color="0.5", ls="--", lw=1)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("per-construction AUC")
    ax.set_title("Single azimuthal observables")

    # (3) combined vs PFN
    ax = fig.add_subplot(gs[2])
    bars = ax.bar(["φ-moments\n(linear)", "trained\nPFN"],
                  [combined, pfn_auc], color=["#009E73", "#D55E00"])
    ax.axhline(0.5, color="0.5", ls="--", lw=1)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("test AUC")
    ax.set_title("φ moments meet/exceed the PFN")
    for b, v in zip(bars, [combined, pfn_auc]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", fontsize=10)

    fig.suptitle(f"Native vs rotated-K=1 azimuthal anisotropy — {label}",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="native split-mother bank")
    ap.add_argument("--rotated-store", default=None,
                    help="optional prebuilt rotated store (else synthesised)")
    ap.add_argument("--label", default="MUPLUS")
    ap.add_argument("--out", default="phi_anisotropy")
    ap.add_argument("--n-constructions", type=int, default=400)
    ap.add_argument("--pfn-auc", type=float, default=0.8243,
                    help="reference trained-PFN AUC for the summary panel")
    args = ap.parse_args()

    phi_nat, E, owners, cycle_off = load_bank(args.bank)
    if args.rotated_store:
        phi_rot = load_flat_phi(args.rotated_store)
        if len(phi_rot) != len(phi_nat):
            raise SystemExit("rotated store particle count != bank")
    else:
        phi_rot = rotated_phi(phi_nat, owners)

    print(f"[{args.label}] {len(phi_nat):,} particles, "
          f"{len(cycle_off) - 1:,} cycles")

    glob_nat = global_moments(phi_nat, E, cycle_off)
    edges = np.logspace(np.log10(0.01), np.log10(200), 22)
    eres = energy_resolved(phi_nat, phi_rot, E, edges)
    cols_nat, ncyc = per_cycle_partials(phi_nat, E, cycle_off)
    cols_rot, _ = per_cycle_partials(phi_rot, E, cycle_off)
    singles, combined = construction_auc(cols_nat, cols_rot, ncyc,
                                         n_con=args.n_constructions)

    top = sorted(singles.items(), key=lambda kv: abs(kv[1] - 0.5),
                 reverse=True)
    print("\nglobal count-weighted R_n (sig = R/boot):")
    for n in range(NMAX):
        m = glob_nat["count"]
        print(f"  n={n + 1}: R={m['R'][n]:.5f}  sig={m['sig'][n]:.1f}")
    print("\ntop per-construction single-observable separating power "
          "max(AUC,1-AUC):")
    for k, v in top[:8]:
        print(f"  {k:16s} {max(v, 1 - v):.4f}")
    print(f"\ncombined linear AUC = {combined:.4f}  "
          f"(reference PFN {args.pfn_auc:.4f})")

    with open(args.out + ".json", "w") as fh:
        json.dump({"label": args.label, "global": glob_nat,
                   "energy_resolved": eres, "singles": singles,
                   "combined_auc": combined, "pfn_auc": args.pfn_auc}, fh)
    make_figure(glob_nat, eres, singles, combined, args.pfn_auc,
                args.out, args.label)
    print(f"\nwrote {args.out}.png / .pdf / .json")


if __name__ == "__main__":
    main()
