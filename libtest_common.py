"""Shared helpers for the norm1-vs-norm42 library-discrimination test.

A "store" is an HDF5 file produced by gen_libtest_make_store.py holding all
GEN particles of one (library, polarity), file by file:

    particles/px,py,pz,E,t,vx,vy,vz   float32, flat over all files
    particles/pdg                     int32
    offsets                           int64, len nfiles+1
    cycle_ids                         int64, len nfiles (sorted ascending)
    filenames                         str, len nfiles

A "unit" (pseudo-crossing) is a list of file positions into one store:
    unique class = n_files norm1 cycles
    reuse  class = n_files / clone_factor norm42 cycles
Both represent the same decay statistics; splits are by cycle so no mother
appears in more than one of train/val/test.
"""

import json
import os
import re

import h5py
import numpy as np

RAW_KEYS = ["px", "py", "pz", "E", "t", "vx", "vy", "vz", "pdg"]


def assign_cycle_ids(paths):
    """Cycle id per file: the integer token in the basename that VARIES
    across the directory (constant tokens like version tags are skipped).

    Returns (ids, token_pos_from_end, n_distinct). Callers should require
    n_distinct == len(paths); anything less means the naming convention is
    ambiguous and needs a hand-picked rule.
    """
    tokens = [re.findall(r"\d+", os.path.basename(p)) for p in paths]
    n_tok = min(len(t) for t in tokens)
    if n_tok == 0:
        raise ValueError("filenames contain no integer tokens")
    best_pos, best_distinct = 1, 0
    for pos in range(1, n_tok + 1):
        distinct = len({t[-pos] for t in tokens})
        if distinct > best_distinct:
            best_pos, best_distinct = pos, distinct
    ids = [int(t[-best_pos]) for t in tokens]
    return ids, best_pos, best_distinct

# Feature sets, anchored on the PFN paper (arXiv:1810.05165):
#   paper = PFN-ID inputs (pT, angle, ID), adapted for BIB: theta instead of
#           rapidity (forward particles), absolute angles via cos/sin phi
#           (no jet axis to center on), log pT (6-decade spectrum).
#   bib   = paper + time/vertex displacement (asinh-compressed), the BIB
#           discriminants of arXiv:2105.09116 / 2203.06773.
PDG_ONEHOT = ["pdg_gamma", "pdg_n", "pdg_e", "pdg_mu", "pdg_other"]
FEATURE_SETS = {
    "paper": ["logpt", "theta", "cosphi", "sinphi"] + PDG_ONEHOT,
    "bib": ["logpt", "theta", "cosphi", "sinphi",
            "asinh_t", "asinh_vz", "asinh_vr"] + PDG_ONEHOT,
}
PHI_FEATURES = ["cosphi", "sinphi"]


class Store:
    """In-RAM view of a particle store."""

    def __init__(self, path):
        self.path = path
        with h5py.File(path, "r") as f:
            self.offsets = f["offsets"][:]
            self.cycle_ids = f["cycle_ids"][:]
            self.raw = {k: f["particles"][k][:] for k in RAW_KEYS}
        self.n_files = len(self.cycle_ids)

    def file_arrays(self, positions):
        """Concatenated raw arrays for the given file positions."""
        segs = [(self.offsets[p], self.offsets[p + 1]) for p in positions]
        out = {}
        for key in RAW_KEYS:
            arr = self.raw[key]
            out[key] = np.concatenate([arr[a:b] for a, b in segs])
        return out


def common_positions(store_a, store_b):
    """Positions (per store) of the cycles present in both stores, sorted."""
    common = np.intersect1d(store_a.cycle_ids, store_b.cycle_ids)
    pos_a = np.searchsorted(store_a.cycle_ids, common)
    pos_b = np.searchsorted(store_b.cycle_ids, common)
    return common, pos_a, pos_b


def split_indices(n_common, fracs=(0.60, 0.15, 0.25)):
    """Split [0, n_common) into train/val/test index arrays by cycle order."""
    n_train = int(round(n_common * fracs[0]))
    n_val = int(round(n_common * fracs[1]))
    idx = np.arange(n_common)
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


def apply_cuts(raw, e_min=0.0, t_abs_max=0.0):
    """Deterministic physics cuts; identical for both classes. 0 = no cut."""
    mask = np.ones(len(raw["E"]), dtype=bool)
    if e_min > 0:
        mask &= raw["E"] >= e_min
    if t_abs_max > 0:
        mask &= np.abs(raw["t"]) <= t_abs_max
    if mask.all():
        return raw
    return {k: v[mask] for k, v in raw.items()}


def feature_names(feature_set="paper", drop_phi=False):
    names = list(FEATURE_SETS[feature_set])
    if drop_phi:
        names = [n for n in names if n not in PHI_FEATURES]
    return names


def build_features(raw, feature_set="paper", drop_phi=False):
    """(N, F) float32 feature array from raw particle arrays."""
    px, py = raw["px"], raw["py"]
    pt = np.hypot(px, py)
    phi = np.arctan2(py, px)

    apdg = np.abs(raw["pdg"])
    onehot = {name: np.zeros(len(px), dtype=np.float32) for name in PDG_ONEHOT}
    onehot["pdg_gamma"][apdg == 22] = 1.0
    onehot["pdg_n"][apdg == 2112] = 1.0
    onehot["pdg_e"][apdg == 11] = 1.0
    onehot["pdg_mu"][apdg == 13] = 1.0
    assigned = sum(onehot[n] for n in PDG_ONEHOT[:4])
    onehot["pdg_other"][assigned == 0] = 1.0

    columns = {
        "logpt": lambda: np.log10(np.maximum(pt, 1e-9)),
        "theta": lambda: np.arctan2(pt, raw["pz"]),
        "cosphi": lambda: np.cos(phi),
        "sinphi": lambda: np.sin(phi),
        "asinh_t": lambda: np.arcsinh(raw["t"]),
        "asinh_vz": lambda: np.arcsinh(raw["vz"]),
        "asinh_vr": lambda: np.arcsinh(np.hypot(raw["vx"], raw["vy"])),
    }
    cols = []
    for name in feature_names(feature_set, drop_phi):
        if name in PDG_ONEHOT:
            cols.append(onehot[name])
        else:
            cols.append(columns[name]())
    return np.column_stack(cols).astype(np.float32)


def compute_norm_stats(feature_arrays):
    """Per-feature mean/std over a list of (N, F) arrays."""
    stacked = np.concatenate(feature_arrays, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def save_norm_stats(path, mean, std, names, latent_scale):
    with open(path, "w") as f:
        json.dump({"names": names, "mean": mean.tolist(), "std": std.tolist(),
                   "latent_scale": latent_scale}, f, indent=1)


def load_norm_stats(path):
    with open(path) as f:
        d = json.load(f)
    return (np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32),
            float(d["latent_scale"]))


def build_pfn(input_dim, latent_scale, phi_sizes=(200, 200, 256),
              f_sizes=(200, 200, 200), lr=0.001):
    """PFN (per-particle Phi MLP -> masked sum -> F MLP) in plain Keras.

    Zero-padded particles (all features exactly 0) are masked out. The
    latent sum is multiplied by the constant latent_scale (typically
    1/median particles-per-unit) so the F network sees O(1) inputs at any
    unit size. A constant scale is class-blind and linear, so relative
    multiplicity information is fully preserved.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, optimizers

    inp = layers.Input(shape=(None, input_dim), name="particles")
    mask = layers.Lambda(
        lambda x: tf.cast(tf.reduce_any(tf.not_equal(x, 0.0), axis=-1), tf.float32),
        name="mask")(inp)
    h = inp
    for i, width in enumerate(phi_sizes):
        h = layers.Dense(width, activation="relu", name=f"phi_{i}")(h)
    summed = layers.Lambda(
        lambda t: tf.reduce_sum(t[0] * t[1][..., None], axis=1) * latent_scale,
        name="scaled_sum")([h, mask])
    g = summed
    for i, width in enumerate(f_sizes):
        g = layers.Dense(width, activation="relu", name=f"f_{i}")(g)
    out = layers.Dense(2, activation="softmax", name="output")(g)
    model = Model(inp, out)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["acc"])
    return model


def sample_unit_positions(rng, split_positions, n_files):
    """Random distinct file positions for one unit."""
    return rng.choice(split_positions, size=n_files, replace=False)


def blocked_unit_positions(split_positions, n_files):
    """Disjoint consecutive blocks of n_files positions (for test eval)."""
    n_blocks = len(split_positions) // n_files
    return [split_positions[i * n_files:(i + 1) * n_files] for i in range(n_blocks)]


def auc_score(y_true, scores):
    """ROC AUC via average ranks (ties handled); no sklearn."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_true).astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc(scores_class0, scores_class1, n_boot=1000, seed=7):
    """AUC uncertainty by resampling disjoint units within each class."""
    rng = np.random.default_rng(seed)
    s0 = np.asarray(scores_class0)
    s1 = np.asarray(scores_class1)
    vals = []
    for _ in range(n_boot):
        b0 = rng.choice(s0, size=len(s0), replace=True)
        b1 = rng.choice(s1, size=len(s1), replace=True)
        y = np.concatenate([np.zeros(len(b0)), np.ones(len(b1))])
        vals.append(auc_score(y, np.concatenate([b0, b1])))
    return float(np.mean(vals)), float(np.std(vals))
