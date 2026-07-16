#!/usr/bin/env python3
"""Mother-level storage and coherent-rotation helpers for variable reuse."""

import h5py
import numpy as np


RAW_KEYS = ("px", "py", "pz", "E", "t", "vx", "vy", "vz", "pdg")


def expand_ranges(starts, counts):
    """Vectorized concatenation of integer ranges [start, start + count)."""
    starts = np.asarray(starts, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    output_positions = np.arange(total, dtype=np.int64)
    group_output_starts = np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(starts, counts) + output_positions - group_output_starts


class MotherStore:
    """In-memory view of one split-by-mother GEN HDF5 bank."""

    def __init__(self, path):
        self.path = path
        with h5py.File(path, "r") as handle:
            self.raw = {key: handle["particles"][key][:] for key in RAW_KEYS}
            self.mother_offsets = handle["mother_offsets"][:]
            self.mother_cycle_ids = handle["mother_cycle_ids"][:]
            self.mother_local_ids = handle["mother_local_ids"][:]
            self.cycle_ids = handle["cycle_ids"][:]
            self.cycle_offsets = handle["cycle_offsets"][:]
        self.n_mothers = len(self.mother_offsets) - 1
        self.n_cycles = len(self.cycle_ids)

    def mothers_for_cycles(self, cycle_ids):
        """Return all mother positions belonging to the requested cycles."""
        requested = np.asarray(cycle_ids, dtype=np.int64).reshape(-1)
        positions = np.searchsorted(self.cycle_ids, requested)
        valid = positions < self.n_cycles
        valid[valid] &= self.cycle_ids[positions[valid]] == requested[valid]
        if not np.all(valid):
            raise KeyError("cycles absent from mother store: {}".format(
                requested[~valid].tolist()))
        starts = self.cycle_offsets[positions]
        counts = self.cycle_offsets[positions + 1] - starts
        return expand_ranges(starts, counts)

    def rotated_mothers(self, mother_positions, angles):
        """Concatenate coherent z-rotations of selected mother events.

        `angles` has shape (number of selected unique mothers, reuse k). Every
        particle belonging to a given mother/copy receives the same angle.
        """
        mothers = np.asarray(mother_positions, dtype=np.int64)
        angles = np.asarray(angles, dtype=np.float64)
        if angles.ndim != 2 or angles.shape[0] != len(mothers):
            raise ValueError("angles must have shape (n_selected_mothers, k)")
        if len(np.unique(mothers)) != len(mothers):
            raise ValueError("source mothers must be unique within a unit")
        if np.any(mothers < 0) or np.any(mothers >= self.n_mothers):
            raise IndexError("mother position outside store")

        starts = self.mother_offsets[mothers]
        counts = self.mother_offsets[mothers + 1] - starts
        particle_positions = expand_ranges(starts, counts)
        owners = np.repeat(np.arange(len(mothers), dtype=np.int64), counts)
        base = {key: values[particle_positions] for key, values in self.raw.items()}

        pieces = {key: [] for key in RAW_KEYS}
        for rotation in range(angles.shape[1]):
            phi = angles[:, rotation][owners]
            cosine = np.cos(phi)
            sine = np.sin(phi)

            px = base["px"]
            py = base["py"]
            vx = base["vx"]
            vy = base["vy"]
            pieces["px"].append((cosine * px - sine * py).astype(np.float32))
            pieces["py"].append((sine * px + cosine * py).astype(np.float32))
            pieces["vx"].append((cosine * vx - sine * vy).astype(np.float32))
            pieces["vy"].append((sine * vx + cosine * vy).astype(np.float32))
            for key in ("pz", "E", "t", "vz", "pdg"):
                pieces[key].append(base[key])

        return {key: np.concatenate(value) for key, value in pieces.items()}


def cycle_split_mothers(store, fractions=(0.5, 0.25, 0.25), seed=1):
    """Cycle-disjoint train/validation/test arrays of mother positions."""
    fractions = np.asarray(fractions, dtype=np.float64)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("split fractions must be positive and sum to one")
    order = np.random.default_rng(seed).permutation(store.n_cycles)
    n_train = int(round(store.n_cycles * fractions[0]))
    n_val = int(round(store.n_cycles * fractions[1]))
    cycle_splits = {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }
    output = {}
    for name, cycles in cycle_splits.items():
        starts = store.cycle_offsets[cycles]
        counts = store.cycle_offsets[cycles + 1] - starts
        output[name] = expand_ranges(starts, counts)
    return output


def sample_definition(rng, mother_pool, reuse_k, mother_equivalents,
                      rotation_policy="all-random"):
    """Select unique sources and angles for one fixed-size pseudo-event."""
    if reuse_k < 1 or mother_equivalents < 1:
        raise ValueError("reuse and event size must be positive")
    if mother_equivalents % reuse_k:
        raise ValueError("mother-equivalents must be divisible by every reuse k")
    n_unique = mother_equivalents // reuse_k
    if len(mother_pool) < n_unique:
        raise ValueError("pool has {} mothers but unit needs {} unique sources"
                         .format(len(mother_pool), n_unique))
    mothers = rng.choice(mother_pool, size=n_unique, replace=False)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(n_unique, reuse_k))
    if rotation_policy == "baseline-unrotated" and reuse_k == 1:
        angles.fill(0.0)
    elif rotation_policy == "include-original":
        angles[:, 0] = 0.0
    elif rotation_policy not in ("all-random", "baseline-unrotated"):
        raise ValueError("unknown rotation policy: {}".format(rotation_policy))
    return {"mothers": mothers, "angles": angles, "reuse_k": int(reuse_k)}
