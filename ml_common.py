import csv
import hashlib
import os

import h5py
import numpy as np


VAL_FRAC = 0.1
TEST_FRAC = 0.3
SEED = 1
PFO_FEATURES = [
    "pt",
    "eta",
    "phi",
    "energy",
    "mass",
    "charge",
    "type",
    "px",
    "py",
    "pz",
]
PFO = {name: i for i, name in enumerate(PFO_FEATURES)}


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def real_pfos(event):
    return event[event[:, 0] > 0]


def event_hash(event):
    return hashlib.sha1(real_pfos(event).tobytes()).hexdigest()[:12]


def read_optional(h5, name, n, default, dtype=None):
    if name in h5:
        values = h5[name][:n]
        return values.astype(dtype) if dtype is not None else values
    return np.full(n, default, dtype=dtype)


def load_h5(path):
    with h5py.File(path, "r") as h5:
        particles = h5["particles"][:]
        class_name = decode(h5.attrs.get("class_name", os.path.basename(path)))
        features = decode(h5.attrs.get("features", ""))
        source_files = read_optional(h5, "source_file", len(particles), "")
        source_events = read_optional(h5, "source_event", len(particles), -1, np.int32)

    meta = []
    for i, event in enumerate(particles):
        pfos = real_pfos(event)
        if particles.shape[2] > PFO["energy"]:
            energy = pfos[:, PFO["energy"]]
        else:
            energy = np.zeros(len(pfos))
        meta.append({
            "sample_path": path,
            "event_index": i,
            "source_file": decode(source_files[i]),
            "source_event": int(source_events[i]),
            "n_particles": int(len(pfos)),
            "sum_pt": float(pfos[:, 0].sum()) if len(pfos) else 0.0,
            "leading_pt": float(pfos[:, 0].max()) if len(pfos) else 0.0,
            "sum_energy": float(np.sum(energy)) if len(pfos) else 0.0,
            "leading_energy": float(np.max(energy)) if len(pfos) else 0.0,
            "event_hash": event_hash(event),
        })

    print(f"  {class_name}: {len(particles)} events, {particles.shape[1]} PFO slots")
    if features:
        print(f"    features: {features}")
    return particles, meta, class_name


def fit_slots(a, b):
    width = max(
        int((a[:, :, 0] > 0).sum(axis=1).max()),
        int((b[:, :, 0] > 0).sum(axis=1).max()),
        1,
    )

    def fit(x):
        if x.shape[1] == width:
            return x
        if x.shape[1] > width:
            return x[:, :width, :]
        out = np.zeros((x.shape[0], width, x.shape[2]), dtype=x.dtype)
        out[:, :x.shape[1], :] = x
        return out

    return fit(a), fit(b)


def split_by_class(y):
    rng = np.random.default_rng(SEED)
    train_idx = []
    val_idx = []
    test_idx = []

    for cls in np.unique(y):
        cls_idx = rng.permutation(np.where(y == cls)[0])
        n_test = max(1, int(len(cls_idx) * TEST_FRAC))
        n_val = max(1, int(len(cls_idx) * VAL_FRAC))
        if n_test + n_val >= len(cls_idx):
            raise SystemExit("Need more events per class for train/val/test split")
        test_idx.extend(cls_idx[:n_test])
        val_idx.extend(cls_idx[n_test:n_test + n_val])
        train_idx.extend(cls_idx[n_test + n_val:])

    return (
        rng.permutation(np.asarray(train_idx, dtype=np.int64)),
        rng.permutation(np.asarray(val_idx, dtype=np.int64)),
        rng.permutation(np.asarray(test_idx, dtype=np.int64)),
    )


def write_rows(path, fields, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
