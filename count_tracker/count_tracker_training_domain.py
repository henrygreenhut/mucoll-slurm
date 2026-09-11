#!/usr/bin/env python3
"""Verify the model's flat training arrays against their source HDF5 table.

The six published training ``.npy`` files are flat and contain no event ID.
Their sibling HDF5 table has an ``event`` column, but the verified file contains
only ``event=0`` and therefore provides no event boundaries. This program proves
that ``inside_bounds`` rows reproduce all six arrays in their original order and
reports the event-column contents and collection counts. It writes no dataset.

Runtime dependency: pandas with PyTables (``tables``), needed for the Blosc
compressed pandas HDFStore.  This script does not submit jobs.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from count_tracker_conditions import COLLECTIONS


MODEL_COLUMNS = ["Edep", "x", "y", "z", "t", "system", "side", "layer", "module", "sensor"]
HDF_COLUMNS = ["event", "collection", *MODEL_COLUMNS, "cellid0"]
SELECTION = "inside_bounds == True"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_strings(values):
    """Return fixed-width HDF strings as ordinary Python strings."""
    return np.asarray([
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ])


def model_rows(frame, mask=None):
    """Convert the ten model columns to the exact flat-array representation."""
    rows = np.column_stack([np.asarray(frame[column]) for column in MODEL_COLUMNS])
    if mask is not None:
        rows = rows[np.asarray(mask)]
    return np.asarray(rows, dtype=np.float64)


def compare_training_chunk(reference, offset, rows, collection):
    """Require an exact, ordered match to one slice of a training array."""
    end = offset + len(rows)
    if end > len(reference):
        raise ValueError(
            f"{collection}: HDF selection has more rows than the training array "
            f"(first overflow at row {offset})"
        )
    expected = np.asarray(reference[offset:end])
    equal = np.equal(rows, expected)
    if not np.all(equal):
        row, column = np.argwhere(~equal)[0]
        raise ValueError(
            f"{collection}: HDF selection does not reproduce the training array at "
            f"row {offset + int(row)}, column {MODEL_COLUMNS[int(column)]}: "
            f"HDF={rows[row, column]!r}, NPY={expected[row, column]!r}"
        )
    return end


def training_array_paths(directory):
    directory = Path(directory).expanduser().resolve()
    result = {}
    for short, (_, name) in COLLECTIONS.items():
        path = directory / f"{name}_SimTrackerHit_conditional_reco9_0.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {short} training array: {path}")
        result[short] = path
    return result


def scan_and_verify(hdf5, npy_directory, chunk_size):
    """Verify the HDF selection against every NPY and count selected hits/event."""
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "Reading this Blosc-compressed pandas HDF5 requires pandas and PyTables "
            "(the Python package 'tables')"
        ) from error

    paths = training_array_paths(npy_directory)
    references = {short: np.load(path, mmap_mode="r", allow_pickle=False)
                  for short, path in paths.items()}
    for short, rows in references.items():
        if rows.ndim != 2 or rows.shape[1] != len(MODEL_COLUMNS) or rows.dtype != np.float64:
            raise ValueError(f"{short}: expected float64 training array with shape (N, 10)")

    name_to_short = {name: short for short, (_, name) in COLLECTIONS.items()}
    offsets = {short: 0 for short in COLLECTIONS}
    event_counts = Counter()
    event_collection_counts = {short: Counter() for short in COLLECTIONS}
    iterator = pd.read_hdf(
        hdf5, key="df", where=SELECTION, columns=HDF_COLUMNS,
        chunksize=chunk_size, iterator=True,
    )
    for chunk in iterator:
        names = collection_strings(chunk["collection"])
        unknown = sorted(set(names) - set(name_to_short))
        if unknown:
            raise ValueError(f"Unknown tracker collections in HDF5: {unknown}")
        events = np.asarray(chunk["event"], dtype=np.int64)
        unique, counts = np.unique(events, return_counts=True)
        event_counts.update({int(event): int(count) for event, count in zip(unique, counts)})
        for name, short in name_to_short.items():
            mask = names == name
            if np.any(mask):
                rows = model_rows(chunk, mask)
                offsets[short] = compare_training_chunk(
                    references[short], offsets[short], rows, short)
                selected_events = events[mask]
                unique, counts = np.unique(selected_events, return_counts=True)
                event_collection_counts[short].update(
                    {int(event): int(count) for event, count in zip(unique, counts)})

    for short, reference in references.items():
        if offsets[short] != len(reference):
            raise ValueError(
                f"{short}: HDF selection reproduced {offsets[short]} of "
                f"{len(reference)} training rows"
            )
    if not event_counts:
        raise ValueError("The verified HDF selection contains no events")
    return event_counts, event_collection_counts, paths, offsets


def count_distribution(counts, events):
    """Compact event-multiplicity summary, including zero-hit collections."""
    values = np.asarray([counts.get(event, 0) for event in events], dtype=np.int64)
    return {
        "min": int(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": int(values.max()),
    }


def verify_only(args):
    hdf5 = Path(args.hdf5).expanduser().resolve()
    if not hdf5.is_file():
        raise FileNotFoundError(f"Missing HDF5 source: {hdf5}")
    event_counts, event_collection_counts, paths, collection_counts = scan_and_verify(
        hdf5, args.training_npys, args.chunk_size)
    events = sorted(event_counts)
    result = {
        "status": "exact ordered match",
        "selection": SELECTION,
        "source_hdf5": str(hdf5),
        "n_source_events": len(event_counts),
        "source_event_range": [min(event_counts), max(event_counts)],
        "selected_hits": int(sum(event_counts.values())),
        "hits_per_source_event": count_distribution(event_counts, events),
        "collections": {
            short: {
                "path": str(paths[short]),
                "rows": collection_counts[short],
                "hits_per_source_event": count_distribution(
                    event_collection_counts[short], events),
            }
            for short in COLLECTIONS
        },
    }
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, help="Event-aware nuGun source HDF5")
    parser.add_argument("--training-npys", required=True, help="Directory with the six flat arrays")
    parser.add_argument("--verify-only", action="store_true", required=True,
                        help="verify HDF-to-NPY identity and write no files")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    verify_only(args)


if __name__ == "__main__":
    main()
