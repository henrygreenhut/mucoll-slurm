#!/usr/bin/env python3
"""Prepare event-level SIM hits and COUNT conditions from the model source HDF5.

The six published training ``.npy`` files are flat and contain no event ID.
Their sibling HDF5 table retains that ID and the original CellID.  This adapter
first proves that ``inside_bounds`` rows from the HDF5 reproduce all six flat
arrays, in their original order.  It aborts on the first discrepancy.  Only
after that closure check does it choose distinct source events, partition them
at event level, and write the inputs consumed by the COUNT tracker pipeline.

This is deliberately labelled ``training_domain``: the new event partition is
an evaluation partition for the reconstructed classifier, not evidence that
the selected hits were held out when the generative model was trained.

Runtime dependency: pandas with PyTables (``tables``), needed for the Blosc
compressed pandas HDFStore.  This script does not submit jobs.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import tempfile

import numpy as np

from count_tracker_conditions import CELL_ID_ENCODING, COLLECTIONS, decode_cell_ids, sensor_counts


MODEL_COLUMNS = ["Edep", "x", "y", "z", "t", "system", "side", "layer", "module", "sensor"]
HDF_COLUMNS = ["event", "collection", *MODEL_COLUMNS, "cellid0"]
SIM_COLUMNS = [*MODEL_COLUMNS, "cellid0"]
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


def choose_event_splits(event_counts, split_sizes, seed):
    """Choose distinct nonempty source events and partition them event-wise."""
    requested = sum(split_sizes.values())
    eligible = sorted(event for event, hits in event_counts.items() if hits > 0)
    if requested > len(eligible):
        raise ValueError(f"Requested {requested} events but only {len(eligible)} are available")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = {}
    cursor = 0
    for split in ("train", "val", "test"):
        size = split_sizes[split]
        selected[split] = eligible[cursor:cursor + size]
        cursor += size
    return selected


def validate_source_event(frame, source_event):
    if len(frame) == 0:
        raise ValueError(f"Source event {source_event} contains no selected tracker hits")
    events = np.asarray(frame["event"], dtype=np.int64)
    if np.any(events != source_event):
        raise ValueError(f"HDF query for event {source_event} returned another event")
    if not np.all(np.isfinite(model_rows(frame))):
        raise ValueError(f"Source event {source_event} contains nonfinite model values")
    if np.any(np.asarray(frame["Edep"], dtype=np.float64) < 0):
        raise ValueError(f"Source event {source_event} contains negative energy deposition")


def event_products(frame, source_event):
    """Build per-collection condition and SIM arrays, checking original CellIDs."""
    validate_source_event(frame, source_event)
    names = collection_strings(frame["collection"])
    products = {}
    for short, (system, name) in COLLECTIONS.items():
        mask = names == name
        selected = frame.loc[mask]
        rows = model_rows(selected)
        ids = np.asarray(selected["cellid0"], dtype=np.int64)
        labels = rows[:, 5:].astype(np.int64)
        if not np.all(rows[:, 5:] == labels):
            raise ValueError(f"{source_event}/{short}: noninteger sensor identifiers")
        decoded = decode_cell_ids(ids, system)
        if not np.array_equal(decoded, labels):
            raise ValueError(f"{source_event}/{short}: HDF labels disagree with cellid0")
        sim = np.column_stack([rows, ids]).astype(np.float64)
        products[short] = (labels, sim)
    if sum(len(sim) for _, sim in products.values()) != len(frame):
        raise ValueError(f"Source event {source_event} contains unrecognized collections")
    return products


def read_source_event(hdf5, source_event):
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("pandas and PyTables are required") from error
    return pd.read_hdf(
        hdf5, key="df", where=[f"event == {source_event}", SELECTION],
        columns=HDF_COLUMNS,
    )


def write_event(work, split, source_event, products):
    event_id = f"training_domain_{source_event:06d}"
    destination = work / split / event_id
    destination.mkdir(parents=True)
    collections = {}
    arrays = {}
    for short, (conditions, sim) in products.items():
        condition_path = destination / f"{short}_conditions.npy"
        sim_path = destination / f"{short}_sim_hits.npy"
        np.save(condition_path, conditions, allow_pickle=False)
        np.save(sim_path, sim, allow_pickle=False)
        collections[short] = {
            "hits": int(len(conditions)),
            "occupied_sensors": int(len(sensor_counts(conditions))),
            "sha256": sha256_file(condition_path),
        }
        arrays[short] = {
            "path": str(sim_path.relative_to(work)),
            "hits": int(len(sim)),
            "sha256": sha256_file(sim_path),
        }
    event = {
        "event_id": event_id, "split": split, "source_event": int(source_event),
        "sim_arrays": arrays,
    }
    summary = {"event_id": event_id, "split": split, "collections": collections}
    return event, summary


def prepare(args):
    hdf5 = Path(args.hdf5).expanduser().resolve()
    if not hdf5.is_file():
        raise FileNotFoundError(f"Missing HDF5 source: {hdf5}")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    event_counts, _, npy_paths, collection_counts = scan_and_verify(
        hdf5, args.training_npys, args.chunk_size)
    splits = choose_event_splits(
        event_counts,
        {"train": args.n_train, "val": args.n_val, "test": args.n_test},
        args.seed,
    )

    with tempfile.TemporaryDirectory(prefix=".count_training_domain_", dir=output.parent) as tmp:
        work = Path(tmp)
        events = []
        summaries = []
        for split in ("train", "val", "test"):
            for source_event in splits[split]:
                frame = read_source_event(hdf5, source_event)
                event, summary = write_event(
                    work, split, source_event, event_products(frame, source_event))
                events.append(event)
                summaries.append(summary)
                print(f"Prepared {split}/{event['event_id']}", flush=True)

        manifest = {
            "schema_version": 2,
            "construction": "training_domain",
            "sampling": {"cohort": "HDF_SOURCE", "seed": args.seed},
            "cell_id_encoding": CELL_ID_ENCODING,
            "source_hdf5": str(hdf5),
            "selection": SELECTION,
            "evaluation_partition": "new deterministic event-level partition",
            "generator_training_holdout": False,
            "events": events,
        }
        report = {
            "manifest": manifest,
            "kind": "count_tracker_training_domain_conditions",
            "condition_columns": ["system", "side", "layer", "module", "sensor"],
            "sim_columns": SIM_COLUMNS,
            "training_array_verification": {
                "status": "exact ordered match",
                "selection": SELECTION,
                "arrays": {
                    short: {"path": str(npy_paths[short]), "rows": collection_counts[short]}
                    for short in COLLECTIONS
                },
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": importlib.metadata.version("pandas"),
                "tables": importlib.metadata.version("tables"),
            },
            "events": summaries,
        }
        (work / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Output appeared during preparation: {output}")
        os.replace(work, output)


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
    parser.add_argument("--verify-only", action="store_true",
                        help="verify HDF-to-NPY identity and write no files")
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-val", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.seed < 0 or args.chunk_size < 1:
        parser.error("--seed must be nonnegative and --chunk-size must be positive")
    cohort_args = (args.n_train, args.n_val, args.n_test, args.output)
    if args.verify_only:
        if any(value is not None for value in cohort_args):
            parser.error("--verify-only must omit split sizes and --output")
        verify_only(args)
    else:
        if any(value is None for value in cohort_args):
            parser.error("preparation requires --n-train, --n-val, --n-test, and --output")
        if min(args.n_train, args.n_val, args.n_test) < 1:
            parser.error("--n-train, --n-val, and --n-test must each be positive")
        prepare(args)


if __name__ == "__main__":
    main()
