#!/usr/bin/env python3
"""Run GenBIB CellID assignment in bounded-memory row chunks.

The native ``assign_actual_cellid.py`` conversion is applied unchanged to
ordered slices of each collection.  Sensor lookup and containment decisions
are independent per hit, so chunking changes only peak memory use.  Completed
NumPy files are published atomically; no partial output is mistaken for a
finished collection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


COUNT_KEYS = (
    "written_rows",
    "invalid_rows",
    "multi_assigned_rows",
    "rescued_rows",
    "lookup_rescued_rows",
)
OUTPUT_COLUMNS = 10


def load_native(path: Path):
    """Load the pinned GenBIB implementation without modifying its checkout."""
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("genbib_assign_actual_cellid", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load native CellID assignment from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_save_empty(path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, np.empty((0, OUTPUT_COLUMNS), dtype=np.float64))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def compact_memmap(source_path: Path, destination_path: Path, rows: int,
                   chunk_size: int) -> None:
    """Publish the filled prefix of a maximum-sized temporary array."""
    if rows == 0:
        atomic_save_empty(destination_path)
        return
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    compact_path = destination_path.with_name(
        f".{destination_path.name}.{os.getpid()}.compact"
    )
    try:
        compact = np.lib.format.open_memmap(
            compact_path, mode="w+", dtype=np.float64,
            shape=(rows, OUTPUT_COLUMNS),
        )
        for start in range(0, rows, chunk_size):
            stop = min(start + chunk_size, rows)
            compact[start:stop] = source[start:stop]
        compact.flush()
        del compact
        os.replace(compact_path, destination_path)
    finally:
        compact_path.unlink(missing_ok=True)


def convert_collection(native, rows, output_path: Path, *, input_format: str,
                       system_id: int, geom, k: int, ref_lookup,
                       chunk_size: int, label: str):
    """Convert one collection with the native per-hit algorithm in slices."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    totals = {key: 0 for key in COUNT_KEYS}
    if total == 0:
        atomic_save_empty(output_path)
        return totals

    staging_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.maximum"
    )
    written = 0
    try:
        staging = np.lib.format.open_memmap(
            staging_path, mode="w+", dtype=np.float64,
            shape=(total, OUTPUT_COLUMNS),
        )
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            converted, counts = native.convert_rows(
                rows[start:stop], input_format, system_id, geom, k, ref_lookup
            )
            if converted.ndim != 2 or converted.shape[1] != OUTPUT_COLUMNS:
                raise ValueError(
                    f"{label}: native conversion returned shape {converted.shape}"
                )
            next_written = written + len(converted)
            staging[written:next_written] = converted
            written = next_written
            for key in COUNT_KEYS:
                totals[key] += int(counts[key])
            print(
                f"{label}: processed {stop:,}/{total:,}; written {written:,}",
                flush=True,
            )
        staging.flush()
        del staging

        if written == total:
            os.replace(staging_path, output_path)
        else:
            compact_memmap(staging_path, output_path, written, chunk_size)
            staging_path.unlink()
    finally:
        staging_path.unlink(missing_ok=True)
    return totals


def write_summary(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def assign(args) -> None:
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    native = load_native(args.native_script)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    geom = native.load_or_build_sensor_geometry(args.geomap_path)
    print(f"loaded sensor geometry: {args.geomap_path}", flush=True)

    summary_rows = []
    for index, (input_stem, output_stem, system_id) in enumerate(
        native.collection_specs(args.input_format)
    ):
        input_path = input_dir / f"{input_stem}.npy"
        output_path = output_dir / f"{output_stem}.npy"
        rows = np.load(input_path, mmap_mode="r", allow_pickle=False)
        selected = native.select_rows(
            rows, args.write_fraction, args.sample_mode, args.random_seed + index
        )
        ref_lookup = None
        if native.REF_H5_PATH is not None:
            collection = output_stem.removesuffix(
                "_SimTrackerHit_conditional_reco9_0"
            )
            ref_lookup = native.load_h5_reference_lookup(
                native.REF_H5_PATH, collection
            )
        counts = convert_collection(
            native, selected, output_path,
            input_format=args.input_format, system_id=system_id,
            geom=geom, k=args.k, ref_lookup=ref_lookup,
            chunk_size=args.chunk_size, label=output_stem,
        )
        summary = {
            "collection": output_stem,
            "input_file": str(input_path),
            "source_rows": len(rows),
            "selected_rows": len(selected),
            "written_rows": counts["written_rows"],
            "unresolved_rows": counts["invalid_rows"],
            "multi_assigned_rows": counts["multi_assigned_rows"],
            "lookup_rescued_rows": counts["lookup_rescued_rows"],
        }
        summary_rows.append(summary)
        print(
            f"{output_stem}: source={len(rows)} selected={len(selected)} "
            f"written={counts['written_rows']} "
            f"multi_assigned={counts['multi_assigned_rows']} "
            f"unresolved={counts['invalid_rows']}",
            flush=True,
        )

    summary_path = output_dir / "assign_actual_cellid_summary.csv"
    write_summary(summary_path, summary_rows)
    native_path = args.native_script.resolve()
    write_json(output_dir / "assign_actual_cellid_manifest.json", {
        "kind": "chunked_genbib_cellid_assignment",
        "native_script": str(native_path),
        "native_script_sha256": sha256_file(native_path),
        "chunk_size": args.chunk_size,
        "candidate_sensors_per_hit": args.k,
        "input_format": args.input_format,
        "write_fraction": args.write_fraction,
        "sample_mode": args.sample_mode,
        "random_seed": args.random_seed,
        "geometry_map": str(args.geomap_path.resolve()),
        "collections": summary_rows,
    })
    print(f"summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-script", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-format", choices=("9col", "10col", "rphi10"), default="9col"
    )
    parser.add_argument("--write-fraction", type=float, default=1.0)
    parser.add_argument("--sample-mode", choices=("first", "random"), default="first")
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--geomap-path", type=Path, required=True)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    assign(parser.parse_args())


if __name__ == "__main__":
    main()
