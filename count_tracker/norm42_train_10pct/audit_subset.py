#!/usr/bin/env python3
"""Audit hit and sensor coverage for a selected Norm42 source cohort."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


COLLECTIONS = ("VBC", "VEC", "ITBC", "ITEC", "OTBC", "OTEC")
POLARITIES = ("MUPLUS", "MUMINUS")
CONDITION_COLUMNS = (5, 6, 7, 8, 9)
CONDITION_NAMES = ("system", "side", "layer", "module", "sensor")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_source_ids(manifest):
    ids = []
    by_polarity = {}
    for polarity in POLARITIES:
        values = [int(row["source_id"]) for row in manifest["sources"][polarity]]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate selected source IDs for {polarity}")
        by_polarity[polarity] = np.asarray(sorted(values), dtype=np.int32)
        ids.extend(values)
    return np.asarray(sorted(ids), dtype=np.int32), by_polarity


def inventory_source_ids(manifest):
    inventory_path = Path(manifest["source_inventory"])
    expected_hash = manifest.get("source_inventory_sha256")
    if expected_hash is not None and sha256(inventory_path) != expected_hash:
        raise ValueError("source inventory hash does not match manifest")
    inventory = json.loads(inventory_path.read_text())
    ids = []
    for polarity in POLARITIES:
        rows = inventory["sources"][polarity]
        ids.extend(int(row["source_id"]) for row in rows)
    values = np.asarray(sorted(ids), dtype=np.int32)
    if len(values) != len(np.unique(values)):
        raise ValueError("duplicate source IDs in source inventory")
    return values


def pack_conditions(values):
    values = np.asarray(values)
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError("condition columns contain non-integral values")
    system, side, layer, module, sensor = rounded.astype(np.int64).T
    bounds = (
        np.all((0 <= system) & (system < 16))
        and np.all((-2 <= side) & (side < 2))
        and np.all((0 <= layer) & (layer < 64))
        and np.all((0 <= module) & (module < 4096))
        and np.all((0 <= sensor) & (sensor < 4096))
    )
    if not bounds:
        raise ValueError("condition value outside the documented packing range")
    return (
        system
        | ((side + 2) << 4)
        | (layer << 6)
        | (module << 12)
        | (sensor << 24)
    ).astype(np.int64)


def unpack_condition(key):
    key = int(key)
    return (
        key & 0xF,
        ((key >> 4) & 0x3) - 2,
        (key >> 6) & 0x3F,
        (key >> 12) & 0xFFF,
        (key >> 24) & 0xFFF,
    )


def add_counts(destination, keys):
    unique, counts = np.unique(keys, return_counts=True)
    for key, count in zip(unique.tolist(), counts.tolist()):
        destination[int(key)] += int(count)


def percentiles(counts):
    if not counts:
        return None
    values = np.asarray(list(counts.values()), dtype=np.int64)
    return {
        "min": int(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": int(values.max()),
    }


def audit_collection(
        data_path, source_path, reference_sources, selected, by_polarity,
        chunk_rows):
    data = np.load(data_path, mmap_mode="r", allow_pickle=False)
    sources = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 2 or data.shape[1] != 10:
        raise ValueError(f"{data_path}: expected shape (N, 10), got {data.shape}")
    if sources.ndim != 1 or len(sources) != len(data):
        raise ValueError(
            f"row mismatch: data {data.shape}, source IDs {sources.shape}"
        )
    if sources.dtype != np.int32:
        raise ValueError(f"{source_path}: expected int32, got {sources.dtype}")

    reference_counts = defaultdict(int)
    selected_counts = defaultdict(int)
    selected_hits = 0
    selected_by_polarity = {polarity: 0 for polarity in POLARITIES}

    for start in range(0, len(data), chunk_rows):
        stop = min(start + chunk_rows, len(data))
        source_chunk = np.asarray(sources[start:stop])
        keys = pack_conditions(np.asarray(data[start:stop, CONDITION_COLUMNS]))
        reference_mask = np.isin(
            source_chunk, reference_sources, assume_unique=False
        )
        if np.any(reference_mask):
            add_counts(reference_counts, keys[reference_mask])

        mask = np.isin(source_chunk, selected, assume_unique=False)
        if np.any(mask):
            add_counts(selected_counts, keys[mask])
            selected_hits += int(mask.sum())
            for polarity, ids in by_polarity.items():
                selected_by_polarity[polarity] += int(
                    np.isin(source_chunk, ids, assume_unique=False).sum()
                )

    full_keys = set(reference_counts)
    selected_keys = set(selected_counts)
    missing = sorted(full_keys - selected_keys)
    hits_in_missing = sum(reference_counts[key] for key in missing)
    reference_hits = sum(reference_counts.values())
    return {
        "rows": int(len(data)),
        "valid_reference_hits": int(reference_hits),
        "selected_hits": selected_hits,
        "selected_fraction_of_valid_reference": (
            selected_hits / reference_hits if reference_hits else None
        ),
        "selected_hits_by_polarity": selected_by_polarity,
        "valid_reference_sensor_conditions": len(full_keys),
        "selected_sensor_conditions": len(selected_keys),
        "missing_sensor_conditions": len(missing),
        "valid_reference_hits_in_missing_conditions": int(hits_in_missing),
        "valid_reference_hit_fraction_in_missing_conditions": (
            hits_in_missing / reference_hits if reference_hits else None
        ),
        "valid_reference_hits_per_condition": percentiles(reference_counts),
        "selected_hits_per_condition": percentiles(selected_counts),
        "missing": [
            {
                **dict(zip(CONDITION_NAMES, unpack_condition(key))),
                "valid_reference_hits": int(reference_counts[key]),
            }
            for key in missing
        ],
    }


def audit(manifest_path, data_root, source_id_root, output, chunk_rows):
    if chunk_rows < 1:
        raise ValueError("chunk rows must be positive")
    manifest_path = Path(manifest_path).resolve()
    data_root = Path(data_root).resolve()
    source_id_root = Path(source_id_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "norm42_10pct_training_development_corpus":
        raise ValueError("unexpected source-manifest kind")
    selected, by_polarity = selected_source_ids(manifest)
    reference_sources = inventory_source_ids(manifest)
    if not np.all(np.isin(selected, reference_sources)):
        raise ValueError("selected source is absent from the recorded inventory")
    expected = manifest["selection"]["files_per_polarity"]
    if any(len(by_polarity[p]) != expected for p in POLARITIES):
        raise ValueError("source count does not match manifest normalization")

    output.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "kind": "norm42_10pct_training_coverage_audit",
        "source_manifest": str(manifest_path),
        "data_root": str(data_root),
        "source_id_root": str(source_id_root),
        "chunk_rows": chunk_rows,
        "selection": manifest["selection"],
        "collections": {},
    }
    missing_rows = []
    for collection in COLLECTIONS:
        print(f"Auditing {collection}...", flush=True)
        result = audit_collection(
            data_root / f"{collection}_data.npy",
            source_id_root / f"{collection}_source_id.npy",
            reference_sources,
            selected,
            by_polarity,
            chunk_rows,
        )
        missing = result.pop("missing")
        for row in missing:
            missing_rows.append({"collection": collection, **row})
        report["collections"][collection] = result
        print(
            f"  {result['selected_hits']:,}/{result['rows']:,} hits; "
            f"{result['selected_sensor_conditions']:,}/"
            f"{result['valid_reference_sensor_conditions']:,} conditions; "
            f"{result['missing_sensor_conditions']:,} missing",
            flush=True,
        )

    total_rows = sum(row["rows"] for row in report["collections"].values())
    total_reference = sum(
        row["valid_reference_hits"] for row in report["collections"].values()
    )
    total_selected = sum(
        row["selected_hits"] for row in report["collections"].values()
    )
    total_missing_hits = sum(
        row["valid_reference_hits_in_missing_conditions"]
        for row in report["collections"].values()
    )
    report["total"] = {
        "rows": total_rows,
        "valid_reference_hits": total_reference,
        "selected_hits": total_selected,
        "selected_fraction_of_valid_reference": total_selected / total_reference,
        "missing_sensor_conditions": sum(
            row["missing_sensor_conditions"]
            for row in report["collections"].values()
        ),
        "valid_reference_hits_in_missing_conditions": total_missing_hits,
        "valid_reference_hit_fraction_in_missing_conditions": (
            total_missing_hits / total_reference
        ),
    }
    report["decision_gate"] = {
        "passed_complete_sensor_coverage": not missing_rows,
        "action": (
            "proceed to reproduce the recorded grouped 80/20 split"
            if not missing_rows
            else "stop and review unsupported sensor conditions before training"
        ),
    }
    (output / "coverage_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (output / "missing_sensor_conditions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "collection", *CONDITION_NAMES, "valid_reference_hits"
            ),
        )
        writer.writeheader()
        writer.writerows(missing_rows)

    print(json.dumps({"total": report["total"], "decision_gate": report["decision_gate"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-id-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    args = parser.parse_args()
    audit(
        args.manifest,
        args.data_root,
        args.source_id_root,
        args.output,
        args.chunk_rows,
    )


if __name__ == "__main__":
    main()
