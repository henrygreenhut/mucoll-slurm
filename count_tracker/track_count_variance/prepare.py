#!/usr/bin/env python3
"""Plan a resumable Norm42 ensemble for SIM-versus-COUNT track statistics."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from count_tracker_conditions import (
    CELL_ID_ENCODING,
    CONSTRUCTIONS,
    NORM42_OCCUPANCY_SCAN_NAME,
    validate_manifest,
)
from count_tracker_manifest import POLARITIES


INVALID = {"MUPLUS": set(), "MUMINUS": {6291}}
FULL_BX_FILES_PER_POLARITY = 1666


def index_sources(sim_root):
    """Index every present, valid source independently for each polarity."""
    sim_root = Path(sim_root).resolve()
    inventory = {}
    for polarity in POLARITIES:
        directory = sim_root / polarity
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        records = {}
        for path in sorted(directory.glob("bib_sim_*.edm4hep.root")):
            match = re.fullmatch(r"bib_sim_(\d+)\.edm4hep\.root", path.name)
            if match is None or not path.is_file():
                raise ValueError(f"unexpected source path: {path}")
            cycle = int(match.group(1))
            if cycle in INVALID[polarity]:
                continue
            if cycle in records:
                raise ValueError(f"duplicate {polarity} cycle {cycle}")
            records[cycle] = {
                "cycle": cycle,
                "path": str(path.resolve()),
                "entry": 0,
                "size_bytes": path.stat().st_size,
            }
        if not records:
            raise ValueError(f"no valid sources found in {directory}")
        inventory[polarity] = records
    return sim_root, inventory


def select_event(inventory, seed, cohort, index, n_files):
    event_id = f"norm42_{cohort}_test_{index:06d}"
    sources = {}
    for polarity in POLARITIES:
        available = sorted(inventory[polarity])
        if len(available) < n_files:
            raise ValueError(f"fewer than {n_files} valid {polarity} sources")
        stream = json.dumps(
            [seed, "norm42", cohort, "test", index, polarity, "all-valid-polarity-files"],
            separators=(",", ":"),
        )
        selected = random.Random(stream).sample(available, n_files)
        sources[polarity] = [dict(inventory[polarity][cycle]) for cycle in selected]
    return {"event_id": event_id, "split": "test", "sources": sources}


def prepare(sim_root, output, events, batch_size, n_files, seed, cohort):
    if events < 2:
        raise ValueError("at least two events are required to estimate a variance")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if n_files < 1:
        raise ValueError("files per polarity must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cohort):
        raise ValueError("cohort must contain letters, digits, '_' or '-'")

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    source_root, inventory = index_sources(sim_root)
    output.mkdir(parents=True)

    inventory_document = {
        "kind": "norm42-independent-polarity-source-inventory",
        "root": str(source_root),
        "selection": "all present files except explicitly invalid files",
        "invalid_excluded": {key: sorted(value) for key, value in INVALID.items()},
        "counts": {polarity: len(inventory[polarity]) for polarity in POLARITIES},
        "sources": {
            polarity: [inventory[polarity][cycle] for cycle in sorted(inventory[polarity])]
            for polarity in POLARITIES
        },
    }
    inventory_path = output / "source_inventory.json"
    inventory_path.write_text(json.dumps(inventory_document, indent=2) + "\n")
    inventory_sha = hashlib.sha256(inventory_path.read_bytes()).hexdigest()

    planned = [
        select_event(inventory, seed, cohort, index, n_files)
        for index in range(events)
    ]
    normalization = {
        "name": NORM42_OCCUPANCY_SCAN_NAME,
        "selected_files_per_polarity": n_files,
        "full_bx_reference_files_per_polarity": FULL_BX_FILES_PER_POLARITY,
        "fraction_of_full_bx_reference": n_files / FULL_BX_FILES_PER_POLARITY,
        "selection": "fixed-size repeated empirical source draw",
    }
    pool = {
        "kind": "all-valid-files-indexed-independently-by-polarity",
        "path": str(source_root),
        "inventory": str(inventory_path),
        "inventory_sha256": inventory_sha,
        "files": {polarity: len(inventory[polarity]) for polarity in POLARITIES},
    }
    sampling = {
        "seed": seed,
        "cohort": cohort,
        "within_event": "without replacement, independently by polarity",
        "across_events": "source reuse permitted",
        "classifier_split": "not applicable; repeated-sampling diagnostic",
        "stream": (
            "random.Random(JSON([seed, norm42, cohort, test, global_event_index, "
            "polarity, all-valid-polarity-files]))"
        ),
    }

    points = []
    for batch_index, start in enumerate(range(0, events, batch_size)):
        batch_events = planned[start:start + batch_size]
        tag = f"batch_{batch_index:04d}"
        destination = output / tag / "sources.json"
        destination.parent.mkdir()
        manifest = {
            "schema_version": 2,
            "cell_id_encoding": CELL_ID_ENCODING,
            "construction": "norm42",
            "n_files_per_polarity": n_files,
            "file_normalization": CONSTRUCTIONS["norm42"]["file_normalization"],
            "norm1_equivalents_per_polarity": (
                n_files * CONSTRUCTIONS["norm42"]["file_normalization"]),
            "source_pool": "all",
            "pool": pool,
            "overlay_normalization": normalization,
            "sampling": sampling,
            "events": batch_events,
        }
        validate_manifest(manifest, destination.parent)
        destination.write_text(json.dumps(manifest, indent=2) + "\n")
        points.append({
            "tag": tag,
            "batch_index": batch_index,
            "first_event_index": start,
            "events": len(batch_events),
            "files_per_polarity": n_files,
            "fraction_of_full_bx_reference": n_files / FULL_BX_FILES_PER_POLARITY,
            "source_manifest": str(destination),
            "source_manifest_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        })

    scan = {
        "kind": "norm42_v3_unsplit_track_count_variance",
        "planned_events": events,
        "batch_size": batch_size,
        "batches": len(points),
        "seed": seed,
        "cohort": cohort,
        "files_per_polarity": n_files,
        "fraction_of_full_bx_reference": n_files / FULL_BX_FILES_PER_POLARITY,
        "source_inventory": str(inventory_path),
        "source_inventory_sha256": inventory_sha,
        "source_selection": (
            "independent per-polarity draws from every valid file; without replacement "
            "within an event and with reuse across events"
        ),
        "hit_selection": "raw-time-lt-1e7-ns",
        "reconstruction": "one unsplit v3 CKF pass per event arm",
        "points": points,
    }
    (output / "scan.json").write_text(json.dumps(scan, indent=2) + "\n")
    print(
        f"Planned {events} events in {len(points)} batches at "
        f"{n_files}/{FULL_BX_FILES_PER_POLARITY} = "
        f"{100 * n_files / FULL_BX_FILES_PER_POLARITY:.4f}% BX"
    )
    print("valid sources: " + ", ".join(
        f"{polarity}={len(inventory[polarity])}" for polarity in POLARITIES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--files-per-polarity", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--cohort", default="onepct_variance")
    args = parser.parse_args()
    prepare(
        args.sim_root, args.output, args.events, args.batch_size,
        args.files_per_polarity, args.seed, args.cohort,
    )


if __name__ == "__main__":
    main()
