#!/usr/bin/env python3
"""Prepare nested Norm42 source manifests for a tracker occupancy scan."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from count_tracker_conditions import (
    CELL_ID_ENCODING,
    CONSTRUCTIONS,
    NORM42_OCCUPANCY_SCAN_NAME,
    validate_manifest,
)
from count_tracker_manifest import POLARITIES, SPLITS, read_pools, select_events

KNOWN_INVALID_CYCLES = {6291}


def read_sim_root(root, minimum_files=1666):
    """Index the authoritative Norm42 banks without inventing a data split."""
    root = Path(root).resolve()
    pools = {
        split: {polarity: {} for polarity in POLARITIES}
        for split in SPLITS
    }
    raw_records = {}
    for polarity in POLARITIES:
        directory = root / polarity
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        records = {}
        for path in sorted(directory.glob("bib_sim_*.edm4hep.root")):
            match = re.fullmatch(r"bib_sim_(\d+)\.edm4hep\.root", path.name)
            if match is None or not path.is_file():
                raise ValueError(f"unexpected Norm42 source path: {path}")
            cycle = int(match.group(1))
            if cycle in KNOWN_INVALID_CYCLES:
                continue
            if cycle in records:
                raise ValueError(f"duplicate {polarity} cycle {cycle}")
            records[cycle] = {
                "cycle": cycle, "path": str(path.resolve()), "entry": 0,
            }
        raw_records[polarity] = records

    shared_cycles = set.intersection(*(
        set(raw_records[polarity]) for polarity in POLARITIES))
    if len(shared_cycles) < minimum_files:
        raise ValueError(
            f"only {len(shared_cycles)} valid cycles are shared by both polarities; "
            f"need at least {minimum_files}")

    inventory = {}
    for polarity in POLARITIES:
        records = {
            cycle: raw_records[polarity][cycle]
            for cycle in sorted(shared_cycles)
        }
        # Put the unsplit shared inventory in one internal bucket.
        # source_pool=all below records that no data split is claimed.
        pools["test"][polarity] = records
        inventory[polarity] = [
            {"cycle": cycle, "path": records[cycle]["path"]}
            for cycle in sorted(records)
        ]
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    provenance = {
        "kind": "authoritative-unsplit-norm42-sim-bank",
        "path": str(root),
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": {polarity: len(inventory[polarity]) for polarity in POLARITIES},
        "known_invalid_cycles_excluded": sorted(KNOWN_INVALID_CYCLES),
        "unpaired_cycles_excluded": {
            polarity: sorted(set(raw_records[polarity]) - shared_cycles)
            for polarity in POLARITIES
        },
    }
    return pools, provenance


def prepare(source, output, file_counts, events, seed, cohort, source_kind="sim-root"):
    if len(file_counts) != len(set(file_counts)):
        raise ValueError("file counts must be unique")
    if file_counts != sorted(file_counts):
        raise ValueError("file counts must be in increasing order")
    if file_counts[0] < 1 or file_counts[-1] > 1666:
        raise ValueError("file counts must lie between 1 and 1666")
    if events < 1:
        raise ValueError("events must be positive")

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if source_kind == "sim-root":
        pools, provenance = read_sim_root(source)
    elif source_kind == "pools":
        pools, provenance = read_pools(source, "norm42")
    else:
        raise ValueError(f"unknown source kind: {source_kind}")
    output.mkdir(parents=True)

    points = []
    for n_files in file_counts:
        tag = f"n{n_files:04d}"
        selected = select_events(
            pools,
            {"train": 0, "val": 0, "test": events},
            seed,
            cohort,
            "norm42",
            source_pool="all",
            n_files_per_polarity=n_files,
            prefix_stable=True,
        )
        normalization = {
            "name": NORM42_OCCUPANCY_SCAN_NAME,
            "selected_files_per_polarity": n_files,
            "full_bx_reference_files_per_polarity": 1666,
            "fraction_of_full_bx_reference": n_files / 1666,
            "selection": "nested deterministic source-order prefix",
        }
        manifest = {
            "schema_version": 2,
            "cell_id_encoding": CELL_ID_ENCODING,
            "construction": "norm42",
            "n_files_per_polarity": n_files,
            "file_normalization": CONSTRUCTIONS["norm42"]["file_normalization"],
            "norm1_equivalents_per_polarity": n_files * 42,
            "source_pool": "all",
            "pool": provenance,
            "overlay_normalization": normalization,
            "sampling": {
                "seed": seed,
                "cohort": cohort,
                "within_event": "without replacement, independently by polarity",
                "across_events": "source reuse permitted",
                "classifier_split": "not applicable; occupancy diagnostic",
                "stream": (
                    "random.Random(JSON([seed, construction, cohort, split, "
                    "event_index, polarity, all-source-cycles]))"
                ),
                "selection_algorithm": (
                    "shuffle complete sorted source pool once; take requested prefix"
                ),
            },
            "events": selected,
        }
        destination = output / tag / "sources.json"
        destination.parent.mkdir()
        validate_manifest(manifest, destination.parent)
        encoded = (json.dumps(manifest, indent=2) + "\n").encode()
        destination.write_bytes(encoded)
        points.append({
            "tag": tag,
            "files_per_polarity": n_files,
            "norm1_equivalents_per_polarity": n_files * 42,
            "fraction_of_full_bx_reference": n_files / 1666,
            "source_manifest": str(destination),
            "source_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        })

    scan = {
        "kind": "norm42_v3_five_theta_occupancy_scan",
        "seed": seed,
        "cohort": cohort,
        "events_per_point": events,
        "source_selection": (
            "nested prefixes of independent deterministic permutations for MUPLUS "
            "and MUMINUS; sources may recur between replicate events"
        ),
        "hit_selection": "raw-time-lt-1e7-ns",
        "theta_regions_deg": [[0, 30], [30, 70], [70, 110], [110, 150], [150, 180]],
        "points": points,
    }
    (output / "scan.json").write_text(json.dumps(scan, indent=2) + "\n")
    print(f"Prepared {len(points)} occupancies x {events} events in {output}")
    for point in points:
        print(
            f"{point['tag']}: {point['files_per_polarity']} files/polarity, "
            f"{point['fraction_of_full_bx_reference']:.6f} of full-BX reference"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sim-root")
    source.add_argument("--pools")
    parser.add_argument("--output", required=True)
    parser.add_argument("--file-counts", required=True, nargs="+", type=int)
    parser.add_argument("--events", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--cohort", default="occupancy_scan")
    args = parser.parse_args()
    prepare(
        args.sim_root or args.pools,
        args.output,
        args.file_counts,
        args.events,
        args.seed,
        args.cohort,
        "sim-root" if args.sim_root else "pools",
    )


if __name__ == "__main__":
    main()
