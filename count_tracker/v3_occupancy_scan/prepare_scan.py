#!/usr/bin/env python3
"""Prepare nested Norm42 source manifests for a tracker occupancy scan."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from count_tracker_conditions import (
    CELL_ID_ENCODING,
    CONSTRUCTIONS,
    NORM42_OCCUPANCY_SCAN_NAME,
    validate_manifest,
)
from count_tracker_manifest import POLARITIES, read_pools, select_events


def prepare(pools_path, output, file_counts, events, seed, cohort):
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
    pools, provenance = read_pools(pools_path, "norm42")
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
    parser.add_argument("--pools", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--file-counts", required=True, nargs="+", type=int)
    parser.add_argument("--events", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--cohort", default="occupancy_scan")
    args = parser.parse_args()
    prepare(args.pools, args.output, args.file_counts, args.events, args.seed, args.cohort)


if __name__ == "__main__":
    main()
