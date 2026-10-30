#!/usr/bin/env python3
"""Prepare one repaired-inventory Norm42 occupancy point above one BX."""

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
from source_disjoint_fullbx.prepare import complete_inventory


POLARITIES = ("MUPLUS", "MUMINUS")
FULL_BX_FILES = 1666


def select_events(plus, minus, cycles, n_files, n_events, seed, cohort):
    """Draw distinct files within each event, independently by polarity."""
    events = []
    banks = {"MUPLUS": plus, "MUMINUS": minus}
    for event_index in range(n_events):
        event_id = f"norm42_{cohort}_test_{event_index:06d}"
        sources = {}
        for polarity in POLARITIES:
            stream = json.dumps(
                [seed, "norm42", cohort, "test", event_index, polarity,
                 "repaired-all-source-cycles"],
                separators=(",", ":"),
            )
            selected = list(cycles)
            random.Random(stream).shuffle(selected)
            selected = selected[:n_files]
            sources[polarity] = [
                {"cycle": cycle, "path": str(banks[polarity][cycle]), "entry": 0}
                for cycle in selected
            ]
        events.append({"event_id": event_id, "split": "test", "sources": sources})
    return events


def prepare(args):
    if args.files_per_polarity < 1:
        raise ValueError("files-per-polarity must be positive")
    if args.events < 1:
        raise ValueError("events must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.cohort):
        raise ValueError("cohort must contain letters, digits, '_' or '-'")

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")

    plus, minus, cycles = complete_inventory(
        args.shared_sim_root, args.recovery_muminus_root)
    if args.files_per_polarity > len(cycles):
        raise ValueError(
            f"requested {args.files_per_polarity} files per polarity but the "
            f"repaired paired inventory contains only {len(cycles)}")

    n_files = args.files_per_polarity
    fraction = n_files / FULL_BX_FILES
    tag = f"n{n_files:04d}"
    events = select_events(
        plus, minus, cycles, n_files, args.events, args.seed, args.cohort)

    inventory_identity = {
        polarity: [[cycle, str((plus if polarity == "MUPLUS" else minus)[cycle])]
                   for cycle in cycles]
        for polarity in POLARITIES
    }
    inventory_hash = hashlib.sha256(json.dumps(
        inventory_identity, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    normalization = {
        "name": NORM42_OCCUPANCY_SCAN_NAME,
        "selected_files_per_polarity": n_files,
        "full_bx_reference_files_per_polarity": FULL_BX_FILES,
        "fraction_of_full_bx_reference": fraction,
        "selection": "independent deterministic permutation prefix",
    }
    manifest = {
        "schema_version": 2,
        "cell_id_encoding": CELL_ID_ENCODING,
        "construction": "norm42",
        "n_files_per_polarity": n_files,
        "file_normalization": CONSTRUCTIONS["norm42"]["file_normalization"],
        "norm1_equivalents_per_polarity": (
            n_files * CONSTRUCTIONS["norm42"]["file_normalization"]),
        "source_pool": "all",
        "pool": {
            "kind": "repaired-authoritative-unsplit-norm42-sim-bank",
            "shared_sim_root": str(Path(args.shared_sim_root).resolve()),
            "recovery_muminus_root": str(Path(args.recovery_muminus_root).resolve()),
            "paired_cycles": len(cycles),
            "inventory_sha256": inventory_hash,
        },
        "overlay_normalization": normalization,
        "sampling": {
            "seed": args.seed,
            "cohort": args.cohort,
            "within_event": "without replacement, independently by polarity",
            "across_events": "source reuse permitted",
            "classifier_split": "not applicable; occupancy diagnostic",
            "selection_algorithm": (
                "independent deterministic shuffle of the repaired shared cycle "
                "inventory for each event and polarity; take requested prefix"),
        },
        "events": events,
    }

    point = output / tag
    point.mkdir(parents=True)
    sources = point / "sources.json"
    validate_manifest(manifest, sources.parent)
    encoded = (json.dumps(manifest, indent=2) + "\n").encode()
    sources.write_bytes(encoded)
    scan = {
        "kind": "norm42_v3_five_theta_high_occupancy_scan",
        "seed": args.seed,
        "cohort": args.cohort,
        "events_per_point": args.events,
        "source_selection": (
            "independent repaired-inventory permutations by event and polarity; "
            "without replacement within an event; reuse permitted across events"),
        "hit_selection": "raw-time-lt-1e7-ns",
        "theta_regions_deg": [[0, 30], [30, 70], [70, 110], [110, 150], [150, 180]],
        "points": [{
            "tag": tag,
            "files_per_polarity": n_files,
            "norm1_equivalents_per_polarity": n_files * 42,
            "fraction_of_full_bx_reference": fraction,
            "source_manifest": str(sources),
            "source_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        }],
    }
    (output / "scan.json").write_text(json.dumps(scan, indent=2) + "\n")
    print(
        f"Prepared {args.events} event(s) at {n_files}/{FULL_BX_FILES} = "
        f"{100 * fraction:.3f}% BX in {output}")
    print(f"repaired paired inventory: {len(cycles)} cycles")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-sim-root", required=True)
    parser.add_argument("--recovery-muminus-root", required=True)
    parser.add_argument("--files-per-polarity", required=True, type=int)
    parser.add_argument("--events", required=True, type=int)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    prepare(args)


if __name__ == "__main__":
    main()
