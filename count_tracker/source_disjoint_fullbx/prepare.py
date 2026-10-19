#!/usr/bin/env python3
"""Build four globally source-disjoint Norm42 full-BX parent events."""

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
    NORM42_FULL_BX,
    validate_manifest,
)


POLARITIES = ("MUPLUS", "MUMINUS")
RECOVERY_CYCLES = (100, 101, 105, 106, 118, 143, 144, 146, 147, 148, 149, 6291)
EVENT_SPLITS = ("train", "train", "val", "test")
FILES_PER_EVENT = 1666
EVENTS = len(EVENT_SPLITS)
REQUIRED_CYCLES = FILES_PER_EVENT * EVENTS
FILENAME = re.compile(r"bib_sim_(\d+)\.edm4hep\.root")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_directory(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    records = {}
    for path in sorted(directory.glob("bib_sim_*.edm4hep.root")):
        match = FILENAME.fullmatch(path.name)
        if match is None or not path.is_file():
            raise ValueError(f"unexpected SIM source path: {path}")
        cycle = int(match.group(1))
        if cycle in records:
            raise ValueError(f"duplicate cycle {cycle} in {directory}")
        records[cycle] = path.resolve()
    return records


def validate_recovered_file(path):
    """Require one readable EDM4hep event with every tracker collection."""
    import uproot

    if path.stat().st_size < 1024:
        raise ValueError(f"recovered SIM file is implausibly small: {path}")
    try:
        with uproot.open(path) as root:
            tree = root["events"]
            if tree.num_entries != 1:
                raise ValueError(
                    f"recovered SIM file must contain one event: {path} has "
                    f"{tree.num_entries}")
            for collection in (
                    "VertexBarrelCollection", "VertexEndcapCollection",
                    "InnerTrackerBarrelCollection", "InnerTrackerEndcapCollection",
                    "OuterTrackerBarrelCollection", "OuterTrackerEndcapCollection"):
                tree[collection]
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"cannot read recovered SIM file {path}: {error}") from error


def complete_inventory(shared_root, recovery_muminus_root, validate_recovery=True):
    shared_root = Path(shared_root).resolve()
    recovery_muminus_root = Path(recovery_muminus_root).resolve()
    plus = index_directory(shared_root / "MUPLUS")
    shared_minus = index_directory(shared_root / "MUMINUS")
    recovered = index_directory(recovery_muminus_root)

    missing_recovery = set(RECOVERY_CYCLES) - set(recovered)
    if missing_recovery:
        raise ValueError(
            f"recovery directory is missing cycles {sorted(missing_recovery)}")
    for cycle in RECOVERY_CYCLES:
        if validate_recovery:
            validate_recovered_file(recovered[cycle])

    minus = dict(shared_minus)
    for cycle in RECOVERY_CYCLES:
        minus[cycle] = recovered[cycle]
    common = sorted(set(plus) & set(minus))
    if len(common) < REQUIRED_CYCLES:
        raise ValueError(
            f"only {len(common)} paired cycles after recovery; need {REQUIRED_CYCLES}")
    return plus, minus, common


def partition_cycles(cycles, seed, cohort):
    ordered = list(cycles)
    stream = json.dumps(
        [seed, "norm42", cohort, "globally-disjoint-full-bx-cycles"],
        separators=(",", ":"),
    )
    random.Random(stream).shuffle(ordered)
    selected = ordered[:REQUIRED_CYCLES]
    groups = [
        selected[index * FILES_PER_EVENT:(index + 1) * FILES_PER_EVENT]
        for index in range(EVENTS)
    ]
    if len(set().union(*map(set, groups))) != REQUIRED_CYCLES:
        raise AssertionError("cycle groups are not globally disjoint")
    return groups, ordered[REQUIRED_CYCLES:]


def prepare(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    plus, minus, cycles = complete_inventory(
        args.shared_sim_root, args.recovery_muminus_root)
    groups, unused = partition_cycles(cycles, args.seed, args.cohort)

    split_index = {"train": 0, "val": 0, "test": 0}
    events = []
    for split, group in zip(EVENT_SPLITS, groups):
        index = split_index[split]
        split_index[split] += 1
        event_id = f"norm42_{args.cohort}_{split}_{index:06d}"
        sources = {
            polarity: [
                {
                    "cycle": cycle,
                    "path": str((plus if polarity == "MUPLUS" else minus)[cycle]),
                    "entry": 0,
                }
                for cycle in group
            ]
            for polarity in POLARITIES
        }
        events.append({"event_id": event_id, "split": split, "sources": sources})

    inventory_identity = {
        polarity: [
            [cycle, str((plus if polarity == "MUPLUS" else minus)[cycle])]
            for cycle in cycles
        ]
        for polarity in POLARITIES
    }
    inventory_hash = hashlib.sha256(json.dumps(
        inventory_identity, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    manifest = {
        "schema_version": 2,
        "cell_id_encoding": CELL_ID_ENCODING,
        "construction": "norm42",
        "n_files_per_polarity": FILES_PER_EVENT,
        "file_normalization": CONSTRUCTIONS["norm42"]["file_normalization"],
        "norm1_equivalents_per_polarity": (
            FILES_PER_EVENT * CONSTRUCTIONS["norm42"]["file_normalization"]),
        "source_pool": "split",
        "pool": {
            "kind": "repaired-authoritative-norm42-sim-bank",
            "shared_sim_root": str(Path(args.shared_sim_root).resolve()),
            "recovery_muminus_root": str(Path(args.recovery_muminus_root).resolve()),
            "paired_cycles": len(cycles),
            "recovered_muminus_cycles": list(RECOVERY_CYCLES),
            "inventory_sha256": inventory_hash,
        },
        "overlay_normalization": NORM42_FULL_BX,
        "sampling": {
            "seed": args.seed,
            "cohort": args.cohort,
            "within_event": (
                "1666 distinct cycle IDs; identical cycle block for MUPLUS and MUMINUS"),
            "across_events": "without replacement globally",
            "classifier_split": (
                "two train, one validation, and one test parent; no cycle crosses parents"),
            "selection_algorithm": (
                "one deterministic shuffle of paired cycles, four consecutive 1666-cycle blocks"),
        },
        "events": events,
    }

    point = output / "n1666"
    point.mkdir(parents=True)
    sources = point / "sources.json"
    validate_manifest(manifest, sources.parent)
    encoded = (json.dumps(manifest, indent=2) + "\n").encode()
    sources.write_bytes(encoded)
    scan = {
        "kind": "norm42_v3_source_disjoint_fullbx",
        "seed": args.seed,
        "cohort": args.cohort,
        "events_per_point": EVENTS,
        "parent_split": {"train": 2, "val": 1, "test": 1},
        "source_selection": (
            "globally disjoint paired cycle blocks; MUPLUS/MUMINUS counterparts remain together"),
        "hit_selection": "raw-time-lt-1e7-ns",
        "theta_regions_deg": [[0, 30], [30, 70], [70, 110], [110, 150], [150, 180]],
        "unused_cycle_ids": unused,
        "points": [{
            "tag": "n1666",
            "files_per_polarity": FILES_PER_EVENT,
            "fraction_of_full_bx_reference": 1.0,
            "source_manifest": str(sources),
            "source_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        }],
    }
    (output / "scan.json").write_text(json.dumps(scan, indent=2) + "\n")
    print(f"Prepared four globally disjoint full-BX parents in {output}")
    print("parent split: 2 train / 1 validation / 1 test")
    print(f"paired cycles: {len(cycles)}; selected: {REQUIRED_CYCLES}; unused: {unused}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-sim-root", required=True)
    parser.add_argument("--recovery-muminus-root", required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--cohort", default="disjoint_fullbx")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    prepare(args)


if __name__ == "__main__":
    main()
