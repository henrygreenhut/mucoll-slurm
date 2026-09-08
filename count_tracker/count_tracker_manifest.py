#!/usr/bin/env python3
"""Select SIM sources independently by polarity from existing split pools.

The user specifies the pool, cohort, event counts, and seed. This script does
not create a new source split or submit production. Pool directories follow
reco_libtest_prepare_pools.py: <construction>/<split>/<polarity>/cycle_<id>__*.root.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

from count_tracker_conditions import CELL_ID_ENCODING, CONSTRUCTIONS, POLARITIES, validate_manifest


SPLITS = ("train", "val", "test")


def read_pools(root, construction):
    if construction not in CONSTRUCTIONS:
        raise ValueError("construction must be norm1 or norm42")
    root = Path(root).resolve()
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    pools = {}
    seen = set()
    for split in SPLITS:
        values = manifest["splits"][split]
        if isinstance(values, dict):
            values = values["cycles"]
        if any(type(cycle) is not int or cycle < 0 for cycle in values):
            raise ValueError("Pool cycles must be nonnegative integers")
        cycles = set(values)
        if len(cycles) != len(values) or seen.intersection(cycles):
            raise ValueError("Duplicate cycles or overlapping pool splits")
        seen.update(cycles)
        pools[split] = {}
        # Audit both libraries against the same cycle split even when selecting
        # only one. A flipped or reused version of a cycle cannot cross splits.
        for library in CONSTRUCTIONS:
            for polarity in POLARITIES:
                directory = root / library / split / polarity
                records = {}
                for path in sorted(directory.glob("*.root")):
                    match = re.fullmatch(r"cycle_(\d+)__.+\.root", path.name)
                    if not match or not path.is_file():
                        raise ValueError(f"Invalid pool filename or broken link: {path}")
                    cycle = int(match.group(1))
                    if cycle in records:
                        raise ValueError(f"Duplicate pool cycle {cycle}: {directory}")
                    records[cycle] = {"cycle": cycle, "path": str(path.resolve()), "entry": 0}
                if set(records) != cycles:
                    raise ValueError(f"Pool directory differs from its manifest: {directory}")
                if library == construction:
                    pools[split][polarity] = records
    provenance = {"path": str(root), "manifest_sha256": hashlib.sha256(raw).hexdigest()}
    return pools, provenance


def select_events(pools, counts, seed, cohort, construction):
    """Use a separate deterministic stream for each event and polarity."""
    if type(seed) is not int or seed < 0:
        raise ValueError("Source seed must be a nonnegative integer")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cohort):
        raise ValueError("Cohort must contain letters, digits, '_' or '-'")
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Supply nonnegative event counts for all three splits")
    if not sum(counts.values()):
        raise ValueError("At least one event is required")
    if construction not in CONSTRUCTIONS:
        raise ValueError("construction must be norm1 or norm42")
    n_files = CONSTRUCTIONS[construction]["n_files_per_polarity"]
    events = []
    for split in SPLITS:
        for index in range(counts[split]):
            event_id = f"{construction}_{cohort}_{index:06d}"
            sources = {}
            for polarity in POLARITIES:
                available = pools[split][polarity]
                if len(available) < n_files:
                    raise ValueError(f"Fewer than {n_files} source files in {split}/{polarity}")
                stream = json.dumps([seed, construction, cohort, split, index, polarity], separators=(",", ":"))
                selected = random.Random(stream).sample(sorted(available), n_files)
                sources[polarity] = [dict(available[cycle]) for cycle in selected]
            events.append({"event_id": event_id, "split": split, "sources": sources})
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--construction", choices=tuple(CONSTRUCTIONS), required=True)
    parser.add_argument("--cohort", required=True, help="Distinct label for each independent SIM cohort")
    parser.add_argument("--seed", required=True, type=int)
    for split in SPLITS:
        parser.add_argument(f"--{split}-events", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pools, provenance = read_pools(args.pools, args.construction)
    counts = {split: getattr(args, f"{split}_events") for split in SPLITS}
    manifest = {
        "schema_version": 2, "cell_id_encoding": CELL_ID_ENCODING,
        "construction": args.construction,
        **CONSTRUCTIONS[args.construction],
        "norm1_equivalents_per_polarity": 420,
        "pool": provenance,
        "sampling": {
            "seed": args.seed, "cohort": args.cohort,
            "within_event": "without replacement, independently by polarity",
            "across_events": "source reuse permitted within each split",
            "stream": "random.Random(JSON([seed, construction, cohort, split, event_index, polarity]))",
        },
        "events": select_events(pools, counts, args.seed, args.cohort, args.construction),
    }
    output = Path(args.output).resolve()
    validate_manifest(manifest, output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest['events'])} explicit source lists to {output}")


if __name__ == "__main__":
    main()
