#!/usr/bin/env python3
"""Prepare one source-disjoint 10%-BX three-arm reconstruction checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from count_tracker_conditions import (  # noqa: E402
    CELL_ID_ENCODING,
    CONSTRUCTIONS,
    NORM42_OCCUPANCY_SCAN_NAME,
    validate_manifest,
)


POLARITIES = ("MUPLUS", "MUMINUS")
FILES_PER_POLARITY = 167
FULL_BX_FILES_PER_POLARITY = 1666


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_sources(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if document.get("kind") != "norm42_10pct_training_development_corpus":
        raise ValueError("training manifest has the wrong kind")
    if document.get("selection", {}).get("files_per_polarity") != 167:
        raise ValueError("training manifest does not describe the 167+167 corpus")
    cycles = {
        polarity: {int(row["cycle"]) for row in document["sources"][polarity]}
        for polarity in POLARITIES
    }
    if any(len(values) != 167 for values in cycles.values()):
        raise ValueError("training manifest contains repeated or missing cycles")
    return path, document, cycles


def load_inventory(training_manifest, document):
    path = Path(document["source_inventory"]).resolve()
    if sha256(path) != document["source_inventory_sha256"]:
        raise ValueError("source inventory checksum differs from training manifest")
    inventory = json.loads(path.read_text())
    if inventory.get("kind") != "norm42-independent-polarity-source-inventory":
        raise ValueError("source inventory has the wrong kind")
    records = {}
    for polarity in POLARITIES:
        indexed = {}
        for row in inventory["sources"][polarity]:
            cycle = int(row["cycle"])
            source = {
                "cycle": cycle,
                "path": str(Path(row["path"]).resolve()),
                "entry": int(row["entry"]),
            }
            if cycle in indexed:
                raise ValueError(f"duplicate {polarity} cycle {cycle}")
            if not Path(source["path"]).is_file():
                raise FileNotFoundError(source["path"])
            indexed[cycle] = source
        records[polarity] = indexed
    return path, inventory, records


def select_evaluation_sources(records, excluded, seed, cohort):
    selected = {}
    for polarity in POLARITIES:
        available = sorted(set(records[polarity]) - excluded)
        if len(available) < FILES_PER_POLARITY:
            raise ValueError(f"only {len(available)} source-disjoint {polarity} cycles")
        stream = json.dumps(
            [seed, "norm42", cohort, "source-disjoint-track-checkpoint", polarity],
            separators=(",", ":"),
        )
        cycles = random.Random(stream).sample(available, FILES_PER_POLARITY)
        selected[polarity] = [dict(records[polarity][cycle]) for cycle in cycles]
    return selected


def prepare(training_sources, output, seed, cohort):
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cohort):
        raise ValueError("cohort must contain letters, digits, '_' or '-'")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")

    training_path, training, training_cycles = load_training_sources(training_sources)
    inventory_path, inventory, records = load_inventory(training_path, training)
    # MUMINUS is a flipped MUPLUS source. Exclude a cycle if either polarity
    # appeared in training, so the evaluation event is physically source-disjoint.
    excluded = set().union(*training_cycles.values())
    selected = select_evaluation_sources(records, excluded, seed, cohort)
    evaluation_cycles = {
        polarity: {row["cycle"] for row in selected[polarity]}
        for polarity in POLARITIES
    }
    if any(values & excluded for values in evaluation_cycles.values()):
        raise AssertionError("evaluation sources overlap the training-cycle union")

    event_id = f"norm42_{cohort}_test_000000"
    normalization = {
        "name": NORM42_OCCUPANCY_SCAN_NAME,
        "selected_files_per_polarity": FILES_PER_POLARITY,
        "full_bx_reference_files_per_polarity": FULL_BX_FILES_PER_POLARITY,
        "fraction_of_full_bx_reference": (
            FILES_PER_POLARITY / FULL_BX_FILES_PER_POLARITY
        ),
        "selection": "source-disjoint 10%-BX reconstruction checkpoint",
    }
    manifest = {
        "schema_version": 2,
        "cell_id_encoding": CELL_ID_ENCODING,
        "construction": "norm42",
        "n_files_per_polarity": FILES_PER_POLARITY,
        "file_normalization": CONSTRUCTIONS["norm42"]["file_normalization"],
        "norm1_equivalents_per_polarity": (
            FILES_PER_POLARITY * CONSTRUCTIONS["norm42"]["file_normalization"]
        ),
        "source_pool": "all",
        "pool": {
            "kind": "training-source-disjoint-norm42-sim-bank",
            "inventory": str(inventory_path),
            "inventory_sha256": sha256(inventory_path),
            "training_source_manifest": str(training_path),
            "training_source_manifest_sha256": sha256(training_path),
            "training_cycles_by_polarity": {
                key: sorted(values) for key, values in training_cycles.items()
            },
            "excluded_physical_cycle_union": sorted(excluded),
            "exclusion_reason": (
                "MUMINUS is the flipped MUPLUS counterpart; exclude a cycle "
                "from both evaluation polarities if either polarity trained on it"
            ),
        },
        "overlay_normalization": normalization,
        "sampling": {
            "seed": seed,
            "cohort": cohort,
            "within_event": "without replacement, independently by polarity",
            "across_events": "not applicable; one checkpoint event",
            "classifier_split": "not applicable; reconstruction checkpoint",
            "training_overlap": "zero against union of training cycle IDs",
        },
        "events": [{
            "event_id": event_id,
            "split": "test",
            "sources": selected,
        }],
    }

    point = output / "n0167"
    point.mkdir(parents=True)
    source_path = point / "sources.json"
    validate_manifest(manifest, source_path.parent)
    source_path.write_text(json.dumps(manifest, indent=2) + "\n")
    scan = {
        "kind": "norm42_10pct_training_scale_track_checkpoint",
        "seed": seed,
        "cohort": cohort,
        "events_per_point": 1,
        "arms": ["SIM", "COUNT_FULL", "COUNT_10PCT"],
        "source_selection": (
            "independent per-polarity draws after excluding the union of cycles "
            "used to train the 10%-corpus model"
        ),
        "training_source_manifest": str(training_path),
        "hit_selection": "raw-time-lt-1e7-ns",
        "theta_regions_deg": [[0, 30], [30, 70], [70, 110], [110, 150], [150, 180]],
        "points": [{
            "tag": "n0167",
            "files_per_polarity": FILES_PER_POLARITY,
            "fraction_of_full_bx_reference": (
                FILES_PER_POLARITY / FULL_BX_FILES_PER_POLARITY
            ),
            "source_manifest": str(source_path),
            "source_manifest_sha256": sha256(source_path),
        }],
    }
    (output / "scan.json").write_text(json.dumps(scan, indent=2) + "\n")
    print(f"Prepared {event_id}")
    print(f"excluded physical training cycles: {len(excluded)}")
    for polarity in POLARITIES:
        print(f"{polarity}: {len(evaluation_cycles[polarity])} selected; overlap=0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-sources", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--cohort", default="train10pct_checkpoint")
    args = parser.parse_args()
    prepare(args.training_sources, args.output, args.seed, args.cohort)


if __name__ == "__main__":
    main()
