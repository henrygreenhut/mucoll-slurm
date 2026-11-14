#!/usr/bin/env python3
"""Select one traceable 10%-BX Norm42 source cohort for retraining."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re


POLARITIES = ("MUPLUS", "MUMINUS")
INVALID = {"MUPLUS": set(), "MUMINUS": {6291}}
FILES_PER_POLARITY = 167
FULL_BX_FILES_PER_POLARITY = 1666
SOURCE_ID_OFFSET = 6666


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_sources(sim_root):
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
            if not 0 <= cycle < SOURCE_ID_OFFSET:
                raise ValueError(f"cycle outside source-ID encoding: {cycle}")
            if cycle in records:
                raise ValueError(f"duplicate {polarity} cycle {cycle}")
            records[cycle] = {
                "cycle": cycle,
                "path": str(path.resolve()),
                "entry": 0,
                "size_bytes": path.stat().st_size,
                "source_id": cycle + (SOURCE_ID_OFFSET if polarity == "MUPLUS" else 0),
            }
        if len(records) < FILES_PER_POLARITY:
            raise ValueError(
                f"{polarity}: found {len(records)} valid sources, need "
                f"{FILES_PER_POLARITY}"
            )
        inventory[polarity] = records
    return sim_root, inventory


def select_sources(inventory, seed, cohort):
    selected = {}
    for polarity in POLARITIES:
        available = sorted(inventory[polarity])
        stream = json.dumps(
            [seed, "norm42", cohort, "training-development-corpus", polarity,
             "all-valid-polarity-files"],
            separators=(",", ":"),
        )
        cycles = random.Random(stream).sample(available, FILES_PER_POLARITY)
        selected[polarity] = [dict(inventory[polarity][cycle]) for cycle in cycles]
    return selected


def prepare(sim_root, output, seed, cohort):
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cohort):
        raise ValueError("cohort must contain only letters, digits, '_' or '-'")

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    source_root, inventory = index_sources(sim_root)
    selected = select_sources(inventory, seed, cohort)
    output.mkdir(parents=True)

    inventory_document = {
        "schema_version": 1,
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

    shared_cycles = sorted(
        set(row["cycle"] for row in selected["MUPLUS"])
        & set(row["cycle"] for row in selected["MUMINUS"])
    )
    manifest = {
        "schema_version": 1,
        "kind": "norm42_10pct_training_development_corpus",
        "cohort": cohort,
        "seed": seed,
        "construction": "norm42",
        "source_id_encoding": {
            "MUMINUS": "cycle",
            "MUPLUS": f"{SOURCE_ID_OFFSET} + cycle",
        },
        "selection": {
            "files_per_polarity": FILES_PER_POLARITY,
            "full_bx_reference_files_per_polarity": FULL_BX_FILES_PER_POLARITY,
            "fraction_of_full_bx_reference": (
                FILES_PER_POLARITY / FULL_BX_FILES_PER_POLARITY
            ),
            "within_polarity": "without replacement",
            "between_polarities": "independent",
            "cross_polarity_cycle_overlap_count": len(shared_cycles),
            "cross_polarity_cycle_overlap": shared_cycles,
            "training_events": "not defined; model inputs are flattened hits",
        },
        "source_inventory": str(inventory_path),
        "source_inventory_sha256": sha256(inventory_path),
        "sources": selected,
    }
    manifest_path = output / "sources.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Prepared {FILES_PER_POLARITY} MUPLUS + {FILES_PER_POLARITY} MUMINUS")
    print(
        f"Nominal occupancy: "
        f"{100 * FILES_PER_POLARITY / FULL_BX_FILES_PER_POLARITY:.4f}% BX"
    )
    print(f"Cross-polarity cycle overlap: {len(shared_cycles)}")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cohort", default="norm42_train10pct")
    args = parser.parse_args()
    prepare(args.sim_root, args.output, args.seed, args.cohort)


if __name__ == "__main__":
    main()
