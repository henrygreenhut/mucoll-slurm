#!/usr/bin/env python3
"""Prepare the direct norm1 mother-muon SIM/COUNT comparison.

MUPLUS and MUMINUS are sampled independently with replacement from all complete
source cycles. Source cycles may be reused within and between events, and every
stored tracker hit in every mother entry is retained. No analysis or model
train/validation split and no pre-digitization timing selection is used.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np

from count_tracker_conditions import CELL_ID_ENCODING, COLLECTIONS, POLARITIES, decode_cell_ids


CONSTRUCTION = "norm1_mother_direct"
FILES_COLUMNS = (
    "file_id", "polarity", "mother_start", "mother_count",
    "hit_count", "row_start", "row_count",
)
TRACKER_FIELDS = ("eDep", "position.x", "position.y", "position.z", "time", "cellID")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cycle_metadata(files):
    """Validate files.npy and return complete cycle/polarity records."""
    files = np.asarray(files)
    if files.ndim != 2 or files.shape[1] != len(FILES_COLUMNS):
        raise ValueError(f"files.npy must have columns {FILES_COLUMNS}")
    records = {}
    for raw in files:
        file_id, polarity, mother_start, mother_count, hit_count, row_start, row_count = map(int, raw)
        if (file_id < 0 or polarity not in (-1, 1) or mother_start < 0
                or mother_count < 1 or hit_count < 0 or row_start < 0
                or row_count < mother_count):
            raise ValueError(f"invalid file metadata row: {raw.tolist()}")
        key = (file_id, polarity)
        if key in records:
            raise ValueError(f"duplicate file metadata for cycle/polarity {key}")
        records[key] = {
            "mother_count": mother_count,
            "source_tracker_hits": hit_count,
            "mother_start": mother_start,
            "row_start": row_start,
            "row_count": row_count,
        }
    cycle_ids = {file_id for file_id, _ in records}
    incomplete = [cycle for cycle in cycle_ids
                  if (cycle, -1) not in records or (cycle, 1) not in records]
    if incomplete:
        raise ValueError(f"cycles missing one polarity: {incomplete[:20]}")
    return records, sorted(cycle_ids)


def select_events(cycles, count, sim_root, records, seed, cohort):
    """Draw 420 cycles with replacement independently for each polarity."""
    if not cycles:
        raise ValueError("source cycle pool is empty")
    events = []
    polarity_code = {"MUPLUS": 1, "MUMINUS": -1}
    for index in range(count):
        event_id = f"{CONSTRUCTION}_{cohort}_{index:06d}"
        sources = {}
        for polarity in POLARITIES:
            stream = json.dumps(
                [seed, CONSTRUCTION, cohort, "test", index, polarity],
                separators=(",", ":"),
            )
            chosen = random.Random(stream).choices(cycles, k=420)
            sources[polarity] = []
            for draw, cycle in enumerate(chosen):
                path = sim_root / polarity / f"bib_sim_{cycle}.edm4hep.root"
                if not path.is_file():
                    raise FileNotFoundError(path)
                sources[polarity].append({
                    "draw": draw, "cycle": cycle, "path": str(path), "entries": "all",
                    **records[(cycle, polarity_code[polarity])],
                })
        events.append({"event_id": event_id, "split": "test", "sources": sources})
    return events


def flatten_branch(tree, collection, field):
    import awkward as ak

    branch = tree[collection][f"{collection}.{field}"].array(library="ak")
    return np.asarray(ak.flatten(branch, axis=1))


def all_tracker_rows(values, system):
    """Build the persisted 11-column schema without selecting tracker hits."""
    if len(values) != 6 or len({len(value) for value in values}) != 1:
        raise ValueError("tracker-hit branch lengths differ")
    edep, x, y, z, time, cell_ids = map(np.asarray, values)
    if (not all(np.all(np.isfinite(value)) for value in (edep, x, y, z, time))
            or np.any(edep < 0)):
        raise ValueError("stored tracker hits contain invalid numeric values")
    labels = decode_cell_ids(cell_ids, system)
    rows = np.column_stack((edep, x, y, z, time, labels, cell_ids)).astype(np.float64)
    return rows, {"all_stored": len(rows)}


def read_cycle(path, expected_entries):
    """Return every stored tracker hit from every mother entry in one cycle."""
    import uproot

    result, totals = {}, {}
    with uproot.open(path) as root:
        if "events" not in root or "podio_metadata" not in root:
            raise ValueError(f"missing events or PODIO metadata: {path}")
        tree = root["events"]
        if tree.num_entries != expected_entries:
            raise ValueError(
                f"{path}: {tree.num_entries} entries, metadata expects {expected_entries}")
        for short, (system, name) in COLLECTIONS.items():
            values = [flatten_branch(tree, name, field) for field in TRACKER_FIELDS]
            try:
                rows, summary = all_tracker_rows(values, system)
            except ValueError as error:
                raise ValueError(f"{path}/{name}: {error}") from error
            result[short] = rows
            totals[short] = summary
    return result, totals


def prepare_event(event, destination):
    chunks = {short: [] for short in COLLECTIONS}
    selection = {short: Counter() for short in COLLECTIONS}
    for polarity in POLARITIES:
        for source in event["sources"][polarity]:
            rows, totals = read_cycle(source["path"], source["mother_count"])
            if sum(item["all_stored"] for item in totals.values()) != source["source_tracker_hits"]:
                raise ValueError(f"{source['path']}: tracker-hit total differs from files.npy")
            for short in COLLECTIONS:
                chunks[short].append(rows[short])
                selection[short].update(totals[short])

    destination.mkdir(parents=True)
    arrays, collections = {}, {}
    for short in COLLECTIONS:
        rows = (np.concatenate(chunks[short], axis=0) if chunks[short]
                else np.empty((0, 11), dtype=np.float64))
        sim_path = destination / f"{short}_sim_hits.npy"
        np.save(sim_path, rows, allow_pickle=False)
        conditions_path = destination / f"{short}_conditions.npy"
        conditions = rows[:, 5:10].astype(np.int64)
        np.save(conditions_path, conditions, allow_pickle=False)
        arrays[short] = {
            "path": str(sim_path), "sha256": sha256_file(sim_path), "hits": len(rows),
        }
        collections[short] = {
            "hits": len(rows),
            "occupied_sensors": len(np.unique(conditions, axis=0)) if len(conditions) else 0,
            "sha256": sha256_file(conditions_path),
            **dict(selection[short]),
        }
    return arrays, collections


def prepare(args):
    sim_root = Path(args.sim_root).resolve()
    metadata = Path(args.metadata).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    for polarity in POLARITIES:
        if not (sim_root / polarity).is_dir():
            raise FileNotFoundError(sim_root / polarity)

    files_path = metadata / "files.npy"
    records, cycles = cycle_metadata(np.load(files_path, allow_pickle=False))
    events = select_events(cycles, args.events, sim_root, records, args.seed, args.cohort)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".count_mother_direct_", dir=output.parent) as tmp:
        work = Path(tmp)
        summaries = []
        for event in events:
            destination = work / event["split"] / event["event_id"]
            arrays, collections = prepare_event(event, destination)
            for saved in arrays.values():
                saved["path"] = str(
                    Path(event["split"]) / event["event_id"] / Path(saved["path"]).name)
            event["sim_arrays"] = arrays
            summaries.append({
                "event_id": event["event_id"], "split": event["split"],
                "collections": collections,
            })
            print(f"Prepared {event['split']}/{event['event_id']}", flush=True)

        manifest = {
            "schema_version": 2,
            "construction": CONSTRUCTION,
            "cell_id_encoding": CELL_ID_ENCODING,
            "source_domain": "original norm1 split-mother unrotated SIM",
            "generator_training_holdout": False,
            "model_split_used": False,
            "analysis_split_used": False,
            "classifier_ready": False,
            "physical_event_boundaries": True,
            "file_normalization": 1,
            "n_files_per_polarity": 420,
            "norm1_equivalents_per_polarity": 420,
            "hit_selection": "all-stored",
            "source_dataset": {
                "path": str(sim_root),
                "files_metadata": {"path": str(files_path), "sha256": sha256_file(files_path)},
            },
            "source_cycle_pool": {
                "kind": "all complete cycles in files.npy",
                "count": len(cycles),
                "cycles": cycles,
            },
            "sampling": {
                "seed": args.seed,
                "cohort": args.cohort,
                "within_event": "420 draws with replacement independently by polarity",
                "across_events": "source reuse permitted",
                "stream": "random.Random(JSON([seed, construction, cohort, split, event_index, polarity]))",
            },
            "events": events,
        }
        report = {
            "kind": "count_tracker_norm1_mother_direct_conditions",
            "hit_selection": "all-stored",
            "selection": (
                "all mother entries and all stored SIM tracker hits; "
                "no time, energy, or spatial cut before digitization"
            ),
            "condition_columns": ["system", "side", "layer", "module", "sensor"],
            "manifest": manifest,
            "events": summaries,
        }
        (work / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"output appeared during preparation: {output}")
        os.replace(work, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", required=True)
    parser.add_argument("--metadata", required=True,
                        help="primary_muon_npy/metadata directory containing files.npy")
    parser.add_argument("--cohort", default="trackcmp")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.seed < 0 or args.events < 1:
        parser.error("seed must be nonnegative and events must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
