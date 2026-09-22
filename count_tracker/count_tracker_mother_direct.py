#!/usr/bin/env python3
"""Prepare the direct norm1 mother-muon SIM/COUNT comparison.

MUPLUS and MUMINUS are sampled independently with replacement from complete
source cycles. Diagnostic mode uses all cycles and one test label. Classifier
mode first makes disjoint train/validation/test cycle pools, then permits reuse
only within a split. Every mother entry is aggregated. The hit selection is
explicit and applied before both persisted SIM rows and COUNT conditions.
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

from count_tracker_conditions import (
    CELL_ID_ENCODING, COLLECTIONS, POLARITIES, TRAINING_RAW_TIME_MAX_NS,
    TRAINING_RAW_TIME_SELECTION, decode_cell_ids, training_raw_time_mask,
)


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


def select_events(cycles, count, sim_root, records, seed, cohort, split=None):
    """Draw 420 cycles with replacement independently for each polarity."""
    if not cycles:
        raise ValueError("source cycle pool is empty")
    event_split = split or "test"
    events = []
    polarity_code = {"MUPLUS": 1, "MUMINUS": -1}
    for index in range(count):
        event_id = (f"{CONSTRUCTION}_{cohort}_{event_split}_{index:06d}"
                    if split else f"{CONSTRUCTION}_{cohort}_{index:06d}")
        sources = {}
        for polarity in POLARITIES:
            stream = json.dumps(
                [seed, CONSTRUCTION, cohort, event_split, index, polarity],
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
        events.append({"event_id": event_id, "split": event_split, "sources": sources})
    return events


def split_cycle_pools(cycles, seed, fractions=(0.6, 0.2, 0.2)):
    """Make deterministic, disjoint analysis pools from complete cycle IDs."""
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("cycle-pool fractions must contain three values summing to one")
    shuffled = list(cycles)
    random.Random(json.dumps(
        [seed, CONSTRUCTION, "analysis_cycle_split"], separators=(",", ":")
    )).shuffle(shuffled)
    n_train = int(len(shuffled) * fractions[0])
    n_val = int(len(shuffled) * fractions[1])
    boundaries = (n_train, n_train + n_val)
    pools = {
        "train": sorted(shuffled[:boundaries[0]]),
        "val": sorted(shuffled[boundaries[0]:boundaries[1]]),
        "test": sorted(shuffled[boundaries[1]:]),
    }
    if any(len(pool) < 1 for pool in pools.values()):
        raise ValueError("cycle split produced an empty pool")
    if set().union(*map(set, pools.values())) != set(cycles):
        raise ValueError("cycle split does not cover the source pool")
    if sum(map(len, pools.values())) != len(cycles):
        raise ValueError("cycle split pools overlap")
    return pools


def flatten_branch(tree, collection, field):
    import awkward as ak

    branch = tree[collection][f"{collection}.{field}"].array(library="ak")
    return np.asarray(ak.flatten(branch, axis=1))


def tracker_rows(values, system, hit_selection="all-stored"):
    """Build the persisted 11-column schema with the declared hit selection."""
    if len(values) != 6 or len({len(value) for value in values}) != 1:
        raise ValueError("tracker-hit branch lengths differ")
    edep, x, y, z, time, cell_ids = map(np.asarray, values)
    if (not all(np.all(np.isfinite(value)) for value in (edep, x, y, z, time))
            or np.any(edep < 0)):
        raise ValueError("stored tracker hits contain invalid numeric values")
    labels = decode_cell_ids(cell_ids, system)
    rows = np.column_stack((edep, x, y, z, time, labels, cell_ids)).astype(np.float64)
    all_stored = len(rows)
    if hit_selection == TRAINING_RAW_TIME_SELECTION:
        rows = rows[training_raw_time_mask(rows[:, 4])]
    elif hit_selection != "all-stored":
        raise ValueError(f"unknown hit selection: {hit_selection}")
    return rows, {"all_stored": all_stored, "selected": len(rows)}


def all_tracker_rows(values, system):
    """Compatibility wrapper for the original all-stored diagnostic."""
    rows, summary = tracker_rows(values, system, "all-stored")
    return rows, {"all_stored": summary["all_stored"]}


def read_cycle(path, expected_entries, hit_selection):
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
                rows, summary = tracker_rows(values, system, hit_selection)
            except ValueError as error:
                raise ValueError(f"{path}/{name}: {error}") from error
            result[short] = rows
            totals[short] = summary
    return result, totals


def prepare_event(event, destination, hit_selection):
    chunks = {short: [] for short in COLLECTIONS}
    selection = {short: Counter() for short in COLLECTIONS}
    for polarity in POLARITIES:
        for source in event["sources"][polarity]:
            rows, totals = read_cycle(
                source["path"], source["mother_count"], hit_selection)
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
    classifier_counts = {
        split: getattr(args, f"{split}_events") for split in ("train", "val", "test")
    }
    classifier_mode = any(value is not None for value in classifier_counts.values())
    if classifier_mode:
        if args.events is not None or any(value is None or value < 1
                                          for value in classifier_counts.values()):
            raise ValueError(
                "classifier mode requires positive train/val/test event counts and no --events")
        cycle_pools = split_cycle_pools(cycles, args.seed)
        events = []
        for split in ("train", "val", "test"):
            events.extend(select_events(
                cycle_pools[split], classifier_counts[split], sim_root, records,
                args.seed, args.cohort, split))
    else:
        if args.events is None or args.events < 1:
            raise ValueError("diagnostic mode requires a positive --events")
        cycle_pools = None
        events = select_events(
            cycles, args.events, sim_root, records, args.seed, args.cohort)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".count_mother_direct_", dir=output.parent) as tmp:
        work = Path(tmp)
        summaries = []
        for event in events:
            destination = work / event["split"] / event["event_id"]
            arrays, collections = prepare_event(
                event, destination, args.hit_selection)
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
            "analysis_split_used": classifier_mode,
            "classifier_ready": classifier_mode,
            "physical_event_boundaries": True,
            "file_normalization": 1,
            "n_files_per_polarity": 420,
            "norm1_equivalents_per_polarity": 420,
            "hit_selection": args.hit_selection,
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
                "across_events": (
                    "source reuse permitted within split; forbidden across splits"
                    if classifier_mode else "source reuse permitted"),
                "stream": "random.Random(JSON([seed, construction, cohort, split, event_index, polarity]))",
            },
            "events": events,
        }
        if classifier_mode:
            manifest["analysis_cycle_pools"] = {
                "seed": args.seed,
                "fractions": {"train": 0.6, "val": 0.2, "test": 0.2},
                "algorithm": (
                    "random.Random(JSON([seed, construction, analysis_cycle_split])).shuffle; "
                    "integer 60/20/remainder boundaries"),
                "pools": {split: {"count": len(pool), "cycles": pool}
                          for split, pool in cycle_pools.items()},
            }
        raw_time_selection = None
        if args.hit_selection == TRAINING_RAW_TIME_SELECTION:
            raw_time_selection = {
                "field": "SimTrackerHit.time", "operator": "<",
                "threshold_ns": TRAINING_RAW_TIME_MAX_NS,
                "flight_corrected": False,
            }
        report = {
            "kind": "count_tracker_norm1_mother_direct_conditions",
            "hit_selection": args.hit_selection,
            "selection": (
                "all mother entries; raw SimTrackerHit time t < 1e7 ns; "
                "no flight correction, energy cut, or spatial cut"
                if raw_time_selection else
                "all mother entries and all stored SIM tracker hits; "
                "no time, energy, or spatial cut before digitization"),
            "raw_time_selection": raw_time_selection,
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
    parser.add_argument("--events", type=int)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-events", type=int)
    parser.add_argument(
        "--hit-selection",
        choices=("all-stored", TRAINING_RAW_TIME_SELECTION),
        default="all-stored")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    prepare(args)


if __name__ == "__main__":
    main()
