#!/usr/bin/env python3
"""Draw empirical SIM hits for norm42 sensor-occupancy templates.

The COUNT training arrays are one flat hit reservoir with no physical event
boundaries.  This program constructs a controlled reconstruction closure:

* an existing norm42 event supplies the exact per-sensor hit multiplicities;
* SIM draws real training rows conditional on those sensors;
* COUNT later generates one hit from the same condition row;
* both samples pass through the usual common digitization and reconstruction.

``--reuse-policy none`` uses every reservoir row at most once in the complete
cohort and is intended for the small pilot. ``within-split`` deterministically
partitions each sensor's reservoir among classifier splits, samples distinct
rows within an event, and permits a row to recur only in different events of
the same split.  A complete sensor-capacity audit runs before any output is
published.  Neither mode restores physical inter-hit or mother-muon grouping.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from count_tracker_conditions import (
    CELL_ID_ENCODING, COLLECTIONS, sensor_counts, validate_manifest,
)
from count_tracker_training_domain import MODEL_COLUMNS, SELECTION, training_array_paths


CONSTRUCTION = "norm42_reservoir"
SPLITS = ("train", "val", "test")
SIM_COLUMNS = [*MODEL_COLUMNS, "cellid0"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rng_for(seed, *identity):
    key = json.dumps([seed, *identity], separators=(",", ":"))
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return np.random.default_rng(value)


def encode_sensor_keys(labels):
    """Pack integer condition rows into the declared 32-bit CellID fields."""
    values = np.asarray(labels)
    if values.ndim != 2 or values.shape[1] != 5 or values.dtype.kind not in "iuf":
        raise ValueError("Sensor labels must have shape (N, 5)")
    if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
        raise ValueError("Sensor labels must be finite integers")
    values = values.astype(np.int64)
    limits = ((0, 31), (-2, 1), (0, 63), (0, 2047), (0, 255))
    for column, (low, high) in enumerate(limits):
        if np.any((values[:, column] < low) | (values[:, column] > high)):
            raise ValueError("Sensor label exceeds the declared CellID field")
    return (values[:, 0].astype(np.uint64)
            | ((values[:, 1] & 3).astype(np.uint64) << np.uint64(5))
            | (values[:, 2].astype(np.uint64) << np.uint64(7))
            | (values[:, 3].astype(np.uint64) << np.uint64(13))
            | (values[:, 4].astype(np.uint64) << np.uint64(24)))


def renamed_event_id(template_id):
    suffix = template_id.removeprefix("norm42_")
    return f"{CONSTRUCTION}_{suffix}"


def load_verification(path, training_paths):
    path = Path(path).expanduser().resolve()
    report = json.loads(path.read_text())
    if report.get("status") != "exact ordered match" or report.get("selection") != SELECTION:
        raise ValueError("Verification report does not establish the HDF-to-NPY identity")
    collections = report.get("collections", {})
    if set(collections) != set(COLLECTIONS):
        raise ValueError("Verification report must contain all six collections")
    for short, training_path in training_paths.items():
        saved = collections[short]
        if Path(saved["path"]).resolve() != training_path:
            raise ValueError(f"Verification report points to a different {short} array")
        rows = np.load(training_path, mmap_mode="r", allow_pickle=False)
        if saved.get("rows") != len(rows):
            raise ValueError(f"Verification report has the wrong {short} row count")
    return path, report


def load_template_events(directory, wanted_splits):
    directory = Path(directory).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    report = json.loads(manifest_path.read_text())
    manifest = validate_manifest(report["manifest"], directory)
    if manifest["construction"] != "norm42":
        raise ValueError("Reservoir templates must come from the norm42 construction")
    summaries = {(event["split"], event["event_id"]): event for event in report["events"]}
    events = []
    seen_ids = set()
    for template in manifest["events"]:
        split = template["split"]
        if split not in wanted_splits:
            continue
        template_id = template["event_id"]
        summary = summaries.get((split, template_id))
        if summary is None:
            raise ValueError(f"Missing conditions summary for {split}/{template_id}")
        event_id = renamed_event_id(template_id)
        if event_id in seen_ids:
            raise ValueError(f"Duplicate reservoir event ID: {event_id}")
        seen_ids.add(event_id)
        events.append({
            "event_id": event_id,
            "template_event_id": template_id,
            "split": split,
            "conditions": {},
        })
        for short, (system, _) in COLLECTIONS.items():
            source = directory / split / template_id / f"{short}_conditions.npy"
            saved = summary["collections"][short]
            if sha256_file(source) != saved["sha256"]:
                raise ValueError(f"Template conditions changed after preparation: {source}")
            rows = np.load(source, allow_pickle=False)
            if rows.shape != (saved["hits"], 5) or rows.dtype.kind not in "iu":
                raise ValueError(f"Invalid template conditions: {source}")
            if len(rows) and np.any(rows[:, 0] != system):
                raise ValueError(f"Wrong system in template conditions: {source}")
            events[-1]["conditions"][short] = rows.astype(np.int64, copy=False)
    if not events:
        raise ValueError(f"No template events for splits {sorted(wanted_splits)}")
    return directory, manifest_path, manifest, events


def validate_training_rows(path, short, system):
    rows = np.load(path, mmap_mode="r", allow_pickle=False)
    if rows.ndim != 2 or rows.shape[1] != 10 or rows.dtype != np.float64:
        raise ValueError(f"{short}: training reservoir must be float64 (N, 10)")
    if not np.all(np.isfinite(rows)) or np.any(rows[:, 0] < 0):
        raise ValueError(f"{short}: training reservoir contains invalid physical values")
    labels = rows[:, 5:10]
    keys = encode_sensor_keys(labels)
    if len(labels) and np.any(labels[:, 0] != system):
        raise ValueError(f"{short}: training reservoir has the wrong system")
    return rows, keys


def requested_by_sensor(events, short):
    result = Counter()
    by_split = {split: Counter() for split in SPLITS}
    max_event = {split: Counter() for split in SPLITS}
    labels = {}
    for event in events:
        rows = event["conditions"][short]
        keys = encode_sensor_keys(rows)
        unique, first, counts = np.unique(keys, return_index=True, return_counts=True)
        for key, position, count in zip(unique, first, counts):
            key_int = int(key)
            n = int(count)
            labels[key_int] = tuple(map(int, rows[position]))
            result[key_int] += n
            by_split[event["split"]][key_int] += n
            max_event[event["split"]][key_int] = max(
                max_event[event["split"]][key_int], n)
    return result, by_split, max_event, labels


def available_by_sensor(keys):
    unique, counts = np.unique(keys, return_counts=True)
    return {int(key): int(count) for key, count in zip(unique, counts)}


def capacity_errors(available, requested, by_split, max_event, reuse_policy):
    errors = []
    for key, total in requested.items():
        have = available.get(key, 0)
        if reuse_policy == "none" and total > have:
            errors.append((key, total, have, "cohort"))
        if reuse_policy == "within-split":
            minimum = sum(max_event[split].get(key, 0) for split in SPLITS)
            if minimum > have:
                errors.append((key, minimum, have, "split-isolated event maxima"))
    return errors


def allocate_split_pool_sizes(available, demand, minima):
    """Allocate a disjoint sensor pool to splits, respecting one-event maxima."""
    sizes = {split: int(minima.get(split, 0)) for split in SPLITS}
    if sum(sizes.values()) > available:
        raise ValueError("Split-isolated sensor pools cannot satisfy one event")
    remaining = available - sum(sizes.values())
    weight = sum(demand.get(split, 0) for split in SPLITS)
    if remaining and weight:
        exact = {split: remaining * demand.get(split, 0) / weight for split in SPLITS}
        additions = {split: int(np.floor(exact[split])) for split in SPLITS}
        left = remaining - sum(additions.values())
        order = sorted(SPLITS, key=lambda split: (-(exact[split] - additions[split]), split))
        for split in order[:left]:
            additions[split] += 1
        for split in SPLITS:
            sizes[split] += additions[split]
    return sizes


def request_layout(events, short):
    lengths = [len(event["conditions"][short]) for event in events]
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    keys = np.concatenate(
        [encode_sensor_keys(event["conditions"][short]) for event in events]
    ) if offsets[-1] else np.empty(0, dtype=np.uint64)
    owner = np.repeat(np.arange(len(events), dtype=np.int64), lengths)
    local = np.concatenate([np.arange(length, dtype=np.int64) for length in lengths]) \
        if offsets[-1] else np.empty(0, dtype=np.int64)
    return keys, owner, local


def draw_indices(pool_keys, events, short, seed, reuse_policy):
    """Return one reservoir row index per condition plus a detailed audit."""
    request_keys, owners, local_positions = request_layout(events, short)
    outputs = [np.empty(len(event["conditions"][short]), dtype=np.int64) for event in events]
    pool_order = np.argsort(pool_keys, kind="stable")
    sorted_pool_keys = pool_keys[pool_order]
    request_order = np.argsort(request_keys, kind="stable")
    sorted_request_keys = request_keys[request_order]
    unique_keys, starts, counts = np.unique(
        sorted_request_keys, return_index=True, return_counts=True)
    audit = []

    for key, start, count in zip(unique_keys, starts, counts):
        key_int = int(key)
        request_positions = request_order[start:start + count]
        low = int(np.searchsorted(sorted_pool_keys, key, side="left"))
        high = int(np.searchsorted(sorted_pool_keys, key, side="right"))
        candidates = pool_order[low:high]
        split_records = {}

        if reuse_policy == "none":
            chosen = rng_for(seed, short, key_int, "cohort").choice(
                candidates, size=len(request_positions), replace=False)
            for position, row_index in zip(request_positions, chosen):
                event_index = int(owners[position])
                outputs[event_index][local_positions[position]] = row_index
            for split in SPLITS:
                split_positions = [p for p in request_positions
                                   if events[int(owners[p])]["split"] == split]
                if split_positions:
                    split_records[split] = {
                        "available_pool": len(candidates),
                        "requested": len(split_positions),
                        "unique_used": len(split_positions),
                        "reused": 0,
                        "unique_fraction": 1.0,
                    }
        else:
            positions_by_split = {
                split: np.asarray([p for p in request_positions
                                   if events[int(owners[p])]["split"] == split], dtype=np.int64)
                for split in SPLITS
            }
            demand = {split: len(positions) for split, positions in positions_by_split.items()}
            maxima = {}
            for split, positions in positions_by_split.items():
                counts_by_event = Counter(int(owners[p]) for p in positions)
                maxima[split] = max(counts_by_event.values(), default=0)
            sizes = allocate_split_pool_sizes(len(candidates), demand, maxima)
            shuffled = rng_for(seed, short, key_int, "split-pools").permutation(candidates)
            cursor = 0
            for split in SPLITS:
                split_pool = shuffled[cursor:cursor + sizes[split]]
                cursor += sizes[split]
                used = set()
                positions = positions_by_split[split]
                for event_index in sorted(set(int(owners[p]) for p in positions)):
                    event_positions = positions[owners[positions] == event_index]
                    chosen = rng_for(
                        seed, short, key_int, split, events[event_index]["event_id"]
                    ).choice(split_pool, size=len(event_positions), replace=False)
                    for position, row_index in zip(event_positions, chosen):
                        outputs[event_index][local_positions[position]] = row_index
                    used.update(map(int, chosen))
                if len(positions):
                    split_records[split] = {
                        "available_pool": int(len(split_pool)),
                        "requested": int(len(positions)),
                        "unique_used": int(len(used)),
                        "reused": int(len(positions) - len(used)),
                        "unique_fraction": float(len(used) / len(positions)),
                    }

        condition = events[int(owners[request_positions[0]])]["conditions"][short][
            local_positions[request_positions[0]]]
        audit.append({
            "condition": list(map(int, condition)),
            "available_global": int(len(candidates)),
            "splits": split_records,
        })
    return outputs, audit


def prepare(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    wanted_splits = set(args.splits)
    training_paths = training_array_paths(args.training_npys)
    verification_path, verification = load_verification(
        args.verification_report, training_paths)
    template_dir, template_manifest_path, template_manifest, events = load_template_events(
        args.templates, wanted_splits)

    # Complete capacity audit before any event product is written or published.
    capacity = {}
    for short, (system, _) in COLLECTIONS.items():
        _, pool_keys = validate_training_rows(training_paths[short], short, system)
        requested, by_split, max_event, labels = requested_by_sensor(events, short)
        available = available_by_sensor(pool_keys)
        errors = capacity_errors(
            available, requested, by_split, max_event, args.reuse_policy)
        capacity[short] = {
            "available_hits": int(len(pool_keys)),
            "requested_hits": int(sum(requested.values())),
            "requested_sensors": int(len(requested)),
            "minimum_available_over_requested_sensor": int(
                min((available.get(key, 0) for key in requested), default=0)),
        }
        if errors:
            details = [
                {"condition": labels.get(key), "required": required,
                 "available": have, "constraint": constraint}
                for key, required, have, constraint in errors[:20]
            ]
            raise ValueError(
                f"{short}: {len(errors)} sensor-capacity failures under "
                f"{args.reuse_policy}: {details}"
            )
        print(f"[{short}] capacity OK: {sum(requested.values()):,} requested from "
              f"{len(pool_keys):,} rows", flush=True)

    with tempfile.TemporaryDirectory(prefix=".count_reservoir_", dir=output.parent) as tmp:
        work = Path(tmp)
        manifest_events = []
        summaries = []
        for event in events:
            destination = work / event["split"] / event["event_id"]
            destination.mkdir(parents=True)
            manifest_events.append({
                "event_id": event["event_id"], "split": event["split"],
                "template_event_id": event["template_event_id"], "sim_arrays": {},
            })
            summaries.append({
                "event_id": event["event_id"], "split": event["split"],
                "collections": {},
            })

        sensor_audit = {
            "construction": CONSTRUCTION,
            "reuse_policy": args.reuse_policy,
            "seed": args.seed,
            "collections": {},
        }
        for short, (system, _) in COLLECTIONS.items():
            pool, pool_keys = validate_training_rows(training_paths[short], short, system)
            indices, audit = draw_indices(
                pool_keys, events, short, args.seed, args.reuse_policy)
            sensor_audit["collections"][short] = {
                **capacity[short], "sensors": audit,
            }
            for event_index, (event, selected) in enumerate(zip(events, indices)):
                conditions = event["conditions"][short]
                selected_rows = np.asarray(pool[selected], dtype=np.float64)
                if not np.array_equal(encode_sensor_keys(selected_rows[:, 5:10]),
                                      encode_sensor_keys(conditions)):
                    raise RuntimeError(f"{event['event_id']}/{short}: sampled wrong sensor")
                sim = np.column_stack(
                    [selected_rows, encode_sensor_keys(conditions)]).astype(np.float64)
                destination = work / event["split"] / event["event_id"]
                condition_path = destination / f"{short}_conditions.npy"
                sim_path = destination / f"{short}_sim_hits.npy"
                np.save(condition_path, conditions, allow_pickle=False)
                np.save(sim_path, sim, allow_pickle=False)
                summaries[event_index]["collections"][short] = {
                    "hits": int(len(conditions)),
                    "occupied_sensors": int(len(sensor_counts(conditions))),
                    "sha256": sha256_file(condition_path),
                }
                manifest_events[event_index]["sim_arrays"][short] = {
                    "path": str(sim_path.relative_to(work)),
                    "hits": int(len(sim)),
                    "sha256": sha256_file(sim_path),
                }
            print(f"[{short}] drew {capacity[short]['requested_hits']:,} SIM hits", flush=True)

        audit_path = work / "sensor_audit.json"
        audit_path.write_text(json.dumps(sensor_audit, indent=2) + "\n")
        manifest = {
            "schema_version": 2,
            "construction": CONSTRUCTION,
            "cell_id_encoding": CELL_ID_ENCODING,
            "sampling": {
                "cohort": template_manifest.get("sampling", {}).get("cohort", "unknown"),
                "seed": args.seed,
                "reuse_policy": args.reuse_policy,
            },
            "template": {
                "construction": "norm42",
                "conditions_dir": str(template_dir),
                "manifest_sha256": sha256_file(template_manifest_path),
            },
            "source_domain": "COUNT model training-data hit reservoir",
            "generator_training_holdout": False,
            "physical_event_boundaries": False,
            "event_correlation_policy": "independent empirical draw conditional on sensor",
            "events": manifest_events,
        }
        report = {
            "kind": "count_tracker_reservoir_conditions",
            "manifest": manifest,
            "condition_columns": ["system", "side", "layer", "module", "sensor"],
            "sim_columns": SIM_COLUMNS,
            "verification_report": {
                "path": str(verification_path),
                "sha256": sha256_file(verification_path),
                "status": verification["status"],
                "selection": verification["selection"],
            },
            "training_arrays": {
                short: {
                    "path": str(path), "rows": int(len(np.load(path, mmap_mode="r"))),
                    "sha256": sha256_file(path),
                }
                for short, path in training_paths.items()
            },
            "sensor_audit": {
                "path": "sensor_audit.json", "sha256": sha256_file(audit_path),
            },
            "events": summaries,
        }
        (work / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Output appeared during preparation: {output}")
        os.replace(work, output)
    print(f"Prepared {len(events)} {CONSTRUCTION} events -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", required=True,
                        help="Prepared norm42 conditions directory")
    parser.add_argument("--training-npys", required=True,
                        help="Directory containing the six verified training arrays")
    parser.add_argument("--verification-report", required=True,
                        help="JSON emitted by count_tracker_training_domain.py --verify-only")
    parser.add_argument("--reuse-policy", required=True, choices=("none", "within-split"))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    prepare(args)


if __name__ == "__main__":
    main()
