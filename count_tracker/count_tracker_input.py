#!/usr/bin/env python3
"""Write one SIM or COUNT tracker input for a COUNT tracker comparison.

Use the same conditions event and neutrino SIM record for a SIM/COUNT pair.
COUNT input arrays must already have geometric CellIDs assigned by GenBIB.
Only the detector environment supplies the PODIO/EDM4hep runtime imports.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from count_tracker_conditions import (
    CELL_ID_ENCODING, COLLECTIONS, CONSTRUCTIONS, IN_TIME_WINDOW_NS, POLARITIES,
    decode_cell_ids, flight_corrected_time, sensor_counts, validate_manifest,
)

ARRAY_SIM_CONSTRUCTIONS = ("norm42_reservoir",)
INPUT_CONSTRUCTIONS = (*CONSTRUCTIONS, *ARRAY_SIM_CONSTRUCTIONS)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_count_rows(rows, system, expected, tolerance=0.0):
    """Check assigned hits without clipping, filtering, or resampling them.

    Geometric CellID assignment (assign_actual_cellid) drops hits whose
    generated position resolves to no sensor and reassigns others to a
    neighbour, so the assigned occupancy no longer matches the conditioned
    occupancy exactly. Rather than repair the hits, we tolerate a small
    mismatch and record it: raise only if the missing fraction exceeds
    ``tolerance``. Returns ``(labels, stats)``. See COUNT_TRACKER_STUDY.md
    decision on the CellID policy; this deficit is COUNT-only and must be
    accounted for when interpreting any SIM-vs-COUNT separation.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 10 or rows.dtype.kind not in "fiu":
        raise ValueError("Assigned hits must be a numeric (N, 10) array")
    if not np.all(np.isfinite(rows)):
        raise ValueError("Assigned hits contain nonfinite values")
    if np.any(rows[:, 0] < 0):
        raise ValueError("Assigned hits contain negative energy deposition")
    labels = rows[:, 5:]
    if not np.all(labels == np.rint(labels)):
        raise ValueError("Assigned hit sensor identifiers must be integers")
    limits = ((0, 31), (-2, 1), (0, 63), (0, 2047), (0, 255))
    for column, (low, high) in enumerate(limits):
        if np.any((labels[:, column] < low) | (labels[:, column] > high)):
            raise ValueError("Assigned sensor identifier exceeds the CellID field")
    labels = labels.astype(np.int64)
    if np.any(labels[:, 0] != system):
        raise ValueError("Assigned hits belong to a different tracker system")
    actual = sensor_counts(labels)
    total_expected = sum(expected.values())
    written = len(labels)
    dropped = total_expected - written                # hits lost entirely (assign 'unresolved')
    reassigned = sum((actual - expected).values())    # hits moved to a neighbour sensor
    drop_fraction = dropped / total_expected if total_expected else 0.0
    # Gate on hits LOST -- drops remove tracking input. Reassignments keep the hit
    # (a one-sensor position smear) so they are recorded but not gated. See the
    # CellID-policy note in COUNT_TRACKER_STUDY.md.
    if drop_fraction > tolerance:
        raise ValueError(
            f"CellID assignment dropped {dropped} of {total_expected} hits "
            f"({drop_fraction:.3%} exceeds tolerance {tolerance:.3%})"
        )
    stats = {"expected": int(total_expected), "written": int(written),
             "dropped": int(dropped), "reassigned": int(reassigned)}
    return labels, stats


def pack_cell_ids(labels):
    """Pack previously validated sensor labels; retain signed side bits."""
    values = np.asarray(labels, dtype=np.int64)
    return (values[:, 0] | ((values[:, 1] & 3) << 5) |
            (values[:, 2] << 7) | (values[:, 3] << 13) |
            (values[:, 4] << 24)).astype(np.uint64)


def load_event(directory, split, event_id, construction):
    """Load the exact source record and check its saved condition digests."""
    directory = Path(directory).resolve()
    report = json.loads((directory / "manifest.json").read_text())
    manifest = report["manifest"]
    if construction == "norm42_reservoir":
        if report.get("kind") != "count_tracker_reservoir_conditions":
            raise ValueError("Expected a norm42 reservoir conditions manifest")
        if manifest.get("schema_version") != 2:
            raise ValueError("Reservoir manifest must be schema_version 2")
        if manifest.get("cell_id_encoding") != CELL_ID_ENCODING:
            raise ValueError("Reservoir manifest has a different CellID encoding")
        if manifest.get("generator_training_holdout") is not False:
            raise ValueError("Reservoir manifest must state that it is not a holdout")
        if manifest.get("physical_event_boundaries") is not False:
            raise ValueError("Reservoir manifest must state that physical boundaries are absent")
        if manifest.get("template", {}).get("construction") != "norm42":
            raise ValueError("Reservoir occupancy template must be norm42")
        if manifest.get("sampling", {}).get("reuse_policy") not in ("none", "within-split"):
            raise ValueError("Reservoir manifest has an unknown reuse policy")
        verification = report.get("verification_report", {})
        if (verification.get("status") != "exact ordered match"
                or verification.get("selection") != "inside_bounds == True"):
            raise ValueError("Reservoir was not based on verified training arrays")
        events = manifest.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError("Reservoir manifest must contain events")
        identities = [(event.get("split"), event.get("event_id")) for event in events]
        template_ids = [event.get("template_event_id") for event in events]
        if any(split not in ("train", "val", "test") for split, _ in identities):
            raise ValueError("Reservoir event has an invalid classifier split")
        if len(identities) != len(set(identities)) or len(template_ids) != len(set(template_ids)):
            raise ValueError("Reservoir template events must be unique across splits")
    else:
        manifest = validate_manifest(manifest, directory)
    if manifest["construction"] != construction:
        raise ValueError("Conditions belong to a different SIM construction")
    matches = [event for event in manifest["events"]
               if (event["split"], event["event_id"]) == (split, event_id)]
    summaries = [event for event in report["events"]
                 if (event["split"], event["event_id"]) == (split, event_id)]
    if len(matches) != 1 or len(summaries) != 1:
        raise ValueError("Conditions must contain exactly one matching event")
    expected = {}
    for short, (system, _) in COLLECTIONS.items():
        path = directory / split / event_id / f"{short}_conditions.npy"
        saved = summaries[0]["collections"][short]
        if sha256_file(path) != saved["sha256"]:
            raise ValueError(f"Conditions changed after preparation: {path}")
        rows = np.load(path, allow_pickle=False)
        expected[short] = sensor_counts(rows)
        if np.any(rows[:, 0] != system) or len(rows) != saved["hits"]:
            raise ValueError(f"Invalid conditions or recorded hit count: {path}")
    event = matches[0]
    if construction in ARRAY_SIM_CONSTRUCTIONS:
        arrays = event.get("sim_arrays")
        if not isinstance(arrays, dict) or set(arrays) != set(COLLECTIONS):
            raise ValueError("Prepared empirical event must declare all six SIM arrays")
        for short, saved in arrays.items():
            path = Path(saved["path"])
            path = (directory / path).resolve() if not path.is_absolute() else path.resolve()
            try:
                path.relative_to(directory)
            except ValueError as error:
                raise ValueError("Prepared SIM array escapes its conditions directory") from error
            if sha256_file(path) != saved["sha256"]:
                raise ValueError(f"Prepared SIM array changed after preparation: {path}")
            if saved.get("hits") != sum(expected[short].values()):
                raise ValueError(f"Prepared {short} SIM and condition counts differ")
            saved["path"] = str(path)
    return event, expected


def read_frame(path, entry):
    from podio.root_io import Reader

    reader = Reader(str(path))
    frames = reader.get("events")
    if entry < 0 or entry >= len(frames):
        raise ValueError(f"Entry {entry} does not exist in {path}")
    # Keep the reader alive while its frame is used.
    return reader, frames[entry]


def hit_counts(collection, system):
    ids = np.fromiter((hit.getCellID() for hit in collection), dtype=np.uint64)
    return sensor_counts(decode_cell_ids(ids, system))


def hit_in_time(hit, window=IN_TIME_WINDOW_NS):
    """Flight-corrected in-time test for one podio SimTrackerHit."""
    position = hit.getPosition()
    tof = flight_corrected_time(hit.getTime(), position.x, position.y, position.z)
    return window[0] <= tof <= window[1]


def append_sim_hits(collections, event):
    """Copy the in-time BIB hits, retaining scalar fields and removing links.

    Only hits inside the flight-corrected window are copied -- the same window
    count_tracker_conditions counts and the domain COUNT was trained on -- so SIM
    and COUNT enter the digitizer on one footing. Out-of-time hits are dropped by
    the digitizer regardless, so SIM's reconstructed tracks are unchanged. BIB
    MCParticles are not merged; clone(False) drops relations into source files.
    """
    for polarity in POLARITIES:
        for source in event["sources"][polarity]:
            reader, frame = read_frame(source["path"], source["entry"])
            for short, (_, name) in COLLECTIONS.items():
                for hit in frame.get(name):
                    if not hit_in_time(hit):
                        continue
                    copied = hit.clone(False)
                    copied.setOverlay(True)
                    collections[short].push_back(copied)
            del frame, reader


def append_array_sim_hits(collections, event, expected):
    """Write prepared empirical SIM rows, with no additional hit selection."""
    import edm4hep

    for short, (system, _) in COLLECTIONS.items():
        path = Path(event["sim_arrays"][short]["path"])
        rows = np.load(path, mmap_mode="r", allow_pickle=False)
        if rows.ndim != 2 or rows.shape[1] != 11 or rows.dtype.kind not in "fiu":
            raise ValueError(f"Prepared {short} SIM rows must have shape (N, 11)")
        if not np.all(np.isfinite(rows)) or np.any(rows[:, 0] < 0):
            raise ValueError(f"Prepared {short} SIM rows contain invalid values")
        labels = rows[:, 5:10]
        if not np.all(labels == np.rint(labels)):
            raise ValueError(f"Prepared {short} has noninteger sensor labels")
        labels = labels.astype(np.int64)
        if sensor_counts(labels) != expected[short]:
            raise ValueError(f"Prepared {short} occupancy differs from conditions")
        ids = rows[:, 10]
        if not np.all(ids == np.rint(ids)):
            raise ValueError(f"Prepared {short} has noninteger CellIDs")
        ids = ids.astype(np.uint64)
        if not np.array_equal(decode_cell_ids(ids, system), labels):
            raise ValueError(f"Prepared {short} CellIDs disagree with sensor labels")
        for row, cell_id in zip(rows, ids):
            hit = collections[short].create()
            hit.setEDep(float(row[0]))
            hit.setPosition(edm4hep.Vector3d(*map(float, row[1:4])))
            hit.setTime(float(row[4]))
            hit.setCellID(int(cell_id))
            hit.setOverlay(True)


def append_count_hits(collections, directory, expected, tolerance=0.0):
    import edm4hep

    provenance = {}
    actual_occupancy = {}
    for short, (system, name) in COLLECTIONS.items():
        path = Path(directory) / f"{name}_SimTrackerHit_conditional_reco9_0.npy"
        rows = np.load(path, mmap_mode="r", allow_pickle=False)
        labels, stats = validate_count_rows(rows, system, expected[short], tolerance)
        if stats["dropped"] or stats["reassigned"]:
            print(f"[{short}] {stats['dropped']} dropped, {stats['reassigned']} reassigned "
                  f"of {stats['expected']} (recorded)", flush=True)
        cell_ids = pack_cell_ids(labels)
        for row, cell_id in zip(rows, cell_ids):
            hit = collections[short].create()
            hit.setEDep(float(row[0]))
            hit.setPosition(edm4hep.Vector3d(*map(float, row[1:4])))
            hit.setTime(float(row[4]))
            hit.setCellID(int(cell_id))
            hit.setOverlay(True)
        actual_occupancy[short] = sensor_counts(labels)
        provenance[short] = {"path": str(path.resolve()), "sha256": sha256_file(path), **stats}
    return provenance, actual_occupancy


def write_input(args):
    import edm4hep
    import podio
    from podio.root_io import Writer

    event, expected = load_event(args.conditions, args.split, args.event_id, args.construction)
    destination = Path(args.output).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}")
    reader, signal = read_frame(args.signal, args.signal_entry)
    particles = signal.get("MCParticles")
    if len(particles) != 1 or particles[0].getPDG() != 14:
        raise ValueError("Expected one PDG-14 neutrino in the signal SIM record")
    if len(particles[0].getParents()) or len(particles[0].getDaughters()):
        raise ValueError("Unexpected neutrino particle relations; explicit handling required")
    for _, name in COLLECTIONS.values():
        if len(signal.get(name)):
            raise ValueError("Neutrino SIM record has tracker hits; refusing to omit them")
    headers = signal.get("EventHeader")
    if len(headers) != 1:
        raise ValueError("Expected one signal EventHeader")

    frame = podio.Frame()
    copied_particles = edm4hep.MCParticleCollection()
    copied_particles.push_back(particles[0].clone(False))
    copied_headers = edm4hep.EventHeaderCollection()
    copied_headers.push_back(headers[0].clone(False))
    frame.put(copied_particles, "MCParticles")
    frame.put(copied_headers, "EventHeader")
    del signal, reader

    collections = {short: edm4hep.SimTrackerHitCollection() for short in COLLECTIONS}
    count_provenance = None
    if args.sample == "SIM":
        if args.construction in ARRAY_SIM_CONSTRUCTIONS:
            append_array_sim_hits(collections, event, expected)
        else:
            append_sim_hits(collections, event)
        targets = expected  # SIM copies real hits, so occupancy must match exactly.
    else:
        count_provenance, targets = append_count_hits(
            collections, args.count_arrays, expected, args.count_tolerance)
    for short, (system, name) in COLLECTIONS.items():
        if hit_counts(collections[short], system) != targets[short]:
            reference = "conditions" if args.sample == "SIM" else "assigned arrays"
            raise ValueError(f"Final {short} occupancy differs from the {reference}")
        frame.put(collections[short], name)

    metadata = podio.Frame()
    for _, name in COLLECTIONS.values():
        metadata.put_parameter(f"{name}__CellIDEncoding", CELL_ID_ENCODING)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".count_input_", dir=destination.parent) as work:
        work = Path(work)
        root_path = work / "input.edm4hep.root"
        writer = Writer(str(root_path))
        writer.write_frame(metadata, "metadata")
        writer.write_frame(frame, "events")
        writer._writer.finish()
        del writer
        # Reopen the serialized data before publishing a completed input.
        check_reader, check = read_frame(root_path, 0)
        for short, (system, name) in COLLECTIONS.items():
            if hit_counts(check.get(name), system) != targets[short]:
                raise ValueError(f"Serialized {short} occupancy changed")
        del check, check_reader
        report = {
            "sample": args.sample, "event": event,
            "construction": args.construction,
            "signal": {"path": str(Path(args.signal).resolve()), "entry": args.signal_entry},
            "conditions_manifest_sha256": sha256_file(Path(args.conditions) / "manifest.json"),
            "count_arrays": count_provenance,
            "overlay_time_selection": False,
            "hits": {short: sum(counts.values()) for short, counts in targets.items()},
            "input_sha256": sha256_file(root_path),
        }
        if args.construction == "norm42_reservoir":
            report.update({
                "source_domain": "COUNT model training-data hit reservoir",
                "generator_training_holdout": False,
                "physical_event_boundaries": False,
                "event_correlation_policy": "independent empirical draw conditional on sensor",
                "template_event_id": event["template_event_id"],
            })
        else:
            report.update({
                **CONSTRUCTIONS[args.construction],
                "norm1_equivalents_per_polarity": 420,
            })
        if args.sample == "COUNT":
            totals = {"expected": 0, "written": 0, "dropped": 0, "reassigned": 0}
            for short in COLLECTIONS:
                for key in totals:
                    totals[key] += count_provenance[short][key]
            report["count_tolerance"] = args.count_tolerance
            report["count_occupancy_totals"] = totals
            report["count_occupancy_exact"] = totals["dropped"] == 0 and totals["reassigned"] == 0
        (work / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        if destination.exists():
            raise FileExistsError(f"Output appeared during writing: {destination}")
        work.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True, help="Prepared condition directory")
    parser.add_argument("--construction", choices=INPUT_CONSTRUCTIONS, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--sample", choices=("SIM", "COUNT"), required=True)
    parser.add_argument("--signal", required=True, help="Neutrino SIM ROOT file")
    parser.add_argument("--signal-entry", required=True, type=int)
    parser.add_argument("--count-arrays", help="Assigned arrays for this single event")
    parser.add_argument(
        "--count-tolerance", type=float, default=0.05,
        help="COUNT only: max per-collection fraction of conditioned hits that CellID "
             "assignment may drop/reassign before the input is rejected (default 0.05). "
             "The actual per-collection deficit is always recorded in the manifest.")
    parser.add_argument("--output", required=True, help="New output directory")
    args = parser.parse_args()
    if (args.sample == "COUNT") != bool(args.count_arrays):
        parser.error("--count-arrays is required for COUNT and must be omitted for SIM")
    if not 0.0 <= args.count_tolerance <= 1.0:
        parser.error("--count-tolerance must be in [0, 1]")
    if args.signal_entry < 0:
        parser.error("--signal-entry must be nonnegative")
    write_input(args)


if __name__ == "__main__":
    main()
