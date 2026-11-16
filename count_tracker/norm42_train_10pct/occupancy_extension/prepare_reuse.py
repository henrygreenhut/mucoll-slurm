#!/usr/bin/env python3
"""Recover exact source manifests for existing SIM occupancy cohorts.

The original occupancy-scan condition arrays may have been cleaned after the
SIM reconstruction was completed. For those points, reproduce the original
source selection from its recorded seed/cohort algorithm. Existing 1% and
200% source manifests are filtered directly. Conditions are deliberately
rebuilt later from these manifests rather than inferred from COUNT output.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from count_tracker_conditions import (  # noqa: E402
    CELL_ID_ENCODING, CONSTRUCTIONS, NORM42_OCCUPANCY_SCAN_NAME,
    validate_manifest,
)
from count_tracker_manifest import select_events  # noqa: E402
from v3_occupancy_scan.prepare_scan import read_sim_root  # noqa: E402


POINTS = {
    "low": (
        ("n0010", "occupancy_scan", "n0010", 10, 5, None, "unsplit"),
        ("n0017", "onepct", "batch_0000", 17, 5, None, "unsplit"),
        ("n0036", "occupancy_scan", "n0036", 36, 5, None, "five-region"),
        ("n0129", "occupancy_scan", "n0129", 129, 5, None, "five-region"),
        ("n0464", "occupancy_scan", "n0464", 464, 5, None, "five-region"),
    ),
    "high": (
        ("n1666", "occupancy_scan", "n1666", 1666, 5, None, "five-region"),
        # Event 000001 is the completed 200% event already used in the plot.
        ("n3332", "high200", "n3332", 3332, 1, (1,), "five-region"),
    ),
}
REGION_TAGS = (
    "theta_000_030", "theta_030_070", "theta_070_110",
    "theta_110_150", "theta_150_180",
)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_training_cycles(path):
    document = json.loads(Path(path).resolve().read_text())
    return set().union(*(
        {int(row["cycle"]) for row in document["sources"][polarity]}
        for polarity in ("MUPLUS", "MUMINUS")
    ))


def regenerated_occupancy_manifest(n_files, pools, provenance):
    selected = select_events(
        pools, {"train": 0, "val": 0, "test": 5}, 12345,
        "occupancy_scan", "norm42", source_pool="all",
        n_files_per_polarity=n_files, prefix_stable=True,
    )
    return {
        "schema_version": 2,
        "cell_id_encoding": CELL_ID_ENCODING,
        "construction": "norm42",
        "n_files_per_polarity": n_files,
        "file_normalization": CONSTRUCTIONS["norm42"]["file_normalization"],
        "norm1_equivalents_per_polarity": n_files * 42,
        "source_pool": "all",
        "pool": provenance,
        "overlay_normalization": {
            "name": NORM42_OCCUPANCY_SCAN_NAME,
            "selected_files_per_polarity": n_files,
            "full_bx_reference_files_per_polarity": 1666,
            "fraction_of_full_bx_reference": n_files / 1666,
            "selection": "nested deterministic source-order prefix",
        },
        "sampling": {
            "seed": 12345,
            "cohort": "occupancy_scan",
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


def select_manifest(source_manifest, event_count, indices):
    manifest = json.loads(Path(source_manifest).read_text())
    if indices is None:
        indices = tuple(range(event_count))
    if len(indices) != event_count:
        raise ValueError("event-index selection has the wrong length")
    manifest = dict(manifest)
    manifest["events"] = [manifest["events"][index] for index in indices]
    return manifest


def write_point(destination, manifest, mode, training_cycles, sim_source_point):
    destination.mkdir(parents=True)
    validate_manifest(manifest, destination)
    source_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    (destination / "sources.json").write_bytes(source_bytes)
    overlaps = []
    for event in manifest["events"]:
        by_polarity = {
            polarity: sorted(
                {int(row["cycle"]) for row in event["sources"][polarity]}
                & training_cycles
            )
            for polarity in ("MUPLUS", "MUMINUS")
        }
        overlaps.append({
            "event_id": event["event_id"],
            "training_cycle_overlap_by_polarity": by_polarity,
            "training_physical_cycle_overlap": sorted(
                set(by_polarity["MUPLUS"]) | set(by_polarity["MUMINUS"])
            ),
        })
    return {
        "events": [
            {"event_id": event["event_id"], "split": event["split"]}
            for event in manifest["events"]
        ],
        "sim_source_point": str(Path(sim_source_point).resolve()),
        "reconstruction": mode,
        "files_per_polarity": manifest["n_files_per_polarity"],
        "fraction_of_full_bx_reference": manifest[
            "overlay_normalization"
        ]["fraction_of_full_bx_reference"],
        "source_manifest_sha256": sha256_bytes(source_bytes),
        "training_overlap": overlaps,
    }


def validate_sim_outputs(sim_source_point, events, mode):
    """Fail preparation unless every existing SIM result to be reused exists."""
    sim_source_point = Path(sim_source_point).resolve()
    missing = []
    for event in events:
        base = sim_source_point / "events" / event["event_id"] / "SIM"
        paths = (
            [base / "unsplit_reco" / "reco_output.edm4hep.root"]
            if mode == "unsplit" else [
                base / "theta_reco" / tag / "reco_output.edm4hep.root"
                for tag in REGION_TAGS
            ]
        )
        missing.extend(str(path) for path in paths if not path.is_file())
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} existing SIM reconstruction output(s) missing:\n{preview}"
        )


def prepare(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    roots = {
        "occupancy_scan": Path(args.occupancy_scan).resolve(),
        "onepct": Path(args.onepct_scan).resolve(),
        "high200": Path(args.high200_scan).resolve(),
    }
    training_cycles = load_training_cycles(args.training_sources)
    needs_occupancy = any(row[1] == "occupancy_scan" for row in POINTS[args.wave])
    pools = provenance = None
    if needs_occupancy:
        pools, provenance = read_sim_root(args.sim_root)
    output.mkdir(parents=True)
    points = []
    for tag, root_key, source_tag, n_files, event_count, indices, mode in POINTS[args.wave]:
        source_point = roots[root_key] / source_tag
        if root_key == "occupancy_scan":
            manifest = regenerated_occupancy_manifest(n_files, pools, provenance)
            if indices is not None:
                manifest["events"] = [manifest["events"][index] for index in indices]
        else:
            manifest = select_manifest(
                source_point / "sources.json", event_count, indices)
        if len(manifest["events"]) != event_count:
            raise ValueError(f"wrong event count for {tag}")
        validate_sim_outputs(source_point, manifest["events"], mode)
        result = write_point(
            output / tag, manifest, mode, training_cycles, source_point)
        points.append({"tag": tag, **result})
    plan = {
        "kind": "norm42_10pct_model_existing_sim_occupancy_extension",
        "wave": args.wave,
        "model_training_regime_bx_fraction": 167 / 1666,
        "training_sources": str(Path(args.training_sources).resolve()),
        "comparison": (
            "recover exact existing SIM source selection; rebuild conditions "
            "from authoritative SIM; regenerate only COUNT with 10%-trained model"
        ),
        "points": points,
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Prepared {args.wave} reuse wave with {len(points)} point(s) -> {output}")
    for point in points:
        print(
            f"{point['tag']}: {len(point['events'])} event(s), "
            f"{100 * point['fraction_of_full_bx_reference']:.4f}% BX, "
            f"{point['reconstruction']}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", choices=tuple(POINTS), required=True)
    parser.add_argument("--sim-root", required=True)
    parser.add_argument("--occupancy-scan", required=True)
    parser.add_argument("--onepct-scan", required=True)
    parser.add_argument("--high200-scan", required=True)
    parser.add_argument("--training-sources", required=True)
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
