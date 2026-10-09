#!/usr/bin/env python3
"""Verify that five regional outputs partition every digitized hit collection."""

import argparse
import json
from pathlib import Path


DIGITIZED = (
    "VXDBarrelHits",
    "VXDEndcapHits",
    "ITBarrelHits",
    "ITEndcapHits",
    "OTBarrelHits",
    "OTEndcapHits",
)
SIMULATED = (
    "VertexBarrelCollection",
    "VertexEndcapCollection",
    "InnerTrackerBarrelCollection",
    "InnerTrackerEndcapCollection",
    "OuterTrackerBarrelCollection",
    "OuterTrackerEndcapCollection",
)
REGIONS = ("theta_000_030", "theta_030_070", "theta_070_110",
           "theta_110_150", "theta_150_180")


def read_counts(path, names):
    from podio.root_io import Reader

    reader = Reader(str(path))
    frames = reader.get("events")
    if len(frames) != 1:
        raise ValueError(f"Expected one event in {path}, found {len(frames)}")
    frame = frames[0]
    available = set(frame.getAvailableCollections())
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"{path}: missing collections {missing}")
    counts = {name: len(frame.get(name)) for name in names}
    del frame, frames, reader
    return counts


def validate(args):
    source_names = DIGITIZED
    split_names = tuple(f"{name}Split" for name in DIGITIZED)
    digi_path = args.digi_file.resolve()
    source = read_counts(digi_path, SIMULATED + DIGITIZED)
    regional = {}
    tracks = {}
    for tag in REGIONS:
        path = args.regional_root.resolve() / tag / "reco_output.edm4hep.root"
        regional[tag] = read_counts(path, split_names)
        tracks[tag] = read_counts(path, ("SiTracks",))["SiTracks"]

    comparison = {}
    for source_name, split_name in zip(source_names, split_names):
        summed = sum(regional[tag][split_name] for tag in REGIONS)
        expected = source[source_name]
        comparison[source_name] = {
            "digitized": expected,
            "regional_sum": summed,
            "difference": summed - expected,
        }
    failures = {name: row for name, row in comparison.items()
                if row["difference"] != 0}
    survival = {}
    for simulated_name, digitized_name in zip(SIMULATED, DIGITIZED):
        entering = source[simulated_name]
        digitized = source[digitized_name]
        survival[simulated_name] = {
            "digitized_collection": digitized_name,
            "entering_digitization": entering,
            "digitized": digitized,
            "survival_fraction": digitized / entering if entering else None,
        }
    total_entering = sum(source[name] for name in SIMULATED)
    total_digitized = sum(source[name] for name in DIGITIZED)
    report = {
        "status": "exact count partition" if not failures else "count mismatch",
        "digi_file": str(digi_path),
        "regional_root": str(args.regional_root.resolve()),
        "digitization_survival": {
            "collections": survival,
            "total": {
                "entering_digitization": total_entering,
                "digitized": total_digitized,
                "survival_fraction": (
                    total_digitized / total_entering if total_entering else None
                ),
            },
        },
        "theta_partition": comparison,
        "si_tracks_by_region": tracks,
        "total_si_tracks": sum(tracks.values()),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"Refusing to replace {args.output}")
        args.output.write_text(rendered)
    print(rendered, end="")
    if failures:
        raise SystemExit(f"Regional hit counts do not partition digitized input: {failures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digi-file", type=Path, required=True)
    parser.add_argument("--regional-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
