#!/usr/bin/env python3
"""Build full-BX theta/phi track patches for the preliminary PFN test.

The five theta regions are the regions used for reconstruction. Within each
regional SiTracks collection, every track is assigned to one fixed sector by
its fitted AtIP momentum phi. The stored phi coordinate is relative to the
sector centre, so the PFN compares local angular structure rather than the
sector's absolute detector azimuth.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from count_tracker_features import pack_store, read_tracks
from count_tracker_track_features import RAW, RAW_FEATURES


REGIONS = (
    ("theta_000_030", 0, 30),
    ("theta_030_070", 30, 70),
    ("theta_070_110", 70, 110),
    ("theta_110_150", 110, 150),
    ("theta_150_180", 150, 180),
)
SAMPLES = ("SIM", "COUNT")
SPLITS = ("train", "val", "test")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap_phi(phi):
    """Map angles to [-pi, pi)."""
    return (np.asarray(phi, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def sector_indices_and_local_phi(phi, n_sectors):
    """Return fixed global-phi sector indices and phi relative to each centre."""
    if n_sectors < 1:
        raise ValueError("n_sectors must be positive")
    wrapped = wrap_phi(phi)
    width = 2 * np.pi / n_sectors
    sectors = np.floor((wrapped + np.pi) / width).astype(np.int64)
    sectors = np.clip(sectors, 0, n_sectors - 1)
    centres = -np.pi + (sectors + 0.5) * width
    local = wrap_phi(wrapped - centres)
    return sectors, local


def partition_tracks(tracks, n_sectors):
    """Partition tracks exactly once and replace global phi with local delta-phi."""
    tracks = np.asarray(tracks, dtype=np.float32)
    if tracks.ndim != 2 or tracks.shape[1] != len(RAW_FEATURES):
        raise ValueError("tracks must have shape (n_tracks, n_raw_features)")
    sectors, local_phi = sector_indices_and_local_phi(tracks[:, RAW["phi"]], n_sectors)
    patches = []
    for sector in range(n_sectors):
        selected = np.flatnonzero(sectors == sector)
        patch = tracks[selected].copy()
        patch[:, RAW["phi"]] = local_phi[selected]
        patches.append(patch)
    if sum(map(len, patches)) != len(tracks):
        raise AssertionError("phi partition did not preserve the track count")
    return patches


def regional_reco(point, event_id, sample, region):
    return (point / "events" / event_id / sample / "theta_reco" / region /
            "reco_output.edm4hep.root")


def assign_splits(events, counts):
    if sum(counts) != len(events) or any(count < 1 for count in counts):
        raise ValueError(
            f"split counts {counts} must be positive and sum to {len(events)} parent events")
    assignment = {}
    offset = 0
    for split, count in zip(SPLITS, counts):
        for event in events[offset:offset + count]:
            assignment[event["event_id"]] = split
        offset += count
    return assignment


def read_regional_tracks(path):
    import uproot

    if not path.is_file():
        raise FileNotFoundError(path)
    with uproot.open(path) as root:
        per_event = read_tracks(root["events"])
    if len(per_event) != 1:
        raise ValueError(f"{path}: expected one event, found {len(per_event)}")
    return per_event[0]


def build(args):
    scan_root = Path(args.scan_root).resolve()
    point = scan_root / args.point
    conditions_path = point / "conditions" / "manifest.json"
    conditions = json.loads(conditions_path.read_text())
    events = conditions["events"]
    assignment = assign_splits(events, tuple(args.split_counts))

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    output.mkdir(parents=True)

    construction = f"norm42_fullbx_theta_phi{args.phi_sectors}"
    condition_digest = sha256_file(conditions_path)
    sector_width = 2 * np.pi / args.phi_sectors
    stores = {split: {sample: [] for sample in SAMPLES} for split in SPLITS}
    records = {split: {sample: [] for sample in SAMPLES} for split in SPLITS}
    parent_totals = {sample: 0 for sample in SAMPLES}

    for event in events:
        event_id = event["event_id"]
        split = assignment[event_id]
        for region, theta_min, theta_max in REGIONS:
            for sample in SAMPLES:
                path = regional_reco(point, event_id, sample, region)
                regional_tracks = read_regional_tracks(path)
                parent_totals[sample] += len(regional_tracks)
                patches = partition_tracks(regional_tracks, args.phi_sectors)
                for sector, patch in enumerate(patches):
                    lower = -np.pi + sector * sector_width
                    upper = lower + sector_width
                    example_id = f"{event_id}__{region}__phi_{sector:03d}"
                    stores[split][sample].append(patch)
                    records[split][sample].append({
                        "event_id": example_id,
                        "parent_event_id": event_id,
                        "theta_region": region,
                        "theta_bounds_deg": [theta_min, theta_max],
                        "phi_sector": sector,
                        "phi_bounds_rad": [float(lower), float(upper)],
                        "phi_center_rad": float(lower + 0.5 * sector_width),
                        "path": str(path),
                        "n_tracks": int(len(patch)),
                    })

    expected_per_parent = len(REGIONS) * args.phi_sectors
    for split in SPLITS:
        parent_ids = [event["event_id"] for event in events
                      if assignment[event["event_id"]] == split]
        expected_examples = len(parent_ids) * expected_per_parent
        for sample in SAMPLES:
            examples = stores[split][sample]
            rows = records[split][sample]
            if len(examples) != expected_examples:
                raise AssertionError(f"{sample}/{split}: wrong number of patches")
            tracks, counts = pack_store(examples)
            prefix = output / f"{construction}_{sample}_{split}"
            np.savez(prefix.with_suffix(".npz"), tracks=tracks, n_tracks=counts)
            manifest = {
                "kind": "count_tracker_angular_patch_store",
                "schema_version": 2,
                "construction": construction,
                "sample": sample,
                "split": split,
                "track_collection": "SiTracks",
                "track_state": "AtIP",
                "features": list(RAW_FEATURES),
                "phi_coordinate": "delta_phi_from_fixed_global_sector_center",
                "patch_definition": {
                    "theta_assignment": "pre-reconstruction digitized-hit position theta",
                    "theta_regions_deg": [[row[1], row[2]] for row in REGIONS],
                    "phi_assignment": "fitted AtIP momentum phi",
                    "phi_range_rad": [-math.pi, math.pi],
                    "phi_sectors": args.phi_sectors,
                    "phi_sector_width_rad": sector_width,
                    "stored_phi": "wrap(global_AtIP_phi - sector_center)",
                    "empty_patches_retained": True,
                },
                "split_policy": {
                    "kind": ("ordered-parent-event-3/1/1"
                             if args.split_counts == [3, 1, 1]
                             else "ordered-parent-event"),
                    "counts": dict(zip(SPLITS, args.split_counts)),
                    "parent_event_ids": parent_ids,
                    "all_patches_from_parent_kept_together": True,
                },
                "example_unit": "parent-event theta-region phi-sector patch",
                "example_parent_ids": [row["parent_event_id"] for row in rows],
                "n_parent_events": len(parent_ids),
                "n_events": len(examples),
                "max_tracks": int(tracks.shape[1]),
                "total_tracks": int(counts.sum()),
                "mean_tracks": float(counts.mean()),
                "conditions_manifest_sha256": condition_digest,
                "scan_root": str(scan_root),
                "scan_point": args.point,
                "events": rows,
                "statistical_scope": (
                    "patches share parent reconstructions; parent events, not patches, "
                    "are the units for splitting and uncertainty"
                ),
            }
            prefix.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                f"{sample}/{split}: {len(examples)} patches from {len(parent_ids)} parent "
                f"events, {int(counts.sum()):,} tracks, width {tracks.shape[1]}"
            )

    for sample in SAMPLES:
        stored = sum(sum(len(patch) for patch in stores[split][sample])
                     for split in SPLITS)
        if stored != parent_totals[sample]:
            raise AssertionError(f"{sample}: stored {stored} tracks but read {parent_totals[sample]}")
    summary = {
        "construction": construction,
        "scan_root": str(scan_root),
        "scan_point": args.point,
        "phi_sectors": args.phi_sectors,
        "patches_per_parent": expected_per_parent,
        "split_counts": dict(zip(SPLITS, args.split_counts)),
        "assignment": assignment,
        "total_tracks": parent_totals,
    }
    (output / "angular_patch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"angular patch stores -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--point", default="n1666")
    parser.add_argument("--phi-sectors", type=int, default=72)
    parser.add_argument("--split-counts", type=int, nargs=3, default=[3, 1, 1],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
