#!/usr/bin/env python3
"""Extract Kinematic-7 track features from reconstructed SIM/COUNT events.

Reads each event's ``reco_output.edm4hep.root`` (one event per file), pulls the
``SiTracks`` subset of ``AllTracks`` and its impact-parameter track state, and
writes one store per (sample, split) as a padded ``.npz`` array plus a ``.json``
manifest. Runs in the v3 container (uproot + numpy); no h5py needed. The trainer
consumes ``<store-prefix>.npz``.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from count_tracker_track_features import RAW_FEATURES, track_row


def choose_state(begin, end, locations):
    """Index of the AtIP track state (location==1), else the first state."""
    if begin < 0 or begin >= end or begin >= len(locations):
        return None
    end = min(end, len(locations))
    at_ip = np.flatnonzero(locations[begin:end] == 1)
    return begin + int(at_ip[0]) if len(at_ip) else begin


def read_tracks(events):
    """Per-event (n_tracks, len(RAW_FEATURES)) arrays for the SiTracks subset."""
    idx = events["SiTracks_objIdx"]["SiTracks_objIdx.index"].array()
    begin = events["AllTracks"]["AllTracks.trackStates_begin"].array()
    end = events["AllTracks"]["AllTracks.trackStates_end"].array()
    states = events["_AllTracks_trackStates"]
    loc = states["_AllTracks_trackStates.location"].array()
    phi = states["_AllTracks_trackStates.phi"].array()
    omega = states["_AllTracks_trackStates.omega"].array()
    tanl = states["_AllTracks_trackStates.tanLambda"].array()
    d0 = states["_AllTracks_trackStates.D0"].array()
    z0 = states["_AllTracks_trackStates.Z0"].array()

    per_event = []
    for i in range(events.num_entries):
        sel = np.asarray(idx[i], dtype=np.int64)
        b = np.asarray(begin[i], dtype=np.int64)
        e = np.asarray(end[i], dtype=np.int64)
        lc = np.asarray(loc[i], dtype=np.int64)
        cols = [np.asarray(c[i], dtype=np.float64) for c in (phi, omega, tanl, d0, z0)]
        rows = []
        for track_index in sel:
            if track_index < 0 or track_index >= len(b):
                continue
            state = choose_state(int(b[track_index]), int(e[track_index]), lc)
            if state is None:
                continue
            row = track_row(*(column[state] for column in cols))
            if row is not None:
                rows.append(row)
        per_event.append(np.asarray(rows, dtype=np.float32).reshape(-1, len(RAW_FEATURES)))
    return per_event


def pack_store(per_event):
    """Zero-pad variable-length per-event track arrays into one dense block."""
    counts = np.asarray([len(tracks) for tracks in per_event], dtype=np.int64)
    width = int(counts.max()) if len(counts) else 0
    tracks = np.zeros((len(per_event), width, len(RAW_FEATURES)), dtype=np.float32)
    for index, event_tracks in enumerate(per_event):
        if len(event_tracks):
            tracks[index, :len(event_tracks)] = event_tracks
    return tracks, counts


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def locate_reco(events_root, split, event_id, sample):
    """Find the event's reco output, split-nested first then flat."""
    events_root = Path(events_root)
    for candidate in (
        events_root / split / event_id / sample / "reco" / "reco_output.edm4hep.root",
        events_root / event_id / sample / "reco" / "reco_output.edm4hep.root",
    ):
        if candidate.is_file():
            return candidate
    return None


def build_store(args):
    import uproot

    conditions_manifest = Path(args.conditions) / "manifest.json"
    report = json.loads(conditions_manifest.read_text())
    construction = report["manifest"]["construction"]
    event_ids = [event["event_id"] for event in report["events"]
                 if event["split"] == args.split]
    if not event_ids:
        raise SystemExit(f"No events for split={args.split} in {conditions_manifest}")

    output = Path(args.output)
    if output.with_suffix(".npz").exists():
        raise FileExistsError(f"Refusing to replace {output.with_suffix('.npz')}")

    per_event, sources, missing = [], [], []
    for event_id in event_ids:
        reco = locate_reco(args.events_root, args.split, event_id, args.sample)
        if reco is None:
            missing.append(event_id)
            continue
        with uproot.open(reco) as handle:
            tracks = read_tracks(handle["events"])
        if len(tracks) != 1:
            raise SystemExit(f"{reco}: expected 1 event, found {len(tracks)}")
        per_event.append(tracks[0])
        sources.append({"event_id": event_id, "path": str(reco.resolve()),
                        "n_tracks": int(len(tracks[0]))})
    if missing:
        raise SystemExit(f"Missing {len(missing)} reco outputs: {missing[:10]}")

    tracks, counts = pack_store(per_event)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output.with_suffix(".npz"), tracks=tracks, n_tracks=counts)
    manifest = {
        "kind": "count_tracker_track_store",
        "construction": construction,
        "sample": args.sample,
        "split": args.split,
        "features": list(RAW_FEATURES),
        "n_events": len(per_event),
        "max_tracks": int(tracks.shape[1]),
        "total_tracks": int(counts.sum()),
        "mean_tracks": float(counts.mean()) if len(counts) else 0.0,
        "conditions_manifest_sha256": sha256_bytes(conditions_manifest.read_bytes()),
        "events": sources,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.sample}/{args.split}: {len(per_event)} events, "
          f"mean {manifest['mean_tracks']:.1f} tracks -> {output.with_suffix('.npz')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True, help="Prepared conditions dir (event/split list)")
    parser.add_argument("--events-root", required=True, help="Root of per-event reco outputs")
    parser.add_argument("--sample", choices=("SIM", "COUNT"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", required=True, help="Store prefix (writes .npz + .json)")
    args = parser.parse_args()
    build_store(args)


if __name__ == "__main__":
    main()
