#!/usr/bin/env python3
"""Audit paper UltraTight thresholds on v3 tracker reconstruction outputs.

The v3 tracker workflow stores duplicate-removed CKF tracks as the ``SiTracks``
subset of ``AllTracks``.  EDM4hep's ``trackerHits`` relation contains the hits
used to create a track, while ``Nholes`` is stored directly.  These files do not
store the number of outliers, so the paper's ``n_outliers < 4`` requirement
cannot be reconstructed from ``trackerHits`` and ``ndf``.  This script reports
the reproducible four-cut subset and states that limitation explicitly.  It is
also not identical to the paper's preselection-and-refit sequence.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from count_tracker_features import choose_state
from count_tracker_track_features import CURVATURE_TO_PT


def ultratight_mask(pt, n_hits, n_holes, n_outliers, reduced_chi2):
    """Return the five paper cuts and their conjunction."""
    cuts = {
        "pt_gt_0p5_GeV": np.asarray(pt) > 0.5,
        "n_hits_ge_9": np.asarray(n_hits) >= 9,
        "n_holes_lt_3": np.asarray(n_holes) < 3,
        "n_outliers_lt_4": np.asarray(n_outliers) < 4,
        "reduced_chi2_lt_3": np.asarray(reduced_chi2) < 3.0,
    }
    cuts["ultratight"] = np.logical_and.reduce(tuple(cuts.values()))
    cuts["all_except_outliers"] = np.logical_and.reduce(tuple(
        mask for name, mask in cuts.items()
        if name not in ("n_outliers_lt_4", "ultratight")
    ))
    return cuts


def ultratight_observable_mask(pt, n_hits, n_holes, reduced_chi2):
    """Return the reproducible UltraTight criteria and their conjunction."""
    cuts = {
        "pt_gt_0p5_GeV": np.asarray(pt) > 0.5,
        "n_hits_ge_9": np.asarray(n_hits) >= 9,
        "n_holes_lt_3": np.asarray(n_holes) < 3,
        "reduced_chi2_lt_3": np.asarray(reduced_chi2) < 3.0,
    }
    cuts["ultratight_observable_subset"] = np.logical_and.reduce(
        tuple(cuts.values()))
    return cuts


def read_events(events):
    """Return one count record for each event in an EDM4hep events tree."""
    selected = events["SiTracks_objIdx"]["SiTracks_objIdx.index"].array()
    tracks = events["AllTracks"]
    hit_begin = tracks["AllTracks.trackerHits_begin"].array()
    hit_end = tracks["AllTracks.trackerHits_end"].array()
    state_begin = tracks["AllTracks.trackStates_begin"].array()
    state_end = tracks["AllTracks.trackStates_end"].array()
    ndf = tracks["AllTracks.ndf"].array()
    chi2 = tracks["AllTracks.chi2"].array()
    n_holes = tracks["AllTracks.Nholes"].array()
    states = events["_AllTracks_trackStates"]
    locations = states["_AllTracks_trackStates.location"].array()
    omegas = states["_AllTracks_trackStates.omega"].array()

    records = []
    for event_index in range(events.num_entries):
        indices = np.asarray(selected[event_index], dtype=np.int64)
        hb = np.asarray(hit_begin[event_index], dtype=np.int64)[indices]
        he = np.asarray(hit_end[event_index], dtype=np.int64)[indices]
        sb = np.asarray(state_begin[event_index], dtype=np.int64)
        se = np.asarray(state_end[event_index], dtype=np.int64)
        nd = np.asarray(ndf[event_index], dtype=np.int64)[indices]
        c2 = np.asarray(chi2[event_index], dtype=np.float64)[indices]
        holes = np.asarray(n_holes[event_index], dtype=np.int64)[indices]
        loc = np.asarray(locations[event_index], dtype=np.int64)
        omega = np.asarray(omegas[event_index], dtype=np.float64)

        if np.any(nd <= 0):
            raise ValueError(f"event {event_index}: expected positive track-fit ndf")
        n_hits = he - hb

        pt = np.empty(len(indices), dtype=np.float64)
        for output_index, track_index in enumerate(indices):
            state = choose_state(int(sb[track_index]), int(se[track_index]), loc)
            if state is None or not np.isfinite(omega[state]) or omega[state] == 0:
                raise ValueError(
                    f"event {event_index}, AllTracks {track_index}: invalid AtIP state"
                )
            pt[output_index] = CURVATURE_TO_PT / abs(omega[state])

        reduced_chi2 = c2 / nd
        cuts = ultratight_observable_mask(pt, n_hits, holes, reduced_chi2)
        records.append({
            "loose": int(len(indices)),
            "ultratight_observable_subset": int(np.count_nonzero(
                cuts["ultratight_observable_subset"])),
            "cutflow": {
                name: int(np.count_nonzero(mask))
                for name, mask in cuts.items()
                if name != "ultratight_observable_subset"
            },
        })
    return records


def find_reco_files(path):
    if glob.has_magic(path):
        return [Path(item) for item in sorted(glob.glob(path, recursive=True))
                if Path(item).is_file()]
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.rglob("reco_output.edm4hep.root"))


def parse_input(value):
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("input must be LABEL=PATH") from exc
    if not label or not path:
        raise argparse.ArgumentTypeError("input must be LABEL=PATH")
    return label, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", type=parse_input, required=True,
        metavar="LABEL=PATH",
        help="ROOT file or directory searched recursively; repeat for each arm",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--summary-only", action="store_true",
                        help="Omit per-event records from the JSON report")
    args = parser.parse_args()

    import uproot

    report = {
        "selection": {
            "pt_GeV": "> 0.5",
            "n_hits": ">= 9",
            "n_holes": "< 3",
            "n_outliers": "< 4 (not reproducible from these EDM4hep files)",
            "reduced_chi2": "< 3",
            "scope": (
                "existing-v3-CKF four-cut subset; paper workflow also requires "
                "<4 outliers and refits before final selection"),
        },
        "samples": {},
    }
    for label, path in args.input:
        files = find_reco_files(path)
        if not files:
            raise FileNotFoundError(f"{label}: no reco_output.edm4hep.root below {path}")
        event_records = []
        for filename in files:
            with uproot.open(filename) as handle:
                records = read_events(handle["events"])
            for record in records:
                record["path"] = str(filename)
                event_records.append(record)
        loose = np.asarray([event["loose"] for event in event_records])
        tight = np.asarray([
            event["ultratight_observable_subset"] for event in event_records])
        report["samples"][label] = {
            "n_files": len(files),
            "n_events": len(event_records),
            "loose_total": int(loose.sum()),
            "loose_mean": float(loose.mean()),
            "ultratight_observable_subset_total": int(tight.sum()),
            "ultratight_observable_subset_mean": float(tight.mean()),
            "ultratight_observable_subset_std": (
                float(tight.std(ddof=1)) if len(tight) > 1 else 0.0),
            "events": event_records,
        }
        if args.summary_only:
            report["samples"][label].pop("events")

    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
