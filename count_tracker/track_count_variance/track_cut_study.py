#!/usr/bin/env python3
"""Compare reconstructed SIM and COUNT multiplicities under track-quality cuts.

The input is a completed track-count-variance cohort with one unsplit v3 CKF
output per event and arm.  Existing tracks are selected after reconstruction;
the script does not refit tracks or modify ROOT files.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


CURVATURE_TO_PT = 0.0015  # pT [GeV] for omega [1/mm] at B = 5 T
SAMPLES = ("SIM", "COUNT")


def choose_state(begin, end, locations):
    """Return the first AtIP state (EDM4hep location 1) in a track's range."""
    if begin < 0 or begin >= end or begin >= len(locations):
        return None
    at_ip = np.flatnonzero(locations[begin:min(end, len(locations))] == 1)
    return begin + int(at_ip[0]) if len(at_ip) else None


def count_selections(events):
    """Return named selection counts for the one event in an events tree."""
    if events.num_entries != 1:
        raise ValueError(f"expected one event, found {events.num_entries}")

    selected = np.asarray(
        events["SiTracks_objIdx"]["SiTracks_objIdx.index"].array()[0],
        dtype=np.int64,
    )
    tracks = events["AllTracks"]
    hit_begin = np.asarray(tracks["AllTracks.trackerHits_begin"].array()[0])
    hit_end = np.asarray(tracks["AllTracks.trackerHits_end"].array()[0])
    state_begin = np.asarray(tracks["AllTracks.trackStates_begin"].array()[0])
    state_end = np.asarray(tracks["AllTracks.trackStates_end"].array()[0])
    ndf = np.asarray(tracks["AllTracks.ndf"].array()[0], dtype=np.int64)[selected]
    chi2 = np.asarray(tracks["AllTracks.chi2"].array()[0], dtype=float)[selected]
    holes = np.asarray(tracks["AllTracks.Nholes"].array()[0], dtype=np.int64)[selected]

    states = events["_AllTracks_trackStates"]
    locations = np.asarray(
        states["_AllTracks_trackStates.location"].array()[0], dtype=np.int64)
    omega = np.asarray(
        states["_AllTracks_trackStates.omega"].array()[0], dtype=float)

    if np.any(ndf <= 0):
        raise ValueError("expected positive track-fit ndf")
    n_hits = hit_end[selected] - hit_begin[selected]

    pt = np.empty(len(selected), dtype=float)
    for output_index, track_index in enumerate(selected):
        state = choose_state(
            int(state_begin[track_index]), int(state_end[track_index]), locations)
        if state is None or not np.isfinite(omega[state]) or omega[state] == 0:
            raise ValueError(f"AllTracks {track_index}: invalid AtIP state")
        pt[output_index] = CURVATURE_TO_PT / abs(omega[state])

    reduced_chi2 = chi2 / ndf
    criteria = {
        **{f"pt_gt_{str(value).replace('.', 'p')}": pt > value
           for value in (0.5, 1.0, 2.0, 5.0, 10.0)},
        **{f"n_hits_ge_{value}": n_hits >= value
           for value in (9, 10, 11, 12, 13)},
        "n_holes_lt_3": holes < 3,
        "n_holes_eq_0": holes == 0,
        **{f"chi2_ndf_lt_{str(value).replace('.', 'p')}": reduced_chi2 < value
           for value in (1.0, 1.5, 2.0, 3.0, 10.0)},
    }
    # Preserve the published-cut names used by the original report.
    criteria["chi2_ndf_lt_3"] = criteria["chi2_ndf_lt_3p0"]
    criteria["chi2_ndf_lt_10"] = criteria["chi2_ndf_lt_10p0"]
    cumulative = np.ones(len(selected), dtype=bool)
    selections = {"loose": int(len(selected))}
    for name in (
        "pt_gt_0p5", "n_hits_ge_9", "n_holes_lt_3", "chi2_ndf_lt_3",
    ):
        cumulative &= criteria[name]
        selections[f"cumulative_through_{name}"] = int(np.count_nonzero(cumulative))
    selections.update({
        "chi2_ndf_lt_10": int(np.count_nonzero(criteria["chi2_ndf_lt_10"])),
        "pt_gt_0p5_and_chi2_ndf_lt_10": int(np.count_nonzero(
            criteria["pt_gt_0p5"] & criteria["chi2_ndf_lt_10"])),
        "ultratight_observable_subset": int(np.count_nonzero(
            criteria["pt_gt_0p5"]
            & criteria["n_hits_ge_9"]
            & criteria["n_holes_lt_3"]
            & criteria["chi2_ndf_lt_3"])),
    })
    # Exploratory post-reconstruction cuts.  The grid keeps the published
    # holes and fit-quality requirements fixed while tightening pT and the
    # number of used hits.  It is a diagnostic scan, not a claim that these
    # are established working points.
    for pt_cut in (0.5, 1.0, 2.0, 5.0, 10.0):
        pt_name = f"pt_gt_{str(pt_cut).replace('.', 'p')}"
        for hit_cut in (9, 10, 11, 12, 13):
            selections[f"scan_{pt_name}_hits_ge_{hit_cut}"] = int(np.count_nonzero(
                criteria[pt_name]
                & criteria[f"n_hits_ge_{hit_cut}"]
                & criteria["n_holes_lt_3"]
                & criteria["chi2_ndf_lt_3"]
            ))
    selections.update({
        "aggressive_pt1_hits10_holes0_chi2lt2": int(np.count_nonzero(
            criteria["pt_gt_1p0"] & criteria["n_hits_ge_10"]
            & criteria["n_holes_eq_0"] & criteria["chi2_ndf_lt_2p0"])),
        "aggressive_pt2_hits11_holes0_chi2lt1p5": int(np.count_nonzero(
            criteria["pt_gt_2p0"] & criteria["n_hits_ge_11"]
            & criteria["n_holes_eq_0"] & criteria["chi2_ndf_lt_1p5"])),
        "aggressive_pt5_hits12_holes0_chi2lt1": int(np.count_nonzero(
            criteria["pt_gt_5p0"] & criteria["n_hits_ge_12"]
            & criteria["n_holes_eq_0"] & criteria["chi2_ndf_lt_1p0"])),
    })
    individual = {name: int(np.count_nonzero(mask))
                  for name, mask in criteria.items()}
    return selections, individual


def reco_files(scan_root, sample):
    pattern = f"batch_*/events/*/{sample}/unsplit_reco/reco_output.edm4hep.root"
    files = sorted(scan_root.glob(pattern))
    return {path.parents[2].name: path for path in files}


def distribution(values):
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else None
    return {
        "n_events": int(len(values)),
        "total": int(values.sum()),
        "mean": float(values.mean()),
        "std_ddof1": std,
        "sem": std / math.sqrt(len(values)) if std is not None else None,
        "median": float(np.median(values)),
        "minimum": int(values.min()),
        "maximum": int(values.max()),
    }


def build_report(scan_root):
    import uproot

    scan_root = Path(scan_root).resolve()
    paths = {sample: reco_files(scan_root, sample) for sample in SAMPLES}
    if not paths["SIM"] or paths["SIM"].keys() != paths["COUNT"].keys():
        raise ValueError("SIM and COUNT reconstructed event IDs are missing or unpaired")

    event_rows = []
    for event_id in sorted(paths["SIM"]):
        row = {"event_id": event_id}
        for sample in SAMPLES:
            with uproot.open(paths[sample][event_id]) as root:
                selections, individual = count_selections(root["events"])
            row[sample] = {
                "selections": selections,
                "individual_criteria": individual,
            }
        event_rows.append(row)

    selection_names = list(event_rows[0]["SIM"]["selections"])
    summary = {}
    for selection in selection_names:
        sim = np.asarray([
            row["SIM"]["selections"][selection] for row in event_rows])
        count = np.asarray([
            row["COUNT"]["selections"][selection] for row in event_rows])
        summary[selection] = {
            "SIM": distribution(sim),
            "COUNT": distribution(count),
            "count_over_sim_ratio_of_means": float(count.mean() / sim.mean())
            if sim.mean() else None,
            "mean_paired_difference": float((count - sim).mean()),
        }

    return {
        "kind": "norm42_unsplit_reconstructed_track_cut_study",
        "scan_root": str(scan_root),
        "events": len(event_rows),
        "scope": "post-reconstruction selection of existing duplicate-removed SiTracks",
        "definitions": {
            "loose": "all SiTracks",
            "chi2_ndf_lt_10": "AllTracks.chi2 / AllTracks.ndf < 10",
            "pt_gt_0p5_and_chi2_ndf_lt_10": "pT > 0.5 GeV and chi2/ndf < 10",
            "ultratight_observable_subset": (
                "pT > 0.5 GeV, >=9 used hits, <3 holes, and chi2/ndf <3; "
                "the paper's <4 outliers requirement is not applied"),
            "outlier_availability": (
                "not stored per track; EDM4hep trackerHits contains hits used to "
                "create the track and cannot be converted to an outlier count from ndf"),
            "aggressive_scan": (
                "exploratory post-reconstruction thresholds; the pT/hit grid also "
                "requires <3 holes and chi2/ndf <3"),
        },
        "caveat": (
            "This audits cuts on existing v3 CKF tracks. It cannot reproduce the "
            "paper's outlier requirement or its preselection-and-refit sequence."),
        "summary": summary,
        "per_event": event_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = build_report(args.scan_root)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
