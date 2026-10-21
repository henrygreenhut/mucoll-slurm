#!/usr/bin/env python3
"""Quantify paired SIM/COUNT track-count means and variances as jobs finish."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from count_tracker_checkpoint_report import COLLECTIONS, collection_size


def arm_paths(point, event_id, sample):
    return {
        "digi": point / "digi" / event_id / sample / "digi" / "digi_output.edm4hep.root",
        "reco": point / "events" / event_id / sample / "unsplit_reco" / "reco_output.edm4hep.root",
    }


def read_arm(paths):
    import uproot

    with uproot.open(paths["digi"]) as root:
        events = root["events"]
        entering = sum(
            collection_size(events, sim_name)
            for sim_name, _ in COLLECTIONS.values())
        digitized = sum(
            collection_size(events, digi_name)
            for _, digi_name in COLLECTIONS.values())
    with uproot.open(paths["reco"]) as root:
        tracks = collection_size(root["events"], "SiTracks_objIdx", "index")
    return {
        "entering_digitization": int(entering),
        "digitized": int(digitized),
        "survival_fraction": digitized / entering if entering else None,
        "si_tracks": int(tracks),
    }


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    variance = float(values.var(ddof=1)) if len(values) > 1 else None
    mean = float(values.mean())
    return {
        "n": int(len(values)),
        "mean": mean,
        "variance_ddof1": variance,
        "std_ddof1": float(np.sqrt(variance)) if variance is not None else None,
        "coefficient_of_variation": (
            float(np.sqrt(variance) / mean)
            if variance is not None and mean != 0 else None),
        "minimum": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def statistics(sim, count):
    sim = np.asarray(sim, dtype=float)
    count = np.asarray(count, dtype=float)
    if sim.shape != count.shape:
        raise ValueError("SIM and COUNT arrays must be paired")
    result = {"SIM": distribution(sim), "COUNT": distribution(count)}
    if not len(sim):
        result["paired"] = None
        return result
    difference = count - sim
    sim_variance = sim.var(ddof=1) if len(sim) > 1 else np.nan
    count_variance = count.var(ddof=1) if len(sim) > 1 else np.nan
    result["paired"] = {
        "difference_count_minus_sim": distribution(difference),
        "ratio_of_means_count_over_sim": (
            float(count.mean() / sim.mean()) if sim.mean() else None),
        "variance_ratio_count_over_sim": (
            float(count_variance / sim_variance)
            if np.isfinite(sim_variance) and sim_variance > 0 else None),
        "pearson_correlation": (
            float(np.corrcoef(sim, count)[0, 1])
            if len(sim) > 1 and sim.std() > 0 and count.std() > 0 else None),
    }
    return result


def bootstrap_intervals(sim, count, draws, seed):
    sim = np.asarray(sim, dtype=float)
    count = np.asarray(count, dtype=float)
    if len(sim) < 2 or draws < 1:
        return None
    rng = np.random.default_rng(seed)
    values = {key: [] for key in (
        "sim_mean", "count_mean", "mean_difference_count_minus_sim",
        "ratio_of_means_count_over_sim", "sim_variance", "count_variance",
        "variance_ratio_count_over_sim",
    )}
    for _ in range(draws):
        index = rng.integers(0, len(sim), len(sim))
        a, b = sim[index], count[index]
        va, vb = a.var(ddof=1), b.var(ddof=1)
        values["sim_mean"].append(a.mean())
        values["count_mean"].append(b.mean())
        values["mean_difference_count_minus_sim"].append((b - a).mean())
        values["ratio_of_means_count_over_sim"].append(
            b.mean() / a.mean() if a.mean() else np.nan)
        values["sim_variance"].append(va)
        values["count_variance"].append(vb)
        values["variance_ratio_count_over_sim"].append(
            vb / va if va > 0 else np.nan)
    intervals = {}
    for key, samples in values.items():
        samples = np.asarray(samples, dtype=float)
        samples = samples[np.isfinite(samples)]
        intervals[key] = (
            [float(x) for x in np.quantile(samples, [0.025, 0.975])]
            if len(samples) else None)
    return {"method": "paired event percentile bootstrap", "draws": draws,
            "seed": seed, "intervals_95": intervals}


def summarize(scan_root, bootstrap_draws=10000, bootstrap_seed=20261021):
    scan_root = Path(scan_root).resolve()
    scan = json.loads((scan_root / "scan.json").read_text())
    completed, incomplete = [], []
    for point_definition in scan["points"]:
        point = scan_root / point_definition["tag"]
        source_manifest = json.loads(Path(point_definition["source_manifest"]).read_text())
        for event in source_manifest["events"]:
            event_id = event["event_id"]
            paths = {
                sample: arm_paths(point, event_id, sample)
                for sample in ("SIM", "COUNT")
            }
            missing = [
                str(path) for sample in paths.values() for path in sample.values()
                if not path.is_file()
            ]
            if missing:
                incomplete.append({"event_id": event_id, "batch": point_definition["tag"],
                                   "missing": missing})
                continue
            row = {
                "event_id": event_id,
                "batch": point_definition["tag"],
                "sources": {
                    polarity: [source["cycle"] for source in event["sources"][polarity]]
                    for polarity in ("MUPLUS", "MUMINUS")
                },
            }
            for sample in ("SIM", "COUNT"):
                row[sample] = read_arm(paths[sample])
            completed.append(row)

    sim_tracks = [row["SIM"]["si_tracks"] for row in completed]
    count_tracks = [row["COUNT"]["si_tracks"] for row in completed]
    return {
        "kind": "norm42_unsplit_track_count_variance_report",
        "scan_root": str(scan_root),
        "planned_events": scan["planned_events"],
        "completed_pairs": len(completed),
        "incomplete_events": len(incomplete),
        "estimand": (
            "variation over repeated fixed-size draws from the empirical per-polarity "
            "Norm42 source inventories plus COUNT generation stochasticity"
        ),
        "track_definition": "SiTracks from one unsplit v3 CKF pass per arm",
        "variance_definition": "unbiased sample variance with ddof=1",
        "track_statistics": statistics(sim_tracks, count_tracks),
        "bootstrap": bootstrap_intervals(
            sim_tracks, count_tracks, bootstrap_draws, bootstrap_seed),
        "completed": completed,
        "incomplete": incomplete,
        "scan": scan,
    }


def write_csv(report, path):
    fields = [
        "event_id", "batch", "muplus_cycles", "muminus_cycles",
        "sim_entering", "count_entering", "sim_digitized", "count_digitized",
        "sim_survival", "count_survival", "sim_tracks", "count_tracks",
        "track_difference_count_minus_sim",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["completed"]:
            writer.writerow({
                "event_id": row["event_id"],
                "batch": row["batch"],
                "muplus_cycles": ";".join(map(str, row["sources"]["MUPLUS"])),
                "muminus_cycles": ";".join(map(str, row["sources"]["MUMINUS"])),
                "sim_entering": row["SIM"]["entering_digitization"],
                "count_entering": row["COUNT"]["entering_digitization"],
                "sim_digitized": row["SIM"]["digitized"],
                "count_digitized": row["COUNT"]["digitized"],
                "sim_survival": row["SIM"]["survival_fraction"],
                "count_survival": row["COUNT"]["survival_fraction"],
                "sim_tracks": row["SIM"]["si_tracks"],
                "count_tracks": row["COUNT"]["si_tracks"],
                "track_difference_count_minus_sim": (
                    row["COUNT"]["si_tracks"] - row["SIM"]["si_tracks"]),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events-csv")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261021)
    args = parser.parse_args()
    if args.bootstrap_draws < 0:
        parser.error("--bootstrap-draws cannot be negative")
    report = summarize(args.scan_root, args.bootstrap_draws, args.bootstrap_seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    csv_path = Path(args.events_csv) if args.events_csv else output.with_suffix(".csv")
    write_csv(report, csv_path)
    print(f"completed paired events: {report['completed_pairs']}/{report['planned_events']}")
    if report["completed_pairs"]:
        stats = report["track_statistics"]
        print(
            f"SIM tracks: mean={stats['SIM']['mean']:.6g}, "
            f"variance={stats['SIM']['variance_ddof1']}")
        print(
            f"COUNT tracks: mean={stats['COUNT']['mean']:.6g}, "
            f"variance={stats['COUNT']['variance_ddof1']}")
        print(
            "COUNT/SIM variance ratio="
            f"{stats['paired']['variance_ratio_count_over_sim']}")
    print(f"JSON -> {output}")
    print(f"events -> {csv_path}")


if __name__ == "__main__":
    main()
