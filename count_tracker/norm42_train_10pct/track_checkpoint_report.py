#!/usr/bin/env python3
"""Report digitization survival and five-region tracks for the three arms."""

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from count_tracker_checkpoint_report import COLLECTIONS, collection_size  # noqa: E402
from prepare_track_checkpoint import overlap_report  # noqa: E402


ARMS = ("SIM", "COUNT_FULL", "COUNT_10PCT")
REGIONS = (
    "theta_000_030", "theta_030_070", "theta_070_110",
    "theta_110_150", "theta_150_180",
)


def read_arm(point, event_id, arm):
    import uproot

    digi = point / "digi" / event_id / arm / "digi" / "digi_output.edm4hep.root"
    with uproot.open(digi) as root:
        events = root["events"]
        entering = sum(
            collection_size(events, sim_name) for sim_name, _ in COLLECTIONS.values()
        )
        digitized = sum(
            collection_size(events, digi_name) for _, digi_name in COLLECTIONS.values()
        )
    regional_tracks = {}
    for region in REGIONS:
        reco = point / "events" / event_id / arm / "theta_reco" / region / "reco_output.edm4hep.root"
        with uproot.open(reco) as root:
            regional_tracks[region] = collection_size(
                root["events"], "SiTracks_objIdx", "index"
            )
    return {
        "entering_digitization": int(entering),
        "digitized": int(digitized),
        "survival_fraction": digitized / entering if entering else None,
        "si_tracks": int(sum(regional_tracks.values())),
        "regional_si_tracks": regional_tracks,
    }


def report(scan_roots):
    scan_roots = [Path(path).resolve() for path in scan_roots]
    scans = [json.loads((root / "scan.json").read_text()) for root in scan_roots]
    source_documents = [
        json.loads((root / "n0167" / "sources.json").read_text())
        for root in scan_roots
    ]
    files_per_polarity = {
        document["n_files_per_polarity"] for document in source_documents
    }
    bx_fractions = {
        document["overlay_normalization"]["fraction_of_full_bx_reference"]
        for document in source_documents
    }
    if len(files_per_polarity) != 1 or len(bx_fractions) != 1:
        raise ValueError("all report inputs must use the same BIB normalization")

    events = []
    source_events = []
    for scan_root, sources in zip(scan_roots, source_documents):
        point = scan_root / "n0167"
        for event in sources["events"]:
            source_events.append(event)
            arms = {
                arm: read_arm(point, event["event_id"], arm) for arm in ARMS
            }
            sim_tracks = arms["SIM"]["si_tracks"]
            events.append({
                "event_id": event["event_id"],
                "scan_root": str(scan_root),
                "arms": arms,
                "track_ratios_to_sim": {
                    arm: arms[arm]["si_tracks"] / sim_tracks if sim_tracks else None
                    for arm in ("COUNT_FULL", "COUNT_10PCT")
                },
            })

    aggregate = {}
    for arm in ARMS:
        entering = sum(event["arms"][arm]["entering_digitization"] for event in events)
        digitized = sum(event["arms"][arm]["digitized"] for event in events)
        regional = {
            region: sum(
                event["arms"][arm]["regional_si_tracks"][region]
                for event in events
            )
            for region in REGIONS
        }
        aggregate[arm] = {
            "entering_digitization": entering,
            "digitized": digitized,
            "survival_fraction": digitized / entering if entering else None,
            "si_tracks": sum(regional.values()),
            "regional_si_tracks": regional,
            "mean_si_tracks_per_event": (
                statistics.mean(event["arms"][arm]["si_tracks"] for event in events)
            ),
        }

    def summarize(values):
        return {
            "mean": statistics.mean(values),
            "sample_standard_deviation": (
                statistics.stdev(values) if len(values) > 1 else None
            ),
            "minimum": min(values),
            "maximum": max(values),
            "values": values,
        }

    ratio_summary = {}
    for arm in ("COUNT_FULL", "COUNT_10PCT"):
        ratio_summary[arm] = {
            "overall": summarize([
                event["track_ratios_to_sim"][arm] for event in events
            ]),
            "by_region": {
                region: summarize([
                    event["arms"][arm]["regional_si_tracks"][region]
                    / event["arms"]["SIM"]["regional_si_tracks"][region]
                    for event in events
                    if event["arms"]["SIM"]["regional_si_tracks"][region]
                ])
                for region in REGIONS
            },
            "ratio_of_aggregate_tracks": (
                aggregate[arm]["si_tracks"] / aggregate["SIM"]["si_tracks"]
                if aggregate["SIM"]["si_tracks"] else None
            ),
        }

    return {
        "kind": "norm42_10pct_training_scale_track_cohort_report",
        "scan_roots": [str(root) for root in scan_roots],
        "n_events": len(events),
        "files_per_polarity": files_per_polarity.pop(),
        "fraction_of_full_bx_reference": bx_fractions.pop(),
        "source_disjoint_from_10pct_training": True,
        "events": events,
        "aggregate_arms": aggregate,
        "track_ratio_summary": ratio_summary,
        "combined_source_overlap": overlap_report(source_events),
        "source_overlap_by_scan": {
            str(root): (
                json.loads((root / "source_overlap.json").read_text())
                if (root / "source_overlap.json").is_file() else None
            )
            for root in scan_roots
        },
        "scans": scans,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-root", action="append", required=True,
        help="May be repeated to combine the checkpoint and follow-up cohort",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.scan_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"events: {result['n_events']}")
    print("arm          entering  digitized  survival  SiTracks  tracks/SIM")
    for arm in ARMS:
        row = result["aggregate_arms"][arm]
        ratio = (1.0 if arm == "SIM" else
                 result["track_ratio_summary"][arm]["ratio_of_aggregate_tracks"])
        print(f"{arm:12s} {row['entering_digitization']:9d} {row['digitized']:10d} "
              f"{row['survival_fraction']:9.4f} {row['si_tracks']:9d} {ratio:11.4f}")


if __name__ == "__main__":
    main()
