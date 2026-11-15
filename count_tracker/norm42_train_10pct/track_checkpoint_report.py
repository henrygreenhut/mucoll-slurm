#!/usr/bin/env python3
"""Report digitization survival and five-region tracks for the three arms."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from count_tracker_checkpoint_report import COLLECTIONS, collection_size  # noqa: E402


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


def report(scan_root):
    scan_root = Path(scan_root).resolve()
    scan = json.loads((scan_root / "scan.json").read_text())
    point = scan_root / "n0167"
    sources = json.loads((point / "sources.json").read_text())
    event = sources["events"][0]
    arms = {arm: read_arm(point, event["event_id"], arm) for arm in ARMS}
    sim_tracks = arms["SIM"]["si_tracks"]
    return {
        "kind": "norm42_10pct_training_scale_track_checkpoint_report",
        "scan_root": str(scan_root),
        "event_id": event["event_id"],
        "files_per_polarity": sources["n_files_per_polarity"],
        "fraction_of_full_bx_reference": sources["overlay_normalization"]["fraction_of_full_bx_reference"],
        "source_disjoint_from_10pct_training": True,
        "arms": arms,
        "track_ratios_to_sim": {
            arm: arms[arm]["si_tracks"] / sim_tracks if sim_tracks else None
            for arm in ("COUNT_FULL", "COUNT_10PCT")
        },
        "scan": scan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.scan_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("arm          entering  digitized  survival  SiTracks  tracks/SIM")
    for arm in ARMS:
        row = result["arms"][arm]
        ratio = 1.0 if arm == "SIM" else result["track_ratios_to_sim"][arm]
        print(f"{arm:12s} {row['entering_digitization']:9d} {row['digitized']:10d} "
              f"{row['survival_fraction']:9.4f} {row['si_tracks']:9d} {ratio:11.4f}")


if __name__ == "__main__":
    main()
