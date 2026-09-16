#!/usr/bin/env python3
"""Report tracker-hit survival and SiTrack counts for two reconstructed arms."""

import argparse
import json
from pathlib import Path


COLLECTIONS = {
    "VBC": ("VertexBarrelCollection", "VXDBarrelHits"),
    "VEC": ("VertexEndcapCollection", "VXDEndcapHits"),
    "ITBC": ("InnerTrackerBarrelCollection", "ITBarrelHits"),
    "ITEC": ("InnerTrackerEndcapCollection", "ITEndcapHits"),
    "OTBC": ("OuterTrackerBarrelCollection", "OTBarrelHits"),
    "OTEC": ("OuterTrackerEndcapCollection", "OTEndcapHits"),
}


def collection_size(events, collection, field="cellID"):
    """Count collection elements across every event in an EDM4hep tree."""
    import awkward as ak

    values = events[collection][f"{collection}.{field}"].array(library="ak")
    return int(ak.sum(ak.num(values, axis=1)))


def arm_report(event_dir):
    """Read one ``SIM`` or ``COUNT`` event directory without modifying it."""
    import uproot

    event_dir = Path(event_dir).resolve()
    digi_path = event_dir / "reco" / "digi_output.edm4hep.root"
    reco_path = event_dir / "reco" / "reco_output.edm4hep.root"
    for path in (digi_path, reco_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    collections = {}
    with uproot.open(digi_path) as root:
        events = root["events"]
        for short, (sim_name, digi_name) in COLLECTIONS.items():
            before = collection_size(events, sim_name)
            after = collection_size(events, digi_name)
            collections[short] = {
                "entering_digitization": before,
                "digitized": after,
                "survival_fraction": after / before if before else None,
            }

    with uproot.open(reco_path) as root:
        tracks = collection_size(root["events"], "SiTracks_objIdx", "index")

    before = sum(row["entering_digitization"] for row in collections.values())
    after = sum(row["digitized"] for row in collections.values())
    return {
        "event_dir": str(event_dir),
        "collections": collections,
        "total": {
            "entering_digitization": before,
            "digitized": after,
            "survival_fraction": after / before if before else None,
            "si_tracks": tracks,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True, help="SIM arm directory")
    parser.add_argument("--count", required=True, help="COUNT arm directory")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    report = {"SIM": arm_report(args.sim), "COUNT": arm_report(args.count)}
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
