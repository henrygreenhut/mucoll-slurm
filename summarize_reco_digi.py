#!/usr/bin/env python3

import argparse
import csv

import awkward as ak
import numpy as np
import uproot


OVERLAY_TRACKER = (
    "OverlayVertexBarrelCollection/OverlayVertexBarrelCollection.eDep",
    "OverlayVertexEndcapCollection/OverlayVertexEndcapCollection.eDep",
    "OverlayInnerTrackerBarrelCollection/OverlayInnerTrackerBarrelCollection.eDep",
    "OverlayInnerTrackerEndcapCollection/OverlayInnerTrackerEndcapCollection.eDep",
    "OverlayOuterTrackerBarrelCollection/OverlayOuterTrackerBarrelCollection.eDep",
    "OverlayOuterTrackerEndcapCollection/OverlayOuterTrackerEndcapCollection.eDep",
)
OVERLAY_CALO = (
    "OverlayECalBarrelCollection/OverlayECalBarrelCollection.energy",
    "OverlayECalEndcapCollection/OverlayECalEndcapCollection.energy",
    "OverlayHCalBarrelCollection/OverlayHCalBarrelCollection.energy",
    "OverlayHCalEndcapCollection/OverlayHCalEndcapCollection.energy",
)
DIGI_TRACKER = (
    "VXDBarrelHits/VXDBarrelHits.eDep",
    "VXDEndcapHits/VXDEndcapHits.eDep",
    "ITBarrelHits/ITBarrelHits.eDep",
    "ITEndcapHits/ITEndcapHits.eDep",
    "OTBarrelHits/OTBarrelHits.eDep",
    "OTEndcapHits/OTEndcapHits.eDep",
)
DIGI_CALO = (
    "EcalBarrelCollectionRec/EcalBarrelCollectionRec.energy",
    "EcalEndcapCollectionRec/EcalEndcapCollectionRec.energy",
    "HcalBarrelCollectionRec/HcalBarrelCollectionRec.energy",
    "HcalEndcapCollectionRec/HcalEndcapCollectionRec.energy",
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    return parser.parse_args()


def summarize(tree, branches):
    counts = np.zeros(tree.num_entries, dtype=np.int64)
    energy = np.zeros(tree.num_entries, dtype=np.float64)
    for branch in branches:
        if branch not in tree:
            raise KeyError("missing branch {}".format(branch))
        values = tree[branch].array(library="ak")
        counts += ak.to_numpy(ak.num(values, axis=1))
        energy += ak.to_numpy(ak.sum(values, axis=1))
    return counts, energy


def main():
    args = arguments()
    with uproot.open(args.input) as root:
        tree = root["events"]
        overlay_tracker = summarize(tree, OVERLAY_TRACKER)
        overlay_calo = summarize(tree, OVERLAY_CALO)
        digi_tracker = summarize(tree, DIGI_TRACKER)
        digi_calo = summarize(tree, DIGI_CALO)

    names = (
        "overlay_tracker_hits", "overlay_tracker_energy_GeV",
        "overlay_calo_hits", "overlay_calo_energy_GeV",
        "digi_tracker_hits", "digi_tracker_energy_GeV",
        "digi_calo_hits", "digi_calo_energy_GeV",
    )
    columns = (
        overlay_tracker[0], overlay_tracker[1],
        overlay_calo[0], overlay_calo[1],
        digi_tracker[0], digi_tracker[1],
        digi_calo[0], digi_calo[1],
    )
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("event",) + names)
        writer.writerows(
            (event,) + tuple(column[event] for column in columns)
            for event in range(len(columns[0]))
        )

    print("DIGI summary events=", len(columns[0]), "->", args.output)


if __name__ == "__main__":
    main()
