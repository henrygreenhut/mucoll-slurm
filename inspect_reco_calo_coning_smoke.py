#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import uproot


CALORIMETERS = (
    "EcalBarrel",
    "EcalEndcap",
    "HcalBarrel",
    "HcalEndcap",
)
CONDITIONS = (("U", "on"), ("U", "off"), ("R", "on"), ("R", "off"))


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory")
    return parser.parse_args()


def first_event_length(tree, branch):
    values = tree[branch].array(entry_start=0, entry_stop=1)
    if len(values) != 1:
        raise RuntimeError("{} did not contain one readable event".format(branch))
    return len(values[0])


def file_tree(path):
    root = uproot.open(path)
    tree = root["events"]
    if tree.num_entries != 1:
        raise RuntimeError("{} contains {} events, expected 1".format(path, tree.num_entries))
    return tree


def condition_counts(directory):
    digi_path = directory / "digi_output.edm4hep.root"
    reco_path = directory / "reco_output.edm4hep.root"
    digi = file_tree(digi_path)
    reco = file_tree(reco_path)
    counts = {
        "digi_mb": digi_path.stat().st_size / 1e6,
        "reco_mb": reco_path.stat().st_size / 1e6,
    }
    for prefix in CALORIMETERS:
        counts[prefix + "Rec"] = first_event_length(
            digi, "{0}CollectionRec/{0}CollectionRec.energy".format(prefix)
        )
        counts[prefix + "Sel"] = first_event_length(
            digi, "{0}CollectionSel_objIdx/{0}CollectionSel_objIdx.index".format(prefix)
        )
    counts["clusters"] = first_event_length(
        reco, "PandoraClusters/PandoraClusters.energy"
    )
    counts["tracks"] = first_event_length(
        reco, "SiTracks_objIdx/SiTracks_objIdx.index"
    )
    counts["pfos"] = first_event_length(reco, "PandoraPFOs/PandoraPFOs.PDG")
    charge = reco["PandoraPFOs/PandoraPFOs.charge"].array(
        entry_start=0, entry_stop=1
    )[0]
    counts["charged_pfos"] = int(np.count_nonzero(np.asarray(charge)))
    track_begin = np.asarray(
        reco["PandoraPFOs/PandoraPFOs.tracks_begin"].array(
            entry_start=0, entry_stop=1
        )[0]
    )
    track_end = np.asarray(
        reco["PandoraPFOs/PandoraPFOs.tracks_end"].array(
            entry_start=0, entry_stop=1
        )[0]
    )
    counts["pfo_track_links"] = int(np.sum(track_end - track_begin))
    return counts


def verify(results):
    failures = []
    for sample in ("U", "R"):
        coned = results[sample + "/on"]
        unconed = results[sample + "/off"]
        for prefix in CALORIMETERS:
            raw = prefix + "Rec"
            selected = prefix + "Sel"
            if coned[raw] != unconed[raw]:
                failures.append(
                    "{} {} differs between coning on/off: {} vs {}".format(
                        sample, raw, coned[raw], unconed[raw]
                    )
                )
            if unconed[selected] < coned[selected]:
                failures.append(
                    "{} {} decreased without coning: {} vs {}".format(
                        sample, selected, coned[selected], unconed[selected]
                    )
                )
        if coned["tracks"] != unconed["tracks"]:
            failures.append(
                "{} track count differs between coning on/off: {} vs {}".format(
                    sample, coned["tracks"], unconed["tracks"]
                )
            )
    if failures:
        raise RuntimeError("\n".join(failures))


def main():
    args = arguments()
    run_directory = Path(args.run_directory).resolve()
    results = {}
    for sample, coning in CONDITIONS:
        key = sample + "/" + coning
        results[key] = condition_counts(run_directory / sample / coning)

    fields = [prefix + suffix for prefix in CALORIMETERS for suffix in ("Rec", "Sel")]
    fields += ["clusters", "tracks", "pfos", "charged_pfos", "pfo_track_links"]
    print("condition " + " ".join(fields))
    for sample, coning in CONDITIONS:
        key = sample + "/" + coning
        print(key + " " + " ".join(str(results[key][field]) for field in fields))

    verify(results)
    output = run_directory / "summary.json"
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print("paired raw-hit and track checks passed")
    print("summary -> {}".format(output))


if __name__ == "__main__":
    main()
