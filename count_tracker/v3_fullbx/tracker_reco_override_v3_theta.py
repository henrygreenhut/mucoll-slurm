#!/usr/bin/env python3
"""Standard v3 tracker digitization and CKF for a theta-partitioned input.

Digitization is identical to the GenBIB-ML v3 tracker override. Reconstruction
is also identical except that ``--split-input`` makes the ordinary mergers read
the collections materialized by the Paper-1 theta splitter.
"""

import sys


TRACKER_COLLECTIONS = (
    # short name, digitized hits, reco-to-sim relations, simulated-hit output
    ("VXDBarrel", "VXDBarrelHits", "VXDBarrelHitsRelations", "VertexBarrelCollection"),
    ("VXDEndcap", "VXDEndcapHits", "VXDEndcapHitsRelations", "VertexEndcapCollection"),
    ("ITBarrel", "ITBarrelHits", "ITBarrelHitsRelations", "InnerTrackerBarrelCollection"),
    ("ITEndcap", "ITEndcapHits", "ITEndcapHitsRelations", "InnerTrackerEndcapCollection"),
    ("OTBarrel", "OTBarrelHits", "OTBarrelHitsRelations", "OuterTrackerBarrelCollection"),
    ("OTEndcap", "OTEndcapHits", "OTEndcapHitsRelations", "OuterTrackerEndcapCollection"),
)


def pop_option(name):
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    try:
        value = sys.argv[index + 1]
    except IndexError as error:
        raise RuntimeError(f"Missing value for {name}") from error
    del sys.argv[index:index + 2]
    return value


def pop_stage():
    stage = pop_option("--stage")
    if stage not in ("digi", "reco"):
        raise RuntimeError("Pass --stage digi or --stage reco")
    return stage


def pop_flag(name):
    if name not in sys.argv:
        return False
    sys.argv.remove(name)
    return True


def tracker_digi_algs(args):
    from TrackerDigi.tracking_vertex import VXDBarrel_cfg, VXDEndcap_cfg
    from TrackerDigi.tracking_inner import ITBarrel_cfg, ITEndcap_cfg
    from TrackerDigi.tracking_outer import OTBarrel_cfg, OTEndcap_cfg

    algorithms = [
        VXDBarrel_cfg(args),
        VXDEndcap_cfg(args),
        ITBarrel_cfg(args),
        ITEndcap_cfg(args),
        OTBarrel_cfg(args),
        OTEndcap_cfg(args),
    ]
    for algorithm in algorithms:
        algorithm.IsStrip = False
        algorithm.ForceHitsOntoSurface = True
    return algorithms


def make_digi_alg_list(args):
    from Common.event_counter import event_counter_cfg

    return [event_counter_cfg()] + tracker_digi_algs(args)


def split_mergers():
    from Configurables import CollectionMerger

    hit_merger = CollectionMerger(
        "MergeSplitHits",
        InputCollections=[f"{row[1]}Split" for row in TRACKER_COLLECTIONS],
        OutputCollection="MergedTrackerHits",
    )
    relation_merger = CollectionMerger(
        "MergeSplitHitRelations",
        InputCollections=[f"{row[2]}Split" for row in TRACKER_COLLECTIONS],
        OutputCollection="MergedTrackerHitsRelations",
    )
    return [hit_merger, relation_merger]


def make_reco_alg_list(args, split_input):
    from Common.event_counter import event_counter_cfg
    from Tracking.CKF_tracking import CKFTracker_cfg, deduper_cfg
    from Tracking.mergers import mergehits_cfg, mergehitsrelations_cfg

    algorithms = [event_counter_cfg()]
    if split_input:
        algorithms += split_mergers()
    else:
        algorithms += [mergehits_cfg(args), mergehitsrelations_cfg(args)]

    deduper = deduper_cfg()
    deduper.OutputTrackCollectionName = ["SiTracks"]
    algorithms += [CKFTracker_cfg(args), deduper]
    return algorithms


stage = pop_stage()
split_input = pop_flag("--split-input")
if stage == "digi" and split_input:
    raise RuntimeError("--split-input applies only to reconstruction")

if stage == "digi":
    from digi_args import get_digi_args
    from Common.steering import build_application

    args = get_digi_args()
    build_application(
        args,
        make_digi_alg_list(args),
        input_files=["bib_flow.edm4hep.root"],
        output_file="digi_output.edm4hep.root",
        histo_file="digi_histograms.root",
    )
else:
    from reco_args import get_reco_args
    from Common.steering import build_application

    args = get_reco_args()
    build_application(
        args,
        make_reco_alg_list(args, split_input),
        input_files=["digi_output.edm4hep.root"],
        output_file="reco_output.edm4hep.root",
        histo_file="reco_histograms.root",
    )
