#!/usr/bin/env python3
"""Tracker-only digitization and CKF reconstruction for the v2.11 image.

This is the v2.11 counterpart of GenBIB-ML/reco/tracker_reco_override.py.
The public collection names and the digitizer settings are intentionally kept
the same.  The implementation uses MarlinProcessorWrapper because the v2.11
image predates the native Gaudi tracker algorithms used by the v3 workflow.
"""

import os
import sys

from Gaudi.Configuration import INFO, WARNING
from Configurables import (
    EDM4hep2LcioTool,
    EventDataSvc,
    Lcio2EDM4hepTool,
    MarlinProcessorWrapper,
)
from k4FWCore import ApplicationMgr, IOSvc


TRACKER_COLLECTIONS = (
    # name, simulated hits, digitized hits, relations, subdetector,
    # spatial resolutions (u, v), timing resolution, timing window
    (
        "VXDBarrel",
        "VertexBarrelCollection",
        "VXDBarrelHits",
        "VXDBarrelHitsRelations",
        "Vertex",
        0.005,
        0.005,
        0.03,
        -0.09,
        0.15,
    ),
    (
        "VXDEndcap",
        "VertexEndcapCollection",
        "VXDEndcapHits",
        "VXDEndcapHitsRelations",
        "Vertex",
        0.005,
        0.005,
        0.03,
        -0.09,
        0.15,
    ),
    (
        "ITBarrel",
        "InnerTrackerBarrelCollection",
        "ITBarrelHits",
        "ITBarrelHitsRelations",
        "InnerTrackers",
        0.007,
        0.09,
        0.06,
        -0.18,
        0.3,
    ),
    (
        "ITEndcap",
        "InnerTrackerEndcapCollection",
        "ITEndcapHits",
        "ITEndcapHitsRelations",
        "InnerTrackers",
        0.007,
        0.09,
        0.06,
        -0.18,
        0.3,
    ),
    (
        "OTBarrel",
        "OuterTrackerBarrelCollection",
        "OTBarrelHits",
        "OTBarrelHitsRelations",
        "OuterTrackers",
        0.007,
        0.09,
        0.06,
        -0.18,
        0.3,
    ),
    (
        "OTEndcap",
        "OuterTrackerEndcapCollection",
        "OTEndcapHits",
        "OTEndcapHitsRelations",
        "OuterTrackers",
        0.007,
        0.09,
        0.06,
        -0.18,
        0.3,
    ),
)

# Historical MAIA_v0 seeding layers from mucoll-benchmarks commit 1b702ec.
# The v2.11 ACTS Marlin processor expects one "volume layer" pair per item.
MAIA_SEEDING_LAYERS = (
    "13 2",
    "13 6",
    "13 10",
    "13 14",
    "14 2",
    "14 6",
    "14 10",
    "14 14",
    "15 2",
    "15 6",
    "15 10",
    "15 14",
    "8 2",
    "17 2",
    "18 2",
)


def pop_stage():
    stage = os.environ.get("V2_STAGE")
    if stage is not None:
        if stage not in ("digi", "reco"):
            raise RuntimeError("V2_STAGE must be digi or reco")
        return stage
    if "--stage" not in sys.argv:
        raise RuntimeError("Set V2_STAGE=digi|reco or pass --stage digi|reco")
    index = sys.argv.index("--stage")
    try:
        stage = sys.argv[index + 1]
    except IndexError as exc:
        raise RuntimeError("Missing value for --stage") from exc
    del sys.argv[index:index + 2]
    if stage not in ("digi", "reco"):
        raise RuntimeError("Stage must be digi or reco")
    return stage


def marlin_processor(name, processor_type, parameters):
    processor = MarlinProcessorWrapper(name)
    processor.OutputLevel = WARNING
    processor.ProcessorType = processor_type
    processor.Parameters = parameters
    return processor


def attach_edm4hep_input(processor):
    converter = EDM4hep2LcioTool(f"{processor.name()}EDM4hepInput")
    converter.convertAll = True
    converter.OutputLevel = WARNING
    processor.EDM4hep2LcioTool = converter


def attach_edm4hep_output(processor, collection_names):
    converter = Lcio2EDM4hepTool(f"{processor.name()}EDM4hepOutput")
    converter.convertAll = False
    converter.collNameMapping = {name: name for name in collection_names}
    converter.OutputLevel = WARNING
    processor.Lcio2EDM4hepTool = converter


def input_converter():
    processor = marlin_processor(
        "InputConverter",
        "Statusmonitor",
        {"HowOften": ["1"]},
    )
    attach_edm4hep_input(processor)
    return processor


def dd4hep_initializer():
    return marlin_processor(
        "InitializeDD4hep",
        "InitializeDD4hep",
        {
            "DD4hepXMLFile": [os.environ["MUCOLL_GEO"]],
            "EncodingStringParameterName": ["GlobalTrackerReadoutID"],
        },
    )


def tracker_digi_algs():
    algorithms = []
    for (
        short,
        simulated_hits,
        tracker_hits,
        relations,
        subdetector,
        resolution_u,
        resolution_v,
        resolution_t,
        time_min,
        time_max,
    ) in TRACKER_COLLECTIONS:
        processor = marlin_processor(
            f"{short}Digitiser",
            "DDPlanarDigiProcessor",
            {
                "CorrectTimesForPropagation": ["true"],
                "ForceHitsOntoSurface": ["true"],
                "IsStrip": ["false"],
                "ResolutionT": [str(resolution_t)],
                "ResolutionU": [str(resolution_u)],
                "ResolutionV": [str(resolution_v)],
                "SimTrackHitCollectionName": [simulated_hits],
                "SimTrkHitRelCollection": [relations],
                "SubDetectorName": [subdetector],
                "TimeWindowMax": [str(time_max)],
                "TimeWindowMin": [str(time_min)],
                "TrackerHitCollectionName": [tracker_hits],
                "UseTimeWindow": ["true"],
            },
        )
        attach_edm4hep_output(processor, (tracker_hits, relations))
        algorithms.append(processor)
    return algorithms


def tracker_reco_algs():
    tracker_hits = [row[2] for row in TRACKER_COLLECTIONS]
    ckf = marlin_processor(
        "CKFTracking",
        "ACTSSeededCKFTrackingProc",
        {
            "CKF_Chi2CutOff": ["10"],
            "CKF_NumMeasurementsCutOff": ["1"],
            "DetectorSchema": ["MAIA_v0"],
            "MatFile": [os.environ["MUCOLL_MATMAP"]],
            "RunCKF": ["true"],
            "SeedCollectionName": ["SeedTracks"],
            "SeedFinding_CollisionRegion": ["6"],
            "SeedFinding_ImpactMax": ["3"],
            "SeedFinding_MinPt": ["500"],
            "SeedFinding_RMax": ["150"],
            "SeedFinding_RadLengthPerSeed": ["0.1"],
            "SeedFinding_SigmaScattering": ["50"],
            "SeedingLayers": list(MAIA_SEEDING_LAYERS),
            "TGeoDescFile": [os.environ["MUCOLL_TGEO_DESC"]],
            "TGeoFile": [os.environ["MUCOLL_TGEO"]],
            "TrackCollectionName": ["AllTracks"],
            "TrackerHitCollectionNames": tracker_hits,
        },
    )
    attach_edm4hep_output(ckf, ("SeedTracks", "AllTracks"))

    deduper = marlin_processor(
        "TrackDeduplication",
        "ACTSDuplicateRemoval",
        {
            "InputTrackCollectionName": ["AllTracks"],
            "OutputTrackCollectionName": ["SiTracks"],
        },
    )
    attach_edm4hep_output(deduper, ("SiTracks",))
    return [ckf, deduper]


def main():
    stage = pop_stage()
    required = (
        "MUCOLL_GEO",
        "V2_INPUT_FILE",
        "V2_OUTPUT_FILE",
    )
    if stage == "reco":
        required += ("MUCOLL_MATMAP", "MUCOLL_TGEO", "MUCOLL_TGEO_DESC")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment: {', '.join(missing)}")

    services = [EventDataSvc("EventDataSvc")]
    io_service = IOSvc()
    io_service.Input = os.environ["V2_INPUT_FILE"]
    io_service.Output = os.environ["V2_OUTPUT_FILE"]
    io_service.outputCommands = ["keep *"]

    algorithms = [input_converter(), dd4hep_initializer()]
    algorithms += tracker_digi_algs() if stage == "digi" else tracker_reco_algs()

    ApplicationMgr(
        TopAlg=algorithms,
        EvtSel="NONE",
        EvtMax=int(os.environ.get("V2_NUM_EVENTS", "1")),
        ExtSvc=services,
        OutputLevel=INFO,
    )


# A k4run steering file is loaded as a configuration module, so it must build
# the ApplicationMgr at module scope rather than behind a __main__ guard.
main()
