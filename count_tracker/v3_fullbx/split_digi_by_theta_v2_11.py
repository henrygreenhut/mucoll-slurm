#!/usr/bin/env python3
"""Materialize one Paper-1 theta region from a digitized tracker event.

This uses the established v2.11 Marlin splitter solely as an EDM4hep
transformation between v3 digitization and v3 reconstruction.
"""

import os

from Gaudi.Configuration import INFO, WARNING
from Configurables import (
    EDM4hep2LcioTool,
    EventDataSvc,
    Lcio2EDM4hepTool,
    MarlinProcessorWrapper,
)
from k4FWCore import ApplicationMgr, IOSvc


COLLECTIONS = (
    ("VXDBarrel", "VXDBarrelHits", "VXDBarrelHitsRelations", "VertexBarrelCollection"),
    ("VXDEndcap", "VXDEndcapHits", "VXDEndcapHitsRelations", "VertexEndcapCollection"),
    ("ITBarrel", "ITBarrelHits", "ITBarrelHitsRelations", "InnerTrackerBarrelCollection"),
    ("ITEndcap", "ITEndcapHits", "ITEndcapHitsRelations", "InnerTrackerEndcapCollection"),
    ("OTBarrel", "OTBarrelHits", "OTBarrelHitsRelations", "OuterTrackerBarrelCollection"),
    ("OTEndcap", "OTEndcapHits", "OTEndcapHitsRelations", "OuterTrackerEndcapCollection"),
)


def marlin_processor(name, processor_type, parameters):
    processor = MarlinProcessorWrapper(name)
    processor.OutputLevel = WARNING
    processor.ProcessorType = processor_type
    processor.Parameters = parameters
    return processor


def input_converter():
    processor = marlin_processor(
        "AIDAInputConverter",
        "AIDAProcessor",
        {"Compress": ["1"], "FileName": ["theta_split_histograms"], "FileType": ["root"]},
    )
    converter = EDM4hep2LcioTool("ThetaSplitEDM4hepInput")
    converter.convertAll = True
    converter.OutputLevel = WARNING
    processor.EDM4hep2LcioTool = converter
    return processor


def splitters(lower, upper):
    algorithms = []
    for short, hits, relations, simulated_hits in COLLECTIONS:
        output_names = (f"{hits}Split", f"{relations}Split", f"{simulated_hits}Split")
        processor = marlin_processor(
            f"{short}ThetaSplitter",
            "SplitCollectionByPolarAngle",
            {
                "FillHistograms": ["false"],
                "PolarAngleLowerLimit": [str(lower)],
                "PolarAngleUpperLimit": [str(upper)],
                "TrackerHitInputCollections": [hits],
                "TrackerHitInputRelations": [relations],
                "TrackerHitOutputCollections": [output_names[0]],
                "TrackerHitOutputRelations": [output_names[1]],
                "TrackerSimHitInputCollections": [simulated_hits],
                "TrackerSimHitOutputCollections": [output_names[2]],
            },
        )
        converter = Lcio2EDM4hepTool(f"{short}ThetaSplitEDM4hepOutput")
        converter.convertAll = False
        converter.collNameMapping = {name: name for name in output_names}
        converter.OutputLevel = WARNING
        processor.Lcio2EDM4hepTool = converter
        algorithms.append(processor)
    return algorithms


lower = float(os.environ["THETA_MIN"])
upper = float(os.environ["THETA_MAX"])
if not 0.0 <= lower < upper <= 180.0:
    raise RuntimeError("Require 0 <= THETA_MIN < THETA_MAX <= 180")

io_service = IOSvc()
io_service.Input = os.environ["SPLIT_INPUT_FILE"]
io_service.Output = os.environ["SPLIT_OUTPUT_FILE"]
io_service.outputCommands = ["keep *"]

ApplicationMgr(
    TopAlg=[input_converter()] + splitters(lower, upper),
    EvtSel="NONE",
    EvtMax=1,
    ExtSvc=[EventDataSvc("EventDataSvc")],
    OutputLevel=INFO,
)
