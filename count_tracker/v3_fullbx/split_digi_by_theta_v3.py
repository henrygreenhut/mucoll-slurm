#!/usr/bin/env python3
"""Copy one theta region from a v3 digitized tracker event with v3 podio.

The previous bridge converted the full event through the v2.11 EDM4hep/LCIO
layer.  At full Norm42 occupancy that converter corrupts its heap.  This
implementation stays in the v3 data model, copies digitized hits selected by
their stored position, and rebuilds only their matching hit-to-SIM links and
referenced SimTrackerHits.  No hit values or reconstruction settings change.
"""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile


CELL_ID_ENCODING = (
    "system:5,side:-2,layer:6,module:11,sensor:8"
)
COLLECTIONS = (
    ("VXDBarrelHits", "VXDBarrelHitsRelations", "VertexBarrelCollection"),
    ("VXDEndcapHits", "VXDEndcapHitsRelations", "VertexEndcapCollection"),
    ("ITBarrelHits", "ITBarrelHitsRelations", "InnerTrackerBarrelCollection"),
    ("ITEndcapHits", "ITEndcapHitsRelations", "InnerTrackerEndcapCollection"),
    ("OTBarrelHits", "OTBarrelHitsRelations", "OuterTrackerBarrelCollection"),
    ("OTEndcapHits", "OTEndcapHitsRelations", "OuterTrackerEndcapCollection"),
)


def object_index(obj):
    identifier = obj.getObjectID()
    if identifier.index < 0:
        raise ValueError("encountered an unavailable EDM4hep object")
    return int(identifier.index)


def theta_degrees(hit):
    position = hit.getPosition()
    return math.degrees(math.atan2(math.hypot(position.x, position.y), position.z))


def in_region(hit, lower, upper):
    theta = theta_degrees(hit)
    return lower <= theta <= upper if upper == 180.0 else lower <= theta < upper


def copy_collection(frame, output, hit_name, relation_name, sim_name, lower, upper):
    import edm4hep

    hits = frame.get(hit_name)
    relations = frame.get(relation_name)
    simulated_hits = frame.get(sim_name)

    selected_hits = edm4hep.TrackerHitPlaneCollection()
    hit_by_source_index = {}
    for hit in hits:
        if in_region(hit, lower, upper):
            copied = hit.clone()
            selected_hits.push_back(copied)
            hit_by_source_index[object_index(hit)] = copied

    selected_sim = edm4hep.SimTrackerHitCollection()
    sim_by_source_index = {}
    selected_relations = type(relations)()
    for relation in relations:
        source_index = object_index(relation.getFrom())
        copied_hit = hit_by_source_index.get(source_index)
        if copied_hit is None:
            continue
        source_sim = relation.getTo()
        sim_index = object_index(source_sim)
        copied_sim = sim_by_source_index.get(sim_index)
        if copied_sim is None:
            copied_sim = source_sim.clone(False)
            selected_sim.push_back(copied_sim)
            sim_by_source_index[sim_index] = copied_sim
        copied_relation = selected_relations.create()
        copied_relation.setFrom(copied_hit)
        copied_relation.setTo(copied_sim)
        copied_relation.setWeight(relation.getWeight())

    output.put(selected_hits, f"{hit_name}Split")
    output.put(selected_relations, f"{relation_name}Split")
    output.put(selected_sim, f"{sim_name}Split")
    return {
        "input_hits": len(hits),
        "selected_hits": len(selected_hits),
        "selected_relations": len(selected_relations),
        "selected_sim_hits": len(selected_sim),
    }


def split_event(input_path, output_path, lower, upper):
    import edm4hep
    import podio
    from podio.root_io import Reader, Writer

    reader = Reader(str(input_path))
    frames = reader.get("events")
    if len(frames) != 1:
        raise ValueError(f"expected one event, found {len(frames)}")
    source = frames[0]
    available = set(source.getAvailableCollections())
    required = {name for row in COLLECTIONS for name in row} | {"EventHeader"}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"input is missing collections: {', '.join(missing)}")

    headers = source.get("EventHeader")
    if len(headers) != 1:
        raise ValueError("expected exactly one EventHeader")
    output = podio.Frame()
    copied_headers = edm4hep.EventHeaderCollection()
    copied_headers.push_back(headers[0].clone(False))
    output.put(copied_headers, "EventHeader")

    counts = {}
    for hit_name, relation_name, sim_name in COLLECTIONS:
        counts[hit_name] = copy_collection(
            source, output, hit_name, relation_name, sim_name, lower, upper
        )

    metadata = podio.Frame()
    for hit_name, _, sim_name in COLLECTIONS:
        metadata.put_parameter(f"{hit_name}Split__CellIDEncoding", CELL_ID_ENCODING)
        metadata.put_parameter(f"{sim_name}Split__CellIDEncoding", CELL_ID_ENCODING)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to replace {output_path}")
    with tempfile.TemporaryDirectory(prefix=".theta_split_", dir=output_path.parent) as work:
        temporary = Path(work) / output_path.name
        writer = Writer(str(temporary))
        writer.write_frame(metadata, "metadata")
        writer.write_frame(output, "events")
        writer._writer.finish()
        del writer
        os.replace(temporary, output_path)

    print(json.dumps({
        "status": "written",
        "input": str(input_path),
        "output": str(output_path),
        "theta_degrees": [lower, upper],
        "collections": counts,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theta-min", type=float, required=True)
    parser.add_argument("--theta-max", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.theta_min < args.theta_max <= 180.0:
        parser.error("require 0 <= theta-min < theta-max <= 180")
    split_event(
        args.input.resolve(), args.output.resolve(), args.theta_min, args.theta_max
    )


if __name__ == "__main__":
    main()
