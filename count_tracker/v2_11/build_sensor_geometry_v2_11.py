#!/usr/bin/env python3
"""Build the MAIA v2.11 sensor map consumed by assign_actual_cellid.py.

This is the geometry-building subset of GenBIB-ML's assignment script.  It is
kept separate because the v2.11 image has DD4hep/ROOT but does not have SciPy;
the subsequent hit assignment is therefore run in the existing count-hdf
environment using this saved map.
"""

import argparse
import os
from array import array
from pathlib import Path

import numpy as np


def decode_system(cell_id):
    return int(cell_id) & 0x1F


def context_center_cm(context):
    point = context.localToWorld(array("d", [0.0, 0.0, 0.0]))
    return float(point.X()), float(point.Y()), float(point.Z())


def local_to_master(matrix, xyz_cm):
    source = array("d", xyz_cm)
    destination = array("d", [0.0, 0.0, 0.0])
    matrix.LocalToMaster(source, destination)
    return np.array(destination, dtype=np.float64)


def shape_half_lengths_cm(shape):
    if shape is None:
        return np.nan, np.nan, np.nan, False
    if hasattr(shape, "ComputeBBox"):
        shape.ComputeBBox()
    values = []
    complete = True
    for method in ("GetDX", "GetDY", "GetDZ"):
        if hasattr(shape, method):
            values.append(float(getattr(shape, method)()))
        else:
            values.append(np.nan)
            complete = False
    return *values, complete


def trd2_params_mm(shape):
    if shape is None or not hasattr(shape, "GetDx1"):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    return (
        float(shape.GetDx1()) * 10.0,
        float(shape.GetDx2()) * 10.0,
        float(shape.GetDy1()) * 10.0,
        float(shape.GetDy2()) * 10.0,
        float(shape.GetDz()) * 10.0,
    )


def build_sensor_geometry(output_path):
    import dd4hep
    import ROOT

    geometry_path = os.environ.get("MUCOLL_GEO")
    if not geometry_path:
        raise RuntimeError("MUCOLL_GEO is not set")

    detector = dd4hep.Detector.getInstance()
    detector.fromXML(geometry_path)
    geometry = ROOT.gGeoManager
    navigator = geometry.GetCurrentNavigator()
    volume_manager = detector.volumeManager()

    paths_by_cell_id = {}
    for system in range(1, 7):
        subdetector = volume_manager.subdetector(system)
        for pair in subdetector.ptr().volumes:
            cell_id = int(pair.first)
            if decode_system(cell_id) != system or cell_id in paths_by_cell_id:
                continue
            x, y, z = context_center_cm(pair.second)
            if not navigator.FindNode(x, y, z):
                continue
            sensor_path = geometry.GetPath()
            if sensor_path != "/world_volume_1" and sensor_path.count("/") >= 3:
                paths_by_cell_id[cell_id] = sensor_path

    cell_ids = []
    systems = []
    centers = []
    axes = []
    half_lengths = []
    shape_names = []
    trapezoids = []
    incomplete_shapes = 0

    for cell_id, sensor_path in sorted(paths_by_cell_id.items()):
        if not geometry.cd(sensor_path):
            continue
        matrix = geometry.GetCurrentMatrix()
        node = geometry.GetCurrentNode()
        volume = node.GetVolume() if node else None
        shape = volume.GetShape() if volume else None

        center_cm = local_to_master(matrix, (0.0, 0.0, 0.0))
        sensor_axes = np.vstack(
            [
                local_to_master(matrix, (1.0, 0.0, 0.0)) - center_cm,
                local_to_master(matrix, (0.0, 1.0, 0.0)) - center_cm,
                local_to_master(matrix, (0.0, 0.0, 1.0)) - center_cm,
            ]
        )
        norms = np.linalg.norm(sensor_axes, axis=1)
        sensor_axes /= np.where(norms[:, None] == 0.0, 1.0, norms[:, None])

        dx, dy, dz, complete = shape_half_lengths_cm(shape)
        if not complete:
            incomplete_shapes += 1
        cell_ids.append(cell_id)
        systems.append(decode_system(cell_id))
        centers.append(center_cm * 10.0)
        axes.append(sensor_axes)
        half_lengths.append(np.array([dx, dy, dz]) * 10.0)
        shape_names.append(shape.ClassName() if shape else "")
        trapezoids.append(trd2_params_mm(shape))

    if not cell_ids:
        raise RuntimeError("No sensitive sensor geometry was found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        cellids=np.asarray(cell_ids, dtype=np.int64),
        systems=np.asarray(systems, dtype=np.int16),
        centers_mm=np.vstack(centers).astype(np.float64),
        axes=np.stack(axes).astype(np.float64),
        half_lengths_mm=np.vstack(half_lengths).astype(np.float64),
        shape_names=np.asarray(shape_names, dtype=str),
        trd2_params_mm=np.asarray(trapezoids, dtype=np.float64),
    )
    print(f"built v2.11 sensor geometry: {output_path} ({len(cell_ids)} sensors)")
    if incomplete_shapes:
        print(f"warning: {incomplete_shapes} sensors have incomplete shape half lengths")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_sensor_geometry(args.output.resolve())


if __name__ == "__main__":
    main()
