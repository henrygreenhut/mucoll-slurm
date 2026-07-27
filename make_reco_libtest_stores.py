#!/usr/bin/env python3
"""Convert N=420 RECO events into PFO, selected-track, and cluster stores."""

import argparse
import glob
import os
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot


SAMPLES = ("U", "R", "null_b")
SPLITS = ("train", "val", "test")
N_FILES = 420
CURVATURE_TO_PT = 0.00015
PFO_FEATURES = (
    "pt", "eta", "phi", "energy", "mass", "charge", "pdg", "px", "py", "pz",
)
TRACK_FEATURES = (
    "pt", "eta", "phi", "d0", "z0", "chi2_ndf", "n_holes", "omega",
    "tan_lambda",
)
CLUSTER_FEATURES = ("energy", "eta", "phi", "r", "z", "n_hits")


def parse_args():
    scratch = os.environ.get("PSCRATCH", "")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reco-dir",
        default=(scratch + "/mucoll/libtest/reco_n420_pfn_simple")
        if scratch else None,
        required=not bool(scratch),
    )
    parser.add_argument(
        "--outdir",
        default=(scratch + "/mucoll/libtest/reco_n420_pfn_stores_simple")
        if scratch else None,
        required=not bool(scratch),
    )
    return parser.parse_args()


def find_root_files(directory):
    pattern = str(Path(directory) / "job_*" / "reco_output_*.edm4hep.root")
    return [Path(path).resolve() for path in sorted(glob.glob(pattern))]


def as_numpy(values, dtype=np.float32):
    return ak.to_numpy(values).astype(dtype)


def read_pfos(events):
    pfos = events["PandoraPFOs"]
    px = pfos["PandoraPFOs.momentum.x"].array()
    py = pfos["PandoraPFOs.momentum.y"].array()
    pz = pfos["PandoraPFOs.momentum.z"].array()
    energy = pfos["PandoraPFOs.energy"].array()
    mass = pfos["PandoraPFOs.mass"].array()
    charge = pfos["PandoraPFOs.charge"].array()
    pdg = pfos["PandoraPFOs.PDG"].array()
    track_begin = pfos["PandoraPFOs.tracks_begin"].array()
    track_end = pfos["PandoraPFOs.tracks_end"].array()

    vectors = []
    link_counts = []
    for i in range(events.num_entries):
        columns = [
            as_numpy(column[i])
            for column in (px, py, pz, energy, mass, charge, pdg)
        ]
        px_i, py_i, pz_i, energy_i, mass_i, charge_i, pdg_i = columns
        pt_i = np.hypot(px_i, py_i)
        good = (
            np.isfinite(pt_i) & np.isfinite(px_i) & np.isfinite(py_i)
            & np.isfinite(pz_i) & np.isfinite(energy_i) & np.isfinite(mass_i)
            & (pt_i > 0)
        )
        if np.any(good):
            pt_i, px_i, py_i, pz_i, energy_i, mass_i, charge_i, pdg_i = [
                column[good] for column in
                (pt_i, px_i, py_i, pz_i, energy_i, mass_i, charge_i, pdg_i)
            ]
            vector = np.stack(
                [
                    pt_i,
                    np.arcsinh(pz_i / np.maximum(pt_i, 1e-12)),
                    np.arctan2(py_i, px_i),
                    energy_i,
                    mass_i,
                    charge_i,
                    pdg_i,
                    px_i,
                    py_i,
                    pz_i,
                ],
                axis=1,
            ).astype(np.float32)
            vector = vector[np.argsort(vector[:, 0])[::-1]]
        else:
            vector = np.zeros((0, len(PFO_FEATURES)), dtype=np.float32)
        vectors.append(vector)
        begins = as_numpy(track_begin[i], np.int64)
        ends = as_numpy(track_end[i], np.int64)
        link_counts.append(int(np.sum(ends - begins)))
    return vectors, np.asarray(link_counts, dtype=np.int32)


def choose_track_state(begin, end, locations):
    """Choose the IP state (location 1), otherwise the first available state."""
    if begin < 0 or begin >= end or begin >= len(locations):
        return None
    end = min(end, len(locations))
    at_ip = np.flatnonzero(locations[begin:end] == 1)
    return begin + int(at_ip[0]) if len(at_ip) else begin


def read_tracks(events):
    tracks = events["AllTracks"]
    chi2 = tracks["AllTracks.chi2"].array()
    ndf = tracks["AllTracks.ndf"].array()
    holes = tracks["AllTracks.Nholes"].array()
    state_begin = tracks["AllTracks.trackStates_begin"].array()
    state_end = tracks["AllTracks.trackStates_end"].array()
    selected = events["SiTracks_objIdx"]["SiTracks_objIdx.index"].array()

    states = events["_AllTracks_trackStates"]
    location = states["_AllTracks_trackStates.location"].array()
    phi = states["_AllTracks_trackStates.phi"].array()
    omega = states["_AllTracks_trackStates.omega"].array()
    tan_lambda = states["_AllTracks_trackStates.tanLambda"].array()
    d0 = states["_AllTracks_trackStates.D0"].array()
    z0 = states["_AllTracks_trackStates.Z0"].array()

    vectors = []
    for i in range(events.num_entries):
        selected_i = as_numpy(selected[i], np.int64)
        begin_i = as_numpy(state_begin[i], np.int64)
        end_i = as_numpy(state_end[i], np.int64)
        location_i = as_numpy(location[i], np.int64)
        state_columns = [
            as_numpy(column[i])
            for column in (phi, omega, tan_lambda, d0, z0)
        ]
        chi2_i = as_numpy(chi2[i])
        ndf_i = as_numpy(ndf[i])
        holes_i = as_numpy(holes[i])
        rows = []
        for track_index in selected_i:
            if track_index < 0 or track_index >= len(begin_i):
                continue
            state_index = choose_track_state(
                int(begin_i[track_index]), int(end_i[track_index]), location_i
            )
            if state_index is None:
                continue
            phi_i, omega_i, tan_i, d0_i, z0_i = (
                float(column[state_index]) for column in state_columns
            )
            if not np.all(np.isfinite([phi_i, omega_i, tan_i, d0_i, z0_i])):
                continue
            pt_i = (
                CURVATURE_TO_PT / abs(omega_i)
                if abs(omega_i) > 1e-12 else 0.0
            )
            rows.append([
                pt_i,
                np.arcsinh(tan_i),
                phi_i,
                d0_i,
                z0_i,
                float(chi2_i[track_index])
                / max(float(ndf_i[track_index]), 1.0),
                float(holes_i[track_index]),
                omega_i,
                tan_i,
            ])
        vectors.append(np.asarray(rows, dtype=np.float32).reshape(
            (-1, len(TRACK_FEATURES))
        ))
    return vectors


def read_clusters(events):
    clusters = events["PandoraClusters"]
    energy = clusters["PandoraClusters.energy"].array()
    x = clusters["PandoraClusters.position.x"].array()
    y = clusters["PandoraClusters.position.y"].array()
    z = clusters["PandoraClusters.position.z"].array()
    hit_begin = clusters["PandoraClusters.hits_begin"].array()
    hit_end = clusters["PandoraClusters.hits_end"].array()

    vectors = []
    for i in range(events.num_entries):
        energy_i, x_i, y_i, z_i = [
            as_numpy(column[i]) for column in (energy, x, y, z)
        ]
        begin_i = as_numpy(hit_begin[i], np.int64)
        end_i = as_numpy(hit_end[i], np.int64)
        r_i = np.hypot(x_i, y_i)
        good = (
            np.isfinite(energy_i) & np.isfinite(x_i) & np.isfinite(y_i)
            & np.isfinite(z_i)
        )
        if np.any(good):
            vector = np.stack(
                [
                    energy_i,
                    np.arcsinh(z_i / np.maximum(r_i, 1e-12)),
                    np.arctan2(y_i, x_i),
                    r_i,
                    z_i,
                    end_i - begin_i,
                ],
                axis=1,
            )[good].astype(np.float32)
            vector = vector[np.argsort(vector[:, 0])[::-1]]
        else:
            vector = np.zeros((0, len(CLUSTER_FEATURES)), dtype=np.float32)
        vectors.append(vector)
    return vectors


def read_root_file(path):
    events = uproot.open(path)["events"]
    pfos, pfo_track_links = read_pfos(events)
    return {
        "particles": pfos,
        "pfo_track_links": pfo_track_links,
        "tracks": read_tracks(events),
        "clusters": read_clusters(events),
    }


def pad_events(events, n_features):
    width = max(max((len(event) for event in events), default=0), 1)
    output = np.zeros((len(events), width, n_features), dtype=np.float32)
    counts = np.asarray([len(event) for event in events], dtype=np.int32)
    for i, event in enumerate(events):
        output[i, :len(event)] = event
    return output, counts


def write_store(directory, output, class_name):
    root_files = find_root_files(directory)
    if not root_files:
        raise SystemExit("No RECO ROOT files found in {}".format(directory))

    objects = {"particles": [], "tracks": [], "clusters": []}
    pfo_track_links = []
    source_files = []
    source_events = []
    for path in root_files:
        result = read_root_file(path)
        n_events = len(result["particles"])
        for name in objects:
            objects[name].extend(result[name])
        pfo_track_links.extend(result["pfo_track_links"])
        source_files.extend([str(path)] * n_events)
        source_events.extend(range(n_events))

    padded = {}
    for name, features in (
        ("particles", PFO_FEATURES),
        ("tracks", TRACK_FEATURES),
        ("clusters", CLUSTER_FEATURES),
    ):
        padded[name] = pad_events(objects[name], len(features))

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h5:
        for name, count_name in (
            ("particles", "n_particles"),
            ("tracks", "n_tracks"),
            ("clusters", "n_clusters"),
        ):
            values, counts = padded[name]
            h5.create_dataset(name, data=values, compression="gzip")
            h5.create_dataset(count_name, data=counts)
        h5.create_dataset(
            "pfo_track_links", data=np.asarray(pfo_track_links, dtype=np.int32)
        )
        h5.create_dataset(
            "source_event", data=np.asarray(source_events, dtype=np.int32)
        )
        h5.create_dataset(
            "source_file",
            data=np.asarray(source_files, dtype=h5py.string_dtype("utf-8")),
        )
        h5.attrs["class_name"] = class_name
        h5.attrs["features"] = ",".join(PFO_FEATURES)
        h5.attrs["track_features"] = ",".join(TRACK_FEATURES)
        h5.attrs["cluster_features"] = ",".join(CLUSTER_FEATURES)
        h5.attrs["pfo_collection"] = "PandoraPFOs"
        h5.attrs["track_collection"] = "SiTracks_objIdx -> AllTracks"
        h5.attrs["cluster_collection"] = "PandoraClusters"

    means = {
        name: np.mean(counts) for name, (_, counts) in padded.items()
    }
    print(
        "{}: {} events; mean PFO/track/cluster = {:.2f}/{:.2f}/{:.2f} -> {}"
        .format(
            class_name, len(source_events), means["particles"],
            means["tracks"], means["clusters"], output,
        )
    )


def main():
    args = parse_args()
    reco_dir = Path(args.reco_dir).resolve()
    outdir = Path(args.outdir).resolve()
    for sample in SAMPLES:
        for split in SPLITS:
            source = (
                reco_dir
                / "reco_libtest_n{}_{}".format(N_FILES, sample)
                / split
            )
            output = outdir / "n{}_{}_{}.h5".format(
                N_FILES, sample, split
            )
            print("\n{} / {}".format(sample, split))
            write_store(source, output, sample)


if __name__ == "__main__":
    main()
