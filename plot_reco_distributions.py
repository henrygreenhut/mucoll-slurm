#!/usr/bin/env python3
"""Plot whole-sample RECO distributions for the fixed N=420 reuse study."""

import argparse
import csv
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reco_libtest_features import FEATURES, RAW_FEATURES, pfn_features

N_FILES = 420
SAMPLES = ("U", "R", "null_b")
SPLITS = ("train", "val", "test")
PFO_FEATURES = RAW_FEATURES
TRACK_FEATURES = (
    "pt", "eta", "phi", "d0", "z0", "chi2_ndf", "n_holes", "omega",
    "tan_lambda",
)
CLUSTER_FEATURES = ("energy", "eta", "phi", "r", "z", "n_hits")
SAMPLE_LABELS = {
    "U": "Unique mothers (420 unrotated files)",
    "R": r"Reused mothers (10 files $\times$ 42)",
    "null_b": "Independent unrotated reconstruction",
}
COLORS = {"U": "#0072B2", "R": "#D55E00", "null_b": "#009E73"}


def parse_args():
    scratch = os.environ.get("PSCRATCH", "")
    default_store = (
        scratch + "/mucoll/libtest/reco_n420_pfn_stores_simple"
        if scratch else None
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--store-dir", default=default_store, required=default_store is None,
        help="directory containing the nine extended N=420 RECO stores",
    )
    parser.add_argument(
        "--outdir", default="plots/reco_n420_directlog_whole_distributions",
        help="output directory",
    )
    return parser.parse_args()


def decode_attr(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def load_objects(h5, dataset, count_name, expected_features, attr_name):
    names = tuple(decode_attr(h5.attrs[attr_name]).split(","))
    if names != expected_features:
        raise ValueError(
            "{} has {}={!r}; expected {!r}".format(
                h5.filename, attr_name, names, expected_features
            )
        )
    values = h5[dataset][:].astype(np.float64)
    counts = h5[count_name][:].astype(np.int64)
    if len(values) != len(counts):
        raise ValueError("{} has inconsistent {}".format(h5.filename, dataset))
    if np.any(counts < 0) or np.any(counts > values.shape[1]):
        raise ValueError("{} has invalid {}".format(h5.filename, count_name))
    mask = np.arange(values.shape[1])[None, :] < counts[:, None]
    return values, counts, mask


def finite(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def event_max(values, mask):
    return np.max(np.where(mask, values, 0.0), axis=1)


def extract_store(path):
    with h5py.File(path, "r") as h5:
        required = {
            "tracks", "n_tracks", "clusters", "n_clusters",
            "pfo_track_links",
        }
        missing = sorted(required - set(h5.keys()))
        if missing:
            raise SystemExit(
                "{} predates the track/cluster extension (missing {}). "
                "Rebuild it with make_reco_libtest_stores.py."
                .format(path, ", ".join(missing))
            )
        pfos, n_pfos, pfo_mask = load_objects(
            h5, "particles", "n_particles", PFO_FEATURES, "features"
        )
        tracks, n_tracks, track_mask = load_objects(
            h5, "tracks", "n_tracks", TRACK_FEATURES, "track_features"
        )
        clusters, n_clusters, cluster_mask = load_objects(
            h5, "clusters", "n_clusters", CLUSTER_FEATURES,
            "cluster_features",
        )
        pfo_track_links = h5["pfo_track_links"][:].astype(np.float64)

    pfo = {name: i for i, name in enumerate(PFO_FEATURES)}
    track = {name: i for i, name in enumerate(TRACK_FEATURES)}
    cluster = {name: i for i, name in enumerate(CLUSTER_FEATURES)}
    pfo_pt = pfos[:, :, pfo["pt"]]
    pfo_energy = pfos[:, :, pfo["energy"]]
    pfo_charge = pfos[:, :, pfo["charge"]]
    pfo_pdg = np.abs(pfos[:, :, pfo["pdg"]].astype(np.int64))
    charged = pfo_mask & (np.abs(pfo_charge) > 0.1)
    photon = pfo_mask & (~charged) & (pfo_pdg == 22)
    neutral = pfo_mask & (~charged) & (~photon)
    track_pt = tracks[:, :, track["pt"]]
    cluster_energy = clusters[:, :, cluster["energy"]]
    transformed_pfos = pfn_features(pfos)
    transformed = {name: i for i, name in enumerate(FEATURES)}
    pfn_mask = pfo_mask & (pfo_pt > 0)

    events = {
        "n_pfos": n_pfos.astype(np.float64),
        "pfo_track_links": pfo_track_links,
        "sum_pfo_pt": np.sum(np.where(pfo_mask, pfo_pt, 0.0), axis=1),
        "leading_pfo_pt": event_max(pfo_pt, pfo_mask),
        "sum_pfo_energy": np.sum(
            np.where(pfo_mask, pfo_energy, 0.0), axis=1
        ),
        "leading_pfo_energy": event_max(pfo_energy, pfo_mask),
        "n_charged_pfos": np.sum(charged, axis=1).astype(np.float64),
        "n_photons": np.sum(photon, axis=1).astype(np.float64),
        "n_neutral_pfos": np.sum(neutral, axis=1).astype(np.float64),
        "n_tracks": n_tracks.astype(np.float64),
        "sum_track_pt": np.sum(
            np.where(track_mask, track_pt, 0.0), axis=1
        ),
        "leading_track_pt": event_max(track_pt, track_mask),
        "n_clusters": n_clusters.astype(np.float64),
        "sum_cluster_energy": np.sum(
            np.where(cluster_mask, cluster_energy, 0.0), axis=1
        ),
        "leading_cluster_energy": event_max(cluster_energy, cluster_mask),
    }
    objects = {
        "pfo_log_pt": transformed_pfos[
            :, :, transformed["log_pt"]
        ][pfn_mask],
        "pfo_eta": transformed_pfos[:, :, transformed["eta"]][pfn_mask],
        "pfo_sin_phi": transformed_pfos[
            :, :, transformed["sin_phi"]
        ][pfn_mask],
        "pfo_cos_phi": transformed_pfos[
            :, :, transformed["cos_phi"]
        ][pfn_mask],
        "pfo_log_energy": transformed_pfos[
            :, :, transformed["log_energy"]
        ][pfn_mask],
        "pfo_charge": transformed_pfos[
            :, :, transformed["charge"]
        ][pfn_mask],
        "pfo_is_charged": transformed_pfos[
            :, :, transformed["is_charged"]
        ][pfn_mask],
        "pfo_is_photon": transformed_pfos[
            :, :, transformed["is_photon"]
        ][pfn_mask],
        "pfo_is_neutral": transformed_pfos[
            :, :, transformed["is_neutral"]
        ][pfn_mask],
        "track_pt": tracks[:, :, track["pt"]][track_mask],
        "track_eta": tracks[:, :, track["eta"]][track_mask],
        "track_phi": tracks[:, :, track["phi"]][track_mask],
        "track_d0": tracks[:, :, track["d0"]][track_mask],
        "track_z0": tracks[:, :, track["z0"]][track_mask],
        "track_chi2_ndf": tracks[:, :, track["chi2_ndf"]][track_mask],
        "track_n_holes": tracks[:, :, track["n_holes"]][track_mask],
        "cluster_energy": clusters[:, :, cluster["energy"]][cluster_mask],
        "cluster_eta": clusters[:, :, cluster["eta"]][cluster_mask],
        "cluster_phi": clusters[:, :, cluster["phi"]][cluster_mask],
        "cluster_n_hits": clusters[:, :, cluster["n_hits"]][cluster_mask],
    }
    return {"events": events, "objects": objects}


def load_all(store_dir):
    by_split = {}
    for sample in SAMPLES:
        by_split[sample] = []
        for split in SPLITS:
            path = store_dir / "n{}_{}_{}.h5".format(
                N_FILES, sample, split
            )
            if not path.is_file():
                raise SystemExit("missing store: {}".format(path))
            by_split[sample].append(extract_store(path))

    combined = {}
    for sample, pieces in by_split.items():
        combined[sample] = {}
        for group in ("events", "objects"):
            combined[sample][group] = {
                key: np.concatenate([piece[group][key] for piece in pieces])
                for key in pieces[0][group]
            }
    return combined


def log10_one_plus(values):
    return np.log10(1.0 + np.maximum(values, 0.0))


def signed_log10_one_plus(values):
    return np.sign(values) * np.log10(1.0 + np.abs(values))


def histogram_bins(arrays, integer, bins=50):
    if integer:
        lowest = min(
            (int(np.floor(np.min(x))) for x in arrays if len(x)), default=0
        )
        highest = max((int(np.max(x)) for x in arrays if len(x)), default=1)
        return np.arange(lowest - 0.5, highest + 1.5, 1.0)
    merged = np.concatenate([finite(x) for x in arrays if len(finite(x))])
    if not len(merged):
        return np.linspace(0.0, 1.0, bins + 1)
    low, high = float(np.min(merged)), float(np.max(merged))
    if low == high:
        return np.linspace(low - 0.5, high + 0.5, bins + 1)
    return np.linspace(low, high, bins + 1)


def plot_distribution(data, samples, group, key, output, xlabel,
                      integer=False, transform=None, log_y=False):
    arrays = []
    for sample in samples:
        values = finite(data[sample][group][key])
        if transform is not None:
            values = transform(values)
        arrays.append(values)
    bins = histogram_bins(arrays, integer)

    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    for sample, values in zip(samples, arrays):
        weights = np.full(len(values), 1.0 / max(len(values), 1))
        axis.hist(
            values, bins=bins, weights=weights, histtype="step",
            linewidth=1.9, color=COLORS[sample], label=SAMPLE_LABELS[sample],
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(
        "Fraction of events" if group == "events" else "Fraction of objects"
    )
    if log_y:
        axis.set_yscale("log")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def rank_auc(first, second):
    """Mann-Whitney AUC with average ranks for ties."""
    values = np.concatenate([finite(first), finite(second)])
    labels = np.concatenate([
        np.zeros(len(finite(first)), dtype=np.int8),
        np.ones(len(finite(second)), dtype=np.int8),
    ])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    n_positive = int(np.sum(labels))
    n_negative = len(labels) - n_positive
    return float(
        (
            np.sum(ranks[labels == 1])
            - n_positive * (n_positive + 1) / 2.0
        )
        / (n_positive * n_negative)
    )


def write_summary(data, path):
    fields = [
        "sample", "events", "mean_pfos", "zero_pfo_fraction",
        "mean_pfo_track_links", "mean_tracks", "zero_track_fraction",
        "mean_clusters", "zero_cluster_fraction",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in SAMPLES:
            values = data[sample]["events"]
            writer.writerow({
                "sample": sample,
                "events": len(values["n_pfos"]),
                "mean_pfos": float(np.mean(values["n_pfos"])),
                "zero_pfo_fraction": float(np.mean(values["n_pfos"] == 0)),
                "mean_pfo_track_links": float(
                    np.mean(values["pfo_track_links"])
                ),
                "mean_tracks": float(np.mean(values["n_tracks"])),
                "zero_track_fraction": float(np.mean(values["n_tracks"] == 0)),
                "mean_clusters": float(np.mean(values["n_clusters"])),
                "zero_cluster_fraction": float(
                    np.mean(values["n_clusters"] == 0)
                ),
            })


def write_event_auc(data, path):
    keys = tuple(data["U"]["events"])
    fields = ("comparison", "observable", "auc")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for class_b in ("R", "null_b"):
            for key in keys:
                writer.writerow({
                    "comparison": "U_vs_{}".format(class_b),
                    "observable": key,
                    "auc": rank_auc(
                        data["U"]["events"][key],
                        data[class_b]["events"][key],
                    ),
                })


def main():
    args = parse_args()
    store_dir = Path(args.store_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_all(store_dir)

    event_plots = {
        "pfo_multiplicity": ("n_pfos", "PFOs per event", True, None, True),
        "pfo_track_links": (
            "pfo_track_links", "PFO-to-track links per event", True, None, True,
        ),
        "track_multiplicity": (
            "n_tracks", "Selected tracks per event", True, None, True,
        ),
        "cluster_multiplicity": (
            "n_clusters", "Pandora clusters per event", True, None, True,
        ),
        "sum_pfo_pt": (
            "sum_pfo_pt", r"$\log_{10}(1+\sum p_T^\mathrm{PFO}/\mathrm{GeV})$",
            False, log10_one_plus, False,
        ),
        "leading_pfo_pt": (
            "leading_pfo_pt",
            r"$\log_{10}(1+p_T^\mathrm{PFO,lead}/\mathrm{GeV})$",
            False, log10_one_plus, False,
        ),
        "sum_pfo_energy": (
            "sum_pfo_energy",
            r"$\log_{10}(1+\sum E^\mathrm{PFO}/\mathrm{GeV})$",
            False, log10_one_plus, False,
        ),
        "sum_track_pt": (
            "sum_track_pt",
            r"$\log_{10}(1+\sum p_T^\mathrm{track}/\mathrm{GeV})$",
            False, log10_one_plus, False,
        ),
        "leading_track_pt": (
            "leading_track_pt",
            r"$\log_{10}(1+p_T^\mathrm{track,lead}/\mathrm{GeV})$",
            False, log10_one_plus, False,
        ),
        "sum_cluster_energy": (
            "sum_cluster_energy",
            r"$\log_{10}(1+\sum E^\mathrm{cluster}/\mathrm{GeV})$",
            False, log10_one_plus, False,
        ),
        "leading_cluster_energy": (
            "leading_cluster_energy",
            r"$\log_{10}(1+E^\mathrm{cluster,lead}/\mathrm{GeV})$",
            False, log10_one_plus, False,
        ),
    }
    object_plots = {
        "pfo_log_pt": (
            "pfo_log_pt", r"$\ln(p_T/\mathrm{GeV})$", False, None, False,
        ),
        "pfo_eta": ("pfo_eta", r"PFO $\eta$", False, None, False),
        "pfo_sin_phi": (
            "pfo_sin_phi", r"PFO $\sin\phi$", False, None, False,
        ),
        "pfo_cos_phi": (
            "pfo_cos_phi", r"PFO $\cos\phi$", False, None, False,
        ),
        "pfo_log_energy": (
            "pfo_log_energy", r"$\ln(E/\mathrm{GeV})$", False, None, False,
        ),
        "pfo_charge": (
            "pfo_charge", r"PFO charge$/e$", True, None, False,
        ),
        "pfo_is_charged": (
            "pfo_is_charged", "PFO charged indicator", True, None, False,
        ),
        "pfo_is_photon": (
            "pfo_is_photon", "PFO photon indicator", True, None, False,
        ),
        "pfo_is_neutral": (
            "pfo_is_neutral", "PFO neutral indicator", True, None, False,
        ),
        "track_pt": ("track_pt", r"$\log_{10}(1+p_T/\mathrm{GeV})$",
                     False, log10_one_plus, False),
        "track_eta": ("track_eta", r"Track $\eta$", False, None, False),
        "track_phi": ("track_phi", r"Track $\phi$", False, None, False),
        "track_d0": (
            "track_d0", r"$\mathrm{sign}(d_0)\log_{10}(1+|d_0|/\mathrm{mm})$",
            False, signed_log10_one_plus, False,
        ),
        "track_z0": (
            "track_z0", r"$\mathrm{sign}(z_0)\log_{10}(1+|z_0|/\mathrm{mm})$",
            False, signed_log10_one_plus, False,
        ),
        "track_chi2_ndf": (
            "track_chi2_ndf", r"$\log_{10}(1+\chi^2/\mathrm{ndf})$",
            False, log10_one_plus, False,
        ),
        "track_n_holes": (
            "track_n_holes", "Track holes", True, None, True,
        ),
        "cluster_energy": (
            "cluster_energy", r"$\log_{10}(1+E/\mathrm{GeV})$",
            False, log10_one_plus, False,
        ),
        "cluster_eta": (
            "cluster_eta", r"Cluster $\eta$", False, None, False,
        ),
        "cluster_phi": (
            "cluster_phi", r"Cluster $\phi$", False, None, False,
        ),
        "cluster_n_hits": (
            "cluster_n_hits", "Calorimeter hits per cluster",
            True, None, True,
        ),
    }

    for comparison, samples in (
        ("main", ("U", "R")),
        ("null", ("U", "null_b")),
    ):
        directory = outdir / comparison
        directory.mkdir(exist_ok=True)
        for name, (key, xlabel, integer, transform, log_y) in event_plots.items():
            plot_distribution(
                data, samples, "events", key, directory / (name + ".pdf"),
                xlabel, integer=integer, transform=transform, log_y=log_y,
            )
        for name, (
            key, xlabel, integer, transform, log_y
        ) in object_plots.items():
            plot_distribution(
                data, samples, "objects", key, directory / (name + ".pdf"),
                xlabel, integer=integer, transform=transform,
                log_y=log_y,
            )

    write_summary(data, outdir / "reco_summary.csv")
    write_event_auc(data, outdir / "full_sample_event_auc.csv")
    print("whole-sample plots and summaries -> {}".format(outdir))


if __name__ == "__main__":
    main()
