#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


MAX_SCATTER_POINTS = 20000

SAMPLE_LABELS = {
    "nu14_bib": "Neutrino gun + BIB",
    "mu_barrel_bib": "Barrel muon + BIB",
    "mu_endcap_bib": "Endcap muon + BIB",
    "mu_barrel_nobib": "Barrel muon, no BIB",
    "mu_endcap_nobib": "Endcap muon, no BIB",
}

TITLE_LABELS = {
    "reco_bib812_job0_10evt": "BIB812 Reconstruction Comparison (10 Events/Sample)",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", nargs="+", help="name=study_dir_or_reco_root")
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", default="plots")
    return parser.parse_args()


def clean(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return text.strip("_") or "sample"


def presentation_label(text):
    if text in SAMPLE_LABELS:
        return SAMPLE_LABELS[text]

    label = re.sub(r"[_-]+", " ", str(text)).strip()
    replacements = {
        "nu14": "neutrino",
        "bib812": "BIB812",
        "nobib": "no BIB",
        "pt10": "pT 10 GeV",
        "pt100": "pT 100 GeV",
        "theta10 170": "theta 10-170 deg",
    }
    for old, new in replacements.items():
        label = re.sub(rf"\b{old}\b", new, label, flags=re.IGNORECASE)
    return label


def plot_title(label, text):
    return text


def parse_sample(text):
    if "=" in text:
        name, value = text.split("=", 1)
        return clean(name), value
    path = Path(text)
    return clean(path.parent.name if path.is_file() else path.name), text


def find_reco_files(inputs):
    files = []
    for item in inputs:
        matches = glob.glob(item) or [item]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                files.extend(path.glob("reco_output_*.edm4hep.root"))
                files.extend(path.glob("job_*/reco_output_*.edm4hep.root"))
            elif path.is_file() and path.name.endswith(".root"):
                files.append(path)
    return sorted({path.resolve() for path in files})


def branch_name(events, collection, field):
    keys = set(events.keys())
    for candidate in [f"{collection}/{collection}.{field}", f"{collection}.{field}"]:
        if candidate in keys:
            return candidate
    return None


def read_branch(events, collection, field):
    branch = branch_name(events, collection, field)
    if branch is None:
        return None
    return events[branch].array()


def as_array(values, dtype=np.float64):
    if values is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(ak.to_numpy(values), dtype=dtype)


def theta_eta_phi(px, py, pz):
    pt = np.hypot(px, py)
    theta = np.degrees(np.arctan2(pt, pz))
    eta = np.arcsinh(pz / np.maximum(pt, 1e-12))
    phi = np.arctan2(py, px)
    return pt, theta, eta, phi


def position_angles(x, y, z):
    r = np.hypot(x, y)
    theta = np.degrees(np.arctan2(r, z))
    eta = np.arcsinh(z / np.maximum(r, 1e-12))
    phi = np.arctan2(y, x)
    return r, theta, eta, phi


def count_tracks(events, event):
    candidates = [
        ("DedupedTracks_objIdx", "index"),
        ("SiTracks_Refitted", "type"),
        ("AllTracks", "type"),
    ]
    for collection, field in candidates:
        arr = read_branch(events, collection, field)
        if arr is not None:
            return int(len(arr[event]))
    states_phi = read_branch(events, "_AllTracks_trackStates", "phi")
    return int(len(states_phi[event])) if states_phi is not None else 0


def pfo_track_counts(events, event, n_pfos):
    begin = read_branch(events, "PandoraPFOs", "tracks_begin")
    end = read_branch(events, "PandoraPFOs", "tracks_end")
    if begin is None or end is None:
        return np.zeros(n_pfos, dtype=np.int32), False
    counts = as_array(end[event], np.int32) - as_array(begin[event], np.int32)
    return counts[:n_pfos], True


def cluster_hit_counts(events, event, n_clusters):
    begin = read_branch(events, "PandoraClusters", "hits_begin")
    end = read_branch(events, "PandoraClusters", "hits_end")
    if begin is None or end is None:
        return np.zeros(n_clusters, dtype=np.int32)
    counts = as_array(end[event], np.int32) - as_array(begin[event], np.int32)
    return counts[:n_clusters]


def read_pfos(events, path, sample, event):
    px_all = read_branch(events, "PandoraPFOs", "momentum.x")
    py_all = read_branch(events, "PandoraPFOs", "momentum.y")
    pz_all = read_branch(events, "PandoraPFOs", "momentum.z")
    if px_all is None or py_all is None or pz_all is None:
        return []

    px = as_array(px_all[event])
    py = as_array(py_all[event])
    pz = as_array(pz_all[event])
    pt, theta, eta, phi = theta_eta_phi(px, py, pz)

    energy = read_branch(events, "PandoraPFOs", "energy")
    mass = read_branch(events, "PandoraPFOs", "mass")
    charge = read_branch(events, "PandoraPFOs", "charge")
    pfo_type = read_branch(events, "PandoraPFOs", "type")

    energy = as_array(energy[event]) if energy is not None else np.zeros_like(pt)
    mass = as_array(mass[event]) if mass is not None else np.zeros_like(pt)
    charge = as_array(charge[event]) if charge is not None else np.zeros_like(pt)
    pfo_type = as_array(pfo_type[event], np.int32) if pfo_type is not None else np.zeros(len(pt), dtype=np.int32)
    track_counts, have_links = pfo_track_counts(events, event, len(pt))

    rows = []
    for i in range(len(pt)):
        if not np.isfinite(pt[i]) or pt[i] <= 0:
            continue
        rows.append({
            "sample": sample,
            "file": str(path),
            "event": event,
            "object_index": i,
            "pt": float(pt[i]),
            "energy": float(energy[i]) if i < len(energy) else 0.0,
            "mass": float(mass[i]) if i < len(mass) else 0.0,
            "charge": float(charge[i]) if i < len(charge) else 0.0,
            "type": int(pfo_type[i]) if i < len(pfo_type) else 0,
            "theta_deg": float(theta[i]),
            "eta": float(eta[i]),
            "phi": float(phi[i]),
            "px": float(px[i]),
            "py": float(py[i]),
            "pz": float(pz[i]),
            "track_count": int(track_counts[i]) if i < len(track_counts) else 0,
            "has_track": int(have_links and i < len(track_counts) and track_counts[i] > 0),
        })
    return rows


def read_clusters(events, path, sample, event):
    energy_all = read_branch(events, "PandoraClusters", "energy")
    x_all = read_branch(events, "PandoraClusters", "position.x")
    y_all = read_branch(events, "PandoraClusters", "position.y")
    z_all = read_branch(events, "PandoraClusters", "position.z")
    if energy_all is None or x_all is None or y_all is None or z_all is None:
        return []

    energy = as_array(energy_all[event])
    x = as_array(x_all[event])
    y = as_array(y_all[event])
    z = as_array(z_all[event])
    r, theta, eta, phi = position_angles(x, y, z)
    hit_counts = cluster_hit_counts(events, event, len(energy))

    rows = []
    for i in range(len(energy)):
        if not np.isfinite(energy[i]):
            continue
        rows.append({
            "sample": sample,
            "file": str(path),
            "event": event,
            "object_index": i,
            "energy": float(energy[i]),
            "theta_deg": float(theta[i]) if i < len(theta) else "",
            "eta": float(eta[i]) if i < len(eta) else "",
            "phi": float(phi[i]) if i < len(phi) else "",
            "x": float(x[i]) if i < len(x) else "",
            "y": float(y[i]) if i < len(y) else "",
            "z": float(z[i]) if i < len(z) else "",
            "r": float(r[i]) if i < len(r) else "",
            "hit_count": int(hit_counts[i]) if i < len(hit_counts) else 0,
        })
    return rows


def event_summary(sample, path, event, pfos, clusters, n_tracks):
    pfo_pt = np.asarray([row["pt"] for row in pfos], dtype=np.float64)
    pfo_energy = np.asarray([row["energy"] for row in pfos], dtype=np.float64)
    cluster_energy = np.asarray([row["energy"] for row in clusters], dtype=np.float64)
    pfo_with_tracks = sum(row["has_track"] for row in pfos)

    return {
        "sample": sample,
        "file": str(path),
        "event": event,
        "n_pfos": int(len(pfos)),
        "n_clusters": int(len(clusters)),
        "n_tracks": int(n_tracks),
        "n_pfos_with_tracks": int(pfo_with_tracks),
        "pfo_track_fraction": float(pfo_with_tracks / len(pfos)) if pfos else 0.0,
        "sum_pfo_pt": float(np.sum(pfo_pt)) if len(pfo_pt) else 0.0,
        "leading_pfo_pt": float(np.max(pfo_pt)) if len(pfo_pt) else 0.0,
        "sum_pfo_energy": float(np.sum(pfo_energy)) if len(pfo_energy) else 0.0,
        "leading_pfo_energy": float(np.max(pfo_energy)) if len(pfo_energy) else 0.0,
        "sum_cluster_energy": float(np.sum(cluster_energy)) if len(cluster_energy) else 0.0,
        "leading_cluster_energy": float(np.max(cluster_energy)) if len(cluster_energy) else 0.0,
        "cluster_hits_total": int(sum(row["hit_count"] for row in clusters)),
    }


def read_reco_file(path, sample):
    events = uproot.open(path)["events"]
    event_rows = []
    pfo_rows = []
    cluster_rows = []

    for event in range(events.num_entries):
        pfos = read_pfos(events, path, sample, event)
        clusters = read_clusters(events, path, sample, event)
        n_tracks = count_tracks(events, event)
        event_rows.append(event_summary(sample, path, event, pfos, clusters, n_tracks))
        pfo_rows.extend(pfos)
        cluster_rows.extend(clusters)

    return event_rows, pfo_rows, cluster_rows


def write_rows(path, rows, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def values(rows, key, sample=None):
    selected = rows if sample is None else [row for row in rows if row["sample"] == sample]
    out = []
    for row in selected:
        value = row.get(key, "")
        if value == "":
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if np.isfinite(value):
            out.append(value)
    return np.asarray(out, dtype=np.float64)


def mean_std(rows, key, sample):
    vals = values(rows, key, sample)
    if len(vals) == 0:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def save_bar(path, event_rows, samples, metrics, ylabel, title):
    width = 0.8 / max(len(samples), 1)
    x = np.arange(len(metrics))
    plt.figure(figsize=(max(6, 1.4 * len(metrics)), 4))
    for i, sample in enumerate(samples):
        means = []
        errs = []
        for _, key in metrics:
            mean, err = mean_std(event_rows, key, sample)
            means.append(mean)
            errs.append(err)
        plt.bar(x + (i - (len(samples) - 1) / 2) * width, means, width, yerr=errs, capsize=3, label=presentation_label(sample))
    plt.xticks(x, [name for name, _ in metrics], rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def bins_for(arrays, nbins=35):
    merged = np.concatenate([arr for arr in arrays if len(arr)]) if any(len(arr) for arr in arrays) else np.asarray([])
    if len(merged) == 0:
        return np.linspace(0, 1, nbins + 1)
    lo = float(np.min(merged))
    hi = float(np.max(merged))
    if lo == hi:
        return np.linspace(lo - 0.5, hi + 0.5, nbins + 1)
    return np.linspace(lo, hi, nbins + 1)


def save_hist(path, rows, samples, key, xlabel, title, nbins=35, log=True):
    arrays = [values(rows, key, sample) for sample in samples]
    bins = bins_for(arrays, nbins)
    plt.figure(figsize=(5.5, 4))
    for sample, arr in zip(samples, arrays):
        if len(arr):
            plt.hist(arr, bins=bins, histtype="step", linewidth=1.7, label=f"{presentation_label(sample)} ({len(arr)})")
    plt.xlabel(xlabel)
    plt.ylabel("objects" if rows and "object_index" in rows[0] else "events")
    if log:
        plt.yscale("log")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_multiplicity(path, event_rows, samples, label):
    metrics = [
        ("PFOs", "n_pfos"),
        ("clusters", "n_clusters"),
        ("tracks", "n_tracks"),
    ]
    plt.figure(figsize=(6, 4))
    for metric_label, key in metrics:
        arrays = [values(event_rows, key, sample) for sample in samples]
        bins = bins_for(arrays, 20)
        for sample, arr in zip(samples, arrays):
            if len(arr):
                plt.hist(arr, bins=bins, histtype="step", linewidth=1.5, label=f"{presentation_label(sample)}: {metric_label}")
    plt.xlabel("objects per event")
    plt.ylabel("events")
    plt.yscale("log")
    plt.title(plot_title(label, "Reconstructed Object Multiplicities"))
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def downsample(x, y):
    if len(x) <= MAX_SCATTER_POINTS:
        return x, y
    indices = np.linspace(0, len(x) - 1, MAX_SCATTER_POINTS, dtype=np.int64)
    return x[indices], y[indices]


def save_scatter(path, rows, samples, xkey, ykey, xlabel, ylabel, title, logy=False):
    plt.figure(figsize=(5.8, 4.3))
    for sample in samples:
        sample_rows = [row for row in rows if row["sample"] == sample]
        x = values(sample_rows, xkey)
        y = values(sample_rows, ykey)
        n = min(len(x), len(y))
        x, y = downsample(x[:n], y[:n])
        if len(x):
            plt.scatter(x, y, s=8, alpha=0.45, label=f"{presentation_label(sample)} ({n})", rasterized=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def sample_summary(event_rows, pfo_rows, cluster_rows, samples):
    rows = []
    for sample in samples:
        row = {"sample": sample}
        for key in [
            "n_pfos",
            "n_clusters",
            "n_tracks",
            "n_pfos_with_tracks",
            "pfo_track_fraction",
            "sum_pfo_pt",
            "leading_pfo_pt",
            "sum_pfo_energy",
            "leading_pfo_energy",
            "sum_cluster_energy",
            "leading_cluster_energy",
            "cluster_hits_total",
        ]:
            vals = values(event_rows, key, sample)
            row[f"{key}_mean"] = float(np.mean(vals)) if len(vals) else 0.0
            row[f"{key}_median"] = float(np.median(vals)) if len(vals) else 0.0
        row["n_events"] = int(len([r for r in event_rows if r["sample"] == sample]))
        row["n_pfo_objects"] = int(len([r for r in pfo_rows if r["sample"] == sample]))
        row["n_cluster_objects"] = int(len([r for r in cluster_rows if r["sample"] == sample]))
        rows.append(row)
    return rows


def main():
    args = parse_args()
    label = clean(args.label)
    outdir = Path(args.outdir) / label
    outdir.mkdir(parents=True, exist_ok=True)

    samples = []
    event_rows = []
    pfo_rows = []
    cluster_rows = []

    for sample_arg in args.samples:
        sample, path_text = parse_sample(sample_arg)
        files = find_reco_files([path_text])
        if not files:
            raise SystemExit(f"No reco ROOT files found for {sample}: {path_text}")
        samples.append(sample)
        print(f"{sample}: {len(files)} RECO files")
        for path in files:
            events, pfos, clusters = read_reco_file(path, sample)
            event_rows.extend(events)
            pfo_rows.extend(pfos)
            cluster_rows.extend(clusters)

    event_fields = [
        "sample",
        "file",
        "event",
        "n_pfos",
        "n_clusters",
        "n_tracks",
        "n_pfos_with_tracks",
        "pfo_track_fraction",
        "sum_pfo_pt",
        "leading_pfo_pt",
        "sum_pfo_energy",
        "leading_pfo_energy",
        "sum_cluster_energy",
        "leading_cluster_energy",
        "cluster_hits_total",
    ]
    pfo_fields = [
        "sample",
        "file",
        "event",
        "object_index",
        "pt",
        "energy",
        "mass",
        "charge",
        "type",
        "theta_deg",
        "eta",
        "phi",
        "px",
        "py",
        "pz",
        "track_count",
        "has_track",
    ]
    cluster_fields = [
        "sample",
        "file",
        "event",
        "object_index",
        "energy",
        "theta_deg",
        "eta",
        "phi",
        "x",
        "y",
        "z",
        "r",
        "hit_count",
    ]
    summary_rows = sample_summary(event_rows, pfo_rows, cluster_rows, samples)

    write_rows(outdir / f"reco_events_{label}.csv", event_rows, event_fields)
    write_rows(outdir / f"reco_pfos_{label}.csv", pfo_rows, pfo_fields)
    write_rows(outdir / f"reco_clusters_{label}.csv", cluster_rows, cluster_fields)
    write_rows(outdir / f"reco_summary_{label}.csv", summary_rows, list(summary_rows[0]))

    plt.rcParams["font.family"] = "serif"

    save_bar(
        outdir / f"reco_object_counts_{label}.pdf",
        event_rows,
        samples,
        [("PFOs", "n_pfos"), ("clusters", "n_clusters"), ("tracks", "n_tracks")],
        "mean per event",
        plot_title(label, "Reconstructed Object Counts"),
    )
    save_bar(
        outdir / f"reco_event_energy_{label}.pdf",
        event_rows,
        samples,
        [("sum PFO E", "sum_pfo_energy"), ("lead PFO E", "leading_pfo_energy"), ("sum cluster E", "sum_cluster_energy"), ("lead cluster E", "leading_cluster_energy")],
        "GeV",
        plot_title(label, "Visible Reconstructed Energy"),
    )
    save_bar(
        outdir / f"pfo_track_fraction_{label}.pdf",
        event_rows,
        samples,
        [("track-link fraction", "pfo_track_fraction")],
        "fraction",
        plot_title(label, "PFO Track Links"),
    )
    save_multiplicity(outdir / f"reco_multiplicity_distributions_{label}.pdf", event_rows, samples, label)
    save_hist(outdir / f"pfo_energy_{label}.pdf", pfo_rows, samples, "energy", "PFO energy [GeV]", plot_title(label, "PFO Energies"))
    save_hist(outdir / f"cluster_energy_{label}.pdf", cluster_rows, samples, "energy", "cluster energy [GeV]", plot_title(label, "Cluster Energies"))
    save_hist(outdir / f"pfo_theta_{label}.pdf", pfo_rows, samples, "theta_deg", "PFO theta [deg]", plot_title(label, "PFO Polar Angles"), log=False)
    save_hist(outdir / f"cluster_theta_{label}.pdf", cluster_rows, samples, "theta_deg", "cluster theta [deg]", plot_title(label, "Cluster Polar Angles"), log=False)
    save_scatter(outdir / f"pfo_energy_vs_theta_{label}.pdf", pfo_rows, samples, "theta_deg", "energy", "PFO theta [deg]", "PFO energy [GeV]", plot_title(label, "PFO Energy vs Theta"), logy=True)
    save_scatter(outdir / f"pfo_pt_vs_theta_{label}.pdf", pfo_rows, samples, "theta_deg", "pt", "PFO theta [deg]", "PFO pT [GeV]", plot_title(label, "PFO pT vs Theta"), logy=True)
    save_scatter(outdir / f"cluster_energy_vs_theta_{label}.pdf", cluster_rows, samples, "theta_deg", "energy", "cluster theta [deg]", "cluster energy [GeV]", plot_title(label, "Cluster Energy vs Theta"), logy=True)

    print(f"Events: {len(event_rows)}")
    print(f"PFOs: {len(pfo_rows)}")
    print(f"Clusters: {len(cluster_rows)}")
    print(f"Output -> {outdir}")


if __name__ == "__main__":
    main()
