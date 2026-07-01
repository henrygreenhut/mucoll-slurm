#!/usr/bin/env python3

import argparse
import csv
import glob
import os
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from ml_common import write_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", default="root_sanity_checks")
    return parser.parse_args()


def find_files(inputs, pattern):
    files = []
    for item in inputs:
        matches = glob.glob(item) or [item]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                files.extend(path.glob(f"job_*/{pattern}"))
            elif path.is_file() and path.name.startswith(pattern.split("*")[0]):
                files.append(path)
    return sorted({path.resolve() for path in files})


def array(events, collection, field):
    return events[collection][f"{collection}.{field}"].array()


def optional_array(events, collection, field, like, default=0):
    try:
        return array(events, collection, field)
    except Exception:
        return ak.zeros_like(like) + default


def as_np(values):
    return ak.to_numpy(values)


def gen_rows(path):
    events = uproot.open(path)["events"]
    pdg = array(events, "MCParticles", "PDG")
    px = array(events, "MCParticles", "momentum.x")
    py = array(events, "MCParticles", "momentum.y")
    pz = array(events, "MCParticles", "momentum.z")
    mass = optional_array(events, "MCParticles", "mass", px)

    rows = []
    for i in range(events.num_entries):
        pdgs = as_np(pdg[i])
        xs = as_np(px[i])
        ys = as_np(py[i])
        zs = as_np(pz[i])
        masses = as_np(mass[i])
        if len(pdgs) == 0:
            continue

        pt = np.hypot(xs, ys)
        hardest = int(np.argmax(pt))
        first_p = np.sqrt(xs[0] * xs[0] + ys[0] * ys[0] + zs[0] * zs[0])
        first_e = np.sqrt(first_p * first_p + masses[0] * masses[0])
        hardest_p = np.sqrt(
            xs[hardest] * xs[hardest]
            + ys[hardest] * ys[hardest]
            + zs[hardest] * zs[hardest]
        )
        hardest_e = np.sqrt(hardest_p * hardest_p + masses[hardest] * masses[hardest])

        rows.append({
            "file": str(path),
            "event": i,
            "n_mc": int(len(pdgs)),
            "first_pdg": int(pdgs[0]),
            "first_pt": float(pt[0]),
            "first_theta_deg": float(np.degrees(np.arctan2(pt[0], zs[0]))),
            "first_energy": float(first_e),
            "hardest_pdg": int(pdgs[hardest]),
            "hardest_pt": float(pt[hardest]),
            "hardest_theta_deg": float(np.degrees(np.arctan2(pt[hardest], zs[hardest]))),
            "hardest_energy": float(hardest_e),
        })
    return rows


def pfo_track_links(events):
    try:
        begin = array(events, "PandoraPFOs", "tracks_begin")
        end = array(events, "PandoraPFOs", "tracks_end")
    except Exception:
        return None

    counts = []
    for starts, stops in zip(begin, end):
        counts.append(int(np.sum(as_np(stops) - as_np(starts))))
    return counts


def reco_rows(path):
    events = uproot.open(path)["events"]
    keys = [str(key) for key in events.keys()]
    if "PandoraPFOs" not in keys:
        return [], keys

    px = array(events, "PandoraPFOs", "momentum.x")
    py = array(events, "PandoraPFOs", "momentum.y")
    energy = optional_array(events, "PandoraPFOs", "energy", px)
    track_links = pfo_track_links(events)

    rows = []
    for i in range(events.num_entries):
        xs = as_np(px[i])
        ys = as_np(py[i])
        es = as_np(energy[i])
        pt = np.hypot(xs, ys)
        good = np.isfinite(pt) & (pt > 0)
        pt = pt[good]
        es = es[good]

        rows.append({
            "file": str(path),
            "event": i,
            "n_pfos": int(len(pt)),
            "sum_pt": float(np.sum(pt)) if len(pt) else 0.0,
            "leading_pt": float(np.max(pt)) if len(pt) else 0.0,
            "std_pt": float(np.std(pt)) if len(pt) else 0.0,
            "sum_energy": float(np.sum(es)) if len(es) else 0.0,
            "leading_energy": float(np.max(es)) if len(es) else 0.0,
            "pfo_track_links": track_links[i] if track_links is not None else "",
        })
    return rows, keys


def summarize(name, values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "quantity": name,
            "n": 0,
            "mean": "",
            "median": "",
            "min": "",
            "max": "",
            "p05": "",
            "p95": "",
        }
    return {
        "quantity": name,
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def main():
    args = parse_args()
    outdir = os.path.join(args.outdir, args.label)
    os.makedirs(outdir, exist_ok=True)

    gen_files = find_files(args.inputs, "gen_output_*.edm4hep.root")
    reco_files = find_files(args.inputs, "reco_output_*.edm4hep.root")
    if not gen_files and not reco_files:
        raise SystemExit("No gen or reco ROOT files found")

    gen = []
    for path in gen_files:
        gen.extend(gen_rows(path))

    reco = []
    branch_keys = set()
    for path in reco_files:
        rows, keys = reco_rows(path)
        reco.extend(rows)
        branch_keys.update(keys)

    if gen:
        write_rows(os.path.join(outdir, f"gen_truth_{args.label}.csv"), list(gen[0]), gen)
    if reco:
        write_rows(os.path.join(outdir, f"reco_pfos_{args.label}.csv"), list(reco[0]), reco)

    summary = []
    if gen:
        summary.extend([
            summarize("first_pt", [row["first_pt"] for row in gen]),
            summarize("first_theta_deg", [row["first_theta_deg"] for row in gen]),
            summarize("first_energy", [row["first_energy"] for row in gen]),
            summarize("hardest_pt", [row["hardest_pt"] for row in gen]),
        ])
    if reco:
        summary.extend([
            summarize("n_pfos", [row["n_pfos"] for row in reco]),
            summarize("leading_pt", [row["leading_pt"] for row in reco]),
            summarize("std_pt", [row["std_pt"] for row in reco]),
            summarize("sum_pt", [row["sum_pt"] for row in reco]),
            summarize("sum_energy", [row["sum_energy"] for row in reco]),
        ])
    if summary:
        write_rows(os.path.join(outdir, f"summary_{args.label}.csv"), list(summary[0]), summary)

    track_like = sorted(key for key in branch_keys if "track" in key.lower())
    write_rows(
        os.path.join(outdir, f"collections_{args.label}.csv"),
        ["collection_or_branch"],
        [{"collection_or_branch": key} for key in track_like],
    )

    first_counts = Counter(row["first_pdg"] for row in gen)
    hardest_counts = Counter(row["hardest_pdg"] for row in gen)

    print(f"GEN files: {len(gen_files)}")
    print(f"RECO files: {len(reco_files)}")
    if gen:
        print(f"GEN events: {len(gen)}")
        print(f"first_pdg counts: {dict(first_counts)}")
        print(f"hardest_pdg counts: {dict(hardest_counts)}")
        print(f"first_pt mean: {np.mean([row['first_pt'] for row in gen]):.3f} GeV")
    if reco:
        print(f"RECO events: {len(reco)}")
        print(f"n_pfos mean: {np.mean([row['n_pfos'] for row in reco]):.2f}")
        print(f"leading_pt mean: {np.mean([row['leading_pt'] for row in reco]):.3f} GeV")
        links = [row["pfo_track_links"] for row in reco if row["pfo_track_links"] != ""]
        if links:
            print(f"PFO track links total: {sum(links)}")
        else:
            print("PFO track link fields not found")
    print(f"Output -> {outdir}")


if __name__ == "__main__":
    main()
