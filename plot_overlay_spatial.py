#!/usr/bin/env python3

import argparse
import csv
import glob
import os
from pathlib import Path


TRACKER_COLLECTIONS = {
    "OverlayVertexBarrelCollection": "eDep",
    "OverlayVertexEndcapCollection": "eDep",
    "OverlayInnerTrackerBarrelCollection": "eDep",
    "OverlayInnerTrackerEndcapCollection": "eDep",
    "OverlayOuterTrackerBarrelCollection": "eDep",
    "OverlayOuterTrackerEndcapCollection": "eDep",
}

CALO_COLLECTIONS = {
    "OverlayECalBarrelCollection": "energy",
    "OverlayECalEndcapCollection": "energy",
    "OverlayHCalBarrelCollection": "energy",
    "OverlayHCalEndcapCollection": "energy",
}

GROUPS = {
    "all": list(TRACKER_COLLECTIONS) + list(CALO_COLLECTIONS),
    "tracker": list(TRACKER_COLLECTIONS),
    "calo": list(CALO_COLLECTIONS),
}

COLORS = {
    "OverlayVertexBarrelCollection": "#1f77b4",
    "OverlayVertexEndcapCollection": "#17becf",
    "OverlayInnerTrackerBarrelCollection": "#2ca02c",
    "OverlayInnerTrackerEndcapCollection": "#98df8a",
    "OverlayOuterTrackerBarrelCollection": "#9467bd",
    "OverlayOuterTrackerEndcapCollection": "#c5b0d5",
    "OverlayECalBarrelCollection": "#ff7f0e",
    "OverlayECalEndcapCollection": "#ffbb78",
    "OverlayHCalBarrelCollection": "#d62728",
    "OverlayHCalEndcapCollection": "#ff9896",
}


ak = None
np = None
plt = None
uproot = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", default="plots")
    parser.add_argument("--max-points-per-collection", type=int, default=5000)
    return parser.parse_args()


def load_libraries():
    global ak, np, plt, uproot
    import awkward
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot
    import numpy
    import uproot as uproot_module

    ak = awkward
    np = numpy
    plt = matplotlib.pyplot
    uproot = uproot_module


def find_digi_files(inputs):
    files = []
    for item in inputs:
        matches = glob.glob(item) or [item]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                files.extend(path.glob("digi_output_*.edm4hep.root"))
                files.extend(path.glob("job_*/digi_output_*.edm4hep.root"))
            elif path.is_file() and path.name.startswith("digi_output_"):
                files.append(path)
    return sorted({path.resolve() for path in files})


def branch_name(events, collection, field):
    candidates = [
        f"{collection}/{collection}.{field}",
        f"{collection}.{field}",
    ]
    keys = set(events.keys())
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def values(events, branch, event):
    if branch is None:
        return np.asarray([], dtype=np.float64)
    return ak.to_numpy(events[branch].array(entry_start=event, entry_stop=event + 1)[0])


def plot_prefix(path):
    return "__".join(path.parts[-3:]).replace(".edm4hep.root", "").replace(".root", "")


def downsample(*arrays, max_points):
    n = len(arrays[0]) if arrays else 0
    if n <= max_points:
        return arrays
    indices = np.linspace(0, n - 1, max_points, dtype=np.int64)
    return tuple(array[indices] for array in arrays)


def collection_payload(events, path, event, collection, value_field, max_points):
    x = values(events, branch_name(events, collection, "position.x"), event) / 10.0
    y = values(events, branch_name(events, collection, "position.y"), event) / 10.0
    z = values(events, branch_name(events, collection, "position.z"), event) / 10.0
    val = values(events, branch_name(events, collection, value_field), event)

    n = min(len(x), len(y), len(z))
    x = x[:n]
    y = y[:n]
    z = z[:n]
    plotted_x, plotted_y, plotted_z = downsample(x, y, z, max_points=max_points)
    r = np.sqrt(plotted_x * plotted_x + plotted_y * plotted_y)

    row = {
        "file": str(path),
        "event": event,
        "collection": collection,
        "value_field": value_field,
        "n_hits": int(n),
        "plotted_hits": int(len(plotted_x)),
        "sum_value": float(np.sum(val)) if len(val) else 0.0,
        "x_min_cm": float(np.min(x)) if n else "",
        "x_max_cm": float(np.max(x)) if n else "",
        "y_min_cm": float(np.min(y)) if n else "",
        "y_max_cm": float(np.max(y)) if n else "",
        "z_min_cm": float(np.min(z)) if n else "",
        "z_max_cm": float(np.max(z)) if n else "",
    }
    points = {
        "collection": collection,
        "x": plotted_x,
        "y": plotted_y,
        "z": plotted_z,
        "r": r,
        "n_hits": n,
        "plotted_hits": len(plotted_x),
    }
    return row, points


def write_rows(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "file",
        "event",
        "collection",
        "value_field",
        "n_hits",
        "plotted_hits",
        "sum_value",
        "x_min_cm",
        "x_max_cm",
        "y_min_cm",
        "y_max_cm",
        "z_min_cm",
        "z_max_cm",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def axis_limits(a, b):
    finite = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    if len(finite) == 0:
        return None
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1.0)
    else:
        pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def draw_projection(points, x_key, y_key, xlabel, ylabel, title, outpath):
    selected = [item for item in points if item["plotted_hits"]]
    if not selected:
        return False

    fig, ax = plt.subplots(figsize=(7, 6))
    all_x = []
    all_y = []
    total_hits = 0
    plotted_hits = 0

    for item in selected:
        x = item[x_key]
        y = item[y_key]
        all_x.append(x)
        all_y.append(y)
        total_hits += item["n_hits"]
        plotted_hits += item["plotted_hits"]
        ax.scatter(
            x,
            y,
            s=4,
            alpha=0.45,
            linewidths=0,
            color=COLORS[item["collection"]],
            label=f"{item['collection'].replace('Overlay', '')} ({item['n_hits']})",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nplotted {plotted_hits:,} of {total_hits:,} hits")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7, frameon=False)

    xlim = axis_limits(np.concatenate(all_x), np.concatenate(all_y))
    if xlim is not None and x_key in {"x", "y"} and y_key in {"x", "y"}:
        ax.set_xlim(*xlim)
        ax.set_ylim(*xlim)
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)
    return True


def draw_group(points_by_collection, group, prefix, outdir):
    group_points = [
        points_by_collection[name]
        for name in GROUPS[group]
        if name in points_by_collection
    ]
    if not any(item["plotted_hits"] for item in group_points):
        return 0

    title = f"{prefix} overlay {group}"
    outputs = [
        ("xy", "x", "y", "x [cm]", "y [cm]"),
        ("xz", "x", "z", "x [cm]", "z [cm]"),
        ("rz", "z", "r", "z [cm]", "r [cm]"),
    ]
    n_written = 0
    for suffix, x_key, y_key, xlabel, ylabel in outputs:
        outpath = os.path.join(outdir, f"{prefix}__overlay_{group}_{suffix}.pdf")
        if draw_projection(group_points, x_key, y_key, xlabel, ylabel, title, outpath):
            n_written += 1
    return n_written


def inspect_file(path, outdir, max_points):
    rows = []
    n_plots = 0
    with uproot.open(path) as root_file:
        events = root_file["events"]
        prefix = plot_prefix(path)
        for event in range(events.num_entries):
            points_by_collection = {}
            event_prefix = prefix if events.num_entries == 1 else f"{prefix}__event_{event}"
            for collection, value_field in {**TRACKER_COLLECTIONS, **CALO_COLLECTIONS}.items():
                row, points = collection_payload(
                    events,
                    path,
                    event,
                    collection,
                    value_field,
                    max_points,
                )
                rows.append(row)
                points_by_collection[collection] = points
            for group in GROUPS:
                n_plots += draw_group(points_by_collection, group, event_prefix, outdir)
    return rows, n_plots


def main():
    args = parse_args()
    load_libraries()
    files = find_digi_files(args.inputs)
    if not files:
        raise SystemExit("No digi ROOT files found")

    outdir = os.path.join(args.outdir, args.label)
    os.makedirs(outdir, exist_ok=True)

    rows = []
    n_plots = 0
    for path in files:
        file_rows, file_plots = inspect_file(
            path,
            outdir,
            args.max_points_per_collection,
        )
        rows.extend(file_rows)
        n_plots += file_plots

    outpath = os.path.join(outdir, f"overlay_spatial_summary_{args.label}.csv")
    write_rows(outpath, rows)

    print(f"DIGI files: {len(files)}")
    print(f"Plots written: {n_plots}")
    print(f"Summary -> {outpath}")
    for group in GROUPS:
        n_hits = sum(row["n_hits"] for row in rows if row["collection"] in GROUPS[group])
        plotted = sum(row["plotted_hits"] for row in rows if row["collection"] in GROUPS[group])
        print(f"{group}: n={n_hits}, plotted={plotted}")


if __name__ == "__main__":
    main()
