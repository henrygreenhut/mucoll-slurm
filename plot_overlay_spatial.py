#!/usr/bin/env python3

import argparse
import csv
import glob
import json
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

SIGNAL_TRACKER_COLLECTIONS = {
    "VertexBarrelCollection": "eDep",
    "VertexEndcapCollection": "eDep",
    "InnerTrackerBarrelCollection": "eDep",
    "InnerTrackerEndcapCollection": "eDep",
    "OuterTrackerBarrelCollection": "eDep",
    "OuterTrackerEndcapCollection": "eDep",
}

SIGNAL_CALO_COLLECTIONS = {
    "ECalBarrelCollection": "energy",
    "ECalEndcapCollection": "energy",
    "HCalBarrelCollection": "energy",
    "HCalEndcapCollection": "energy",
}

SIGNAL_COLLECTIONS = {
    **SIGNAL_TRACKER_COLLECTIONS,
    **SIGNAL_CALO_COLLECTIONS,
}

GROUPS = {
    "all": list(TRACKER_COLLECTIONS) + list(CALO_COLLECTIONS) + list(SIGNAL_COLLECTIONS),
    "tracker": list(TRACKER_COLLECTIONS) + list(SIGNAL_TRACKER_COLLECTIONS),
    "calo": list(CALO_COLLECTIONS) + list(SIGNAL_CALO_COLLECTIONS),
}

SIGNAL_COLOR = "#00ff66"

COLORS = {
    "OverlayVertexBarrelCollection": "#0066cc",
    "OverlayVertexEndcapCollection": "#00a6d6",
    "OverlayInnerTrackerBarrelCollection": "#008f5a",
    "OverlayInnerTrackerEndcapCollection": "#9a8700",
    "OverlayOuterTrackerBarrelCollection": "#6f3fb5",
    "OverlayOuterTrackerEndcapCollection": "#8f4b2e",
    "OverlayECalBarrelCollection": "#f26b00",
    "OverlayECalEndcapCollection": "#d81b60",
    "OverlayHCalBarrelCollection": "#d00000",
    "OverlayHCalEndcapCollection": "#111111",
    **{name: SIGNAL_COLOR for name in SIGNAL_COLLECTIONS},
}

ENVELOPE_OVERRIDES = {
    "OverlayVertexBarrelCollection": {"type": "barrel", "rin": 3.0, "rout": 11.5, "zmax": 6.5, "segments": 64},
    "OverlayVertexEndcapCollection": {"type": "endcap", "rin": 3.0, "rout": 11.5, "zin": 8.0, "zout": 28.5, "segments": 64},
    "OverlayInnerTrackerBarrelCollection": {"type": "barrel", "rin": 12.0, "rout": 58.0, "zmax": 70.0, "segments": 64},
    "OverlayInnerTrackerEndcapCollection": {"type": "endcap", "rin": 6.0, "rout": 58.0, "zin": 80.0, "zout": 230.6, "segments": 64},
    "OverlayOuterTrackerBarrelCollection": {"type": "barrel", "rin": 58.0, "rout": 150.0, "zmax": 130.0, "segments": 64},
    "OverlayOuterTrackerEndcapCollection": {"type": "endcap", "rin": 6.0, "rout": 150.0, "zin": 131.0, "zout": 230.6, "segments": 64},
    "OverlayECalBarrelCollection": {"type": "barrel", "rin": 185.7, "rout": 212.45, "zmax": 230.7, "segments": 12},
    "OverlayECalEndcapCollection": {"type": "endcap", "rin": 31.0, "rout": 212.45, "zin": 230.7, "zout": 257.45, "segments": 12},
    "OverlayHCalBarrelCollection": {"type": "barrel", "rin": 212.6, "rout": 411.35, "zmax": 257.45, "segments": 12},
    "OverlayHCalEndcapCollection": {"type": "endcap", "rin": 31.0, "rout": 411.35, "zin": 257.55, "zout": 460.0, "segments": 12},
}

NOZZLE = {"angle_deg": 10.0, "zin": 6.0, "zout": 600.0}
HTML_SAMPLE_PERCENT = 10.0


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
    parser.add_argument("--plot-percent", type=float, default=None)
    parser.add_argument("--geometry", choices=["envelope", "off"], default="envelope")
    args = parser.parse_args()
    if args.plot_percent is not None and not (0 < args.plot_percent <= 100):
        parser.error("--plot-percent must be greater than 0 and at most 100")
    return args


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
    if field == "eDep":
        candidates.extend([
            f"{collection}/{collection}.EDep",
            f"{collection}.EDep",
        ])
    return first_branch(events, candidates)


def first_branch(events, candidates):
    keys = set(events.keys())
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def values(events, branch, event):
    if branch is None:
        return np.asarray([], dtype=np.float64)
    try:
        return ak.to_numpy(events[branch].array(entry_start=event, entry_stop=event + 1)[0])
    except Exception:
        return np.asarray([], dtype=np.float64)


def contribution_collection_names(collection):
    names = []
    if collection.endswith("Collection"):
        names.append(collection[:-len("Collection")] + "ContributionCollection")
    names.extend([
        f"{collection}Contributions",
        f"{collection}ContributionCollection",
    ])
    return list(dict.fromkeys(names))


def int_values(events, branch, event):
    try:
        return np.asarray(values(events, branch, event), dtype=np.int64).reshape(-1)
    except Exception:
        return np.asarray([], dtype=np.int64)


def contribution_link_indices(events, collection, event):
    link_collection = f"_{collection}_contributions"
    branch = first_branch(events, [
        f"{link_collection}/{link_collection}.index",
        f"{link_collection}.index",
    ])
    return int_values(events, branch, event)


def time_values(events, collection, event, n_hits):
    direct_branch = branch_name(events, collection, "time")
    if direct_branch is not None:
        direct = values(events, direct_branch, event)
        if len(direct) >= n_hits:
            return direct[:n_hits], "hit.time"
        if len(direct):
            time = np.full(n_hits, np.nan, dtype=np.float64)
            time[:min(n_hits, len(direct))] = direct[:n_hits]
            return time, "hit.time_partial"

    contribution_times = np.asarray([], dtype=np.float64)
    contribution_source = None
    for contribution_collection in contribution_collection_names(collection):
        contribution_time_branch = branch_name(events, contribution_collection, "time")
        contribution_times = values(events, contribution_time_branch, event)
        if len(contribution_times):
            contribution_source = contribution_collection
            break

    if len(contribution_times) == n_hits:
        return contribution_times[:n_hits], "contribution.time"

    begin_branch = first_branch(events, [
        f"{collection}/{collection}.contributions_begin",
        f"{collection}.contributions_begin",
        f"_{collection}_contributions_begin",
        f"_{collection}_contributions.begin",
    ])
    end_branch = first_branch(events, [
        f"{collection}/{collection}.contributions_end",
        f"{collection}.contributions_end",
        f"_{collection}_contributions_end",
        f"_{collection}_contributions.end",
    ])
    try:
        begins = values(events, begin_branch, event).astype(np.int64, copy=False)
        ends = values(events, end_branch, event).astype(np.int64, copy=False)
    except Exception:
        begins = np.asarray([], dtype=np.int64)
        ends = np.asarray([], dtype=np.int64)
    if len(contribution_times) and len(begins) >= n_hits and len(ends) >= n_hits:
        link_indices = contribution_link_indices(events, collection, event)
        time = np.full(n_hits, np.nan, dtype=np.float64)
        for i, (begin, end) in enumerate(zip(begins[:n_hits], ends[:n_hits])):
            if len(link_indices) and 0 <= begin < end <= len(link_indices):
                indices = link_indices[begin:end]
                indices = indices[(0 <= indices) & (indices < len(contribution_times))]
                if len(indices):
                    time[i] = float(np.nanmin(contribution_times[indices]))
            elif 0 <= begin < end <= len(contribution_times):
                time[i] = float(np.nanmin(contribution_times[begin:end]))
        source = "contribution_link.time_min" if len(link_indices) else "contribution.time_min"
        if contribution_source:
            source = f"{source}:{contribution_source}"
        return time, source

    if len(contribution_times):
        source = "contribution.time_unmapped"
        if contribution_source:
            source = f"{source}:{contribution_source}"
        return np.full(n_hits, np.nan, dtype=np.float64), source

    return np.full(n_hits, np.nan, dtype=np.float64), "missing"


def plot_prefix(path):
    return "__".join(path.parts[-3:]).replace(".edm4hep.root", "").replace(".root", "")


def plotted_count(n, max_points, plot_percent):
    if n == 0:
        return 0
    if plot_percent is not None:
        return max(1, min(n, int(np.ceil(n * plot_percent / 100.0))))
    return min(n, max_points)


def downsample(*arrays, n_points):
    n = len(arrays[0]) if arrays else 0
    if n <= n_points:
        return arrays
    indices = np.linspace(0, n - 1, n_points, dtype=np.int64)
    return tuple(array[indices] for array in arrays)


def finite_values(*arrays):
    mask = None
    for array in arrays:
        current = np.isfinite(array)
        mask = current if mask is None else mask & current
    if mask is None:
        return ()
    return tuple(array[mask] for array in arrays)


def percentile_bounds(values, lo=0.5, hi=99.5):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def collection_envelope(name, x, y, z):
    if name in ENVELOPE_OVERRIDES:
        return dict(ENVELOPE_OVERRIDES[name])

    x, y, z = finite_values(x, y, z)
    if not len(x):
        return None

    r = np.sqrt(x * x + y * y)
    rbounds = percentile_bounds(r)
    zbounds = percentile_bounds(np.abs(z))
    if rbounds is None or zbounds is None:
        return None

    if "Barrel" in name:
        return {
            "type": "barrel",
            "rin": rbounds[0],
            "rout": rbounds[1],
            "zmax": zbounds[1],
            "segments": 64,
        }
    return {
        "type": "endcap",
        "rin": rbounds[0],
        "rout": rbounds[1],
        "zin": zbounds[0],
        "zout": zbounds[1],
        "segments": 64,
    }


def display_name(collection, role):
    if role == "signal":
        return f"Signal {collection}"
    return collection.replace("Overlay", "")


def collection_payload(events, path, event, collection, value_field, max_points, plot_percent, role):
    x = values(events, branch_name(events, collection, "position.x"), event) / 10.0
    y = values(events, branch_name(events, collection, "position.y"), event) / 10.0
    z = values(events, branch_name(events, collection, "position.z"), event) / 10.0
    val = values(events, branch_name(events, collection, value_field), event)

    n = min(len(x), len(y), len(z))
    x = x[:n]
    y = y[:n]
    z = z[:n]
    r_full = np.sqrt(x * x + y * y)
    envelope = collection_envelope(collection, x, y, z) if role == "bib" else None
    time, time_source = time_values(events, collection, event, n)
    n_plot = plotted_count(n, max_points, plot_percent)
    plotted_x, plotted_y, plotted_z, plotted_r, plotted_time = downsample(
        x,
        y,
        z,
        r_full,
        time,
        n_points=n_plot,
    )
    cap_x, cap_y, cap_z, cap_time = downsample(
        x,
        y,
        z,
        time,
        n_points=plotted_count(n, max_points, None),
    )
    percent_x, percent_y, percent_z, percent_time = downsample(
        x,
        y,
        z,
        time,
        n_points=plotted_count(n, max_points, HTML_SAMPLE_PERCENT),
    )
    finite_time = time[np.isfinite(time)]

    row = {
        "file": str(path),
        "event": event,
        "role": role,
        "collection": collection,
        "value_field": value_field,
        "time_source": time_source,
        "n_hits": int(n),
        "plotted_hits": int(len(plotted_x)),
        "sum_value": float(np.sum(val)) if len(val) else 0.0,
        "time_min": float(np.min(finite_time)) if len(finite_time) else "",
        "time_max": float(np.max(finite_time)) if len(finite_time) else "",
        "x_min_cm": float(np.min(x)) if n else "",
        "x_max_cm": float(np.max(x)) if n else "",
        "y_min_cm": float(np.min(y)) if n else "",
        "y_max_cm": float(np.max(y)) if n else "",
        "z_min_cm": float(np.min(z)) if n else "",
        "z_max_cm": float(np.max(z)) if n else "",
        "r_min_cm": float(np.min(r_full)) if n else "",
        "r_max_cm": float(np.max(r_full)) if n else "",
    }
    points = {
        "collection": collection,
        "name": display_name(collection, role),
        "role": role,
        "color": COLORS[collection],
        "x": plotted_x,
        "y": plotted_y,
        "z": plotted_z,
        "time": plotted_time,
        "r": plotted_r,
        "n_hits": n,
        "plotted_hits": len(plotted_x),
        "time_source": time_source,
        "envelope": envelope,
        "html_samples": {
            "cap": {"x": cap_x, "y": cap_y, "z": cap_z, "time": cap_time},
            "percent": {"x": percent_x, "y": percent_y, "z": percent_z, "time": percent_time},
        },
    }
    return row, points


def write_rows(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "file",
        "event",
        "role",
        "collection",
        "value_field",
        "time_source",
        "n_hits",
        "plotted_hits",
        "sum_value",
        "time_min",
        "time_max",
        "x_min_cm",
        "x_max_cm",
        "y_min_cm",
        "y_max_cm",
        "z_min_cm",
        "z_max_cm",
        "r_min_cm",
        "r_max_cm",
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


def set_equal_3d(ax, x, y, z):
    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    finite_z = z[np.isfinite(z)]
    if not len(finite_x) or not len(finite_y) or not len(finite_z):
        return

    centers = [
        0.5 * (float(np.min(finite_x)) + float(np.max(finite_x))),
        0.5 * (float(np.min(finite_y)) + float(np.max(finite_y))),
        0.5 * (float(np.min(finite_z)) + float(np.max(finite_z))),
    ]
    spans = [
        float(np.max(finite_x)) - float(np.min(finite_x)),
        float(np.max(finite_y)) - float(np.min(finite_y)),
        float(np.max(finite_z)) - float(np.min(finite_z)),
    ]
    radius = 0.5 * max(max(spans), 1.0)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def envelope_projection_points(envelope, x_key, y_key):
    if envelope is None:
        return None
    rout = envelope["rout"]
    if {x_key, y_key} == {"x", "y"}:
        return np.asarray([-rout, rout]), np.asarray([-rout, rout])
    if x_key == "z" and y_key == "r":
        if envelope["type"] == "barrel":
            return np.asarray([-envelope["zmax"], envelope["zmax"]]), np.asarray([envelope["rin"], rout])
        return np.asarray([-envelope["zout"], envelope["zout"]]), np.asarray([envelope["rin"], rout])
    return None


def draw_ring_xy(ax, radius, segments, color, alpha):
    if segments <= 16:
        from matplotlib.patches import Polygon
        radius = radius / np.cos(np.pi / segments)
        phi = np.linspace(0, 2 * np.pi, segments, endpoint=False) + np.pi / segments
        xy = np.column_stack([radius * np.cos(phi), radius * np.sin(phi)])
        ax.add_patch(Polygon(xy, closed=True, fill=False, edgecolor=color, linewidth=1.0, alpha=alpha))
    else:
        from matplotlib.patches import Circle
        ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor=color, linewidth=1.0, alpha=alpha))


def draw_envelope_projection(ax, envelope, color, x_key, y_key):
    if envelope is None:
        return
    alpha = 0.28
    if {x_key, y_key} == {"x", "y"}:
        draw_ring_xy(ax, envelope["rin"], envelope["segments"], color, alpha)
        draw_ring_xy(ax, envelope["rout"], envelope["segments"], color, alpha)
    elif x_key == "z" and y_key == "r":
        from matplotlib.patches import Rectangle
        if envelope["type"] == "barrel":
            ax.add_patch(Rectangle(
                (-envelope["zmax"], envelope["rin"]),
                2 * envelope["zmax"],
                envelope["rout"] - envelope["rin"],
                fill=False,
                edgecolor=color,
                linewidth=1.0,
                alpha=alpha,
            ))
        else:
            for sign in (-1, 1):
                x0 = sign * envelope["zin"] if sign > 0 else -envelope["zout"]
                width = envelope["zout"] - envelope["zin"]
                ax.add_patch(Rectangle(
                    (x0, envelope["rin"]),
                    width,
                    envelope["rout"] - envelope["rin"],
                    fill=False,
                    edgecolor=color,
                    linewidth=1.0,
                    alpha=alpha,
                ))


def draw_projection(points, x_key, y_key, xlabel, ylabel, title, outpath, geometry=True):
    selected = [item for item in points if item["plotted_hits"]]
    if not selected:
        return False

    fig, ax = plt.subplots(figsize=(7, 6))
    all_x = []
    all_y = []
    total_hits = 0
    plotted_hits = 0

    for item in selected:
        color = item["color"]
        is_signal = item["role"] == "signal"
        if geometry and not is_signal:
            draw_envelope_projection(
                ax,
                item.get("envelope"),
                color,
                x_key,
                y_key,
            )
            extents = envelope_projection_points(item.get("envelope"), x_key, y_key)
            if extents is not None:
                all_x.append(extents[0])
                all_y.append(extents[1])

        x = item[x_key]
        y = item[y_key]
        all_x.append(x)
        all_y.append(y)
        total_hits += item["n_hits"]
        plotted_hits += item["plotted_hits"]
        ax.scatter(
            x,
            y,
            s=18 if is_signal else 4,
            alpha=0.95 if is_signal else 0.45,
            linewidths=0.35 if is_signal else 0,
            edgecolors="#111111" if is_signal else "none",
            color=color,
            zorder=4 if is_signal else 2,
            label=f"{item['name']} ({item['n_hits']})",
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


def draw_xyz(points, title, outpath):
    selected = [item for item in points if item["plotted_hits"]]
    if not selected:
        return False

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    all_x = []
    all_y = []
    all_z = []
    total_hits = 0
    plotted_hits = 0

    for item in selected:
        is_signal = item["role"] == "signal"
        x = item["x"]
        y = item["y"]
        z = item["z"]
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)
        total_hits += item["n_hits"]
        plotted_hits += item["plotted_hits"]
        ax.scatter(
            x,
            y,
            z,
            s=18 if is_signal else 3,
            alpha=0.95 if is_signal else 0.35,
            linewidths=0.25 if is_signal else 0,
            edgecolors="#111111" if is_signal else "none",
            color=item["color"],
            label=f"{item['name']} ({item['n_hits']})",
            depthshade=not is_signal,
            rasterized=True,
        )

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)
    set_equal_3d(ax, all_x, all_y, all_z)
    ax.view_init(elev=22, azim=-55)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_zlabel("z [cm]")
    ax.set_title(f"{title} xyz\nplotted {plotted_hits:,} of {total_hits:,} hits")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close(fig)
    return True


def clean_number_list(values, digits=3):
    out = []
    for value in values:
        if np.isfinite(value):
            out.append(round(float(value), digits))
        else:
            out.append(None)
    return out


def clean_xyzt_arrays(x, y, z, time):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if len(time) != len(x):
        time = np.full(len(x), np.nan, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return (
        np.round(x[mask], 3).tolist(),
        np.round(y[mask], 3).tolist(),
        np.round(z[mask], 3).tolist(),
        clean_number_list(time[mask], digits=4),
    )


def clean_xyzt(item):
    return clean_xyzt_arrays(item["x"], item["y"], item["z"], item["time"])


def clean_html_sample(sample):
    x, y, z, time = clean_xyzt_arrays(sample["x"], sample["y"], sample["z"], sample["time"])
    return {
        "x": x,
        "y": y,
        "z": z,
        "time": time,
        "count": len(x),
    }


def interactive_html(payload):
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{payload["title"]}</title>
<style>
html, body {{ margin: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f7f7; color: #222; }}
#toolbar {{ position: fixed; left: 12px; top: 10px; z-index: 2; display: flex; align-items: center; gap: 12px; padding: 8px 10px; background: rgba(255,255,255,0.9); border: 1px solid #ddd; border-radius: 6px; box-shadow: 0 1px 8px rgba(0,0,0,0.08); }}
#title {{ font-weight: 600; }}
#help {{ font-size: 12px; color: #555; }}
button {{ border: 1px solid #bbb; background: white; border-radius: 4px; padding: 4px 8px; cursor: pointer; }}
button:disabled {{ color: #999; cursor: default; }}
.toggle {{ display: flex; align-items: center; gap: 4px; font-size: 12px; color: #333; user-select: none; }}
.sample-control {{ display: flex; align-items: center; gap: 5px; font-size: 12px; color: #333; }}
.sample-control select {{ font-size: 12px; }}
.time-control {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: #333; }}
#time-slider {{ width: 170px; }}
#time-label {{ min-width: 110px; color: #444; }}
#legend {{ position: fixed; left: 12px; bottom: 12px; z-index: 2; max-width: 390px; padding: 9px 10px; background: rgba(255,255,255,0.9); border: 1px solid #ddd; border-radius: 6px; font-size: 12px; line-height: 1.45; }}
#legend-help {{ margin-bottom: 5px; color: #555; }}
.legend-row {{ display: flex; align-items: center; gap: 7px; padding: 2px 3px; border-radius: 4px; cursor: pointer; user-select: none; }}
.legend-row:hover {{ background: rgba(0,0,0,0.06); }}
.legend-row.off {{ opacity: 0.35; text-decoration: line-through; }}
.swatch {{ width: 11px; height: 11px; border-radius: 50%; flex: 0 0 auto; box-shadow: 0 0 0 1px rgba(0,0,0,0.25); }}
canvas {{ width: 100vw; height: 100vh; display: block; cursor: grab; }}
canvas.dragging {{ cursor: grabbing; }}
</style>
</head>
<body>
<div id="toolbar">
  <span id="title"></span>
  <button id="reset">Reset</button>
  <label class="toggle"><input id="frame-toggle" type="checkbox" checked> Box axes</label>
  <label class="toggle"><input id="geom-toggle" type="checkbox" checked> Geometry</label>
  <label class="sample-control">Points
    <select id="sample-mode">
      <option value="cap">5k each</option>
      <option value="percent">10% each</option>
    </select>
  </label>
  <span class="time-control">
    <button id="time-play" type="button">Play</button>
    <input id="time-slider" type="range" min="0" max="1000" value="1000" disabled>
    <span id="time-label">time unavailable</span>
  </span>
  <span id="help">Drag rotate · wheel zoom · shift/right-drag pan · click legend hide/show</span>
</div>
<canvas id="view"></canvas>
<div id="legend"></div>
<script>
const data = {data};
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const title = document.getElementById("title");
const legend = document.getElementById("legend");
const frameToggle = document.getElementById("frame-toggle");
const geomToggle = document.getElementById("geom-toggle");
const sampleModeSelect = document.getElementById("sample-mode");
const playButton = document.getElementById("time-play");
const timeSlider = document.getElementById("time-slider");
const timeLabel = document.getElementById("time-label");
const defaultBasis = {{
  right: [0, 0, 1],
  up: [0, 1, 0],
  forward: [-1, 0, 0]
}};
const state = {{
  basis: cloneBasis(defaultBasis),
  zoom: 1.0,
  panX: 0,
  panY: 0,
  showFrame: true,
  showGeometry: true,
  sampleMode: "cap",
  timeAvailable: false,
  timeMin: 0,
  timeMax: 0,
  timeCut: 0,
  playing: false,
  playStart: null,
  playFrom: 0,
  drag: false,
  mode: "rotate",
  lastX: 0,
  lastY: 0
}};
let width = 0;
let height = 0;
let baseScale = 1;
let radius = 1;
let bounds = {{ x: [-1, 1], y: [-1, 1], z: [-1, 1] }};
let framePending = false;
title.textContent = data.title;
data.traces.forEach(t => {{
  t.hidden = false;
  if (!t.samples) t.samples = {{ cap: {{ x: t.x || [], y: t.y || [], z: t.z || [], time: t.time || [], count: (t.x || []).length }} }};
  if (!t.samples.percent) t.samples.percent = t.samples.cap;
  if (!Array.isArray(t.time)) t.time = [];
}});
if (!Array.isArray(data.geometry)) data.geometry = [];
computeTimeRange();
updateTimeControls();
buildLegend();

function buildLegend() {{
  legend.innerHTML = `<div id="legend-help">Click detector part to hide/show</div>` + data.traces.map((t, i) => {{
    const sample = activeSample(t);
    return `<div class="legend-row ${{t.hidden ? "off" : ""}}" data-index="${{i}}" title="time: ${{t.time_source || "missing"}}"><span class="swatch" style="background:${{t.color}}"></span><span>${{t.name}} (${{t.total.toLocaleString()}} hits, ${{sample.x.length.toLocaleString()}} plotted)</span></div>`;
  }}).join("");
  for (const row of legend.querySelectorAll(".legend-row")) {{
    row.addEventListener("click", () => {{
      const trace = data.traces[Number(row.dataset.index)];
      trace.hidden = !trace.hidden;
      row.classList.toggle("off", trace.hidden);
      scheduleRender();
    }});
  }}
}}

function activeSample(trace) {{
  return trace.samples[state.sampleMode] || trace.samples.cap || trace;
}}

function finiteTime(value) {{
  return typeof value === "number" && Number.isFinite(value);
}}

function computeTimeRange() {{
  let lo = Infinity;
  let hi = -Infinity;
  for (const t of data.traces) {{
    const sample = activeSample(t);
    for (const time of sample.time) {{
      if (!finiteTime(time)) continue;
      lo = Math.min(lo, time);
      hi = Math.max(hi, time);
    }}
  }}
  state.timeAvailable = lo !== Infinity && hi !== -Infinity;
  if (state.timeAvailable) {{
    state.timeMin = lo;
    state.timeMax = hi;
    state.timeCut = hi;
    state.playFrom = lo;
  }}
}}

function formatTime(value) {{
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(2);
}}

function sliderToTime() {{
  const frac = Number(timeSlider.value) / 1000;
  return state.timeMin + frac * (state.timeMax - state.timeMin);
}}

function syncTimeSlider() {{
  const span = state.timeMax - state.timeMin || 1;
  timeSlider.value = Math.round(1000 * (state.timeCut - state.timeMin) / span);
}}

function updateTimeControls() {{
  if (!state.timeAvailable) {{
    playButton.disabled = true;
    timeSlider.disabled = true;
    timeLabel.textContent = "time unavailable";
    return;
  }}
  playButton.disabled = false;
  timeSlider.disabled = false;
  syncTimeSlider();
  timeLabel.textContent = `t <= ${{formatTime(state.timeCut)}} ns`;
}}

function setPlaying(playing) {{
  state.playing = playing;
  state.playStart = null;
  state.playFrom = state.timeCut;
  playButton.textContent = playing ? "Pause" : "Play";
}}

function playbackStep(timestamp) {{
  if (!state.playing) return;
  if (state.playStart === null) state.playStart = timestamp;
  const duration = 6000;
  const progress = Math.min((timestamp - state.playStart) / duration, 1);
  state.timeCut = state.playFrom + progress * (state.timeMax - state.playFrom);
  updateTimeControls();
  scheduleRender();
  if (progress < 1) {{
    window.requestAnimationFrame(playbackStep);
  }} else {{
    setPlaying(false);
    updateTimeControls();
  }}
}}

function togglePlayback() {{
  if (!state.timeAvailable) return;
  if (state.playing) {{
    setPlaying(false);
    return;
  }}
  if (state.timeCut >= state.timeMax) state.timeCut = state.timeMin;
  setPlaying(true);
  updateTimeControls();
  scheduleRender();
  window.requestAnimationFrame(playbackStep);
}}

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  computeScale();
  scheduleRender();
}}

function computeScale() {{
  let maxAbs = 1;
  const mins = {{ x: 0, y: 0, z: 0 }};
  const maxes = {{ x: 0, y: 0, z: 0 }};
  let found = false;
  for (const t of data.traces) {{
    const sample = activeSample(t);
    for (let i = 0; i < sample.x.length; i++) {{
      const x = sample.x[i];
      const y = sample.y[i];
      const z = sample.z[i];
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y), Math.abs(z));
      if (!found) {{
        mins.x = maxes.x = x;
        mins.y = maxes.y = y;
        mins.z = maxes.z = z;
        found = true;
      }} else {{
        mins.x = Math.min(mins.x, x);
        mins.y = Math.min(mins.y, y);
        mins.z = Math.min(mins.z, z);
        maxes.x = Math.max(maxes.x, x);
        maxes.y = Math.max(maxes.y, y);
        maxes.z = Math.max(maxes.z, z);
      }}
    }}
  }}
  if (!found) {{
    mins.x = mins.y = mins.z = -1;
    maxes.x = maxes.y = maxes.z = 1;
  }}
  for (const axis of ["x", "y", "z"]) {{
    mins[axis] = Math.min(mins[axis], 0);
    maxes[axis] = Math.max(maxes[axis], 0);
    const span = maxes[axis] - mins[axis] || 1;
    const pad = span * 0.04;
    bounds[axis] = [mins[axis] - pad, maxes[axis] + pad];
  }}
  radius = maxAbs;
  baseScale = 0.42 * Math.min(width, height) / radius;
}}

function niceStep(rawStep) {{
  const exponent = Math.floor(Math.log10(rawStep || 1));
  const scale = Math.pow(10, exponent);
  const fraction = rawStep / scale;
  if (fraction <= 1) return scale;
  if (fraction <= 2) return 2 * scale;
  if (fraction <= 5) return 5 * scale;
  return 10 * scale;
}}

function ticksFor(lo, hi, target = 6) {{
  const step = niceStep((hi - lo) / target);
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let value = start; value <= hi + step * 0.5; value += step) {{
    if (value >= lo - step * 0.5) ticks.push(Math.abs(value) < step * 1e-6 ? 0 : value);
  }}
  return ticks;
}}

function formatTick(value) {{
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(0);
  if (abs >= 1) return value.toFixed(1).replace(/\\.0$/, "");
  return value.toFixed(2).replace(/0+$/, "").replace(/\\.$/, "");
}}

function cloneBasis(basis) {{
  return {{
    right: basis.right.slice(),
    up: basis.up.slice(),
    forward: basis.forward.slice()
  }};
}}

function dot(a, b) {{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}}

function dotPoint(axis, x, y, z) {{
  return axis[0] * x + axis[1] * y + axis[2] * z;
}}

function cross(a, b) {{
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}}

function normalize(v) {{
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}}

function subtractProjection(v, axis) {{
  const d = dot(v, axis);
  return [v[0] - d * axis[0], v[1] - d * axis[1], v[2] - d * axis[2]];
}}

function rotateVector(v, axis, angle) {{
  const a = normalize(axis);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const d = dot(a, v);
  const axv = cross(a, v);
  return [
    v[0] * c + axv[0] * s + a[0] * d * (1 - c),
    v[1] * c + axv[1] * s + a[1] * d * (1 - c),
    v[2] * c + axv[2] * s + a[2] * d * (1 - c)
  ];
}}

function orthonormalizeBasis() {{
  const right = normalize(state.basis.right);
  const up = normalize(subtractProjection(state.basis.up, right));
  state.basis.right = right;
  state.basis.up = up;
  state.basis.forward = normalize(cross(right, up));
}}

function rotateBasis(axis, angle) {{
  state.basis.right = rotateVector(state.basis.right, axis, angle);
  state.basis.up = rotateVector(state.basis.up, axis, angle);
  state.basis.forward = rotateVector(state.basis.forward, axis, angle);
  orthonormalizeBasis();
}}

function project(x, y, z) {{
  const scale = baseScale * state.zoom;
  const sx = dotPoint(state.basis.right, x, y, z);
  const sy = dotPoint(state.basis.up, x, y, z);
  const depth = dotPoint(state.basis.forward, x, y, z);
  return {{
    x: width / 2 + state.panX + sx * scale,
    y: height / 2 + state.panY - sy * scale,
    d: depth
  }};
}}

function drawAxes() {{
  if (state.showFrame) {{
    drawFrameAxes();
    return;
  }}
  const axes = [
    ["x", radius, 0, 0, "#444"],
    ["y", 0, radius, 0, "#444"],
    ["z", 0, 0, radius, "#444"]
  ];
  ctx.save();
  ctx.lineWidth = 1;
  ctx.font = "12px sans-serif";
  ctx.fillStyle = "#333";
  for (const [label, x, y, z, color] of axes) {{
    const a = project(0, 0, 0);
    const b = project(x, y, z);
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.fillText(label, b.x + 5, b.y - 5);
  }}
  ctx.restore();
}}

function point3(x, y, z) {{
  return {{ x, y, z }};
}}

function drawLine3(a, b, color, width = 1, alpha = 1) {{
  const pa = project(a.x, a.y, a.z);
  const pb = project(b.x, b.y, b.z);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.stroke();
  ctx.restore();
}}

function drawPoly3(points, color, alpha = 0.22, width = 1) {{
  if (points.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  const first = project(points[0].x, points[0].y, points[0].z);
  ctx.moveTo(first.x, first.y);
  for (let i = 1; i < points.length; i++) {{
    const p = project(points[i].x, points[i].y, points[i].z);
    ctx.lineTo(p.x, p.y);
  }}
  ctx.stroke();
  ctx.restore();
}}

function drawText3(text, p, dx = 0, dy = 0, color = "#555", align = "center") {{
  const q = project(p.x, p.y, p.z);
  ctx.save();
  ctx.fillStyle = color;
  ctx.font = "12px sans-serif";
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  ctx.fillText(text, q.x + dx, q.y + dy);
  ctx.restore();
}}

function drawLine2(x0, y0, x1, y1, color = "#666", width = 1) {{
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.restore();
}}

function drawTickAt(p, dx, dy) {{
  const q = project(p.x, p.y, p.z);
  drawLine2(q.x, q.y, q.x + dx, q.y + dy, "#666", 1);
}}

function drawFrameAxes() {{
  const x0 = bounds.x[0], x1 = bounds.x[1];
  const y0 = bounds.y[0], y1 = bounds.y[1];
  const z0 = bounds.z[0], z1 = bounds.z[1];
  const corners = [
    point3(x0, y0, z0), point3(x1, y0, z0), point3(x1, y1, z0), point3(x0, y1, z0),
    point3(x0, y0, z1), point3(x1, y0, z1), point3(x1, y1, z1), point3(x0, y1, z1)
  ];
  const edges = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]];
  for (const [a, b] of edges) drawLine3(corners[a], corners[b], "#c9c9c9", 1, 0.65);

  drawLine3(point3(x0, y0, z0), point3(x1, y0, z0), "#666", 1.5, 1);
  drawLine3(point3(x1, y0, z0), point3(x1, y1, z0), "#666", 1.5, 1);
  drawLine3(point3(x1, y1, z0), point3(x1, y1, z1), "#666", 1.5, 1);

  for (const x of ticksFor(x0, x1)) {{
    const p = point3(x, y0, z0);
    drawTickAt(p, 0, 6);
    drawText3(formatTick(x), p, 0, 17, "#666");
  }}
  for (const y of ticksFor(y0, y1)) {{
    const p = point3(x1, y, z0);
    drawTickAt(p, 6, 0);
    drawText3(formatTick(y), p, 14, 0, "#666", "left");
  }}
  for (const z of ticksFor(z0, z1)) {{
    const p = point3(x1, y1, z);
    drawTickAt(p, 6, 0);
    drawText3(formatTick(z), p, 14, 0, "#666", "left");
  }}

  drawText3("x [cm]", point3((x0 + x1) / 2, y0, z0), 0, 34, "#333");
  drawText3("y [cm]", point3(x1, (y0 + y1) / 2, z0), 34, 0, "#333", "left");
  drawText3("z [cm]", point3(x1, y1, (z0 + z1) / 2), 34, 0, "#333", "left");
}}

function ringPoints(radius, z, segments) {{
  const points = [];
  const offset = segments <= 16 ? Math.PI / segments : 0;
  const drawRadius = segments <= 16 ? radius / Math.cos(Math.PI / segments) : radius;
  for (let i = 0; i <= segments; i++) {{
    const phi = offset + 2 * Math.PI * i / segments;
    points.push(point3(drawRadius * Math.cos(phi), drawRadius * Math.sin(phi), z));
  }}
  return points;
}}

function drawGeometryRing(radius, z, segments, color, alpha) {{
  drawPoly3(ringPoints(radius, z, segments), color, alpha, 1);
}}

function drawBarrelGeometry(g) {{
  const segments = g.segments || 64;
  const alpha = 0.22;
  for (const radius of [g.rin, g.rout]) {{
    drawGeometryRing(radius, -g.zmax, segments, g.color, alpha);
    drawGeometryRing(radius, g.zmax, segments, g.color, alpha);
  }}
  for (let i = 0; i < 6; i++) {{
    const phi = 2 * Math.PI * i / 6;
    const x = g.rout * Math.cos(phi);
    const y = g.rout * Math.sin(phi);
    drawLine3(point3(x, y, -g.zmax), point3(x, y, g.zmax), g.color, 1, alpha);
  }}
}}

function drawEndcapGeometry(g) {{
  const segments = g.segments || 64;
  const alpha = 0.22;
  for (const sign of [-1, 1]) {{
    for (const z of [sign * g.zin, sign * g.zout]) {{
      drawGeometryRing(g.rin, z, segments, g.color, alpha);
      drawGeometryRing(g.rout, z, segments, g.color, alpha);
    }}
    for (let i = 0; i < 4; i++) {{
      const phi = 2 * Math.PI * i / 4;
      const c = Math.cos(phi);
      const s = Math.sin(phi);
      for (const z of [sign * g.zin, sign * g.zout]) {{
        drawLine3(point3(g.rin * c, g.rin * s, z), point3(g.rout * c, g.rout * s, z), g.color, 1, alpha);
      }}
    }}
  }}
}}

function drawNozzleGeometry(g) {{
  const segments = g.segments || 64;
  const alpha = 0.18;
  const tanAngle = Math.tan((g.angle_deg || 10) * Math.PI / 180);
  const zLimit = Math.min(Math.abs(bounds.z[0]), Math.abs(bounds.z[1]), g.zout);
  if (zLimit <= g.zin) return;
  const zs = [g.zin, g.zin + 0.33 * (zLimit - g.zin), g.zin + 0.66 * (zLimit - g.zin), zLimit];
  for (const sign of [-1, 1]) {{
    for (const zabs of zs) drawGeometryRing(zabs * tanAngle, sign * zabs, segments, g.color, alpha);
    for (let i = 0; i < 4; i++) {{
      const phi = 2 * Math.PI * i / 4;
      const c = Math.cos(phi);
      const s = Math.sin(phi);
      const a = zs[0] * tanAngle;
      const b = zs[zs.length - 1] * tanAngle;
      drawLine3(point3(a * c, a * s, sign * zs[0]), point3(b * c, b * s, sign * zs[zs.length - 1]), g.color, 1, alpha);
    }}
  }}
}}

function drawGeometry() {{
  if (!state.showGeometry) return;
  for (const g of data.geometry) {{
    if (g.trace >= 0 && data.traces[g.trace]?.hidden) continue;
    if (g.type === "barrel") drawBarrelGeometry(g);
    else if (g.type === "endcap") drawEndcapGeometry(g);
    else if (g.type === "nozzle") drawNozzleGeometry(g);
  }}
}}

function render() {{
  framePending = false;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f7f7f7";
  ctx.fillRect(0, 0, width, height);
  drawAxes();
  drawGeometry();
  const points = [];
  const timeFiltering = state.timeAvailable && state.timeCut < state.timeMax;
  for (const t of data.traces) {{
    if (t.hidden) continue;
    const sample = activeSample(t);
    for (let i = 0; i < sample.x.length; i++) {{
      const hitTime = sample.time[i];
      let alpha = t.role === "signal" ? 0.95 : 0.5;
      if (state.timeAvailable) {{
        if (finiteTime(hitTime) && hitTime > state.timeCut) continue;
        if (!finiteTime(hitTime) && timeFiltering) alpha = t.role === "signal" ? 0.85 : 0.12;
      }}
      const p = project(sample.x[i], sample.y[i], sample.z[i]);
      points.push([p.d, p.x, p.y, t.color, alpha, t.size || 2, t.role || "bib"]);
    }}
  }}
  const bibPoints = points.filter(p => p[6] !== "signal");
  const signalPoints = points.filter(p => p[6] === "signal");
  if (!state.playing && !timeFiltering) bibPoints.sort((a, b) => a[0] - b[0]);
  if (!state.playing && !timeFiltering) signalPoints.sort((a, b) => a[0] - b[0]);
  for (const p of [...bibPoints, ...signalPoints]) {{
    ctx.globalAlpha = p[4];
    ctx.fillStyle = p[3];
    const size = p[5];
    ctx.fillRect(p[1] - size / 2, p[2] - size / 2, size, size);
  }}
  ctx.globalAlpha = 1;
}}

function scheduleRender() {{
  if (framePending) return;
  framePending = true;
  window.requestAnimationFrame(render);
}}

function resetView() {{
  state.basis = cloneBasis(defaultBasis);
  state.zoom = 1.0;
  state.panX = 0;
  state.panY = 0;
  scheduleRender();
}}

canvas.addEventListener("mousedown", event => {{
  state.drag = true;
  state.mode = event.shiftKey || event.button === 2 ? "pan" : "rotate";
  state.lastX = event.clientX;
  state.lastY = event.clientY;
  canvas.classList.add("dragging");
}});
window.addEventListener("mouseup", () => {{
  state.drag = false;
  canvas.classList.remove("dragging");
}});
window.addEventListener("mousemove", event => {{
  if (!state.drag) return;
  const dx = event.clientX - state.lastX;
  const dy = event.clientY - state.lastY;
  state.lastX = event.clientX;
  state.lastY = event.clientY;
  if (state.mode === "pan") {{
    state.panX += dx;
    state.panY += dy;
  }} else {{
    const rightAxis = state.basis.right.slice();
    const upAxis = state.basis.up.slice();
    rotateBasis(upAxis, dx * 0.006);
    rotateBasis(rightAxis, -dy * 0.006);
  }}
  scheduleRender();
}});
canvas.addEventListener("wheel", event => {{
  event.preventDefault();
  state.zoom *= Math.exp(-event.deltaY * 0.001);
  state.zoom = Math.max(0.05, Math.min(40, state.zoom));
  scheduleRender();
}}, {{ passive: false }});
canvas.addEventListener("dblclick", resetView);
canvas.addEventListener("contextmenu", event => event.preventDefault());
document.getElementById("reset").addEventListener("click", resetView);
frameToggle.addEventListener("change", () => {{
  state.showFrame = frameToggle.checked;
  scheduleRender();
}});
geomToggle.addEventListener("change", () => {{
  state.showGeometry = geomToggle.checked;
  scheduleRender();
}});
sampleModeSelect.addEventListener("change", () => {{
  state.sampleMode = sampleModeSelect.value;
  setPlaying(false);
  computeTimeRange();
  updateTimeControls();
  buildLegend();
  computeScale();
  scheduleRender();
}});
playButton.addEventListener("click", togglePlayback);
timeSlider.addEventListener("input", () => {{
  setPlaying(false);
  state.timeCut = sliderToTime();
  updateTimeControls();
  scheduleRender();
}});
window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
"""


def write_interactive_xyz(points, title, outpath, geometry=True):
    selected = [item for item in points if item["plotted_hits"]]
    if not selected:
        return False

    traces = []
    geometry_entries = []
    for item in selected:
        x, y, z, time = clean_xyzt(item)
        samples = {
            "cap": clean_html_sample(item["html_samples"]["cap"]),
            "percent": clean_html_sample(item["html_samples"]["percent"]),
        }
        trace_index = len(traces)
        traces.append({
            "name": item["name"],
            "role": item["role"],
            "color": item["color"],
            "size": 5 if item["role"] == "signal" else 2,
            "total": item["n_hits"],
            "time_source": item["time_source"],
            "x": x,
            "y": y,
            "z": z,
            "time": time,
            "samples": samples,
        })
        if geometry and item.get("envelope") is not None:
            geometry_entries.append({
                "trace": trace_index,
                "color": item["color"],
                **item["envelope"],
            })

    if geometry:
        geometry_entries.append({
            "trace": -1,
            "type": "nozzle",
            "color": "#666666",
            "segments": 64,
            **NOZZLE,
        })

    payload = {
        "title": f"{title} interactive xyz",
        "traces": traces,
        "geometry": geometry_entries,
    }
    with open(outpath, "w", encoding="utf-8") as handle:
        handle.write(interactive_html(payload))
    return True


def draw_group(points_by_collection, group, prefix, outdir, geometry=True):
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
        if draw_projection(group_points, x_key, y_key, xlabel, ylabel, title, outpath, geometry=geometry):
            n_written += 1
    outpath = os.path.join(outdir, f"{prefix}__overlay_{group}_xyz.pdf")
    if draw_xyz(group_points, title, outpath):
        n_written += 1
    outpath = os.path.join(outdir, f"{prefix}__overlay_{group}_xyz.html")
    if write_interactive_xyz(group_points, title, outpath, geometry=geometry):
        n_written += 1
    return n_written


def inspect_file(path, outdir, max_points, plot_percent, geometry=True):
    rows = []
    n_plots = 0
    with uproot.open(path) as root_file:
        events = root_file["events"]
        prefix = plot_prefix(path)
        for event in range(events.num_entries):
            points_by_collection = {}
            event_prefix = prefix if events.num_entries == 1 else f"{prefix}__event_{event}"
            collection_sets = [
                ("bib", {**TRACKER_COLLECTIONS, **CALO_COLLECTIONS}),
                ("signal", SIGNAL_COLLECTIONS),
            ]
            for role, collections in collection_sets:
                for collection, value_field in collections.items():
                    row, points = collection_payload(
                        events,
                        path,
                        event,
                        collection,
                        value_field,
                        max_points,
                        plot_percent,
                        role,
                    )
                    rows.append(row)
                    points_by_collection[collection] = points
            for group in GROUPS:
                n_plots += draw_group(
                    points_by_collection,
                    group,
                    event_prefix,
                    outdir,
                    geometry=geometry,
                )
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
            args.plot_percent,
            geometry=args.geometry == "envelope",
        )
        rows.extend(file_rows)
        n_plots += file_plots

    outpath = os.path.join(outdir, f"overlay_spatial_summary_{args.label}.csv")
    write_rows(outpath, rows)

    print(f"DIGI files: {len(files)}")
    print(f"Artifacts written: {n_plots}")
    print(f"Summary -> {outpath}")
    for group in GROUPS:
        n_hits = sum(row["n_hits"] for row in rows if row["collection"] in GROUPS[group])
        plotted = sum(row["plotted_hits"] for row in rows if row["collection"] in GROUPS[group])
        print(f"{group}: n={n_hits}, plotted={plotted}")
    signal_hits = sum(row["n_hits"] for row in rows if row["role"] == "signal")
    signal_plotted = sum(row["plotted_hits"] for row in rows if row["role"] == "signal")
    print(f"signal: n={signal_hits}, plotted={signal_plotted}")


if __name__ == "__main__":
    main()
