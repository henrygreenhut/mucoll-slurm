#!/usr/bin/env python3
"""Spatial displays of DIGI-stage BIB overlay hits and signal hits.

Reads Overlay* (accepted BIB) and matching signal collections from
digi_output_*.edm4hep.root files and writes, per view group (all/tracker/calo):

  - static PDF projections (xy, xz, rz) with detector envelope outlines
  - a static 3D PDF
  - a self-contained interactive HTML viewer with:
      * MAIA_v0 detector envelopes + nozzle wireframes (from k4geo)
      * hit time slider / playback (tracker: hit.time; calo: earliest
        contribution time via the podio link vectors)
      * color by detector part, energy (log scale), or time
      * dark/light themes with CVD-validated palettes
      * hover tooltip per hit, preset cameras, zoom-to-cursor, PNG export
      * signal hits drawn on top in a high-contrast color

Sampling is proportional to collection size by default (global --max-points
budget with a per-collection floor) so relative hit densities across detector
parts are preserved; --sample equal restores per-collection caps.
"""

import argparse
import csv
import glob
import json
import os
import zlib
from pathlib import Path


OVERLAY_TRACKER = {
    "OverlayVertexBarrelCollection": "eDep",
    "OverlayVertexEndcapCollection": "eDep",
    "OverlayInnerTrackerBarrelCollection": "eDep",
    "OverlayInnerTrackerEndcapCollection": "eDep",
    "OverlayOuterTrackerBarrelCollection": "eDep",
    "OverlayOuterTrackerEndcapCollection": "eDep",
}

OVERLAY_CALO = {
    "OverlayECalBarrelCollection": "energy",
    "OverlayECalEndcapCollection": "energy",
    "OverlayHCalBarrelCollection": "energy",
    "OverlayHCalEndcapCollection": "energy",
}

SIGNAL_TRACKER = {
    "VertexBarrelCollection": "eDep",
    "VertexEndcapCollection": "eDep",
    "InnerTrackerBarrelCollection": "eDep",
    "InnerTrackerEndcapCollection": "eDep",
    "OuterTrackerBarrelCollection": "eDep",
    "OuterTrackerEndcapCollection": "eDep",
}

SIGNAL_CALO = {
    "ECalBarrelCollection": "energy",
    "ECalEndcapCollection": "energy",
    "HCalBarrelCollection": "energy",
    "HCalEndcapCollection": "energy",
}

# name -> (value_field, kind)
ALL_COLLECTIONS = {}
for _name, _field in OVERLAY_TRACKER.items():
    ALL_COLLECTIONS[_name] = (_field, "bib")
for _name, _field in OVERLAY_CALO.items():
    ALL_COLLECTIONS[_name] = (_field, "bib")
for _name, _field in SIGNAL_TRACKER.items():
    ALL_COLLECTIONS[_name] = (_field, "signal")
for _name, _field in SIGNAL_CALO.items():
    ALL_COLLECTIONS[_name] = (_field, "signal")

GROUPS = {
    "all": list(OVERLAY_TRACKER) + list(OVERLAY_CALO) + list(SIGNAL_TRACKER) + list(SIGNAL_CALO),
    "tracker": list(OVERLAY_TRACKER) + list(SIGNAL_TRACKER),
    "calo": list(OVERLAY_CALO) + list(SIGNAL_CALO),
}

# Categorical palettes validated (lightness band, chroma floor, CVD
# separation, contrast) against light #f7f7f7 and dark #0e1117 surfaces.
LIGHT_COLORS = {
    "OverlayVertexBarrelCollection": "#1f6fd6",
    "OverlayVertexEndcapCollection": "#0087a8",
    "OverlayInnerTrackerBarrelCollection": "#1e8a4c",
    "OverlayInnerTrackerEndcapCollection": "#8a6d00",
    "OverlayOuterTrackerBarrelCollection": "#6f42c1",
    "OverlayOuterTrackerEndcapCollection": "#a04a1f",
    "OverlayECalBarrelCollection": "#c25e00",
    "OverlayECalEndcapCollection": "#c2185b",
    "OverlayHCalBarrelCollection": "#c62828",
    "OverlayHCalEndcapCollection": "#3f63b8",
}

DARK_COLORS = {
    "OverlayVertexBarrelCollection": "#3d85e8",
    "OverlayVertexEndcapCollection": "#1596b8",
    "OverlayInnerTrackerBarrelCollection": "#35a263",
    "OverlayInnerTrackerEndcapCollection": "#b3902e",
    "OverlayOuterTrackerBarrelCollection": "#9674e8",
    "OverlayOuterTrackerEndcapCollection": "#c9784a",
    "OverlayECalBarrelCollection": "#cf7d28",
    "OverlayECalEndcapCollection": "#d95a92",
    "OverlayHCalBarrelCollection": "#e05a5a",
    "OverlayHCalEndcapCollection": "#6788d8",
}

SIGNAL_COLOR_LIGHT = "#111111"
SIGNAL_COLOR_DARK = "#ffffff"


def overlay_name(collection):
    return collection if collection.startswith("Overlay") else f"Overlay{collection}"


def collection_colors(collection):
    key = overlay_name(collection)
    return LIGHT_COLORS.get(key, "#555555"), DARK_COLORS.get(key, "#aaaaaa")


# MAIA_v0 envelopes in cm, from k4geo MuColl/MAIA/compact/MAIA_v0/MAIA_v0.xml.
# For 12-segment shapes, rin/rout are inradii (distance to the flat face) --
# the DD4hep polyhedra convention; the drawing code converts to circumradius.
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


ak = None
np = None
plt = None
uproot = None

_warned_branches = set()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", default="plots")
    parser.add_argument("--max-points", type=int, default=60000,
                        help="global point budget per event across all collections")
    parser.add_argument("--min-points-per-collection", type=int, default=1000,
                        help="floor so sparse collections stay legible (proportional mode)")
    parser.add_argument("--sample", choices=["proportional", "equal"], default="proportional")
    parser.add_argument("--geometry", choices=["envelope", "off"], default="envelope")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark",
                        help="initial theme for the interactive viewer")
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


def first_branch(events, candidates):
    keys = set(events.keys())
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def branch_name(events, collection, field):
    return first_branch(events, [
        f"{collection}/{collection}.{field}",
        f"{collection}.{field}",
    ])


def values(events, branch, event):
    if branch is None:
        return np.asarray([], dtype=np.float64)
    try:
        return ak.to_numpy(events[branch].array(entry_start=event, entry_stop=event + 1)[0])
    except Exception as exc:
        if branch not in _warned_branches:
            _warned_branches.add(branch)
            print(f"WARNING: failed to read {branch}: {exc}")
        return np.asarray([], dtype=np.float64)


def contribution_collection_candidates(collection):
    # signal calo: ECalBarrelCollection -> ECalBarrelCollectionContributions
    # overlay calo: OverlayECalBarrelCollection -> OverlayECalBarrelContributionCollection
    candidates = [f"{collection}Contributions"]
    if collection.endswith("Collection"):
        candidates.append(collection[: -len("Collection")] + "ContributionCollection")
    return candidates


def time_values(events, collection, event, n_hits):
    if n_hits == 0:
        return np.asarray([], dtype=np.float64), "empty"

    direct_branch = branch_name(events, collection, "time")
    if direct_branch is not None:
        direct = values(events, direct_branch, event)
        if len(direct) >= n_hits:
            return np.asarray(direct[:n_hits], dtype=np.float64), "hit.time"

    contribution_times = None
    for candidate in contribution_collection_candidates(collection):
        time_branch = branch_name(events, candidate, "time")
        if time_branch is not None:
            contribution_times = values(events, time_branch, event)
            break
    if contribution_times is None or not len(contribution_times):
        return np.full(n_hits, np.nan), "missing"

    begins = values(events, branch_name(events, collection, "contributions_begin"), event)
    ends = values(events, branch_name(events, collection, "contributions_end"), event)
    link = values(events, first_branch(events, [
        f"_{collection}_contributions/_{collection}_contributions.index",
        f"_{collection}_contributions.index",
    ]), event)

    time = np.full(n_hits, np.nan)
    if len(begins) < n_hits or len(ends) < n_hits:
        return time, "contribution.unmapped"

    begins = begins.astype(np.int64, copy=False)
    ends = ends.astype(np.int64, copy=False)
    use_link = len(link) > 0
    if use_link:
        link = link.astype(np.int64, copy=False)
    n_contrib = len(contribution_times)
    for i in range(n_hits):
        begin, end = begins[i], ends[i]
        if not 0 <= begin < end:
            continue
        if use_link:
            if end > len(link):
                continue
            sel = link[begin:end]
            sel = sel[(sel >= 0) & (sel < n_contrib)]
            if len(sel):
                time[i] = float(np.min(contribution_times[sel]))
        elif end <= n_contrib:
            time[i] = float(np.min(contribution_times[begin:end]))
    return time, "contribution.time_min"


def plot_prefix(path):
    return "__".join(path.parts[-3:]).replace(".edm4hep.root", "").replace(".root", "")


def allocate_points(counts, budget, floor, mode):
    """Split a global point budget across collections.

    proportional: per-collection floor, remainder proportional to hit count,
    so relative densities across detector parts survive downsampling.
    equal: legacy behavior, an equal share per non-empty collection.
    """
    alloc = {name: 0 for name in counts}
    active = {name: n for name, n in counts.items() if n > 0}
    if not active:
        return alloc

    if mode == "equal":
        share = max(1, budget // len(active))
        for name, n in active.items():
            alloc[name] = min(n, share)
        return alloc

    for name, n in active.items():
        alloc[name] = min(n, floor)
    remaining = budget - sum(alloc.values())
    if remaining > 0:
        pool = {name: active[name] - alloc[name] for name in active if active[name] > alloc[name]}
        total = sum(pool.values())
        if total > 0:
            scale = min(1.0, remaining / total)
            for name, room in pool.items():
                alloc[name] += int(room * scale)
    return alloc


def downsample_indices(n, k, name):
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return np.sort(rng.choice(n, size=k, replace=False))


def finite_values(*arrays):
    mask = None
    for array in arrays:
        current = np.isfinite(array)
        mask = current if mask is None else mask & current
    if mask is None:
        return ()
    return tuple(array[mask] for array in arrays)


def percentile_bounds(vals, lo=0.5, hi=99.5):
    finite = vals[np.isfinite(vals)]
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
        return {"type": "barrel", "rin": rbounds[0], "rout": rbounds[1], "zmax": zbounds[1], "segments": 64}
    return {"type": "endcap", "rin": rbounds[0], "rout": rbounds[1], "zin": zbounds[0], "zout": zbounds[1], "segments": 64}


def read_collection(events, event, collection, value_field, kind):
    x = values(events, branch_name(events, collection, "position.x"), event) / 10.0
    y = values(events, branch_name(events, collection, "position.y"), event) / 10.0
    z = values(events, branch_name(events, collection, "position.z"), event) / 10.0
    val = values(events, branch_name(events, collection, value_field), event)

    n = min(len(x), len(y), len(z))
    x, y, z = x[:n], y[:n], z[:n]
    aligned_val = np.full(n, np.nan)
    aligned_val[: min(n, len(val))] = val[: min(n, len(val))]
    time, time_source = time_values(events, collection, event, n)

    return {
        "collection": collection,
        "kind": kind,
        "value_field": value_field,
        "n": n,
        "x": x,
        "y": y,
        "z": z,
        "r": np.sqrt(x * x + y * y),
        "val": aligned_val,
        "sum_value": float(np.sum(val)) if len(val) else 0.0,
        "time": time,
        "time_source": time_source,
    }


def build_points(raw, n_plot):
    idx = downsample_indices(raw["n"], n_plot, raw["collection"]) if raw["n"] else np.arange(0)
    envelope = None
    if raw["kind"] == "bib":  # signal shares the overlay envelopes
        envelope = collection_envelope(raw["collection"], raw["x"], raw["y"], raw["z"])
    return {
        "collection": raw["collection"],
        "kind": raw["kind"],
        "x": raw["x"][idx],
        "y": raw["y"][idx],
        "z": raw["z"][idx],
        "r": raw["r"][idx],
        "val": raw["val"][idx],
        "time": raw["time"][idx],
        "n_hits": raw["n"],
        "plotted_hits": int(len(idx)),
        "time_source": raw["time_source"],
        "envelope": envelope,
    }


def summary_row(path, event, raw, points):
    n = raw["n"]

    def rng(key):
        if not n:
            return "", ""
        finite = raw[key][np.isfinite(raw[key])]
        if not len(finite):
            return "", ""
        return float(np.min(finite)), float(np.max(finite))

    x_min, x_max = rng("x")
    y_min, y_max = rng("y")
    z_min, z_max = rng("z")
    r_min, r_max = rng("r")
    t_min, t_max = rng("time")
    return {
        "file": str(path),
        "event": event,
        "collection": raw["collection"],
        "kind": raw["kind"],
        "value_field": raw["value_field"],
        "time_source": raw["time_source"],
        "n_hits": n,
        "plotted_hits": points["plotted_hits"],
        "plotted_fraction": round(points["plotted_hits"] / n, 4) if n else "",
        "sum_value": raw["sum_value"],
        "time_min": t_min,
        "time_max": t_max,
        "x_min_cm": x_min,
        "x_max_cm": x_max,
        "y_min_cm": y_min,
        "y_max_cm": y_max,
        "z_min_cm": z_min,
        "z_max_cm": z_max,
        "r_min_cm": r_min,
        "r_max_cm": r_max,
    }


CSV_FIELDS = [
    "file", "event", "collection", "kind", "value_field", "time_source",
    "n_hits", "plotted_hits", "plotted_fraction", "sum_value",
    "time_min", "time_max",
    "x_min_cm", "x_max_cm", "y_min_cm", "y_max_cm",
    "z_min_cm", "z_max_cm", "r_min_cm", "r_max_cm",
]


def write_rows(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Static matplotlib plots
# --------------------------------------------------------------------------

def axis_limits(a, b):
    finite = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    if len(finite) == 0:
        return None
    lo, hi = float(np.min(finite)), float(np.max(finite))
    pad = max(abs(lo) * 0.05, 1.0) if lo == hi else 0.05 * (hi - lo)
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


def polygon_circumradius(radius, segments):
    # rin/rout for polyhedra are inradii; drawn vertices sit at r/cos(pi/n)
    if segments <= 16:
        return radius / np.cos(np.pi / segments)
    return radius


def draw_ring_xy(ax, radius, segments, color, alpha):
    if segments <= 16:
        from matplotlib.patches import Polygon
        rr = polygon_circumradius(radius, segments)
        phi = np.linspace(0, 2 * np.pi, segments, endpoint=False) + np.pi / segments
        xy = np.column_stack([rr * np.cos(phi), rr * np.sin(phi)])
        ax.add_patch(Polygon(xy, closed=True, fill=False, edgecolor=color, linewidth=1.0, alpha=alpha))
    else:
        from matplotlib.patches import Circle
        ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor=color, linewidth=1.0, alpha=alpha))


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
                2 * envelope["zmax"], envelope["rout"] - envelope["rin"],
                fill=False, edgecolor=color, linewidth=1.0, alpha=alpha,
            ))
        else:
            for sign in (-1, 1):
                x0 = envelope["zin"] if sign > 0 else -envelope["zout"]
                ax.add_patch(Rectangle(
                    (x0, envelope["rin"]),
                    envelope["zout"] - envelope["zin"], envelope["rout"] - envelope["rin"],
                    fill=False, edgecolor=color, linewidth=1.0, alpha=alpha,
                ))


def scatter_style(item):
    if item["kind"] == "signal":
        return {"s": 10, "alpha": 0.9, "color": SIGNAL_COLOR_LIGHT, "zorder": 5}
    light, _dark = collection_colors(item["collection"])
    return {"s": 4, "alpha": 0.45, "color": light, "zorder": 2}


def item_label(item):
    name = item["collection"].replace("Overlay", "")
    tag = "signal" if item["kind"] == "signal" else "BIB"
    frac = 100.0 * item["plotted_hits"] / item["n_hits"] if item["n_hits"] else 0.0
    return f"{name} [{tag}] ({item['n_hits']:,} hits, {frac:.0f}% shown)"


def draw_projection(points, x_key, y_key, xlabel, ylabel, title, outpath, geometry=True):
    selected = [item for item in points if item["plotted_hits"]]
    if not selected:
        return False
    selected.sort(key=lambda item: item["kind"] == "signal")  # signal drawn last

    fig, ax = plt.subplots(figsize=(7, 6))
    all_x, all_y = [], []
    total_hits = plotted_hits = 0

    for item in selected:
        if geometry and item["kind"] == "bib":
            light, _dark = collection_colors(item["collection"])
            draw_envelope_projection(ax, item.get("envelope"), light, x_key, y_key)
            extents = envelope_projection_points(item.get("envelope"), x_key, y_key)
            if extents is not None:
                all_x.append(extents[0])
                all_y.append(extents[1])

        x, y = item[x_key], item[y_key]
        all_x.append(x)
        all_y.append(y)
        total_hits += item["n_hits"]
        plotted_hits += item["plotted_hits"]
        ax.scatter(x, y, linewidths=0, rasterized=True,
                   label=item_label(item), **scatter_style(item))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nplotted {plotted_hits:,} of {total_hits:,} hits")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=6, frameon=False)

    xlim = axis_limits(np.concatenate(all_x), np.concatenate(all_y))
    if xlim is not None and x_key in {"x", "y"} and y_key in {"x", "y"}:
        ax.set_xlim(*xlim)
        ax.set_ylim(*xlim)
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    return True


def draw_xyz(points, title, outpath):
    selected = [item for item in points if item["plotted_hits"]]
    if not selected:
        return False
    selected.sort(key=lambda item: item["kind"] == "signal")

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    all_x, all_y, all_z = [], [], []
    total_hits = plotted_hits = 0

    for item in selected:
        all_x.append(item["x"])
        all_y.append(item["y"])
        all_z.append(item["z"])
        total_hits += item["n_hits"]
        plotted_hits += item["plotted_hits"]
        style = scatter_style(item)
        style["s"] = max(3, style["s"] - 1)
        ax.scatter(item["x"], item["y"], item["z"], linewidths=0, rasterized=True,
                   label=item_label(item), **style)

    set_equal_3d(ax, np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_z))
    ax.view_init(elev=22, azim=-55)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_zlabel("z [cm]")
    ax.set_title(f"{title} xyz\nplotted {plotted_hits:,} of {total_hits:,} hits")
    ax.legend(loc="upper left", fontsize=6, frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
# Interactive HTML viewer
# --------------------------------------------------------------------------

def sig_round(value, sig=4):
    if value is None or not np.isfinite(value):
        return None
    return float(f"{float(value):.{sig}g}")


def clean_trace_arrays(item):
    x = np.asarray(item["x"], dtype=np.float64)
    y = np.asarray(item["y"], dtype=np.float64)
    z = np.asarray(item["z"], dtype=np.float64)
    t = np.asarray(item["time"], dtype=np.float64)
    v = np.asarray(item["val"], dtype=np.float64)
    if len(t) != len(x):
        t = np.full(len(x), np.nan)
    if len(v) != len(x):
        v = np.full(len(x), np.nan)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return (
        np.round(x[mask], 3).tolist(),
        np.round(y[mask], 3).tolist(),
        np.round(z[mask], 3).tolist(),
        [None if not np.isfinite(val) else round(float(val), 4) for val in t[mask]],
        [sig_round(val) for val in v[mask]],
    )


def write_interactive_xyz(points, title, outpath, theme="dark", geometry=True):
    selected = [item for item in points if item["plotted_hits"]]
    if not selected:
        return False

    traces = []
    geometry_entries = []
    for item in selected:
        x, y, z, t, v = clean_trace_arrays(item)
        light, dark = collection_colors(item["collection"])
        trace_index = len(traces)
        traces.append({
            "name": item["collection"].replace("Overlay", ""),
            "kind": item["kind"],
            "colorLight": light,
            "colorDark": dark,
            "total": item["n_hits"],
            "time_source": item["time_source"],
            "x": x, "y": y, "z": z, "t": t, "v": v,
        })
        if geometry and item["kind"] == "bib" and item.get("envelope") is not None:
            geometry_entries.append({
                "trace": trace_index,
                "colorLight": light,
                "colorDark": dark,
                **item["envelope"],
            })

    if geometry:
        geometry_entries.append({
            "trace": -1, "type": "nozzle",
            "colorLight": "#8a8a8a", "colorDark": "#7d8590",
            "segments": 64, **NOZZLE,
        })

    payload = {
        "title": f"{title} interactive xyz",
        "theme": theme,
        "traces": traces,
        "geometry": geometry_entries,
    }
    html = (VIEWER_TEMPLATE
            .replace("__TITLE__", json.dumps(payload["title"]))
            .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))))
    with open(outpath, "w", encoding="utf-8") as handle:
        handle.write(html)
    return True


VIEWER_TEMPLATE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title id="page-title"></title>
<style>
:root {
  --bg: #f7f7f7; --panel: rgba(255,255,255,0.92); --border: #d5d5d5;
  --text: #1c1c1c; --text2: #555; --btn-bg: #fff; --btn-border: #bbb;
  --hover: rgba(0,0,0,0.06); --accent: #1f6fd6;
}
:root[data-theme="dark"] {
  --bg: #0e1117; --panel: rgba(22,27,34,0.92); --border: #30363d;
  --text: #e6edf3; --text2: #9aa4af; --btn-bg: #21262d; --btn-border: #444c56;
  --hover: rgba(255,255,255,0.08); --accent: #3d85e8;
}
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
#toolbar { position: fixed; left: 10px; top: 8px; right: 10px; z-index: 3;
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  padding: 7px 10px; background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; font-size: 12px; }
#title { font-weight: 600; font-size: 12px; max-width: 320px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
button, select { border: 1px solid var(--btn-border); background: var(--btn-bg);
  color: var(--text); border-radius: 5px; padding: 3px 8px; cursor: pointer;
  font-size: 12px; }
button:hover, select:hover { border-color: var(--accent); }
button:disabled { opacity: 0.45; cursor: default; }
button.active { border-color: var(--accent); color: var(--accent); }
.grp { display: flex; align-items: center; gap: 4px; }
.sep { width: 1px; height: 18px; background: var(--border); }
.toggle { display: flex; align-items: center; gap: 4px; user-select: none;
  color: var(--text2); }
#time-slider { width: 150px; accent-color: var(--accent); }
#time-label { min-width: 92px; color: var(--text2); font-variant-numeric: tabular-nums; }
#scale-box { display: none; align-items: center; gap: 6px; color: var(--text2); }
#scale-bar { width: 110px; height: 10px; border-radius: 3px; border: 1px solid var(--border); }
#legend { position: fixed; left: 10px; bottom: 10px; z-index: 3; max-width: 380px;
  padding: 8px 10px; background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; font-size: 12px; line-height: 1.45; }
.legend-head { display: flex; align-items: baseline; gap: 8px; margin: 3px 0 2px;
  color: var(--text2); font-weight: 600; font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.04em; }
.legend-head a { color: var(--accent); cursor: pointer; font-weight: 400;
  text-transform: none; letter-spacing: 0; }
.legend-row { display: flex; align-items: center; gap: 7px; padding: 1px 3px;
  border-radius: 4px; cursor: pointer; user-select: none; }
.legend-row:hover { background: var(--hover); }
.legend-row.off { opacity: 0.35; text-decoration: line-through; }
.legend-row .n { color: var(--text2); margin-left: auto; padding-left: 10px;
  font-variant-numeric: tabular-nums; }
.swatch { width: 11px; height: 11px; border-radius: 50%; flex: 0 0 auto;
  box-shadow: 0 0 0 1px var(--border); }
#tooltip { position: fixed; z-index: 4; display: none; pointer-events: none;
  padding: 6px 9px; background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; font-size: 12px; line-height: 1.4; max-width: 260px; }
#tooltip .tt-name { font-weight: 600; }
#tooltip .tt-dim { color: var(--text2); }
canvas { width: 100vw; height: 100vh; display: block; cursor: grab; }
canvas.dragging { cursor: grabbing; }
</style>
</head>
<body>
<div id="toolbar">
  <span id="title"></span>
  <span class="sep"></span>
  <span class="grp">
    <button id="view-side" class="active" title="beam-side view, z horizontal">Side</button>
    <button id="view-face" title="face-on x-y view">Face</button>
    <button id="view-three" title="three-quarter view">&frac34;</button>
    <button id="reset" title="reset camera (double-click canvas)">Reset</button>
  </span>
  <span class="sep"></span>
  <span class="grp">Color
    <select id="color-mode">
      <option value="detector">Detector</option>
      <option value="energy">Energy (log)</option>
      <option value="time">Time</option>
    </select>
    <span id="scale-box"><span id="scale-lo"></span><span id="scale-bar"></span><span id="scale-hi"></span></span>
  </span>
  <span class="sep"></span>
  <label class="toggle"><input id="frame-toggle" type="checkbox" checked> Axes</label>
  <label class="toggle"><input id="geom-toggle" type="checkbox" checked> Geometry</label>
  <span class="sep"></span>
  <span class="grp">
    <button id="time-play" type="button">Play</button>
    <input id="time-slider" type="range" min="0" max="1000" value="1000" disabled>
    <span id="time-label">time unavailable</span>
  </span>
  <span class="sep"></span>
  <button id="theme-toggle">Light</button>
  <button id="png-export" title="download current view as PNG">PNG</button>
</div>
<canvas id="view"></canvas>
<div id="legend"></div>
<div id="tooltip"></div>
<script>
"use strict";
const data = __PAYLOAD__;

const THEMES = {
  dark:  { bg: "#0e1117", frame: "#8b949e", frameDim: "#2d333b", text: "#c9d1d9",
           signal: "#ffffff", missing: "#616a75", pointAlpha: 0.6 },
  light: { bg: "#f7f7f7", frame: "#666666", frameDim: "#cccccc", text: "#333333",
           signal: "#111111", missing: "#a5a5a5", pointAlpha: 0.5 }
};
// viridis approximation for energy/time ramps
const RAMP = [[68,1,84],[71,44,122],[59,81,139],[44,113,142],[33,144,141],
              [39,173,129],[92,200,99],[170,220,50],[253,231,37]];
const RAMP_LUT = Array.from({ length: 256 }, (_, i) => {
  const u = i / 255 * (RAMP.length - 1);
  const k = Math.min(RAMP.length - 2, Math.floor(u));
  const f = u - k;
  const c = [0, 1, 2].map(j => Math.round(RAMP[k][j] + f * (RAMP[k + 1][j] - RAMP[k][j])));
  return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
});

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const el = id => document.getElementById(id);
const titleEl = el("title"), legend = el("legend"), tooltip = el("tooltip");
const frameToggle = el("frame-toggle"), geomToggle = el("geom-toggle");
const playButton = el("time-play"), timeSlider = el("time-slider"), timeLabel = el("time-label");
const colorModeSel = el("color-mode"), themeButton = el("theme-toggle");
const scaleBox = el("scale-box"), scaleBar = el("scale-bar");
const scaleLo = el("scale-lo"), scaleHi = el("scale-hi");

const defaultBasis = { right: [0, 0, 1], up: [0, 1, 0], forward: [-1, 0, 0] };
const state = {
  theme: (data.theme === "light") ? "light" : "dark",
  basis: cloneBasis(defaultBasis),
  zoom: 1, panX: 0, panY: 0,
  showFrame: true, showGeometry: true,
  colorMode: "detector",
  timeAvailable: false, timeMin: 0, timeMax: 0, timeCut: 0,
  playing: false, playStart: null, playFrom: 0,
  drag: false, mode: "rotate", lastX: 0, lastY: 0
};
let width = 0, height = 0, baseScale = 1, radius = 1;
let bounds = { x: [-1, 1], y: [-1, 1], z: [-1, 1] };
let framePending = false;
let energyLogMin = 0, energyLogMax = 1;

let totalPlotted = 0;
data.traces.forEach(t => {
  t.hidden = false;
  if (!Array.isArray(t.t)) t.t = [];
  if (!Array.isArray(t.v)) t.v = [];
  t.colorCache = null;
  totalPlotted += t.x.length;
});
if (!Array.isArray(data.geometry)) data.geometry = [];

// hit-test buffers filled during render
const hitPX = new Float32Array(totalPlotted);
const hitPY = new Float32Array(totalPlotted);
const hitTrace = new Int16Array(totalPlotted);
const hitIndex = new Int32Array(totalPlotted);
let hitCount = 0;

function theme() { return THEMES[state.theme]; }
function traceColor(t) {
  if (t.kind === "signal") return theme().signal;
  return state.theme === "dark" ? t.colorDark : t.colorLight;
}
function geomColor(g) { return state.theme === "dark" ? g.colorDark : g.colorLight; }

function computeEnergyRange() {
  let lo = Infinity, hi = -Infinity;
  for (const t of data.traces) {
    for (const v of t.v) {
      if (typeof v === "number" && v > 0) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
  }
  if (lo < hi) { energyLogMin = Math.log10(lo); energyLogMax = Math.log10(hi); }
  else { energyLogMin = 0; energyLogMax = 1; }
}

function invalidateColors() { for (const t of data.traces) t.colorCache = null; }

function traceColors(t) {
  if (t.colorCache) return t.colorCache;
  const n = t.x.length;
  const cols = new Array(n);
  const miss = theme().missing;
  if (state.colorMode === "detector") {
    cols.fill(traceColor(t));
  } else if (state.colorMode === "energy") {
    const span = (energyLogMax - energyLogMin) || 1;
    for (let i = 0; i < n; i++) {
      const v = t.v[i];
      cols[i] = (typeof v === "number" && v > 0)
        ? RAMP_LUT[Math.max(0, Math.min(255, Math.round(255 * (Math.log10(v) - energyLogMin) / span)))]
        : miss;
    }
  } else {
    const span = (state.timeMax - state.timeMin) || 1;
    for (let i = 0; i < n; i++) {
      const tv = t.t[i];
      cols[i] = (typeof tv === "number" && Number.isFinite(tv))
        ? RAMP_LUT[Math.max(0, Math.min(255, Math.round(255 * (tv - state.timeMin) / span)))]
        : miss;
    }
  }
  t.colorCache = cols;
  return cols;
}

function fmtEnergy(v) {
  if (!(typeof v === "number") || !Number.isFinite(v)) return "n/a";
  if (v >= 1) return v.toPrecision(3) + " GeV";
  if (v >= 1e-3) return (v * 1e3).toPrecision(3) + " MeV";
  return (v * 1e6).toPrecision(3) + " keV";
}
function fmtTime(v) {
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

function updateScaleBox() {
  if (state.colorMode === "detector") { scaleBox.style.display = "none"; return; }
  scaleBox.style.display = "flex";
  const stops = [0, 0.25, 0.5, 0.75, 1]
    .map(u => RAMP_LUT[Math.round(255 * u)] + " " + (u * 100) + "%").join(",");
  scaleBar.style.background = "linear-gradient(90deg," + stops + ")";
  if (state.colorMode === "energy") {
    scaleLo.textContent = fmtEnergy(Math.pow(10, energyLogMin));
    scaleHi.textContent = fmtEnergy(Math.pow(10, energyLogMax));
  } else {
    scaleLo.textContent = fmtTime(state.timeMin) + " ns";
    scaleHi.textContent = fmtTime(state.timeMax) + " ns";
  }
}

function buildLegend() {
  const sections = [["signal", "Signal"], ["bib", "BIB overlay"]];
  let html = "";
  for (const [kind, label] of sections) {
    const idx = data.traces.map((t, i) => [t, i]).filter(([t]) => t.kind === kind);
    if (!idx.length) continue;
    const all = kind === "bib"
      ? ' <a data-act="all">all</a> <a data-act="none">none</a>' : "";
    html += '<div class="legend-head">' + label + all + "</div>";
    for (const [t, i] of idx) {
      const pct = t.total ? Math.round(100 * t.x.length / t.total) : 0;
      html += '<div class="legend-row" data-index="' + i + '" title="time source: ' +
        (t.time_source || "missing") + '">' +
        '<span class="swatch" style="background:' + traceColor(t) + '"></span>' +
        "<span>" + t.name + "</span>" +
        '<span class="n">' + t.total.toLocaleString() + " · " + pct + "%</span></div>";
    }
  }
  legend.innerHTML = html;
  for (const row of legend.querySelectorAll(".legend-row")) {
    row.classList.toggle("off", data.traces[Number(row.dataset.index)].hidden);
    row.addEventListener("click", () => {
      const trace = data.traces[Number(row.dataset.index)];
      trace.hidden = !trace.hidden;
      row.classList.toggle("off", trace.hidden);
      scheduleRender();
    });
  }
  for (const a of legend.querySelectorAll(".legend-head a")) {
    a.addEventListener("click", () => {
      const hide = a.dataset.act === "none";
      data.traces.forEach(t => { if (t.kind === "bib") t.hidden = hide; });
      buildLegend();
      scheduleRender();
    });
  }
}

function finiteTime(value) { return typeof value === "number" && Number.isFinite(value); }

function computeTimeRange() {
  let lo = Infinity, hi = -Infinity;
  for (const t of data.traces) {
    for (const tv of t.t) {
      if (!finiteTime(tv)) continue;
      lo = Math.min(lo, tv);
      hi = Math.max(hi, tv);
    }
  }
  state.timeAvailable = lo !== Infinity;
  if (state.timeAvailable) {
    state.timeMin = lo; state.timeMax = hi;
    state.timeCut = hi; state.playFrom = lo;
  }
}

function sliderToTime() {
  return state.timeMin + (Number(timeSlider.value) / 1000) * (state.timeMax - state.timeMin);
}
function syncTimeSlider() {
  const span = state.timeMax - state.timeMin || 1;
  timeSlider.value = Math.round(1000 * (state.timeCut - state.timeMin) / span);
}
function updateTimeControls() {
  if (!state.timeAvailable) {
    playButton.disabled = true;
    timeSlider.disabled = true;
    timeLabel.textContent = "time unavailable";
    return;
  }
  playButton.disabled = false;
  timeSlider.disabled = false;
  syncTimeSlider();
  timeLabel.textContent = state.timeCut >= state.timeMax
    ? "all t" : "t ≤ " + fmtTime(state.timeCut) + " ns";
}
function setPlaying(playing) {
  state.playing = playing;
  state.playStart = null;
  state.playFrom = state.timeCut;
  playButton.textContent = playing ? "Pause" : "Play";
}
function playbackStep(timestamp) {
  if (!state.playing) return;
  if (state.playStart === null) state.playStart = timestamp;
  const duration = 6000;
  const progress = Math.min((timestamp - state.playStart) / duration, 1);
  state.timeCut = state.playFrom + progress * (state.timeMax - state.playFrom);
  updateTimeControls();
  scheduleRender();
  if (progress < 1) window.requestAnimationFrame(playbackStep);
  else { setPlaying(false); updateTimeControls(); }
}
function togglePlayback() {
  if (!state.timeAvailable) return;
  if (state.playing) { setPlaying(false); return; }
  if (state.timeCut >= state.timeMax) state.timeCut = state.timeMin;
  setPlaying(true);
  updateTimeControls();
  scheduleRender();
  window.requestAnimationFrame(playbackStep);
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  computeScale();
  scheduleRender();
}

function computeScale() {
  let maxAbs = 1;
  const mins = { x: 0, y: 0, z: 0 }, maxes = { x: 0, y: 0, z: 0 };
  let found = false;
  for (const t of data.traces) {
    for (let i = 0; i < t.x.length; i++) {
      const x = t.x[i], y = t.y[i], z = t.z[i];
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y), Math.abs(z));
      if (!found) { mins.x = maxes.x = x; mins.y = maxes.y = y; mins.z = maxes.z = z; found = true; }
      else {
        mins.x = Math.min(mins.x, x); maxes.x = Math.max(maxes.x, x);
        mins.y = Math.min(mins.y, y); maxes.y = Math.max(maxes.y, y);
        mins.z = Math.min(mins.z, z); maxes.z = Math.max(maxes.z, z);
      }
    }
  }
  if (!found) { mins.x = mins.y = mins.z = -1; maxes.x = maxes.y = maxes.z = 1; }
  for (const axis of ["x", "y", "z"]) {
    mins[axis] = Math.min(mins[axis], 0);
    maxes[axis] = Math.max(maxes[axis], 0);
    const span = maxes[axis] - mins[axis] || 1;
    bounds[axis] = [mins[axis] - span * 0.04, maxes[axis] + span * 0.04];
  }
  radius = maxAbs;
  baseScale = 0.42 * Math.min(width, height) / radius;
}

function niceStep(rawStep) {
  const exponent = Math.floor(Math.log10(rawStep || 1));
  const scale = Math.pow(10, exponent);
  const fraction = rawStep / scale;
  if (fraction <= 1) return scale;
  if (fraction <= 2) return 2 * scale;
  if (fraction <= 5) return 5 * scale;
  return 10 * scale;
}
function ticksFor(lo, hi, target = 6) {
  const step = niceStep((hi - lo) / target);
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let value = start; value <= hi + step * 0.5; value += step) {
    if (value >= lo - step * 0.5) ticks.push(Math.abs(value) < step * 1e-6 ? 0 : value);
  }
  return ticks;
}
function formatTick(value) {
  const abs = Math.abs(value);
  if (abs >= 10) return value.toFixed(0);
  if (abs >= 1) return value.toFixed(1).replace(/\.0$/, "");
  return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function cloneBasis(basis) {
  return { right: basis.right.slice(), up: basis.up.slice(), forward: basis.forward.slice() };
}
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function dotPoint(axis, x, y, z) { return axis[0] * x + axis[1] * y + axis[2] * z; }
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalize(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}
function subtractProjection(v, axis) {
  const d = dot(v, axis);
  return [v[0] - d * axis[0], v[1] - d * axis[1], v[2] - d * axis[2]];
}
function rotateVector(v, axis, angle) {
  const a = normalize(axis), c = Math.cos(angle), s = Math.sin(angle);
  const d = dot(a, v), axv = cross(a, v);
  return [
    v[0] * c + axv[0] * s + a[0] * d * (1 - c),
    v[1] * c + axv[1] * s + a[1] * d * (1 - c),
    v[2] * c + axv[2] * s + a[2] * d * (1 - c)
  ];
}
function orthonormalizeBasis() {
  const right = normalize(state.basis.right);
  const up = normalize(subtractProjection(state.basis.up, right));
  state.basis.right = right;
  state.basis.up = up;
  state.basis.forward = normalize(cross(right, up));
}
function rotateBasis(axis, angle) {
  state.basis.right = rotateVector(state.basis.right, axis, angle);
  state.basis.up = rotateVector(state.basis.up, axis, angle);
  state.basis.forward = rotateVector(state.basis.forward, axis, angle);
  orthonormalizeBasis();
}

function project(x, y, z) {
  const scale = baseScale * state.zoom;
  return {
    x: width / 2 + state.panX + dotPoint(state.basis.right, x, y, z) * scale,
    y: height / 2 + state.panY - dotPoint(state.basis.up, x, y, z) * scale,
    d: dotPoint(state.basis.forward, x, y, z)
  };
}

function point3(x, y, z) { return { x, y, z }; }

function drawLine3(a, b, color, lineWidth = 1, alpha = 1) {
  const pa = project(a.x, a.y, a.z), pb = project(b.x, b.y, b.z);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.stroke();
  ctx.restore();
}
function drawPoly3(points, color, alpha = 0.22, lineWidth = 1) {
  if (points.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  const first = project(points[0].x, points[0].y, points[0].z);
  ctx.moveTo(first.x, first.y);
  for (let i = 1; i < points.length; i++) {
    const p = project(points[i].x, points[i].y, points[i].z);
    ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
  ctx.restore();
}
function drawText3(text, p, dx = 0, dy = 0, color, align = "center") {
  const q = project(p.x, p.y, p.z);
  ctx.save();
  ctx.fillStyle = color || theme().frame;
  ctx.font = "12px sans-serif";
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  ctx.fillText(text, q.x + dx, q.y + dy);
  ctx.restore();
}
function drawLine2(x0, y0, x1, y1, color, lineWidth = 1) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.restore();
}
function drawTickAt(p, dx, dy) {
  const q = project(p.x, p.y, p.z);
  drawLine2(q.x, q.y, q.x + dx, q.y + dy, theme().frame, 1);
}

function drawFrameAxes() {
  const th = theme();
  const x0 = bounds.x[0], x1 = bounds.x[1];
  const y0 = bounds.y[0], y1 = bounds.y[1];
  const z0 = bounds.z[0], z1 = bounds.z[1];
  const corners = [
    point3(x0, y0, z0), point3(x1, y0, z0), point3(x1, y1, z0), point3(x0, y1, z0),
    point3(x0, y0, z1), point3(x1, y0, z1), point3(x1, y1, z1), point3(x0, y1, z1)
  ];
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  for (const [a, b] of edges) drawLine3(corners[a], corners[b], th.frameDim, 1, 0.9);
  drawLine3(point3(x0, y0, z0), point3(x1, y0, z0), th.frame, 1.5, 1);
  drawLine3(point3(x1, y0, z0), point3(x1, y1, z0), th.frame, 1.5, 1);
  drawLine3(point3(x1, y1, z0), point3(x1, y1, z1), th.frame, 1.5, 1);
  for (const x of ticksFor(x0, x1)) {
    const p = point3(x, y0, z0);
    drawTickAt(p, 0, 6);
    drawText3(formatTick(x), p, 0, 17);
  }
  for (const y of ticksFor(y0, y1)) {
    const p = point3(x1, y, z0);
    drawTickAt(p, 6, 0);
    drawText3(formatTick(y), p, 14, 0, null, "left");
  }
  for (const z of ticksFor(z0, z1)) {
    const p = point3(x1, y1, z);
    drawTickAt(p, 6, 0);
    drawText3(formatTick(z), p, 14, 0, null, "left");
  }
  drawText3("x [cm]", point3((x0 + x1) / 2, y0, z0), 0, 34, th.text);
  drawText3("y [cm]", point3(x1, (y0 + y1) / 2, z0), 34, 0, th.text, "left");
  drawText3("z [cm]", point3(x1, y1, (z0 + z1) / 2), 34, 0, th.text, "left");
}

function drawSimpleAxes() {
  const th = theme();
  for (const [label, x, y, z] of [["x", radius, 0, 0], ["y", 0, radius, 0], ["z", 0, 0, radius]]) {
    drawLine3(point3(0, 0, 0), point3(x, y, z), th.frame, 1, 1);
    drawText3(label, point3(x, y, z), 8, -8, th.text);
  }
}

function ringPoints(radius, z, segments) {
  // for polygons, radius is the inradius -> vertices at r/cos(pi/n)
  const rr = segments <= 16 ? radius / Math.cos(Math.PI / segments) : radius;
  const offset = segments <= 16 ? Math.PI / segments : 0;
  const points = [];
  for (let i = 0; i <= segments; i++) {
    const phi = offset + 2 * Math.PI * i / segments;
    points.push(point3(rr * Math.cos(phi), rr * Math.sin(phi), z));
  }
  return points;
}
function drawGeometryRing(radius, z, segments, color, alpha) {
  drawPoly3(ringPoints(radius, z, segments), color, alpha, 1);
}
function drawBarrelGeometry(g, color) {
  const segments = g.segments || 64, alpha = 0.22;
  for (const r of [g.rin, g.rout]) {
    drawGeometryRing(r, -g.zmax, segments, color, alpha);
    drawGeometryRing(r, g.zmax, segments, color, alpha);
  }
  const rr = segments <= 16 ? g.rout / Math.cos(Math.PI / segments) : g.rout;
  for (let i = 0; i < 6; i++) {
    const phi = 2 * Math.PI * i / 6;
    const x = rr * Math.cos(phi), y = rr * Math.sin(phi);
    drawLine3(point3(x, y, -g.zmax), point3(x, y, g.zmax), color, 1, alpha);
  }
}
function drawEndcapGeometry(g, color) {
  const segments = g.segments || 64, alpha = 0.22;
  for (const sign of [-1, 1]) {
    for (const z of [sign * g.zin, sign * g.zout]) {
      drawGeometryRing(g.rin, z, segments, color, alpha);
      drawGeometryRing(g.rout, z, segments, color, alpha);
    }
    for (let i = 0; i < 4; i++) {
      const phi = 2 * Math.PI * i / 4;
      const c = Math.cos(phi), s = Math.sin(phi);
      for (const z of [sign * g.zin, sign * g.zout]) {
        drawLine3(point3(g.rin * c, g.rin * s, z), point3(g.rout * c, g.rout * s, z), color, 1, alpha);
      }
    }
  }
}
function drawNozzleGeometry(g, color) {
  const segments = g.segments || 64, alpha = 0.18;
  const tanAngle = Math.tan((g.angle_deg || 10) * Math.PI / 180);
  const zLimit = Math.min(Math.abs(bounds.z[0]), Math.abs(bounds.z[1]), g.zout);
  if (zLimit <= g.zin) return;
  const zs = [g.zin, g.zin + 0.33 * (zLimit - g.zin), g.zin + 0.66 * (zLimit - g.zin), zLimit];
  for (const sign of [-1, 1]) {
    for (const zabs of zs) drawGeometryRing(zabs * tanAngle, sign * zabs, segments, color, alpha);
    for (let i = 0; i < 4; i++) {
      const phi = 2 * Math.PI * i / 4;
      const c = Math.cos(phi), s = Math.sin(phi);
      const a = zs[0] * tanAngle, b = zs[zs.length - 1] * tanAngle;
      drawLine3(point3(a * c, a * s, sign * zs[0]),
                point3(b * c, b * s, sign * zs[zs.length - 1]), color, 1, alpha);
    }
  }
}
function drawGeometry() {
  if (!state.showGeometry) return;
  for (const g of data.geometry) {
    if (g.trace >= 0 && data.traces[g.trace] && data.traces[g.trace].hidden) continue;
    const color = geomColor(g);
    if (g.type === "barrel") drawBarrelGeometry(g, color);
    else if (g.type === "endcap") drawEndcapGeometry(g, color);
    else if (g.type === "nozzle") drawNozzleGeometry(g, color);
  }
}

function render() {
  framePending = false;
  const th = theme();
  ctx.fillStyle = th.bg;
  ctx.fillRect(0, 0, width, height);
  if (state.showFrame) drawFrameAxes(); else drawSimpleAxes();
  drawGeometry();

  const filtering = state.timeAvailable && state.timeCut < state.timeMax - 1e-9;
  const bib = [], sig = [], ghosts = [];
  hitCount = 0;
  for (let ti = 0; ti < data.traces.length; ti++) {
    const t = data.traces[ti];
    if (t.hidden) continue;
    const cols = traceColors(t);
    const isSignal = t.kind === "signal";
    for (let i = 0; i < t.x.length; i++) {
      const tv = t.t[i];
      if (filtering) {
        if (finiteTime(tv)) {
          if (tv > state.timeCut) continue;
        } else {
          // timeless hits are ghosted during filtering, never silently hidden
          const p = project(t.x[i], t.y[i], t.z[i]);
          ghosts.push([p.x, p.y]);
          continue;
        }
      }
      const p = project(t.x[i], t.y[i], t.z[i]);
      (isSignal ? sig : bib).push([p.d, p.x, p.y, cols[i]]);
      hitPX[hitCount] = p.x;
      hitPY[hitCount] = p.y;
      hitTrace[hitCount] = ti;
      hitIndex[hitCount] = i;
      hitCount++;
    }
  }
  if (!state.drag && !state.playing) bib.sort((a, b) => a[0] - b[0]);

  ctx.globalAlpha = 0.16;
  ctx.fillStyle = th.missing;
  for (const g of ghosts) ctx.fillRect(g[0], g[1], 2, 2);
  ctx.globalAlpha = th.pointAlpha;
  for (const p of bib) { ctx.fillStyle = p[3]; ctx.fillRect(p[1], p[2], 2, 2); }
  ctx.globalAlpha = 0.95;
  for (const p of sig) { ctx.fillStyle = p[3]; ctx.fillRect(p[1] - 2, p[2] - 2, 4, 4); }
  ctx.globalAlpha = 1;
}

function scheduleRender() {
  if (framePending) return;
  framePending = true;
  window.requestAnimationFrame(render);
}

function resetView() {
  state.basis = cloneBasis(defaultBasis);
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  setActivePreset("view-side");
  scheduleRender();
}
function setActivePreset(id) {
  for (const pid of ["view-side", "view-face", "view-three"]) {
    el(pid).classList.toggle("active", pid === id);
  }
}
function applyPreset(name) {
  state.basis = cloneBasis(defaultBasis);
  if (name === "face") {
    state.basis = { right: [1, 0, 0], up: [0, 1, 0], forward: [0, 0, 1] };
  } else if (name === "three") {
    rotateBasis(state.basis.up.slice(), -0.6);
    rotateBasis(state.basis.right.slice(), -0.35);
  }
  state.panX = 0;
  state.panY = 0;
  setActivePreset("view-" + (name === "side" ? "side" : name));
  scheduleRender();
}

// ---- tooltip -------------------------------------------------------------
let hoverPending = false, hoverX = 0, hoverY = 0;
function showTooltip(clientX, clientY) {
  if (state.drag || !hitCount) { tooltip.style.display = "none"; return; }
  let best = -1, bestDist = 100; // 10px radius
  for (let k = 0; k < hitCount; k++) {
    const dx = hitPX[k] - clientX, dy = hitPY[k] - clientY;
    let d2 = dx * dx + dy * dy;
    if (data.traces[hitTrace[k]].kind === "signal") d2 -= 25; // prefer signal
    if (d2 < bestDist) { bestDist = d2; best = k; }
  }
  if (best < 0) { tooltip.style.display = "none"; return; }
  const t = data.traces[hitTrace[best]], i = hitIndex[best];
  const r = Math.hypot(t.x[i], t.y[i]);
  const tv = t.t[i], v = t.v[i];
  tooltip.innerHTML =
    '<div class="tt-name">' + t.name + " <span class=\"tt-dim\">[" +
    (t.kind === "signal" ? "signal" : "BIB") + "]</span></div>" +
    '<div>r = ' + r.toFixed(1) + " cm, z = " + t.z[i].toFixed(1) + " cm</div>" +
    '<div class="tt-dim">x = ' + t.x[i].toFixed(1) + ", y = " + t.y[i].toFixed(1) + " cm</div>" +
    "<div>E = " + fmtEnergy(v) + (finiteTime(tv) ? ", t = " + fmtTime(tv) + " ns" : "") + "</div>";
  tooltip.style.display = "block";
  const pad = 14;
  tooltip.style.left = Math.min(clientX + pad, width - 270) + "px";
  tooltip.style.top = Math.min(clientY + pad, height - 90) + "px";
}

// ---- theme ---------------------------------------------------------------
function applyTheme(name) {
  state.theme = name;
  document.documentElement.dataset.theme = name;
  themeButton.textContent = name === "dark" ? "Light" : "Dark";
  invalidateColors();
  buildLegend();
  updateScaleBox();
  scheduleRender();
}

// ---- events ----------------------------------------------------------------
canvas.addEventListener("mousedown", event => {
  state.drag = true;
  state.mode = event.shiftKey || event.button === 2 ? "pan" : "rotate";
  state.lastX = event.clientX;
  state.lastY = event.clientY;
  canvas.classList.add("dragging");
  tooltip.style.display = "none";
});
window.addEventListener("mouseup", () => {
  if (state.drag) { state.drag = false; scheduleRender(); }
  canvas.classList.remove("dragging");
});
window.addEventListener("mousemove", event => {
  if (state.drag) {
    const dx = event.clientX - state.lastX;
    const dy = event.clientY - state.lastY;
    state.lastX = event.clientX;
    state.lastY = event.clientY;
    if (state.mode === "pan") {
      state.panX += dx;
      state.panY += dy;
    } else {
      const rightAxis = state.basis.right.slice();
      const upAxis = state.basis.up.slice();
      rotateBasis(upAxis, dx * 0.006);
      rotateBasis(rightAxis, -dy * 0.006);
    }
    scheduleRender();
    return;
  }
  hoverX = event.clientX;
  hoverY = event.clientY;
  if (!hoverPending) {
    hoverPending = true;
    window.requestAnimationFrame(() => { hoverPending = false; showTooltip(hoverX, hoverY); });
  }
});
canvas.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
canvas.addEventListener("wheel", event => {
  event.preventDefault();
  const factor = Math.exp(-event.deltaY * 0.001);
  const next = Math.max(0.05, Math.min(40, state.zoom * factor));
  const ratio = next / state.zoom;
  const mx = event.clientX - width / 2, my = event.clientY - height / 2;
  state.panX = mx - (mx - state.panX) * ratio;
  state.panY = my - (my - state.panY) * ratio;
  state.zoom = next;
  scheduleRender();
}, { passive: false });
canvas.addEventListener("dblclick", resetView);
canvas.addEventListener("contextmenu", event => event.preventDefault());

el("reset").addEventListener("click", resetView);
el("view-side").addEventListener("click", () => applyPreset("side"));
el("view-face").addEventListener("click", () => applyPreset("face"));
el("view-three").addEventListener("click", () => applyPreset("three"));
frameToggle.addEventListener("change", () => { state.showFrame = frameToggle.checked; scheduleRender(); });
geomToggle.addEventListener("change", () => { state.showGeometry = geomToggle.checked; scheduleRender(); });
playButton.addEventListener("click", togglePlayback);
timeSlider.addEventListener("input", () => {
  setPlaying(false);
  state.timeCut = sliderToTime();
  updateTimeControls();
  scheduleRender();
});
colorModeSel.addEventListener("change", () => {
  state.colorMode = colorModeSel.value;
  invalidateColors();
  updateScaleBox();
  scheduleRender();
});
themeButton.addEventListener("click", () => applyTheme(state.theme === "dark" ? "light" : "dark"));
el("png-export").addEventListener("click", () => {
  canvas.toBlob(blob => {
    if (!blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (data.title || "event-display").replace(/[^\w.-]+/g, "_") + ".png";
    a.click();
    URL.revokeObjectURL(a.href);
  });
});
window.addEventListener("resize", resize);

// ---- init ------------------------------------------------------------------
titleEl.textContent = data.title;
document.title = data.title;
computeTimeRange();
computeEnergyRange();
applyTheme(state.theme);
updateTimeControls();
resize();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def draw_group(points_by_collection, group, prefix, outdir, theme, geometry=True):
    group_points = [
        points_by_collection[name]
        for name in GROUPS[group]
        if name in points_by_collection
    ]
    if not any(item["plotted_hits"] for item in group_points):
        return 0

    title = f"{prefix} overlay {group}"
    n_written = 0
    for suffix, x_key, y_key, xlabel, ylabel in [
        ("xy", "x", "y", "x [cm]", "y [cm]"),
        ("xz", "x", "z", "x [cm]", "z [cm]"),
        ("rz", "z", "r", "z [cm]", "r [cm]"),
    ]:
        outpath = os.path.join(outdir, f"{prefix}__overlay_{group}_{suffix}.pdf")
        if draw_projection(group_points, x_key, y_key, xlabel, ylabel, title, outpath, geometry=geometry):
            n_written += 1
    outpath = os.path.join(outdir, f"{prefix}__overlay_{group}_xyz.pdf")
    if draw_xyz(group_points, title, outpath):
        n_written += 1
    outpath = os.path.join(outdir, f"{prefix}__overlay_{group}_xyz.html")
    if write_interactive_xyz(group_points, title, outpath, theme=theme, geometry=geometry):
        n_written += 1
    return n_written


def inspect_file(path, outdir, args):
    rows = []
    n_plots = 0
    geometry = args.geometry == "envelope"
    with uproot.open(path) as root_file:
        events = root_file["events"]
        prefix = plot_prefix(path)
        for event in range(events.num_entries):
            raw = {
                name: read_collection(events, event, name, field, kind)
                for name, (field, kind) in ALL_COLLECTIONS.items()
            }
            counts = {name: item["n"] for name, item in raw.items()}
            alloc = allocate_points(counts, args.max_points,
                                    args.min_points_per_collection, args.sample)
            points_by_collection = {}
            for name, item in raw.items():
                points = build_points(item, alloc[name])
                points_by_collection[name] = points
                rows.append(summary_row(path, event, item, points))

            event_prefix = prefix if events.num_entries == 1 else f"{prefix}__event_{event}"
            for group in GROUPS:
                n_plots += draw_group(points_by_collection, group, event_prefix,
                                      outdir, args.theme, geometry=geometry)
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
        file_rows, file_plots = inspect_file(path, outdir, args)
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
    for kind in ("bib", "signal"):
        n_hits = sum(row["n_hits"] for row in rows if row["kind"] == kind)
        print(f"{kind}: n={n_hits}")


if __name__ == "__main__":
    main()
