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
            s=3,
            alpha=0.35,
            linewidths=0,
            color=COLORS[item["collection"]],
            label=f"{item['collection'].replace('Overlay', '')} ({item['n_hits']})",
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
    plt.savefig(outpath)
    plt.close(fig)
    return True


def clean_xyz(item):
    x = np.asarray(item["x"], dtype=np.float64)
    y = np.asarray(item["y"], dtype=np.float64)
    z = np.asarray(item["z"], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return (
        np.round(x[mask], 3).tolist(),
        np.round(y[mask], 3).tolist(),
        np.round(z[mask], 3).tolist(),
    )


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
#legend {{ position: fixed; left: 12px; bottom: 12px; z-index: 2; max-width: 390px; padding: 9px 10px; background: rgba(255,255,255,0.9); border: 1px solid #ddd; border-radius: 6px; font-size: 12px; line-height: 1.45; }}
.row {{ display: flex; align-items: center; gap: 7px; }}
.swatch {{ width: 10px; height: 10px; border-radius: 50%; flex: 0 0 auto; }}
canvas {{ width: 100vw; height: 100vh; display: block; cursor: grab; }}
canvas.dragging {{ cursor: grabbing; }}
</style>
</head>
<body>
<div id="toolbar">
  <span id="title"></span>
  <button id="reset">Reset</button>
  <span id="help">Drag rotate · wheel zoom · shift/right-drag pan · double-click reset</span>
</div>
<canvas id="view"></canvas>
<div id="legend"></div>
<script>
const data = {data};
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const title = document.getElementById("title");
const legend = document.getElementById("legend");
const state = {{ yaw: -0.65, pitch: -0.35, zoom: 1.0, panX: 0, panY: 0, drag: false, mode: "rotate", lastX: 0, lastY: 0 }};
let width = 0;
let height = 0;
let baseScale = 1;
let radius = 1;
let framePending = false;
title.textContent = data.title;
legend.innerHTML = data.traces.map(t => `<div class="row"><span class="swatch" style="background:${{t.color}}"></span><span>${{t.name}} (${{t.total.toLocaleString()}} hits, ${{t.x.length.toLocaleString()}} plotted)</span></div>`).join("");

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
  for (const t of data.traces) {{
    for (let i = 0; i < t.x.length; i++) {{
      maxAbs = Math.max(maxAbs, Math.abs(t.x[i]), Math.abs(t.y[i]), Math.abs(t.z[i]));
    }}
  }}
  radius = maxAbs;
  baseScale = 0.42 * Math.min(width, height) / radius;
}}

function rotatePoint(x, y, z) {{
  const cy = Math.cos(state.yaw);
  const sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch);
  const sp = Math.sin(state.pitch);
  const x1 = cy * x - sy * y;
  const y1 = sy * x + cy * y;
  const y2 = cp * y1 - sp * z;
  const z2 = sp * y1 + cp * z;
  return [x1, y2, z2];
}}

function project(x, y, z) {{
  const p = rotatePoint(x, y, z);
  const scale = baseScale * state.zoom;
  return {{
    x: width / 2 + state.panX + p[0] * scale,
    y: height / 2 + state.panY - p[1] * scale,
    d: p[2]
  }};
}}

function drawAxes() {{
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

function render() {{
  framePending = false;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f7f7f7";
  ctx.fillRect(0, 0, width, height);
  drawAxes();
  const points = [];
  for (const t of data.traces) {{
    for (let i = 0; i < t.x.length; i++) {{
      const p = project(t.x[i], t.y[i], t.z[i]);
      points.push([p.d, p.x, p.y, t.color]);
    }}
  }}
  points.sort((a, b) => a[0] - b[0]);
  ctx.globalAlpha = 0.5;
  for (const p of points) {{
    ctx.fillStyle = p[3];
    ctx.fillRect(p[1], p[2], 2, 2);
  }}
  ctx.globalAlpha = 1;
}}

function scheduleRender() {{
  if (framePending) return;
  framePending = true;
  window.requestAnimationFrame(render);
}}

function resetView() {{
  state.yaw = -0.65;
  state.pitch = -0.35;
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
    state.yaw += dx * 0.006;
    state.pitch += dy * 0.006;
    state.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, state.pitch));
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
window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
"""


def write_interactive_xyz(points, title, outpath):
    selected = [item for item in points if item["plotted_hits"]]
    if not selected:
        return False

    traces = []
    for item in selected:
        x, y, z = clean_xyz(item)
        traces.append({
            "name": item["collection"].replace("Overlay", ""),
            "color": COLORS[item["collection"]],
            "total": item["n_hits"],
            "x": x,
            "y": y,
            "z": z,
        })

    payload = {"title": f"{title} interactive xyz", "traces": traces}
    with open(outpath, "w", encoding="utf-8") as handle:
        handle.write(interactive_html(payload))
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
    outpath = os.path.join(outdir, f"{prefix}__overlay_{group}_xyz.pdf")
    if draw_xyz(group_points, title, outpath):
        n_written += 1
    outpath = os.path.join(outdir, f"{prefix}__overlay_{group}_xyz.html")
    if write_interactive_xyz(group_points, title, outpath):
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
    print(f"Artifacts written: {n_plots}")
    print(f"Summary -> {outpath}")
    for group in GROUPS:
        n_hits = sum(row["n_hits"] for row in rows if row["collection"] in GROUPS[group])
        plotted = sum(row["plotted_hits"] for row in rows if row["collection"] in GROUPS[group])
        print(f"{group}: n={n_hits}, plotted={plotted}")


if __name__ == "__main__":
    main()
