import argparse
import json
from pathlib import Path

import numpy as np


SEED = 1701
CYCLES = 420
STEM = "photon_entry_xy_n420"


def extract(bank_path, output):
    import h5py

    with h5py.File(bank_path, "r") as bank:
        cycle_ids = bank["cycle_ids"][:]
        if len(cycle_ids) < CYCLES:
            raise ValueError("The source bank has fewer than 420 cycles")
        chosen = np.sort(np.random.default_rng(SEED).choice(len(cycle_ids), CYCLES,
                                                           replace=False))
        offsets = bank["mother_offsets"][:][bank["cycle_offsets"][:]]
        particles = bank["particles"]
        photon_indices = np.flatnonzero(particles["pdg"][:] == 22)
        photon_cycles = np.searchsorted(offsets, photon_indices, side="right") - 1
        selected = photon_indices[np.isin(photon_cycles, chosen)]
        x = particles["vx"][:][selected].astype(float) / 10.0
        y = particles["vy"][:][selected].astype(float) / 10.0
        selected_ids = cycle_ids[chosen]
    if not len(x) or not np.all(np.isfinite(x) & np.isfinite(y)):
        raise ValueError("No photons or invalid photon entry coordinates")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / f"{STEM}.npz", x_cm=x, y_cm=y,
                         cycle_ids=selected_ids)
    nonzero = (x != 0) | (y != 0)
    metadata = {
        "bank": str(bank_path), "polarity": "MUPLUS", "seed": SEED,
        "cycles": CYCLES, "cycle_ids": selected_ids.tolist(),
        "construction": "420 source cycles sampled without replacement; all their photons",
        "coordinate": "GEN vx, vy: FLUKA detector-entry positions, converted from mm to cm",
        "photons": len(x), "rotation": False, "energy_cut": None,
        "z_cut": None, "downsampling": False,
        "position_c2": float(np.cos(2 * np.arctan2(y[nonzero], x[nonzero])).mean()),
        "position_c2_excluded_on_axis": int(np.count_nonzero(~nonzero)),
        "zoom_limits_cm": [-5, 5],
        "zoom_photons": int(np.count_nonzero((np.abs(x) <= 5) & (np.abs(y) <= 5))),
        "display": "Actual XY entry-density map, logarithmic photon counts; not trajectories or momentum directions",
    }
    (output / f"{STEM}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{len(x):,} photons from {CYCLES} cycles", flush=True)


def plot(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_overlay_spatial as display

    display.np = np
    display.plt = plt
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "savefig.dpi": 200})
    with np.load(output / f"{STEM}.npz") as data:
        x, y = data["x_cm"], data["y_cm"]
    layer = {"name": "BIB photons", "role": "bib", "color": "#0072B2",
             "x": x, "y": y, "n_hits": len(x), "plotted_hits": len(x)}
    extent = float(max(np.max(np.abs(x)), np.max(np.abs(y)))) * 1.05
    for suffix, limits in (("", (-extent, extent)), ("_zoom", (-5, 5))):
        visible = (x >= limits[0]) & (x <= limits[1]) & (y >= limits[0]) & (y <= limits[1])
        title = f"BIB photon entry positions\nN=420 · {np.count_nonzero(visible):,} photons"
        for extension in ("pdf", "png"):
            display.draw_projection([layer], "x", "y", "x [cm]", "y [cm]", title,
                                    output / f"{STEM}{suffix}.{extension}", geometry=False,
                                    limits=limits, compact=True, density=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.no_plots and args.plot_only:
        parser.error("--no-plots and --plot-only cannot be combined")
    if not args.plot_only:
        if args.bank is None:
            parser.error("--bank is required for extraction")
        extract(args.bank, args.output)
    if not args.no_plots:
        plot(args.output)


if __name__ == "__main__":
    main()
