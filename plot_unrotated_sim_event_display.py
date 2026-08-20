#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import plot_overlay_spatial as display


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_directory")
    parser.add_argument("--n-files", type=int, default=420)
    parser.add_argument("--energy-cut", type=float, default=5.0)
    parser.add_argument("--max-points-per-layer", type=int, default=5000)
    parser.add_argument("--output", default="plots/sim_n420_unrotated_event_display")
    return parser.parse_args()


def cycle_id(path):
    match = re.search(r"bib_sim_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def input_files(directory, count):
    paths = sorted(Path(directory).glob("bib_sim_*.edm4hep.root"), key=cycle_id)
    if len(paths) < count:
        raise SystemExit(f"found only {len(paths)} SIM files; need {count}")
    return paths[:count]


def detector_color(collection):
    overlay_name = f"Overlay{collection}"
    return display.COLORS.get(overlay_name, "#555555")


def detector_envelope(collection):
    return display.ENVELOPE_OVERRIDES.get(f"Overlay{collection}")


def layer(name, role, color, arrays, maximum, envelope=None):
    item = display.category_points(name, role, color, arrays, maximum, None)
    item["envelope"] = dict(envelope) if envelope else None
    return item


def main():
    args = arguments()
    display.load_libraries()
    files = input_files(args.sim_directory, args.n_files)
    collections = display.SIM_TRACKER_COLLECTIONS + display.SIM_CALO_COLLECTIONS
    detector_hits = {name: [] for name in collections}
    soft_muon_hits = []
    energetic_muon_hits = []
    particle_count = 0
    muon_count = 0
    energetic_muon_count = 0

    display.classify_hits.energy_cut = args.energy_cut
    for number, path in enumerate(files, start=1):
        with display.uproot.open(path) as root_file:
            events = root_file["events"]
            particles = display.mc_particle_arrays(events, 0)
            muon_energy = display.muon_ancestor_energies(particles)
            pdg = display.np.asarray(particles["pdg"], dtype=display.np.int64)
            particle_count += len(pdg)
            muon_count += int(display.np.sum(display.np.abs(pdg) == 13))
            energetic_muon_count += int(
                display.np.sum((display.np.abs(pdg) == 13) & (muon_energy > args.energy_cut))
            )

            for collection in collections:
                if display.branch_name(events, collection, "position.x") is None:
                    continue
                x, y, z, time, categories = display.classify_hits(
                    events, collection, 0, muon_energy
                )
                detector_hits[collection].append((x, y, z, time))
                soft = categories == 1
                energetic = categories == 2
                if display.np.any(soft):
                    soft_muon_hits.append((x[soft], y[soft], z[soft], time[soft]))
                if display.np.any(energetic):
                    energetic_muon_hits.append(
                        (x[energetic], y[energetic], z[energetic], time[energetic])
                    )
        if number % 50 == 0 or number == len(files):
            print(f"read {number}/{len(files)} SIM files", flush=True)

    layers = []
    for collection in collections:
        if not detector_hits[collection]:
            continue
        layers.append(
            layer(
                display.COLLECTION_LABELS.get(collection, collection),
                "bib",
                detector_color(collection),
                detector_hits[collection],
                args.max_points_per_layer,
                detector_envelope(collection),
            )
        )
    layers.extend([
        layer(
            rf"Muon-induced hits ($E_\mu\leq{args.energy_cut:g}$ GeV)",
            "signal",
            "#2b8cbe",
            soft_muon_hits,
            args.max_points_per_layer,
        ),
        layer(
            rf"Muon-induced hits ($E_\mu>{args.energy_cut:g}$ GeV)",
            "signal",
            "#d7301f",
            energetic_muon_hits,
            args.max_points_per_layer,
        ),
    ])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    title = f"Native unrotated SIM BIB, N={args.n_files}"
    pdf = output / f"sim_n{args.n_files}_unrotated_bib_xyz.pdf"
    html = output / f"sim_n{args.n_files}_unrotated_bib_xyz.html"
    display.draw_xyz(layers, title, pdf)
    display.write_interactive_xyz(layers, title, html, geometry=True)

    print(f"source files: {len(files)}")
    print(f"MC particles: {particle_count:,}")
    print(f"muons: {muon_count:,}")
    print(f"muons above {args.energy_cut:g} GeV: {energetic_muon_count:,}")
    print(f"PDF -> {pdf}")
    print(f"HTML -> {html}")


if __name__ == "__main__":
    main()
