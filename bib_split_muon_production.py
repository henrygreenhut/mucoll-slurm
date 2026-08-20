#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


COMPONENTS = (
    ("bulk-norm42", 42),
    ("decays-containing-muon-norm1-norot", 1),
)
POLARITIES = ("MUPLUS", "MUMINUS")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("items", "status"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--cycle", type=int, action="append")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != "bib-split-muon-v1":
        raise SystemExit("unexpected manifest schema")
    return manifest


def validate_cycles(manifest, cycles):
    if not cycles:
        return
    available = {
        int(record["cycle"])
        for polarity in POLARITIES
        for record in manifest["polarities"][polarity]
    }
    missing = set(cycles) - available
    if missing:
        raise SystemExit("manifest is missing cycles {}".format(sorted(missing)))


def paths(root, component, polarity, cycle):
    name = "bib_{}_{}.edm4hep.root"
    gen = root / component / "GEN" / polarity / name.format("gen", cycle)
    sim = root / component / "SIM" / polarity / name.format("sim", cycle)
    return gen, sim


def work_items(root, manifest, cycles=None):
    items = []
    selected_cycles = set(cycles) if cycles else None
    for polarity in POLARITIES:
        for record in manifest["polarities"][polarity]:
            cycle = int(record["cycle"])
            if selected_cycles is not None and cycle not in selected_cycles:
                continue
            for component, reuse in COMPONENTS:
                if component == "bulk-norm42":
                    entries = 1
                    particles = int(record["bulk_particles"]) * reuse
                else:
                    entries = len(record["muon_entries"])
                    particles = int(record["muon_component_particles"])
                    if not entries:
                        continue
                gen, sim = paths(root, component, polarity, cycle)
                weight = particles + 10000 * entries
                items.append((weight, component, polarity, cycle, entries, gen, sim))
    return items


def assignments(items, tasks):
    groups = [[] for _ in range(tasks)]
    loads = [0] * tasks
    for item in sorted(items, reverse=True):
        rank = min(range(tasks), key=loads.__getitem__)
        groups[rank].append(item)
        loads[rank] += item[0]
    return groups


def print_items(args, manifest):
    if args.tasks < 1 or not 0 <= args.rank < args.tasks:
        raise SystemExit("rank must satisfy 0 <= rank < tasks")
    root = Path(args.output_root).resolve()
    for item in assignments(work_items(root, manifest, args.cycle), args.tasks)[args.rank]:
        _, component, polarity, cycle, entries, gen, sim = item
        print("\t".join(map(str, (component, polarity, cycle, entries, gen, sim))))


def print_status(args, manifest):
    root = Path(args.output_root).resolve()
    complete = True
    items = work_items(root, manifest, args.cycle)
    for component, _ in COMPONENTS:
        print(component)
        for polarity in POLARITIES:
            expected = [item for item in items
                        if item[1] == component and item[2] == polarity]
            gen = sum(item[5].is_file() and item[5].stat().st_size > 0 for item in expected)
            sim = sum(item[6].is_file() and item[6].stat().st_size > 0 for item in expected)
            complete &= gen == len(expected) and sim == len(expected)
            print("  {} GEN {}/{} SIM {}/{}".format(polarity, gen, len(expected), sim, len(expected)))
    if args.require_complete and not complete:
        raise SystemExit(1)


def main():
    args = arguments()
    manifest = load_manifest(args.manifest)
    validate_cycles(manifest, args.cycle)
    if args.command == "items":
        print_items(args, manifest)
    else:
        print_status(args, manifest)


if __name__ == "__main__":
    main()
