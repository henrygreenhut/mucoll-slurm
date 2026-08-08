#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path


REUSE_FACTORS = (7, 21)
POLARITIES = ("MUPLUS", "MUMINUS")
SPLITS = ("train", "val", "test")


def arguments():
    default_base = Path(os.environ.get("PSCRATCH", "/tmp")) / "mucoll" / "reco_cycle_k"
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("items", "status"))
    parser.add_argument("--base", default=str(default_base))
    parser.add_argument("--manifest")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def manifest_path(args):
    return Path(args.manifest or Path(args.base) / "pools" / "manifest.json").resolve()


def load_manifest(args):
    path = manifest_path(args)
    if not path.is_file():
        raise SystemExit("missing {}".format(path))
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "reco-cycle-k-v1":
        raise SystemExit("unexpected manifest schema in {}".format(path))
    return manifest


def cycles(manifest, split):
    value = manifest["splits"][split]
    return value["cycles"] if isinstance(value, dict) else value


def paths(base, k, split, polarity, cycle):
    name = "cycle_{:06d}.edm4hep.root".format(cycle)
    gen = base / "GEN" / "k{}".format(k) / split / polarity / ("bib_gen_" + name)
    sim = base / "pools" / "k{}".format(k) / split / polarity / ("bib_sim_" + name)
    return gen, sim


def work_items(base, manifest):
    for split in SPLITS:
        for polarity in POLARITIES:
            for cycle in cycles(manifest, split):
                for k in REUSE_FACTORS:
                    gen, sim = paths(base, k, split, polarity, cycle)
                    yield k, split, polarity, int(cycle), gen, sim


def print_items(args, manifest):
    if args.tasks < 1 or not 0 <= args.rank < args.tasks:
        raise SystemExit("rank must satisfy 0 <= rank < tasks")
    base = Path(args.base).resolve()
    for index, item in enumerate(work_items(base, manifest)):
        if index % args.tasks == args.rank:
            print("\t".join(map(str, item)))


def existing_cycles(directory, prefix):
    if not directory.is_dir():
        return set()
    start = prefix + "_cycle_"
    end = ".edm4hep.root"
    output = set()
    for path in directory.glob("*.root"):
        if path.name.startswith(start) and path.name.endswith(end) and path.stat().st_size:
            output.add(int(path.name[len(start):-len(end)]))
    return output


def print_status(args, manifest):
    base = Path(args.base).resolve()
    complete = True
    for k in REUSE_FACTORS:
        print("K={}".format(k))
        for split in SPLITS:
            expected = set(map(int, cycles(manifest, split)))
            for polarity in POLARITIES:
                gen_dir = base / "GEN" / "k{}".format(k) / split / polarity
                sim_dir = base / "pools" / "k{}".format(k) / split / polarity
                gen = existing_cycles(gen_dir, "bib_gen")
                sim = existing_cycles(sim_dir, "bib_sim")
                okay = gen == expected and sim == expected
                complete &= okay
                print(
                    "  {:5s} {:7s} GEN {:4d}/{:4d}  SIM {:4d}/{:4d}".format(
                        split, polarity, len(gen & expected), len(expected),
                        len(sim & expected), len(expected)
                    )
                )
                if gen - expected or sim - expected:
                    print(
                        "    unexpected: GEN={} SIM={}".format(
                            len(gen - expected), len(sim - expected)
                        )
                    )
    if args.require_complete and not complete:
        raise SystemExit(1)


def main():
    args = arguments()
    manifest = load_manifest(args)
    if args.command == "items":
        print_items(args, manifest)
    else:
        print_status(args, manifest)


if __name__ == "__main__":
    main()
