#!/usr/bin/env python3
"""Audit paired SIM libraries and build immutable reco-libtest file pools."""

import argparse
import json
import os
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm1-sim", required=True,
                        help="norm1 SIM directory containing polarity folders")
    parser.add_argument("--norm42-sim", required=True,
                        help="norm42 SIM directory containing polarity folders")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fallback-muminus-from-muplus", action="store_true",
                        help="use MUPLUS files for both overlay polarities if "
                             "paired MUMINUS coverage is unavailable")
    parser.add_argument("--exclude-cycle", type=int, action="append", default=[],
                        help="cycle ID to exclude (repeatable)")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="replace existing pool symlinks and manifest")
    return parser.parse_args()


def root_files(directory):
    path = Path(directory)
    if not path.is_dir():
        return []
    return sorted(p.resolve() for p in path.glob("*.root") if p.is_file())


def assign_cycle_ids(paths):
    """Select the varying integer token in filenames, as in the GEN audit."""
    if not paths:
        return {}
    tokens = [re.findall(r"\d+", path.name) for path in paths]
    n_tokens = min(map(len, tokens))
    if not n_tokens:
        raise SystemExit("filenames contain no integer cycle token")
    best_position = None
    best_distinct = -1
    for position in range(1, n_tokens + 1):
        distinct = len({row[-position] for row in tokens})
        if distinct > best_distinct:
            best_position, best_distinct = position, distinct
    ids = [int(row[-best_position]) for row in tokens]
    if len(set(ids)) != len(ids):
        raise SystemExit(
            "cycle-id token is not unique: {} distinct for {} files in {}"
            .format(len(set(ids)), len(ids), paths[0].parent))
    return dict(zip(ids, paths))


def split_cycles(cycles):
    n_cycles = len(cycles)
    n_train = round(0.60 * n_cycles)
    n_val = round(0.15 * n_cycles)
    splits = {
        "train": cycles[:n_train],
        "val": cycles[n_train:n_train + n_val],
        "test": cycles[n_train + n_val:],
    }
    test = splits["test"]
    midpoint = len(test) // 2
    splits["test_a"] = test[:midpoint]
    splits["test_b"] = test[midpoint:]
    return splits


def link_file(source, destination, force=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        if not force:
            raise SystemExit("conflicting symlink: {}".format(destination))
        destination.unlink()
    elif destination.exists():
        raise SystemExit("refusing to replace non-symlink: {}".format(destination))
    destination.symlink_to(source)


def populate(outdir, library, split, polarity, cycles, mapping, force):
    directory = outdir / library / split / polarity
    directory.mkdir(parents=True, exist_ok=True)
    expected = set()
    for cycle in cycles:
        source = mapping[cycle]
        name = "cycle_{:06d}__{}".format(cycle, source.name)
        expected.add(name)
        link_file(source, directory / name, force)
    stale = [path for path in directory.iterdir()
             if path.is_symlink() and path.name not in expected]
    if stale and not force:
        raise SystemExit("stale pool links in {}; rerun with --force".format(directory))
    for path in stale:
        path.unlink()


def main():
    args = parse_args()
    source = {
        library: {
            polarity: assign_cycle_ids(root_files(Path(base) / polarity))
            for polarity in ("MUPLUS", "MUMINUS")
        }
        for library, base in (("norm1", args.norm1_sim),
                              ("norm42", args.norm42_sim))
    }
    for library in source:
        for polarity in source[library]:
            print("{}/{}: {} SIM files".format(
                library, polarity, len(source[library][polarity])))

    use_fallback = False
    required = [(lib, pol) for lib in ("norm1", "norm42")
                for pol in ("MUPLUS", "MUMINUS")]
    if all(source[lib][pol] for lib, pol in required):
        common = set.intersection(*(set(source[lib][pol]) for lib, pol in required))
        polarity_sources = {
            lib: {pol: source[lib][pol] for pol in ("MUPLUS", "MUMINUS")}
            for lib in ("norm1", "norm42")
        }
    elif args.fallback_muminus_from_muplus:
        if not source["norm1"]["MUPLUS"] or not source["norm42"]["MUPLUS"]:
            raise SystemExit("fallback requires both MUPLUS libraries")
        use_fallback = True
        common = set(source["norm1"]["MUPLUS"]) & set(source["norm42"]["MUPLUS"])
        polarity_sources = {
            lib: {pol: source[lib]["MUPLUS"] for pol in ("MUPLUS", "MUMINUS")}
            for lib in ("norm1", "norm42")
        }
    else:
        raise SystemExit(
            "both polarities are not available; either wait or pass "
            "--fallback-muminus-from-muplus")

    excluded = set(args.exclude_cycle)
    cycles = sorted(common - excluded)
    if not cycles:
        raise SystemExit("the selected SIM libraries have no paired cycle IDs")
    splits = split_cycles(cycles)
    print("paired cycles: {} ({} .. {})".format(len(cycles), cycles[0], cycles[-1]))
    if excluded:
        print("excluded cycles: {}".format(
            ", ".join(map(str, sorted(excluded)))))
    print("split counts: {}".format(
        ", ".join("{}={}".format(name, len(values))
                  for name, values in splits.items())))
    if use_fallback:
        print("NOTE: MUPLUS files will be used for both overlay polarities")
    if args.audit_only:
        return

    outdir = Path(args.outdir).expanduser().resolve()
    for library in ("norm1", "norm42"):
        for split, selected in splits.items():
            for polarity in ("MUPLUS", "MUMINUS"):
                populate(outdir, library, split, polarity, selected,
                         polarity_sources[library][polarity], args.force)

    # Null classes use interleaved halves of norm1 cycles within every split.
    for split, selected in splits.items():
        for null_name, null_cycles in (("null_a", selected[::2]),
                                       ("null_b", selected[1::2])):
            for polarity in ("MUPLUS", "MUMINUS"):
                populate(outdir, null_name, split, polarity, null_cycles,
                         polarity_sources["norm1"][polarity], args.force)

    manifest = {
        "norm1_sim": str(Path(args.norm1_sim).resolve()),
        "norm42_sim": str(Path(args.norm42_sim).resolve()),
        "fallback_muminus_from_muplus": use_fallback,
        "excluded_cycles": sorted(excluded),
        "n_paired_cycles": len(cycles),
        "cycles": cycles,
        "splits": {name: values for name, values in splits.items()},
        "null_partition": "alternating cycle IDs within each split",
    }
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists() and not args.force:
        old = json.loads(manifest_path.read_text())
        if old != manifest:
            raise SystemExit("manifest differs; inspect and rerun with --force")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("pools -> {}".format(outdir))


if __name__ == "__main__":
    main()
