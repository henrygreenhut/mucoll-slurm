#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np


FIELDS = ("px", "py", "pz", "E", "t", "vx", "vy", "vz", "pdg")
SPLITS = ("train", "val", "test")
REUSE_FACTORS = (7, 21)
ROTATION_SEED = 1701
ANGLE_SLOTS = 42
SOURCE_CYCLE_COUNT = 6666
SOURCE_SPLIT_SEED = 12345
SOURCE_EXCLUDED_CYCLES = (
    100, 101, 105, 106, 118, 143, 144, 146, 147, 148, 149, 6291
)
SOURCE_SPLIT_FRACTIONS = {"train": 0.5, "val": 0.25, "test": 0.25}
SOURCE_SPLIT_SHA256 = "51a3cdc0b3453ee6079fb06f5c7b823c81e4e5f5b2f4c5c6a549c89159653311"


def arguments():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("make-source-split")
    source.add_argument("--output", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-pool-manifest", required=True)
    prepare.add_argument("--muplus-bank", required=True)
    prepare.add_argument("--muminus-bank", required=True)
    prepare.add_argument("--output", required=True)

    write = commands.add_parser("write-gen")
    write.add_argument("--bank", required=True)
    write.add_argument("--manifest", required=True)
    write.add_argument("--benchmarks-dir", required=True)
    write.add_argument("--split", choices=SPLITS, required=True)
    write.add_argument("--reuse-k", type=int, choices=REUSE_FACTORS, required=True)
    write.add_argument("--output-dir", required=True)
    write.add_argument("--shard-index", type=int, default=0)
    write.add_argument("--num-shards", type=int, default=1)
    write.add_argument("--max-cycles", type=int, default=0)
    write.add_argument("--validate", action="store_true")

    inspect = commands.add_parser("inspect")
    inspect.add_argument("--manifest", required=True)
    return parser.parse_args()


def digest_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode())
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_digest(splits):
    payload = json.dumps(
        splits, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def frozen_source_split():
    excluded = set(SOURCE_EXCLUDED_CYCLES)
    cycles = [cycle for cycle in range(SOURCE_CYCLE_COUNT) if cycle not in excluded]
    shuffled = list(cycles)
    random.Random(SOURCE_SPLIT_SEED).shuffle(shuffled)
    n_val = round(SOURCE_SPLIT_FRACTIONS["val"] * len(shuffled))
    n_test = round(SOURCE_SPLIT_FRACTIONS["test"] * len(shuffled))
    n_train = len(shuffled) - n_val - n_test
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }
    digest = split_digest(splits)
    if digest != SOURCE_SPLIT_SHA256:
        raise RuntimeError("frozen source split digest changed: {}".format(digest))
    return {
        "schema": "reco-cycle-source-split-v1",
        "construction": "exact source-cycle split used by the OSCAR val25 K=1 baseline",
        "excluded_cycles": list(SOURCE_EXCLUDED_CYCLES),
        "n_paired_cycles": len(cycles),
        "cycles": cycles,
        "split_seed": SOURCE_SPLIT_SEED,
        "split_fractions": SOURCE_SPLIT_FRACTIONS,
        "split_sha256": digest,
        "splits": splits,
    }


def write_immutable_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise SystemExit("existing file differs: {}".format(path))
        print("validated existing {}".format(path))
        return
    temporary = path.with_name("." + path.name + ".partial")
    temporary.write_text(text)
    os.replace(temporary, path)
    print("wrote {}".format(path))


def bank_index(path):
    path = Path(path).resolve()
    with h5py.File(path, "r") as handle:
        schema = handle.attrs.get("schema", "")
        polarity = handle.attrs.get("polarity", "")
        if isinstance(schema, bytes):
            schema = schema.decode()
        if isinstance(polarity, bytes):
            polarity = polarity.decode()
        if schema != "split-mother-gen-v1":
            raise ValueError("{} has schema {!r}".format(path, schema))
        cycle_ids = handle["cycle_ids"][:].astype(np.int64)
        cycle_offsets = handle["cycle_offsets"][:].astype(np.int64)
        mother_cycle_ids = handle["mother_cycle_ids"][:].astype(np.int64)
        mother_local_ids = handle["mother_local_ids"][:].astype(np.int32)
    return {
        "path": str(path),
        "polarity": polarity,
        "cycle_ids": cycle_ids,
        "cycle_offsets": cycle_offsets,
        "mother_cycle_ids": mother_cycle_ids,
        "mother_local_ids": mother_local_ids,
        "identity_sha256": digest_arrays(
            cycle_ids, cycle_offsets, mother_cycle_ids, mother_local_ids
        ),
    }


def split_cycles(source, split):
    item = source["splits"][split]
    if isinstance(item, dict):
        item = item["cycles"]
    return np.asarray(item, dtype=np.int64)


def validate_cycles(bank, cycles):
    positions = np.searchsorted(bank["cycle_ids"], cycles)
    valid = positions < len(bank["cycle_ids"])
    valid[valid] &= bank["cycle_ids"][positions[valid]] == cycles[valid]
    if not np.all(valid):
        raise ValueError(
            "bank is missing cycles {}".format(cycles[~valid].astype(int).tolist())
        )


def make_manifest(source_path, plus_path, minus_path):
    source_path = Path(source_path).resolve()
    source = json.loads(source_path.read_text())
    banks = {
        "MUPLUS": bank_index(plus_path),
        "MUMINUS": bank_index(minus_path),
    }
    for polarity, bank in banks.items():
        if bank["polarity"] != polarity:
            raise ValueError("{} bank reports polarity {!r}".format(polarity, bank["polarity"]))
    if banks["MUPLUS"]["identity_sha256"] != banks["MUMINUS"]["identity_sha256"]:
        raise ValueError("MUPLUS and MUMINUS mother identities differ")

    splits = {}
    all_cycles = []
    for split in SPLITS:
        cycles = split_cycles(source, split)
        validate_cycles(banks["MUPLUS"], cycles)
        validate_cycles(banks["MUMINUS"], cycles)
        splits[split] = {"cycles": cycles.astype(int).tolist(), "count": int(len(cycles))}
        all_cycles.extend(cycles.tolist())
    if len(all_cycles) != len(set(all_cycles)):
        raise ValueError("source-cycle partitions overlap")

    return {
        "schema": "reco-cycle-k-v1",
        "construction": "one synthetic GEN and SIM file per original FLUKA source cycle",
        "n_file_equivalents_per_pseudocrossing_per_polarity": 420,
        "reuse_factors": list(REUSE_FACTORS),
        "files_per_pseudocrossing_per_polarity": {
            str(k): 420 // k for k in REUSE_FACTORS
        },
        "rotation": {
            "seed": ROTATION_SEED,
            "angle_slots": ANGLE_SLOTS,
            "policy": "first K fixed uniform azimuthal angles per mother within each cycle",
            "changed": ["px", "py", "vx", "vy"],
            "unchanged": ["pz", "E", "t", "vz", "pdg", "mass", "charge"],
        },
        "source_pool_manifest": str(source_path),
        "source_pool_manifest_sha256": file_digest(source_path),
        "source_split_seed": source.get("split_seed"),
        "source_split_fractions": source.get("split_fractions"),
        "source_split_sha256": source.get("split_sha256", split_digest({
            split: splits[split]["cycles"] for split in SPLITS
        })),
        "excluded_cycles": source.get("excluded_cycles", []),
        "n_paired_cycles": source.get("n_paired_cycles", len(all_cycles)),
        "source_identity_sha256": banks["MUPLUS"]["identity_sha256"],
        "banks": {
            polarity: {
                "path": bank["path"],
                "cycles": int(len(bank["cycle_ids"])),
                "mothers": int(len(bank["mother_cycle_ids"])),
            }
            for polarity, bank in banks.items()
        },
        "splits": splits,
    }


def prepare(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(
        args.source_pool_manifest, args.muplus_bank, args.muminus_bank
    )
    write_immutable_json(output, manifest)
    print_manifest(manifest)


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != "reco-cycle-k-v1":
        raise ValueError("unsupported manifest schema")
    return manifest


def print_manifest(manifest):
    print("cycle-preserving variable-K library")
    print("  paired cycles: {}".format(manifest["n_paired_cycles"]))
    for split in SPLITS:
        print("  {}: {} cycles".format(split, manifest["splits"][split]["count"]))
    print(
        "  overlay per polarity: {}".format(
            ", ".join(
                "k{}={}".format(k, manifest["files_per_pseudocrossing_per_polarity"][str(k)])
                for k in REUSE_FACTORS
            )
        )
    )


def particle_properties(benchmarks_dir):
    directory = Path(benchmarks_dir).resolve() / "generation" / "bib"
    sys.path.insert(0, str(directory))
    from bib_pdgs import PDG_PROPS
    return PDG_PROPS


def cycle_mothers(handle, cycle):
    cycle_ids = handle["cycle_ids"][:]
    position = int(np.searchsorted(cycle_ids, cycle))
    if position >= len(cycle_ids) or int(cycle_ids[position]) != int(cycle):
        raise KeyError("cycle {} is absent".format(cycle))
    first = int(handle["cycle_offsets"][position])
    last = int(handle["cycle_offsets"][position + 1])
    mothers = np.arange(first, last, dtype=np.int64)
    local_ids = handle["mother_local_ids"][mothers].astype(np.int64)
    if not np.array_equal(local_ids, np.arange(len(mothers), dtype=np.int64)):
        raise ValueError("cycle {} mother order is not local-ID order".format(cycle))
    return mothers


def read_mothers(handle, mothers):
    offsets = handle["mother_offsets"]
    starts = offsets[mothers]
    stops = offsets[mothers + 1]
    counts = stops - starts
    particles = {
        field: np.concatenate(
            [handle["particles"][field][start:stop] for start, stop in zip(starts, stops)]
        )
        for field in FIELDS
    }
    owners = np.repeat(np.arange(len(mothers), dtype=np.int64), counts)
    return particles, owners


def cycle_angles(cycle, number_of_mothers):
    rng = np.random.default_rng(np.random.SeedSequence([ROTATION_SEED, int(cycle)]))
    return rng.uniform(0.0, 2.0 * np.pi, size=(number_of_mothers, ANGLE_SLOTS))


def rotate(particles, owners, angles, reuse_k):
    output = {field: [] for field in FIELDS}
    for rotation in range(reuse_k):
        phi = angles[:, rotation][owners]
        cosine = np.cos(phi)
        sine = np.sin(phi)
        output["px"].append((cosine * particles["px"] - sine * particles["py"]).astype(np.float32))
        output["py"].append((sine * particles["px"] + cosine * particles["py"]).astype(np.float32))
        output["vx"].append((cosine * particles["vx"] - sine * particles["vy"]).astype(np.float32))
        output["vy"].append((sine * particles["vx"] + cosine * particles["vy"]).astype(np.float32))
        for field in ("pz", "E", "t", "vz", "pdg"):
            output[field].append(particles[field])
    return {field: np.concatenate(parts) for field, parts in output.items()}


def add_mass_charge(particles, properties):
    missing = sorted(set(map(int, particles["pdg"])) - set(properties))
    if missing:
        raise ValueError("missing PDG properties for {}".format(missing))
    particles["charge"] = np.asarray(
        [properties[int(pdg)][0] for pdg in particles["pdg"]], dtype=np.float32
    )
    particles["mass"] = np.asarray(
        [properties[int(pdg)][1] for pdg in particles["pdg"]], dtype=np.float32
    )


def write_root(particles, path):
    import cppyy
    import edm4hep
    import podio
    from podio.root_io import Writer

    writer = Writer(str(path))
    collection = edm4hep.MCParticleCollection()
    record = podio.Frame()
    record.put_parameter("eventNumber", "0")
    for index in range(len(particles["pdg"])):
        particle = collection.create()
        particle.setPDG(int(particles["pdg"][index]))
        particle.setGeneratorStatus(1)
        particle.setCharge(float(particles["charge"][index]))
        particle.setMass(float(particles["mass"][index]))
        particle.setTime(float(particles["t"][index]))
        particle.getMomentum().x = float(particles["px"][index])
        particle.getMomentum().y = float(particles["py"][index])
        particle.getMomentum().z = float(particles["pz"][index])
        particle.getVertex().x = float(particles["vx"][index])
        particle.getVertex().y = float(particles["vy"][index])
        particle.getVertex().z = float(particles["vz"][index])
    record.put(cppyy.gbl.std.move(collection), "MCParticles")
    writer.write_frame(record, "events")
    writer._writer.finish()


def validate_root(path, expected):
    import awkward as ak
    import uproot

    branches = {
        "pdg": "MCParticles.PDG",
        "px": "MCParticles.momentum.x",
        "py": "MCParticles.momentum.y",
        "pz": "MCParticles.momentum.z",
        "t": "MCParticles.time",
        "vx": "MCParticles.vertex.x",
        "vy": "MCParticles.vertex.y",
        "vz": "MCParticles.vertex.z",
        "mass": "MCParticles.mass",
        "charge": "MCParticles.charge",
    }
    with uproot.open(path) as handle:
        tree = handle["events"]
        if tree.num_entries != 1:
            raise ValueError("{} does not contain one GEN record".format(path))
        arrays = tree.arrays(list(branches.values()), library="ak")
    for field, branch in branches.items():
        observed = ak.to_numpy(ak.flatten(arrays[branch], axis=None))
        if field == "pdg":
            equal = np.array_equal(observed, expected[field])
        else:
            equal = np.allclose(observed, expected[field], rtol=2e-6, atol=2e-6)
        if not equal:
            raise ValueError("{} disagrees in {}".format(path, field))


def write_gen(args):
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard index")
    manifest = load_manifest(args.manifest)
    bank = bank_index(args.bank)
    if bank["identity_sha256"] != manifest["source_identity_sha256"]:
        raise ValueError("mother bank does not match manifest")
    cycles = np.asarray(manifest["splits"][args.split]["cycles"], dtype=np.int64)
    cycles = cycles[args.shard_index::args.num_shards]
    if args.max_cycles:
        cycles = cycles[:args.max_cycles]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    properties = particle_properties(args.benchmarks_dir)
    print(
        "{} {} k={}: shard {}/{} writes {} cycles".format(
            bank["polarity"], args.split, args.reuse_k,
            args.shard_index, args.num_shards, len(cycles)
        )
    )

    with h5py.File(args.bank, "r") as handle:
        for number, cycle in enumerate(cycles, start=1):
            output = output_dir / "bib_gen_cycle_{:06d}.edm4hep.root".format(int(cycle))
            if output.is_file() and output.stat().st_size > 0:
                continue
            mothers = cycle_mothers(handle, int(cycle))
            particles, owners = read_mothers(handle, mothers)
            angles = cycle_angles(int(cycle), len(mothers))
            particles = rotate(particles, owners, angles, args.reuse_k)
            add_mass_charge(particles, properties)
            temporary = output.with_name("." + output.name + ".partial.{}".format(os.getpid()))
            try:
                write_root(particles, temporary)
                if args.validate:
                    validate_root(temporary, particles)
                os.replace(temporary, output)
            finally:
                if temporary.exists():
                    temporary.unlink()
            if number % 25 == 0 or number == len(cycles):
                print(
                    "  {}/{} cycles; cycle={} mothers={} primaries={}".format(
                        number, len(cycles), int(cycle), len(mothers), len(particles["pdg"])
                    ),
                    flush=True,
                )


def main():
    args = arguments()
    if args.command == "make-source-split":
        source = frozen_source_split()
        write_immutable_json(args.output, source)
        print("source cycles: {}".format(source["n_paired_cycles"]))
        print("split sha256: {}".format(source["split_sha256"]))
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "write-gen":
        write_gen(args)
    else:
        print_manifest(load_manifest(args.manifest))


if __name__ == "__main__":
    main()
