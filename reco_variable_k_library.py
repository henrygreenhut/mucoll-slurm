#!/usr/bin/env python3
"""Build exact mother-level GEN chunks for variable-reuse RECO studies."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


REUSE_FACTORS = (1, 5, 7, 10, 21)
MOTHER_EQUIVALENTS = 29400
CHUNK_MOTHERS = 140
SOURCE_SPLIT = (0.50, 0.25, 0.25)
DATA_SEED = 1701
MAX_ROTATIONS = max(REUSE_FACTORS)
PARTICLE_FIELDS = ("px", "py", "pz", "E", "t", "vx", "vy", "vz", "pdg")
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--muplus-bank", required=True)
    prepare.add_argument("--muminus-bank", required=True)
    prepare.add_argument("--outdir", required=True)
    prepare.add_argument("--force", action="store_true")

    write = commands.add_parser("write-gen")
    write.add_argument("--bank", required=True)
    write.add_argument("--manifest-dir", required=True)
    write.add_argument("--benchmarks-dir", required=True)
    write.add_argument("--split", choices=SPLITS, required=True)
    write.add_argument("--reuse-k", type=int, choices=REUSE_FACTORS, required=True)
    write.add_argument("--output-dir", required=True)
    write.add_argument("--shard-index", type=int, default=0)
    write.add_argument("--num-shards", type=int, default=1)
    write.add_argument("--max-chunks", type=int, default=0)
    write.add_argument("--validate", action="store_true")

    inspect = commands.add_parser("inspect")
    inspect.add_argument("--manifest-dir", required=True)

    return parser.parse_args()


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode())
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def bank_index(path):
    with h5py.File(path, "r") as handle:
        schema = handle.attrs.get("schema", "")
        if isinstance(schema, bytes):
            schema = schema.decode()
        if schema != "split-mother-gen-v1":
            raise ValueError("{} has unsupported schema {!r}".format(path, schema))
        cycle_ids = handle["cycle_ids"][:].astype(np.int64)
        cycle_offsets = handle["cycle_offsets"][:].astype(np.int64)
        mother_cycle_ids = handle["mother_cycle_ids"][:].astype(np.int64)
        mother_local_ids = handle["mother_local_ids"][:].astype(np.int32)
        polarity = handle.attrs.get("polarity", "")
        if isinstance(polarity, bytes):
            polarity = polarity.decode()
    return {
        "path": str(Path(path).resolve()),
        "polarity": polarity,
        "cycle_ids": cycle_ids,
        "cycle_offsets": cycle_offsets,
        "mother_cycle_ids": mother_cycle_ids,
        "mother_local_ids": mother_local_ids,
        "source_identity_sha256": array_digest(
            cycle_ids, cycle_offsets, mother_cycle_ids, mother_local_ids
        ),
    }


def split_cycles(cycle_ids):
    order = np.random.default_rng(DATA_SEED).permutation(cycle_ids)
    n_train = int(round(len(order) * SOURCE_SPLIT[0]))
    n_val = int(round(len(order) * SOURCE_SPLIT[1]))
    return {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }


def mother_positions(cycle_ids, cycle_offsets, selected_cycles):
    positions = np.searchsorted(cycle_ids, selected_cycles)
    if np.any(positions >= len(cycle_ids)):
        raise ValueError("split contains a cycle absent from the bank")
    if not np.array_equal(cycle_ids[positions], selected_cycles):
        raise ValueError("split contains a cycle absent from the bank")
    pieces = [
        np.arange(cycle_offsets[index], cycle_offsets[index + 1], dtype=np.int64)
        for index in positions
    ]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def prepare_manifest(args):
    banks = {
        "MUPLUS": bank_index(args.muplus_bank),
        "MUMINUS": bank_index(args.muminus_bank),
    }
    if banks["MUPLUS"]["polarity"] != "MUPLUS":
        raise ValueError("the MUPLUS bank has the wrong polarity attribute")
    if banks["MUMINUS"]["polarity"] != "MUMINUS":
        raise ValueError("the MUMINUS bank has the wrong polarity attribute")
    if banks["MUPLUS"]["source_identity_sha256"] != banks["MUMINUS"]["source_identity_sha256"]:
        raise ValueError("MUPLUS and MUMINUS banks do not contain identical source mothers")

    reference = banks["MUPLUS"]
    cycle_splits = split_cycles(reference["cycle_ids"])
    chunks = {}
    discarded = {}
    for split_number, split in enumerate(SPLITS):
        positions = mother_positions(
            reference["cycle_ids"], reference["cycle_offsets"], cycle_splits[split]
        )
        rng = np.random.default_rng(np.random.SeedSequence([DATA_SEED, split_number]))
        positions = rng.permutation(positions)
        keep = len(positions) // CHUNK_MOTHERS * CHUNK_MOTHERS
        chunks[split] = positions[:keep].reshape(-1, CHUNK_MOTHERS)
        discarded[split] = int(len(positions) - keep)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    arrays_path = outdir / "chunks.npz"
    manifest_path = outdir / "manifest.json"
    if not args.force and (arrays_path.exists() or manifest_path.exists()):
        raise SystemExit("manifest output exists; use --force to replace it")

    temporary = outdir / "chunks.npz.partial"
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                **{"{}_chunks".format(split): chunks[split] for split in SPLITS},
                **{"{}_cycles".format(split): cycle_splits[split] for split in SPLITS},
            }
        )
    os.replace(temporary, arrays_path)

    manifest = {
        "schema": "reco-variable-k-chunks-v1",
        "data_seed": DATA_SEED,
        "source_split": dict(zip(SPLITS, SOURCE_SPLIT)),
        "mother_equivalents_per_event_per_polarity": MOTHER_EQUIVALENTS,
        "mothers_per_chunk": CHUNK_MOTHERS,
        "reuse_factors": list(REUSE_FACTORS),
        "files_per_event": {
            str(k): MOTHER_EQUIVALENTS // (CHUNK_MOTHERS * k)
            for k in REUSE_FACTORS
        },
        "source_identity_sha256": reference["source_identity_sha256"],
        "n_paired_cycles": int(len(reference["cycle_ids"])),
        "banks": {
            polarity: {
                "path": bank["path"],
                "polarity": bank["polarity"],
                "cycles": int(len(bank["cycle_ids"])),
                "mothers": int(len(bank["mother_cycle_ids"])),
            }
            for polarity, bank in banks.items()
        },
        "splits": {
            split: {
                "cycles": cycle_splits[split].astype(int).tolist(),
                "cycle_count": int(len(cycle_splits[split])),
                "chunk_count": int(len(chunks[split])),
                "mothers_used": int(chunks[split].size),
                "mothers_discarded": discarded[split],
            }
            for split in SPLITS
        },
        "chunk_arrays": str(arrays_path),
        "chunk_arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        "rotation": {
            "policy": "native unrotated k=1; nested first-k random angles for k>1",
            "seed_key": ["data_seed", "mother_cycle_id", "mother_local_id"],
            "changed": ["px", "py", "vx", "vy"],
            "unchanged": ["pz", "E", "t", "vz", "pdg", "mass", "charge"],
        },
    }
    temporary_json = outdir / "manifest.json.partial"
    temporary_json.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary_json, manifest_path)
    print_manifest(manifest)


def load_manifest(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    arrays_path = directory / "chunks.npz"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "reco-variable-k-chunks-v1":
        raise ValueError("unsupported variable-k manifest schema")
    digest = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    if digest != manifest["chunk_arrays_sha256"]:
        raise ValueError("chunks.npz does not match manifest.json")
    return manifest, arrays_path


def print_manifest(manifest):
    print(
        "variable-k chunks: {} mothers/chunk, {} mother-equivalents/event/polarity"
        .format(
            manifest["mothers_per_chunk"],
            manifest["mother_equivalents_per_event_per_polarity"],
        )
    )
    for split in SPLITS:
        item = manifest["splits"][split]
        print(
            "  {}: {} cycles, {} chunks, {} mothers discarded".format(
                split, item["cycle_count"], item["chunk_count"],
                item["mothers_discarded"]
            )
        )
    print(
        "  files/event: {}".format(
            ", ".join(
                "k{}={}".format(k, manifest["files_per_event"][str(k)])
                for k in REUSE_FACTORS
            )
        )
    )


def load_particle_properties(benchmarks_dir):
    bib_dir = Path(benchmarks_dir).resolve() / "generation" / "bib"
    sys.path.insert(0, str(bib_dir))
    from bib_pdgs import PDG_PROPS
    return PDG_PROPS


def mother_angles(cycle_ids, local_ids):
    angles = np.empty((len(cycle_ids), MAX_ROTATIONS), dtype=np.float64)
    for index, (cycle, local) in enumerate(zip(cycle_ids, local_ids)):
        seed = np.random.SeedSequence([DATA_SEED, int(cycle), int(local)])
        angles[index] = np.random.default_rng(seed).uniform(
            0.0, 2.0 * np.pi, size=MAX_ROTATIONS
        )
    return angles


def read_chunk(handle, mothers):
    mothers = np.sort(np.asarray(mothers, dtype=np.int64))
    offsets = handle["mother_offsets"]
    starts = offsets[mothers]
    stops = offsets[mothers + 1]
    counts = stops - starts
    raw = {
        field: np.concatenate(
            [handle["particles"][field][start:stop] for start, stop in zip(starts, stops)]
        )
        for field in PARTICLE_FIELDS
    }
    owners = np.repeat(np.arange(len(mothers), dtype=np.int64), counts)
    cycle_ids = handle["mother_cycle_ids"][mothers]
    local_ids = handle["mother_local_ids"][mothers]
    return raw, owners, cycle_ids, local_ids


def rotate_chunk(raw, owners, angles, reuse_k):
    output = {field: [] for field in PARTICLE_FIELDS}
    for rotation in range(reuse_k):
        if reuse_k == 1:
            phi = np.zeros(len(owners), dtype=np.float64)
        else:
            phi = angles[:, rotation][owners]
        cosine = np.cos(phi)
        sine = np.sin(phi)
        output["px"].append((cosine * raw["px"] - sine * raw["py"]).astype(np.float32))
        output["py"].append((sine * raw["px"] + cosine * raw["py"]).astype(np.float32))
        output["vx"].append((cosine * raw["vx"] - sine * raw["vy"]).astype(np.float32))
        output["vy"].append((sine * raw["vx"] + cosine * raw["vy"]).astype(np.float32))
        for field in ("pz", "E", "t", "vz", "pdg"):
            output[field].append(raw[field])
    return {field: np.concatenate(pieces) for field, pieces in output.items()}


def add_mass_charge(particles, properties):
    missing = sorted(set(map(int, particles["pdg"])) - set(properties))
    if missing:
        raise ValueError("PDG properties are missing for {}".format(missing))
    particles["charge"] = np.asarray(
        [properties[int(pdg)][0] for pdg in particles["pdg"]], dtype=np.float32
    )
    particles["mass"] = np.asarray(
        [properties[int(pdg)][1] for pdg in particles["pdg"]], dtype=np.float32
    )


def write_gen_file(particles, path):
    import cppyy
    import edm4hep
    import podio
    from podio.root_io import Writer

    writer = Writer(str(path))
    collection = edm4hep.MCParticleCollection()
    event = podio.Frame()
    event.put_parameter("eventNumber", "0")
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
    event.put(cppyy.gbl.std.move(collection), "MCParticles")
    writer.write_frame(event, "events")
    writer._writer.finish()


def validate_gen_file(path, expected):
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
        events = handle["events"]
        if events.num_entries != 1:
            raise ValueError("{} contains {} events".format(path, events.num_entries))
        arrays = events.arrays(list(branches.values()), library="ak")
    for field, branch in branches.items():
        observed = ak.to_numpy(ak.flatten(arrays[branch], axis=None))
        target = expected[field]
        if field == "pdg":
            equal = np.array_equal(observed, target)
        else:
            equal = np.allclose(observed, target, rtol=2e-6, atol=2e-6)
        if not equal:
            raise ValueError("{} disagrees with expected {}".format(path, field))


def write_gen(args):
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    manifest, arrays_path = load_manifest(args.manifest_dir)
    bank = bank_index(args.bank)
    if bank["source_identity_sha256"] != manifest["source_identity_sha256"]:
        raise ValueError("mother bank does not match the chunk manifest")

    with np.load(arrays_path) as arrays:
        chunks = arrays["{}_chunks".format(args.split)]
    indices = np.arange(len(chunks))[args.shard_index::args.num_shards]
    if args.max_chunks:
        indices = indices[:args.max_chunks]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    properties = load_particle_properties(args.benchmarks_dir)
    print(
        "{} {} k={}: shard {}/{} writes {} of {} chunks".format(
            bank["polarity"], args.split, args.reuse_k, args.shard_index,
            args.num_shards, len(indices), len(chunks)
        )
    )

    with h5py.File(args.bank, "r") as handle:
        for number, chunk_index in enumerate(indices, start=1):
            output = output_dir / "bib_gen_chunk_{:06d}.edm4hep.root".format(
                chunk_index
            )
            if output.is_file() and output.stat().st_size > 0:
                print("skip {}".format(output.name))
                continue
            raw, owners, cycle_ids, local_ids = read_chunk(
                handle, chunks[chunk_index]
            )
            angles = mother_angles(cycle_ids, local_ids)
            particles = rotate_chunk(raw, owners, angles, args.reuse_k)
            add_mass_charge(particles, properties)
            temporary = output.with_name(
                "." + output.name + ".partial.{}".format(os.getpid())
            )
            try:
                write_gen_file(particles, temporary)
                if args.validate:
                    validate_gen_file(temporary, particles)
                os.replace(temporary, output)
            finally:
                if temporary.exists():
                    temporary.unlink()
            if number % 25 == 0 or number == len(indices):
                print(
                    "  {}/{} chunks; last={} primaries".format(
                        number, len(indices), len(particles["pdg"])
                    ),
                    flush=True,
                )


def main():
    args = parse_args()
    if args.command == "prepare":
        prepare_manifest(args)
    elif args.command == "write-gen":
        write_gen(args)
    else:
        manifest, _ = load_manifest(args.manifest_dir)
        print_manifest(manifest)


if __name__ == "__main__":
    main()
