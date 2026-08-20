#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


POLARITIES = ("MUPLUS", "MUMINUS")
REUSE_FACTOR = 42
ROTATION_SEED = 42042

BRANCHES = {
    "pdg": "MCParticles.PDG",
    "generator_status": "MCParticles.generatorStatus",
    "simulator_status": "MCParticles.simulatorStatus",
    "charge": "MCParticles.charge",
    "time": "MCParticles.time",
    "mass": "MCParticles.mass",
    "vertex_x": "MCParticles.vertex.x",
    "vertex_y": "MCParticles.vertex.y",
    "vertex_z": "MCParticles.vertex.z",
    "endpoint_x": "MCParticles.endpoint.x",
    "endpoint_y": "MCParticles.endpoint.y",
    "endpoint_z": "MCParticles.endpoint.z",
    "momentum_x": "MCParticles.momentum.x",
    "momentum_y": "MCParticles.momentum.y",
    "momentum_z": "MCParticles.momentum.z",
    "endpoint_momentum_x": "MCParticles.momentumAtEndpoint.x",
    "endpoint_momentum_y": "MCParticles.momentumAtEndpoint.y",
    "endpoint_momentum_z": "MCParticles.momentumAtEndpoint.z",
    "helicity": "MCParticles.helicity",
}


def arguments():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan")
    scan.add_argument("--source", required=True)
    scan.add_argument("--output", required=True)
    scan.add_argument("--workers", type=int, default=32)

    write = commands.add_parser("write-gen")
    write.add_argument("--manifest", required=True)
    write.add_argument("--output-root", required=True)
    write.add_argument("--polarity", choices=POLARITIES, required=True)
    write.add_argument("--shard-index", type=int, default=0)
    write.add_argument("--num-shards", type=int, default=1)
    write.add_argument("--cycle", type=int, action="append")

    return parser.parse_args()


def cycle_from_name(name):
    match = re.fullmatch(r"bib_gen_(\d+)\.edm4hep\.root", name)
    if not match:
        raise ValueError("unexpected GEN filename: {}".format(name))
    return int(match.group(1))


def atomic_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError("existing manifest differs: {}".format(path))
        print("validated existing {}".format(path))
        return
    temporary = path.with_name("." + path.name + ".partial")
    temporary.write_text(text)
    os.replace(temporary, path)
    print("wrote {}".format(path))


def flat_entry(array, entry):
    import awkward as ak

    return ak.to_numpy(array[entry])


def scan_file(path):
    import awkward as ak
    import uproot

    path = Path(path)
    with uproot.open(path) as root:
        if "events" not in root or "podio_metadata" not in root:
            raise ValueError("missing events or podio_metadata in {}".format(path))
        tree = root["events"]
        required = [
            BRANCHES["pdg"],
            BRANCHES["mass"],
            BRANCHES["momentum_x"],
            BRANCHES["momentum_y"],
            BRANCHES["momentum_z"],
        ]
        arrays = tree.arrays(required, library="ak")
        for relation in ("_MCParticles_parents.index", "_MCParticles_daughters.index"):
            if relation in tree:
                values = tree[relation].array(library="ak")
                if int(ak.sum(ak.num(values, axis=1))):
                    raise ValueError("nonempty MCParticle relation {} in {}".format(relation, path))

    muon_entries = []
    total_particles = 0
    muon_particles = 0
    muon_component_particles = 0
    for entry in range(len(arrays[BRANCHES["pdg"]])):
        pdg = flat_entry(arrays[BRANCHES["pdg"]], entry).astype(np.int32)
        total_particles += len(pdg)
        mask = np.abs(pdg) == 13
        number_of_muons = int(np.count_nonzero(mask))
        if not number_of_muons:
            continue
        px = flat_entry(arrays[BRANCHES["momentum_x"]], entry)[mask]
        py = flat_entry(arrays[BRANCHES["momentum_y"]], entry)[mask]
        pz = flat_entry(arrays[BRANCHES["momentum_z"]], entry)[mask]
        mass = flat_entry(arrays[BRANCHES["mass"]], entry)[mask]
        energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
        muon_entries.append({
            "entry": entry,
            "particles": len(pdg),
            "muons": number_of_muons,
            "maximum_muon_energy_GeV": float(np.max(energy)),
        })
        muon_particles += number_of_muons
        muon_component_particles += len(pdg)

    return {
        "cycle": cycle_from_name(path.name),
        "filename": path.name,
        "bytes": path.stat().st_size,
        "entries": len(arrays[BRANCHES["pdg"]]),
        "particles": total_particles,
        "bulk_entries": len(arrays[BRANCHES["pdg"]]) - len(muon_entries),
        "bulk_particles": total_particles - muon_component_particles,
        "muon_entries": muon_entries,
        "muon_particles": muon_particles,
        "muon_component_particles": muon_component_particles,
    }


def scan_polarity(source, polarity, workers):
    directory = source / polarity
    files = sorted(directory.glob("bib_gen_*.edm4hep.root"), key=lambda path: cycle_from_name(path.name))
    if not files:
        raise RuntimeError("no split-mother GEN files in {}".format(directory))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(scan_file, files))
    cycles = [record["cycle"] for record in records]
    if cycles != list(range(cycles[-1] + 1)):
        raise RuntimeError("{} cycles are not contiguous from zero".format(polarity))
    return records


def summarize(records):
    keys = (
        "entries",
        "particles",
        "bulk_entries",
        "bulk_particles",
        "muon_particles",
        "muon_component_particles",
    )
    summary = {key: sum(record[key] for record in records) for key in keys}
    summary["files"] = len(records)
    summary["muon_component_entries"] = sum(len(record["muon_entries"]) for record in records)
    if summary["entries"] != summary["bulk_entries"] + summary["muon_component_entries"]:
        raise RuntimeError("entry partition does not close")
    if summary["particles"] != summary["bulk_particles"] + summary["muon_component_particles"]:
        raise RuntimeError("particle partition does not close")
    return summary


def inventory_digest(polarities):
    compact = {
        polarity: [
            [record["cycle"], record["filename"], record["bytes"], record["entries"], record["particles"]]
            for record in records
        ]
        for polarity, records in polarities.items()
    }
    payload = json.dumps(compact, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def scan(args):
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    source = Path(args.source).resolve()
    polarities = {}
    for polarity in POLARITIES:
        print("scanning {}".format(polarity), flush=True)
        polarities[polarity] = scan_polarity(source, polarity, args.workers)
        print(json.dumps(summarize(polarities[polarity]), sort_keys=True), flush=True)

    manifest = {
        "schema": "bib-split-muon-v1",
        "source": str(source),
        "definition": "a complete mother-muon decay is selected when its GEN entry contains abs(PDG)==13",
        "components": {
            "bulk": "mother-muon decays containing no detector-bound muon",
            "decays-containing-muon": "complete mother-muon decays containing at least one detector-bound muon",
        },
        "bulk_reuse_factor": REUSE_FACTOR,
        "bulk_rotation": {
            "seed": ROTATION_SEED,
            "copy_zero_unrotated": False,
            "angle_distribution": "uniform [0, 2pi) for all copies",
            "policy": "one common azimuthal angle for every particle in a mother-copy",
            "rotated_fields": [
                "momentum.x/y",
                "vertex.x/y",
                "endpoint.x/y",
                "momentumAtEndpoint.x/y",
            ],
        },
        "polarities": polarities,
        "summaries": {polarity: summarize(records) for polarity, records in polarities.items()},
    }
    manifest["source_inventory_sha256"] = inventory_digest(polarities)
    atomic_json(args.output, manifest)


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != "bib-split-muon-v1":
        raise ValueError("unsupported manifest schema")
    if manifest.get("bulk_reuse_factor") != REUSE_FACTOR:
        raise ValueError("unexpected bulk reuse factor")
    return manifest


def read_file(path):
    import uproot

    with uproot.open(path) as root:
        return root["events"].arrays(list(BRANCHES.values()), library="ak")


def entry_values(arrays, entry):
    return {name: flat_entry(arrays[branch], entry) for name, branch in BRANCHES.items()}


def rotation_angles(polarity, cycle, entry, reuse_factor=REUSE_FACTOR):
    if reuse_factor < 1:
        raise ValueError("reuse factor must be positive")
    code = 0 if polarity == "MUPLUS" else 1
    rng = np.random.default_rng(np.random.SeedSequence([ROTATION_SEED, code, cycle, entry]))
    return rng.uniform(0.0, 2.0 * np.pi, reuse_factor)


def rotate_pair(x, y, angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return cosine * x - sine * y, sine * x + cosine * y


def put_vector(particle, getter, x, y, z):
    vector = getattr(particle, getter)()
    vector.x = float(x)
    vector.y = float(y)
    vector.z = float(z)


def append_particles(collection, values, angle=0.0):
    momentum_x, momentum_y = rotate_pair(values["momentum_x"], values["momentum_y"], angle)
    vertex_x, vertex_y = rotate_pair(values["vertex_x"], values["vertex_y"], angle)
    endpoint_x, endpoint_y = rotate_pair(values["endpoint_x"], values["endpoint_y"], angle)
    endpoint_momentum_x, endpoint_momentum_y = rotate_pair(
        values["endpoint_momentum_x"], values["endpoint_momentum_y"], angle
    )
    for index in range(len(values["pdg"])):
        particle = collection.create()
        particle.setPDG(int(values["pdg"][index]))
        particle.setGeneratorStatus(int(values["generator_status"][index]))
        particle.setSimulatorStatus(int(values["simulator_status"][index]))
        particle.setCharge(float(values["charge"][index]))
        particle.setTime(float(values["time"][index]))
        particle.setMass(float(values["mass"][index]))
        particle.setHelicity(float(values["helicity"][index]))
        put_vector(particle, "getVertex", vertex_x[index], vertex_y[index], values["vertex_z"][index])
        put_vector(particle, "getEndpoint", endpoint_x[index], endpoint_y[index], values["endpoint_z"][index])
        put_vector(particle, "getMomentum", momentum_x[index], momentum_y[index], values["momentum_z"][index])
        put_vector(
            particle,
            "getMomentumAtEndpoint",
            endpoint_momentum_x[index],
            endpoint_momentum_y[index],
            values["endpoint_momentum_z"][index],
        )


def finish_writer(writer):
    writer._writer.finish()


def write_bulk(path, arrays, bulk_entries, polarity, cycle):
    import cppyy
    import edm4hep
    import podio
    from podio.root_io import Writer

    writer = Writer(str(path))
    collection = edm4hep.MCParticleCollection()
    for entry in bulk_entries:
        values = entry_values(arrays, entry)
        for angle in rotation_angles(polarity, cycle, entry):
            append_particles(collection, values, angle)
    frame = podio.Frame()
    frame.put_parameter("eventNumber", "0")
    frame.put_parameter("sourceCycle", str(cycle))
    frame.put_parameter("component", "bulk-norm42")
    frame.put_parameter("reuseFactor", str(REUSE_FACTOR))
    frame.put(cppyy.gbl.std.move(collection), "MCParticles")
    writer.write_frame(frame, "events")
    finish_writer(writer)


def write_muon_component(path, arrays, muon_entries, cycle):
    import cppyy
    import edm4hep
    import podio
    from podio.root_io import Writer

    writer = Writer(str(path))
    for entry in muon_entries:
        collection = edm4hep.MCParticleCollection()
        append_particles(collection, entry_values(arrays, entry), 0.0)
        frame = podio.Frame()
        frame.put_parameter("eventNumber", str(entry))
        frame.put_parameter("sourceCycle", str(cycle))
        frame.put_parameter("sourceEntry", str(entry))
        frame.put_parameter("component", "decays-containing-muon-norm1-norot")
        frame.put(cppyy.gbl.std.move(collection), "MCParticles")
        writer.write_frame(frame, "events")
    finish_writer(writer)


def validate_output(path, entries, particles):
    import awkward as ak
    import uproot

    with uproot.open(path) as root:
        if "podio_metadata" not in root:
            raise ValueError("missing podio_metadata in {}".format(path))
        tree = root["events"]
        if tree.num_entries != entries:
            raise ValueError("{} has {} entries, expected {}".format(path, tree.num_entries, entries))
        observed = int(ak.sum(ak.num(tree[BRANCHES["pdg"]].array(library="ak"), axis=1)))
        if observed != particles:
            raise ValueError("{} has {} particles, expected {}".format(path, observed, particles))


def atomic_write(path, write_function, entries, particles):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".partial.{}".format(os.getpid()))
    if temporary.exists():
        temporary.unlink()
    try:
        write_function(temporary)
        validate_output(temporary, entries, particles)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_cycle(source, output_root, polarity, record):
    cycle = int(record["cycle"])
    source_path = source / polarity / record["filename"]
    arrays = read_file(source_path)
    selected = [int(item["entry"]) for item in record["muon_entries"]]
    selected_set = set(selected)
    bulk_entries = [entry for entry in range(record["entries"]) if entry not in selected_set]

    bulk_dir = output_root / "bulk-norm42" / "GEN" / polarity
    muon_dir = output_root / "decays-containing-muon-norm1-norot" / "GEN" / polarity
    name = "bib_gen_{}.edm4hep.root".format(cycle)
    bulk_path = bulk_dir / name
    muon_path = muon_dir / name

    expected_bulk_particles = int(record["bulk_particles"]) * REUSE_FACTOR
    if not bulk_path.is_file() or not bulk_path.stat().st_size:
        atomic_write(
            bulk_path,
            lambda temporary: write_bulk(temporary, arrays, bulk_entries, polarity, cycle),
            1,
            expected_bulk_particles,
        )

    if selected and (not muon_path.is_file() or not muon_path.stat().st_size):
        atomic_write(
            muon_path,
            lambda temporary: write_muon_component(temporary, arrays, selected, cycle),
            len(selected),
            int(record["muon_component_particles"]),
        )

    print(
        "{} cycle {}: bulk mothers={} particles={} muon mothers={} particles={}".format(
            polarity,
            cycle,
            len(bulk_entries),
            expected_bulk_particles,
            len(selected),
            record["muon_component_particles"],
        ),
        flush=True,
    )


def write_gen(args):
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard index must satisfy 0 <= index < num shards")
    manifest = load_manifest(args.manifest)
    source = Path(manifest["source"])
    output_root = Path(args.output_root).resolve()
    records = manifest["polarities"][args.polarity]
    if args.cycle:
        requested = set(args.cycle)
        records = [record for record in records if int(record["cycle"]) in requested]
        found = {int(record["cycle"]) for record in records}
        if found != requested:
            raise SystemExit("manifest is missing cycles {}".format(sorted(requested - found)))
    records = records[args.shard_index::args.num_shards]
    for record in records:
        write_cycle(source, output_root, args.polarity, record)


def main():
    args = arguments()
    if args.command == "scan":
        scan(args)
    else:
        write_gen(args)


if __name__ == "__main__":
    main()
