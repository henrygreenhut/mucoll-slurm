import argparse
import shlex
import sys
from pathlib import Path

import h5py
import numpy as np


PARTICLE_FIELDS = ("px", "py", "pz", "E", "t", "vx", "vy", "vz")

# One FLUKA record describes one detector-bound particle.
FLUKA_RECORD = np.dtype([
    ("fid", np.int32),
    ("fid_mo", np.int32),
    ("e_kin", np.float64),
    ("x", np.float64),
    ("y", np.float64),
    ("z", np.float64),
    ("cx", np.float64),
    ("cy", np.float64),
    ("cz", np.float64),
    ("time", np.float64),
    ("x_mu", np.float64),
    ("y_mu", np.float64),
    ("z_mu", np.float64),
])


def get_arguments():
    default_benchmarks = Path(__file__).resolve().parent.parent / "mucoll-benchmarks"
    parser = argparse.ArgumentParser()
    parser.add_argument("--fluka-dir", required=True)
    parser.add_argument("--tasklist", required=True)
    parser.add_argument("--benchmarks-dir", default=str(default_benchmarks))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--polarity", choices=("MUPLUS", "MUMINUS"), default="MUPLUS"
    )
    parser.add_argument(
        "--exclude-cycle", type=int, action="append", default=[]
    )
    return parser.parse_args()


def load_particle_tables(benchmarks_directory):
    bib_directory = Path(benchmarks_directory).resolve() / "generation" / "bib"
    if not bib_directory.is_dir():
        raise RuntimeError(f"Missing BIB configuration: {bib_directory}")

    sys.path.insert(0, str(bib_directory))
    from bib_pdgs import FLUKA_PIDS, PDG_PROPS

    return FLUKA_PIDS, PDG_PROPS


def find_cycle_files(tasklist, fluka_directory, polarity):
    output_marker = f"/GEN/{polarity}/bib_gen_"
    cycle_files = {}

    with open(tasklist) as lines:
        for line in lines:
            if output_marker not in line:
                continue

            tokens = shlex.split(line)
            inputs = [token for token in tokens if token.endswith(".dat")]
            outputs = [
                token
                for token in tokens
                if output_marker in token and token.endswith(".root")
            ]
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError(f"Cannot parse tasklist line: {line}")

            output_name = Path(outputs[0]).name
            prefix = "bib_gen_"
            suffix = ".edm4hep.root"
            if not (
                output_name.startswith(prefix) and output_name.endswith(suffix)
            ):
                raise RuntimeError(f"Unexpected GEN filename: {output_name}")

            cycle = int(output_name[len(prefix):-len(suffix)])
            input_path = Path(fluka_directory) / Path(inputs[0]).name
            if cycle in cycle_files:
                raise RuntimeError(f"Duplicate cycle: {cycle}")
            if not input_path.is_file():
                raise RuntimeError(f"Missing FLUKA source: {input_path}")
            cycle_files[cycle] = input_path

    if not cycle_files:
        raise RuntimeError(f"No {polarity} cycles found in {tasklist}")

    cycles = sorted(cycle_files)
    if cycles != list(range(cycles[-1] + 1)):
        raise RuntimeError("Cycle IDs are not contiguous from zero")

    return [(cycle, cycle_files[cycle]) for cycle in cycles]


def group_particles_by_mother(records):
    # Exact mother-decay position identifies particles from the same decay.
    positions = records[["x_mu", "y_mu", "z_mu"]]
    _, first_particle, sorted_group = np.unique(
        positions, return_index=True, return_inverse=True
    )

    groups_by_appearance = np.argsort(first_particle)
    group_number = np.empty(len(groups_by_appearance), dtype=np.int64)
    group_number[groups_by_appearance] = np.arange(len(groups_by_appearance))
    mother_number = group_number[sorted_group]

    particle_order = np.argsort(mother_number, kind="stable")
    particles_per_mother = np.bincount(mother_number).astype(np.int64)
    return particle_order, particles_per_mother


def convert_cycle(records, fluka_to_pdg, pdg_properties, invert_z):
    pdg = np.asarray(
        [fluka_to_pdg.get(int(fluka_id), 0) for fluka_id in records["fid"]],
        dtype=np.int32,
    )
    known = np.asarray([int(particle) in pdg_properties for particle in pdg])
    skipped = int(np.count_nonzero(~known))
    records = records[known]
    pdg = pdg[known]

    particle_order, mother_counts = group_particles_by_mother(records)
    records = records[particle_order]
    pdg = pdg[particle_order]

    masses = np.asarray(
        [pdg_properties[int(particle)][1] for particle in pdg],
        dtype=np.float64,
    )
    kinetic_energy = records["e_kin"]
    momentum = np.sqrt(
        kinetic_energy**2 + 2.0 * kinetic_energy * masses
    )
    z_sign = -1.0 if invert_z else 1.0

    # Positions become mm and times become ns.
    particles = {
        "pdg": pdg,
        "px": (records["cx"] * momentum).astype(np.float32),
        "py": (records["cy"] * momentum).astype(np.float32),
        "pz": (z_sign * records["cz"] * momentum).astype(np.float32),
        "E": (kinetic_energy + masses).astype(np.float32),
        "t": (records["time"] * 1e9).astype(np.float32),
        "vx": (records["x"] * 10.0).astype(np.float32),
        "vy": (records["y"] * 10.0).astype(np.float32),
        "vz": (z_sign * records["z"] * 10.0).astype(np.float32),
    }
    return mother_counts, particles, skipped


def create_particle_datasets(output):
    group = output.create_group("particles")
    datasets = {}
    for field in PARTICLE_FIELDS + ("pdg",):
        dtype = np.int32 if field == "pdg" else np.float32
        datasets[field] = group.create_dataset(
            field, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(1 << 20)
        )
    return datasets


def append_particles(datasets, particles, start):
    stop = start + len(particles["pdg"])
    for field, dataset in datasets.items():
        dataset.resize((stop,))
        dataset[start:stop] = particles[field]


def write_dataset(output, name, values, dtype):
    output.create_dataset(name, data=np.asarray(values, dtype=dtype))


def build_mother_bank(arguments):
    fluka_to_pdg, pdg_properties = load_particle_tables(
        arguments.benchmarks_dir
    )
    cycle_files = find_cycle_files(
        arguments.tasklist, arguments.fluka_dir, arguments.polarity
    )
    excluded = set(arguments.exclude_cycle)
    cycle_files = [
        (cycle, path) for cycle, path in cycle_files if cycle not in excluded
    ]

    output_path = Path(arguments.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    invert_z = arguments.polarity == "MUMINUS"

    mother_offsets = [0]
    cycle_offsets = [0]
    mother_cycle_ids = []
    mother_local_ids = []
    skipped_particles = 0

    with h5py.File(output_path, "w") as output:
        particle_datasets = create_particle_datasets(output)

        for number, (cycle, input_path) in enumerate(cycle_files, start=1):
            file_size = input_path.stat().st_size
            if file_size % FLUKA_RECORD.itemsize:
                raise RuntimeError(
                    f"{input_path} is not a whole number of "
                    f"{FLUKA_RECORD.itemsize}-byte FLUKA records"
                )

            records = np.fromfile(input_path, dtype=FLUKA_RECORD)
            mother_counts, particles, skipped = convert_cycle(
                records, fluka_to_pdg, pdg_properties, invert_z
            )

            particle_start = mother_offsets[-1]
            append_particles(particle_datasets, particles, particle_start)
            mother_offsets.extend(
                (particle_start + np.cumsum(mother_counts)).tolist()
            )

            number_of_mothers = len(mother_counts)
            mother_cycle_ids.extend([cycle] * number_of_mothers)
            mother_local_ids.extend(range(number_of_mothers))
            cycle_offsets.append(cycle_offsets[-1] + number_of_mothers)
            skipped_particles += skipped

            if number % 250 == 0 or number == len(cycle_files):
                print(
                    f"  {number}/{len(cycle_files)} cycles, "
                    f"{cycle_offsets[-1]:,} mothers, "
                    f"{mother_offsets[-1]:,} particles",
                    flush=True,
                )

        cycles = [cycle for cycle, _ in cycle_files]
        filenames = [path.name for _, path in cycle_files]

        # Mother i owns particles mother_offsets[i]:mother_offsets[i + 1].
        write_dataset(output, "mother_offsets", mother_offsets, np.int64)
        write_dataset(output, "mother_cycle_ids", mother_cycle_ids, np.int64)
        write_dataset(output, "mother_local_ids", mother_local_ids, np.int32)
        write_dataset(output, "cycle_ids", cycles, np.int64)
        write_dataset(output, "cycle_offsets", cycle_offsets, np.int64)
        output.create_dataset(
            "filenames",
            data=np.asarray(filenames, dtype=object),
            dtype=h5py.string_dtype(),
        )

        output.attrs["input_dir"] = str(Path(arguments.fluka_dir).resolve())
        output.attrs["tasklist"] = str(Path(arguments.tasklist).resolve())
        output.attrs["polarity"] = arguments.polarity
        output.attrs["schema"] = "split-mother-gen-v1"
        output.attrs["source_schema"] = "fluka-format-2-direct-v1"

    counts = np.diff(mother_offsets)
    print(
        f"done: {len(cycle_files)} cycles, {len(counts):,} mothers, "
        f"{mother_offsets[-1]:,} particles; mother particles "
        f"min/median/max = {counts.min()}/{np.median(counts):.0f}/"
        f"{counts.max()}; skipped {skipped_particles} -> {output_path}"
    )


def main():
    build_mother_bank(get_arguments())


if __name__ == "__main__":
    main()
