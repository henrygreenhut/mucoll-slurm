#!/usr/bin/env python3

import argparse
from pathlib import Path

import h5py
import numpy as np

from variable_reuse_common import MotherStore, RAW_KEYS


ANGLE_SLOTS = 42


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args()


def cycle_angles(number_of_mothers, cycle, reuse_k, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(cycle)]))
    angles = rng.uniform(
        0.0, 2.0 * np.pi, size=(number_of_mothers, ANGLE_SLOTS)
    )
    return angles[:, :reuse_k]


def create_particle_datasets(output):
    group = output.create_group("particles")
    datasets = {}
    for field in RAW_KEYS:
        dtype = np.int32 if field == "pdg" else np.float32
        datasets[field] = group.create_dataset(
            field,
            shape=(0,),
            maxshape=(None,),
            dtype=dtype,
            chunks=(1 << 20),
        )
    return datasets


def append_particles(datasets, particles, start):
    stop = start + len(particles["pdg"])
    for field, dataset in datasets.items():
        dataset.resize((stop,))
        dataset[start:stop] = particles[field]
    return stop


def build_fixed_reuse_store(input_path, output_path, reuse_k, seed=1701):
    if reuse_k < 2 or reuse_k > ANGLE_SLOTS:
        raise ValueError("reuse-k must be between 2 and {}".format(ANGLE_SLOTS))

    source = MotherStore(input_path)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError("output already exists: {}".format(output_path))

    offsets = [0]
    filenames = []
    with h5py.File(output_path, "w") as output:
        datasets = create_particle_datasets(output)

        for index, cycle in enumerate(source.cycle_ids):
            first = source.cycle_offsets[index]
            last = source.cycle_offsets[index + 1]
            mothers = np.arange(first, last, dtype=np.int64)
            angles = cycle_angles(len(mothers), cycle, reuse_k, seed)
            particles = source.rotated_mothers(mothers, angles)
            offsets.append(append_particles(datasets, particles, offsets[-1]))
            filenames.append("bib_gen_{}.edm4hep.root".format(int(cycle)))

            if (index + 1) % 250 == 0 or index + 1 == source.n_cycles:
                print(
                    "  {}/{} cycles, {:,} stored particles".format(
                        index + 1, source.n_cycles, offsets[-1]
                    ),
                    flush=True,
                )

        output.create_dataset(
            "offsets", data=np.asarray(offsets, dtype=np.int64)
        )
        output.create_dataset(
            "cycle_ids", data=np.asarray(source.cycle_ids, dtype=np.int64)
        )
        output.create_dataset(
            "filenames",
            data=np.asarray(filenames, dtype=object),
            dtype=h5py.string_dtype(),
        )
        output.attrs["schema"] = "fixed-mother-reuse-gen-v1"
        output.attrs["source"] = str(Path(input_path).resolve())
        output.attrs["reuse_k"] = reuse_k
        output.attrs["rotation_seed"] = seed
        output.attrs["angle_slots"] = ANGLE_SLOTS
        output.attrs["rotation_policy"] = "fixed uniform random z rotations"

    expected_particles = reuse_k * len(source.raw["pdg"])
    if offsets[-1] != expected_particles:
        raise RuntimeError(
            "stored {:,} particles; expected {:,}".format(
                offsets[-1], expected_particles
            )
        )

    print(
        "done: k={} | {} cycles | {:,} particles -> {}".format(
            reuse_k, source.n_cycles, offsets[-1], output_path
        )
    )


def main():
    args = get_arguments()
    build_fixed_reuse_store(
        args.input, args.output, args.reuse_k, args.seed
    )


if __name__ == "__main__":
    main()
