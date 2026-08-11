#!/usr/bin/env python3

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

import libtest_common as lc
from pfn_libtest_train import (
    UnitSampler,
    balanced_chunks,
    class_layout,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--events-per-class", type=int, default=1)
    return parser.parse_args()


def selected_events(config, samplers, epoch, events_per_class):
    rng = np.random.default_rng(config["data_seed"] * 100003 + epoch)
    definitions = [
        [
            (class_id, samplers[class_id].random_unit(rng, "train"))
            for _ in range(config["units_per_epoch"])
        ]
        for class_id in (0, 1)
    ]
    ordered = [
        definition
        for batch in balanced_chunks(definitions, config["batch_size"], rng)
        for definition in batch
    ]
    selected = []
    counts = [0, 0]
    for definition in ordered:
        class_id = definition[0]
        if counts[class_id] < events_per_class:
            selected.append(definition)
            counts[class_id] += 1
        if counts == [events_per_class, events_per_class]:
            break
    return selected


def event_particles(sampler, positions, mean, std):
    raw = sampler.store.file_arrays(positions)
    cycle_ids = np.concatenate([
        np.full(
            sampler.store.offsets[position + 1]
            - sampler.store.offsets[position],
            sampler.store.cycle_ids[position],
            dtype=np.int64,
        )
        for position in positions
    ])
    threshold = sampler.exclude_muons_above_gev
    if sampler.exclude_all_muons or threshold > 0:
        muon = np.abs(raw["pdg"]) == 13
        keep = ~muon if sampler.exclude_all_muons else ~(
            muon & (raw["E"] > threshold))
        raw = {name: values[keep] for name, values in raw.items()}
        cycle_ids = cycle_ids[keep]
    features = lc.build_features(raw, sampler.feature_set)
    return cycle_ids, (features - mean) / std


def open_output(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "wt", newline="", compresslevel=6)
    return open(path, "w", newline="")


def main():
    args = arguments()
    result_dir = Path(args.result_dir).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "config.json") as stream:
        config = json.load(stream)

    unique = lc.Store(config["norm1_store"])
    reuse = lc.Store(config["norm42_store"])
    store_a, store_b, files_a, files_b = class_layout(
        unique,
        reuse,
        config["n_files"],
        config["clone_factor"],
        config["null_test"],
        config["null_source"],
    )
    common, positions_a, positions_b = lc.common_positions(store_a, store_b)
    with np.load(result_dir / "source_split.npz") as split_file:
        train_cycles = split_file["train"]
    split_positions = np.searchsorted(common, train_cycles)
    samplers = [
        UnitSampler(
            store_a,
            {"train": positions_a[split_positions]},
            files_a,
            config["features"],
            config["exclude_muons_above_gev"],
            config.get("exclude_muons", False),
        ),
        UnitSampler(
            store_b,
            {"train": positions_b[split_positions]},
            files_b,
            config["features"],
            config["exclude_muons_above_gev"],
            config.get("exclude_muons", False),
        ),
    ]
    mean, std, _ = lc.load_norm_stats(result_dir / "norm_stats.json")
    names = lc.feature_names(config["features"])
    events = selected_events(
        config, samplers, args.epoch, args.events_per_class
    )

    manifest_path = output.with_suffix("").with_suffix(".manifest.csv")
    metadata_path = output.with_suffix("").with_suffix(".metadata.json")
    metadata = {
        "result_dir": str(result_dir),
        "epoch": args.epoch,
        "events_per_class": args.events_per_class,
        "particle_rows": "complete unpadded PFN inputs",
        "feature_values": "(transformed feature - training mean) / training std",
        "features": names,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "norm1_store": config["norm1_store"],
        "norm42_store": config["norm42_store"],
        "n_files": config["n_files"],
        "clone_factor": config["clone_factor"],
        "exclude_muons_above_gev": config["exclude_muons_above_gev"],
        "exclude_muons": config.get("exclude_muons", False),
        "data_seed": config["data_seed"],
        "batch_size": config["batch_size"],
        "units_per_epoch": config["units_per_epoch"],
    }
    with open(metadata_path, "w") as stream:
        json.dump(metadata, stream, indent=2)

    class_counts = [0, 0]
    with open_output(output) as particle_stream, open(
        manifest_path, "w", newline=""
    ) as manifest_stream:
        particle_writer = csv.writer(particle_stream)
        manifest_writer = csv.writer(manifest_stream)
        particle_writer.writerow(
            [
                "epoch",
                "training_order",
                "class",
                "class_name",
                "event_in_class",
                "particle_id",
                "source_cycle_id",
            ]
            + names
        )
        manifest_writer.writerow(
            [
                "epoch",
                "training_order",
                "class",
                "class_name",
                "event_in_class",
                "particle_count",
                "source_cycle_ids",
            ]
        )

        for training_order, (class_id, positions) in enumerate(events):
            class_name = "unrotated" if class_id == 0 else "rotated_once"
            event_in_class = class_counts[class_id]
            class_counts[class_id] += 1
            cycle_ids, features = event_particles(
                samplers[class_id], positions, mean, std
            )
            source_cycles = samplers[class_id].store.cycle_ids[positions]
            manifest_writer.writerow(
                [
                    args.epoch,
                    training_order,
                    class_id,
                    class_name,
                    event_in_class,
                    len(features),
                    ";".join(str(int(value)) for value in source_cycles),
                ]
            )
            prefix = [
                args.epoch,
                training_order,
                class_id,
                class_name,
                event_in_class,
            ]
            for particle_id, (cycle_id, feature_row) in enumerate(
                zip(cycle_ids, features)
            ):
                particle_writer.writerow(
                    prefix
                    + [particle_id, int(cycle_id)]
                    + [format(float(value), ".9g") for value in feature_row]
                )
            print(
                "exported {} event {}: {:,} particles".format(
                    class_name, event_in_class, len(features)
                ),
                flush=True,
            )

    print(output)
    print(manifest_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
