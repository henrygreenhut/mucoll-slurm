#!/usr/bin/env python3
"""Drive GenBIB TabDDPM sampling over prepared COUNT conditions.

This is a self-contained vendored copy of the GenBIB-ML sampling logic
(derived from github.com/ShiyuP1/GenBIB-ML ``sample.py`` @ a483be7), plus an
outer loop over many events. Vendoring keeps the validated inference logic in
one file we own and pins it for reproducibility; the actual generative model
code (``paper1-inference``) stays an external dependency loaded at runtime by
``load_paper1`` from the model root.

For each prepared event it writes one ``tabddpm_<SHORT>_samples.npy`` per
collection in the 9-column ``logE, time, r, phi, z, side, layer, module,
sensor`` layout that ``assign_actual_cellid.py --input-format 9col`` expects.
One generated hit is produced per requested condition, so per-sensor occupancy
is preserved exactly before geometric CellID assignment.

Runs in the GenBIB conda env (torch cu111 + paper1-inference), on GPU.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

# The collection short -> (system_id, EDM4hep name) map is our repo's single
# source of truth and matches GenBIB's COLLECTIONS exactly.
from count_tracker_conditions import COLLECTIONS


# --- Vendored from GenBIB-ML sample.py (unchanged numerics) ------------------

FEATURE_ORDER = [0, 4, 1, 2, 3, 6, 7, 8, 9]
FEATURE_INDICES = {"r": 2, "phi": 3, "z": 4, "side": 5, "layer": 6,
                   "module": 7, "sensor": 8}


def require_file(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def require_directory(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def resolve_device(spec):
    import torch

    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_paper1(paper1_root):
    """Import the Paper 1 TabDDPM inference code from the model root.

    Returns ``(sample, inverse_geometry_transform, build_xy_z_lookup,
    snap_z_to_detector_xy, apply_material_map_hybrid)``.
    """
    paper1_root = require_directory(paper1_root, "Paper 1 repository")
    tabddpm_root = paper1_root / "diffusion" / "tabddpm_official"
    scripts_dir = tabddpm_root / "scripts"
    sampler_path = require_file(scripts_dir / "sample.py", "Paper 1 TabDDPM sampler")

    for path in (scripts_dir, tabddpm_root, paper1_root):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)

    spec = importlib.util.spec_from_file_location("paper1_tabddpm_sample", sampler_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from helpers.data_transforms import inverse_geometry_transform
    from helpers.flow import build_xy_z_lookup, snap_z_to_detector_xy
    from helpers.material_map import apply_material_map_hybrid

    return (
        module.sample,
        inverse_geometry_transform,
        build_xy_z_lookup,
        snap_z_to_detector_xy,
        apply_material_map_hybrid,
    )


def map_conditions_to_classes(conditions, y_lookup):
    class_by_condition = {tuple(row): index for index, row in enumerate(y_lookup)}
    class_ids = np.empty(len(conditions), dtype=np.int64)
    missing = []
    for index, row in enumerate(conditions[:, 1:]):
        key = tuple(row)
        if key not in class_by_condition:
            missing.append(conditions[index].tolist())
        else:
            class_ids[index] = class_by_condition[key]
    if missing:
        raise ValueError(f"Conditions not present in y_lookup.npy: {missing[:10]}")
    return class_ids


def build_endcap_z_lookup(model_dir, y_lookup, collection_name,
                          inverse_geometry_transform, build_xy_z_lookup):
    reference_parts = []
    dataset_dir = Path(model_dir) / "dataset"
    for split in ("train", "val"):
        features = np.load(
            require_file(dataset_dir / f"X_num_{split}.npy", f"dataset X_num_{split}.npy")
        ).astype(np.float32)
        class_ids = np.load(
            require_file(dataset_dir / f"y_{split}.npy", f"dataset y_{split}.npy")
        ).astype(np.int64).reshape(-1)
        local_hits = np.column_stack([features, y_lookup[class_ids]]).astype(np.float32)
        reference_parts.append(
            inverse_geometry_transform(local_hits, "local_phi", collection_name, FEATURE_ORDER)
        )
    reference_hits = np.concatenate(reference_parts)
    z_lookup = build_xy_z_lookup(
        reference_hits,
        FEATURE_INDICES["side"], FEATURE_INDICES["layer"],
        FEATURE_INDICES["r"], FEATURE_INDICES["phi"], FEATURE_INDICES["z"],
    )
    del reference_parts, reference_hits
    gc.collect()
    return z_lookup


class CollectionSampler:
    """Load one collection's TabDDPM model once, then sample it repeatedly."""

    def __init__(self, short, model_dir, device, paper1):
        if short not in COLLECTIONS:
            raise ValueError(f"collection must be one of: {', '.join(COLLECTIONS)}")
        self.short = short
        self.system_id, self.collection_name = COLLECTIONS[short]
        self.device = device
        self.paper1 = paper1
        self.model_dir = Path(model_dir).resolve()

        model_path = require_file(self.model_dir / "model.pt", "model.pt")
        config_path = require_file(self.model_dir / "run_config.json", "run_config.json")
        lookup_path = require_file(self.model_dir / "y_lookup.npy", "y_lookup.npy")
        info_path = require_file(self.model_dir / "dataset" / "info.json", "dataset/info.json")

        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        with info_path.open(encoding="utf-8") as handle:
            dataset_info = json.load(handle)
        if config["BASIS"] != "local_phi" or config["Y_MODE"] != "cond":
            raise ValueError("This interface requires a local_phi, condition-trained model")

        y_lookup = np.load(lookup_path).astype(np.int64)
        if y_lookup.ndim != 2 or y_lookup.shape[1] != 4:
            raise ValueError(f"Expected y_lookup.npy shape (N, 4); found {y_lookup.shape}")
        self.y_lookup = y_lookup
        self.num_features = int(dataset_info["n_num_features"])

        model_params = {
            "num_classes": int(config["n_classes"]),
            "is_y_cond": bool(config["is_y_cond"]),
            "rtdl_params": {
                "d_layers": [int(v) for v in config["D_LAYERS"].split(",")],
                "dropout": 0.0,
            },
            "dim_t": int(config["DIM_T"]),
        }
        transform_config = {
            "seed": int(config["SEED"]),
            "normalization": config["NORMALIZATION"],
            "num_nan_policy": None, "cat_nan_policy": None,
            "cat_min_frequency": None, "cat_encoding": None, "y_policy": "default",
        }
        self.sample_job_common = {
            "real_data_path": str(self.model_dir / "dataset"),
            "batch_size": int(config["SAMPLE_BATCH_SIZE"]),
            "model_type": "mlp",
            "model_params": model_params,
            "model_path": str(model_path),
            "num_timesteps": int(config["NUM_TIMESTEPS"]),
            "gaussian_loss_type": "mse",
            "scheduler": config["SCHEDULER"],
            "T_dict": transform_config,
            "num_numerical_features": self.num_features,
            "disbalance": None,
            "device": device,
            "change_val": False,
        }
        self.z_lookup = None
        if "Endcap" in self.collection_name:
            self.z_lookup = build_endcap_z_lookup(
                self.model_dir, y_lookup, self.collection_name,
                paper1[1], paper1[2],  # inverse_geometry_transform, build_xy_z_lookup
            )

    def sample(self, conditions, seed, **kwargs):
        return run_rejection_loop(self, conditions, seed, **kwargs)


def run_rejection_loop(sampler, conditions, seed, *, oversample=1,
                       max_rejection_rounds=10000, work_dir=None, label=""):
    """Fill every requested condition via the TabDDPM rejection sampler.

    Mirrors GenBIB main()'s loop exactly around the paper1 callables. Returns an
    ``(N * oversample, num_features + 4)`` float32 array whose trailing four
    columns are the requested side/layer/module/sensor.
    """
    tabddpm_sample, inverse_geometry_transform, _build_xy_z_lookup, \
        snap_z_to_detector_xy, apply_material_map_hybrid = sampler.paper1

    conditions = np.asarray(conditions, dtype=np.int64)
    if conditions.ndim != 2 or conditions.shape[1] != 5:
        raise ValueError("conditions must have shape (N, 5)")
    if len(conditions) and np.any(conditions[:, 0] != sampler.system_id):
        raise ValueError(f"conditions belong to system != {sampler.system_id}")
    if len(conditions) == 0:
        return np.empty((0, sampler.num_features + 4), dtype=np.float32)

    class_ids = map_conditions_to_classes(conditions, sampler.y_lookup)
    conditions = np.repeat(conditions, oversample, axis=0)
    class_ids = np.repeat(class_ids, oversample)

    output = np.empty((len(class_ids), sampler.num_features + 4), dtype=np.float32)
    unfilled = np.ones(len(class_ids), dtype=bool)
    work_parent = Path(work_dir) if work_dir is not None else Path.cwd()
    work_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="genbib_tabddpm_", dir=work_parent) as work:
        sample_job = dict(sampler.sample_job_common, parent_dir=work)
        for round_id in range(max_rejection_rounds):
            remaining = np.flatnonzero(unfilled)
            if len(remaining) == 0:
                break
            requested_ids = class_ids[remaining]
            print(
                f"{label}round {round_id + 1}: {len(remaining):,}/{len(class_ids):,} remaining",
                flush=True,
            )
            tabddpm_sample(
                **sample_job,
                num_samples=len(remaining),
                seed=seed + round_id,
                y_to_sample=requested_ids,
            )
            generated_features = np.load(Path(work) / "X_num_train.npy").astype(np.float32)
            returned_ids = np.load(Path(work) / "y_train.npy").astype(np.int64).reshape(-1)
            if not np.array_equal(returned_ids, requested_ids):
                raise RuntimeError("TabDDPM changed the requested condition order")

            generated_conditions = sampler.y_lookup[returned_ids]
            generated_hits = np.column_stack(
                [generated_features, generated_conditions]
            ).astype(np.float32)
            generated_hits = inverse_geometry_transform(
                generated_hits, "local_phi", sampler.collection_name, FEATURE_ORDER
            )
            if sampler.z_lookup is not None:
                generated_hits[:, FEATURE_INDICES["z"]] = snap_z_to_detector_xy(
                    generated_hits[:, FEATURE_INDICES["r"]],
                    generated_hits[:, FEATURE_INDICES["phi"]],
                    generated_hits[:, FEATURE_INDICES["side"]],
                    generated_hits[:, FEATURE_INDICES["layer"]],
                    sampler.z_lookup,
                )
            mask = np.asarray(
                apply_material_map_hybrid(
                    {sampler.collection_name: generated_hits},
                    None, sampler.collection_name, FEATURE_INDICES,
                ),
                dtype=bool,
            )
            passing_slots = remaining[mask]
            output[passing_slots] = generated_hits[mask]
            unfilled[passing_slots] = False

            del generated_features, returned_ids, generated_conditions, generated_hits, mask
            gc.collect()
            if getattr(sampler.device, "type", None) == "cuda":
                import torch

                torch.cuda.empty_cache()

    if np.any(unfilled):
        error = RuntimeError(
            f"{int(unfilled.sum())} conditions remain after {max_rejection_rounds} rounds"
        )
        error.unfilled_conditions = conditions[unfilled]
        raise error
    return output


# --- Driver ------------------------------------------------------------------

def resolve_model_dir(model_root, short):
    """Pick the canonical <SHORT>_TABDDPM_*_dim2048_b4096 model directory.

    A stray truncated sibling (e.g. VBC_..._x4096x40) is filtered out by
    requiring a real directory that contains model.pt and y_lookup.npy.
    """
    model_root = Path(model_root).expanduser().resolve()
    for candidate in sorted(model_root.glob(f"{short}_TABDDPM_*_dim2048_b4096")):
        if candidate.is_dir() and (candidate / "model.pt").is_file() and (
            candidate / "y_lookup.npy"
        ).is_file():
            return candidate
    raise FileNotFoundError(f"No usable {short} model under {model_root}")


def event_seed(seed_base, construction, cohort, event_id, short):
    """Deterministic per-(event, collection) base seed within uint31."""
    key = "|".join([str(seed_base), construction, cohort, event_id, short])
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_conditions_report(conditions_dir):
    report = json.loads((Path(conditions_dir) / "manifest.json").read_text())
    manifest = report["manifest"]
    if manifest.get("schema_version") != 2:
        raise ValueError("Conditions manifest must be schema_version 2")
    return report, manifest


def sample_events(args):
    conditions_dir = Path(args.conditions).resolve()
    report, manifest = load_conditions_report(conditions_dir)
    construction = manifest["construction"]
    cohort = manifest.get("sampling", {}).get("cohort", "unknown")

    wanted = [
        summary for summary in report["events"]
        if summary["split"] == args.split
        and (args.event_id is None or summary["event_id"] == args.event_id)
    ]
    if not wanted:
        raise ValueError(f"No events for split={args.split} event_id={args.event_id}")
    # Interleaved shard so several GPU jobs can share one output directory.
    if args.num_shards > 1:
        wanted = wanted[args.shard_index::args.num_shards]
    if not wanted:
        print(f"shard {args.shard_index}/{args.num_shards}: no events; nothing to do")
        return
    shorts = args.collections or list(COLLECTIONS)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)  # shared and resumable across shards

    paper1_root = args.paper1_root or (Path(args.model_root) / "paper1-inference")
    device = resolve_device(args.device)
    paper1 = None  # loaded lazily once a model is actually needed

    model_dirs = {}
    records = {summary["event_id"]: {} for summary in wanted}
    # Collection-outer, event-inner: load each model at most once; skip events
    # whose output already exists so reruns and parallel shards are safe.
    for short in shorts:
        model_dir = resolve_model_dir(args.model_root, short)
        model_dirs[short] = str(model_dir)
        pending = [s for s in wanted if not (
            output / args.split / s["event_id"] / f"tabddpm_{short}_samples.npy").exists()]
        if not pending:
            print(f"[{short}] all {len(wanted)} events already sampled; skipping", flush=True)
            continue
        if paper1 is None:
            paper1 = load_paper1(paper1_root)
        sampler = CollectionSampler(short, model_dir, device, paper1)
        print(f"[{short}] model {model_dir.name}: {len(pending)}/{len(wanted)} to sample",
              flush=True)
        for summary in pending:
            event_id = summary["event_id"]
            destination = output / args.split / event_id
            out_path = destination / f"tabddpm_{short}_samples.npy"
            source = conditions_dir / args.split / event_id / f"{short}_conditions.npy"
            conditions = np.load(source, allow_pickle=False)
            expected_hits = len(conditions)
            seed = event_seed(args.seed_base, construction, cohort, event_id, short)
            out_rows = run_rejection_loop(
                sampler, conditions, seed, oversample=1,
                max_rejection_rounds=args.max_rounds, work_dir=output,
                label=f"[{short} {event_id}] ")
            if len(out_rows) != expected_hits:
                raise RuntimeError(
                    f"[{short} {event_id}] produced {len(out_rows)} rows for "
                    f"{expected_hits} conditions")
            destination.mkdir(parents=True, exist_ok=True)
            tmp = destination / f".tabddpm_{short}_samples.{os.getpid()}.npy"
            np.save(tmp, out_rows.astype(np.float32), allow_pickle=False)
            os.replace(tmp, out_path)  # atomic; safe against concurrent shards
            records[event_id][short] = {"hits": int(expected_hits), "seed": int(seed)}

    shard_manifest = {
        "kind": "count_tracker_samples_shard",
        "construction": construction, "cohort": cohort, "split": args.split,
        "collections": shorts, "seed_base": args.seed_base, "max_rounds": args.max_rounds,
        "device": str(device), "vendored_from": "GenBIB-ML sample.py @ a483be7",
        "paper1_root": str(Path(paper1_root).resolve()), "model_dirs": model_dirs,
        "conditions_dir": str(conditions_dir),
        "conditions_manifest_sha256": sha256_file(conditions_dir / "manifest.json"),
        "shard": {"index": args.shard_index, "num": args.num_shards},
        "events": [{"event_id": eid, "split": args.split, "collections": records[eid]}
                   for eid in (s["event_id"] for s in wanted)],
    }
    tag = f".shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    (output / f"manifest{tag}.json").write_text(json.dumps(shard_manifest, indent=2) + "\n")
    print(f"shard {args.shard_index}/{args.num_shards}: sampled {len(wanted)} event(s) -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True, help="Prepared conditions directory")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--event-id", help="Sample only this event (default: all in split)")
    parser.add_argument(
        "--model-root", required=True,
        help="Directory holding the six <SHORT>_TABDDPM_* model dirs",
    )
    parser.add_argument("--paper1-root", help="Defaults to <model-root>/paper1-inference")
    parser.add_argument(
        "--collections", nargs="+", choices=tuple(COLLECTIONS),
        help="Subset of collections (default: all six)",
    )
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=10000)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="split the event list across this many parallel GPU jobs")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="which interleaved shard this job handles (0-based)")
    parser.add_argument("--output", required=True, help="Output directory (shared, resumable)")
    args = parser.parse_args()
    if args.seed_base < 0:
        parser.error("--seed-base must be nonnegative")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard-index must be in [0, --num-shards)")
    sample_events(args)


if __name__ == "__main__":
    main()
