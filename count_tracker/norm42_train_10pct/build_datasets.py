#!/usr/bin/env python3
"""Build reference-format TabDDPM datasets from the approved 10%-BX cohort."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


PAPER1_COMMIT = "90bd576c619416dbd7889eb1900ec80ac687ab92"
COLLECTIONS = {
    "VBC": "VertexBarrelCollection",
    "VEC": "VertexEndcapCollection",
    "ITBC": "InnerTrackerBarrelCollection",
    "ITEC": "InnerTrackerEndcapCollection",
    "OTBC": "OuterTrackerBarrelCollection",
    "OTEC": "OuterTrackerEndcapCollection",
}
POLARITIES = ("MUPLUS", "MUMINUS")
FEATURE_ORDER = (0, 4, 1, 2, 3, 6, 7, 8, 9)
FEATURE_LABELS = ("logE", "t", "r", "local_phi", "z")
CONDITION_LABELS = ("side", "layer", "module", "sensor")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recorded_commit(root):
    root = Path(root).resolve()
    marker = root / ".source_commit"
    if marker.is_file():
        return marker.read_text().strip()
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"{root} is neither a marked archive nor a readable git checkout"
        ) from error


def load_local_phi_transform(paper1_root):
    paper1_root = Path(paper1_root).resolve()
    commit = recorded_commit(paper1_root)
    if commit != PAPER1_COMMIT:
        raise ValueError(
            f"Paper1 checkout is {commit}, expected recorded commit {PAPER1_COMMIT}"
        )
    sys.path.insert(0, str(paper1_root))
    try:
        module = importlib.import_module("helpers.data_transforms")
    finally:
        sys.path.pop(0)
    return module.local_phi_transformation


def selected_ids(manifest):
    values = []
    for polarity in POLARITIES:
        rows = manifest["sources"][polarity]
        values.extend(int(row["source_id"]) for row in rows)
    values = np.asarray(sorted(values), dtype=np.int32)
    if len(values) != len(np.unique(values)):
        raise ValueError("selected source IDs are not unique")
    return values


def transform_rows(raw, collection, local_phi_transform):
    if raw.ndim != 2 or raw.shape[1] != 10:
        raise ValueError(f"expected raw shape (N, 10), got {raw.shape}")
    values = np.asarray(raw, dtype=np.float32).copy()
    if np.any(values[:, 0] <= 0):
        raise ValueError("energy deposition must be positive before log transform")
    values[:, 0] = np.log(values[:, 0])
    x = values[:, 1].copy()
    y = values[:, 2].copy()
    values[:, 1] = np.sqrt(x * x + y * y)
    values[:, 2] = np.arctan2(y, x)
    phi_index = values[:, 8] if "Barrel" in collection else values[:, 9]
    values = local_phi_transform(
        values,
        layers=values[:, 7],
        phi_index=phi_index,
        collection=collection,
        direction="forward",
    )
    values = np.asarray(values[:, FEATURE_ORDER], dtype=np.float32)
    conditions = values[:, -len(CONDITION_LABELS):]
    rounded = np.rint(conditions)
    if not np.array_equal(conditions, rounded):
        raise ValueError("condition columns contain non-integral values")
    condition_rows = rounded.astype(np.int64)
    y_lookup, y = np.unique(condition_rows, axis=0, return_inverse=True)
    return values[:, :len(FEATURE_LABELS)], y.astype(np.int64), y_lookup


def split_indices(y, seed):
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y)
    singleton_classes = np.flatnonzero(counts == 1)
    singleton_mask = np.isin(y, singleton_classes, assume_unique=False)
    forced_train = np.flatnonzero(singleton_mask)
    eligible = np.flatnonzero(~singleton_mask)
    train, validation = train_test_split(
        eligible,
        test_size=0.2,
        random_state=seed,
        stratify=y[eligible],
    )
    train = np.concatenate((np.asarray(train, dtype=np.int64), forced_train))
    validation = np.asarray(validation, dtype=np.int64)
    if len(np.intersect1d(train, validation)):
        raise AssertionError("training and validation indices overlap")
    if len(train) + len(validation) != len(y):
        raise AssertionError("split does not cover every selected row")
    if len(np.unique(y[train])) != len(counts):
        raise AssertionError("training split does not cover every condition")
    return train, validation, singleton_classes, forced_train


def save_array(path, values):
    np.save(path, values, allow_pickle=False)
    return {
        "file": Path(path).name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": sha256(path),
    }


def build_collection(
        short, data_path, source_path, selected, expected_hits, destination,
        split_seed, local_phi_transform):
    data = np.load(data_path, mmap_mode="r", allow_pickle=False)
    sources = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if data.shape != (len(sources), 10):
        raise ValueError(f"{short}: data/source shape mismatch")
    mask = np.isin(sources, selected, assume_unique=False)
    full_row_indices = np.flatnonzero(mask).astype(np.int64)
    selected_sources = np.asarray(sources[mask], dtype=np.int32)
    raw = np.asarray(data[mask], dtype=np.float32)
    if len(raw) != expected_hits:
        raise ValueError(
            f"{short}: selected {len(raw):,} hits, audit recorded "
            f"{expected_hits:,}"
        )

    x_num, y, y_lookup = transform_rows(
        raw, COLLECTIONS[short], local_phi_transform
    )
    train, validation, singleton_classes, forced_train = split_indices(
        y, split_seed
    )

    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(destination if destination.exists() else partial)
    dataset = partial / "dataset"
    provenance = partial / "provenance"
    dataset.mkdir(parents=True)
    provenance.mkdir()

    files = {}
    files["X_num_train"] = save_array(
        dataset / "X_num_train.npy", x_num[train].astype(np.float32)
    )
    files["X_num_val"] = save_array(
        dataset / "X_num_val.npy", x_num[validation].astype(np.float32)
    )
    files["X_num_test"] = save_array(
        dataset / "X_num_test.npy", x_num[validation].astype(np.float32)
    )
    files["y_train"] = save_array(
        dataset / "y_train.npy", y[train].astype(np.int64)
    )
    files["y_val"] = save_array(
        dataset / "y_val.npy", y[validation].astype(np.int64)
    )
    files["y_test"] = save_array(
        dataset / "y_test.npy", y[validation].astype(np.int64)
    )
    files["y_lookup"] = save_array(
        partial / "y_lookup.npy", y_lookup.astype(np.int64)
    )
    files["train_full_row_index"] = save_array(
        provenance / "train_full_row_index.npy", full_row_indices[train]
    )
    files["val_full_row_index"] = save_array(
        provenance / "val_full_row_index.npy", full_row_indices[validation]
    )
    files["train_source_id"] = save_array(
        provenance / "train_source_id.npy", selected_sources[train]
    )
    files["val_source_id"] = save_array(
        provenance / "val_source_id.npy", selected_sources[validation]
    )

    info = {
        "name": f"norm42_10pct_{short}",
        "id": f"norm42_10pct_{short}",
        "task_type": "multiclass",
        "n_num_features": len(FEATURE_LABELS),
        "n_cat_features": 0,
        "train_size": int(len(train)),
        "val_size": int(len(validation)),
        "test_size": int(len(validation)),
        "n_classes": int(len(y_lookup)),
    }
    (dataset / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    files["info"] = {
        "file": "info.json",
        "sha256": sha256(dataset / "info.json"),
    }

    singleton_conditions = []
    for class_id in singleton_classes:
        rows = np.flatnonzero(y == class_id)
        if len(rows) != 1:
            raise AssertionError("singleton class does not have exactly one row")
        row = int(rows[0])
        singleton_conditions.append({
            **dict(zip(CONDITION_LABELS, map(int, y_lookup[class_id]))),
            "class_id": int(class_id),
            "selected_row": row,
            "full_row": int(full_row_indices[row]),
            "source_id": int(selected_sources[row]),
        })
    split_manifest = {
        "schema_version": 1,
        "collection": short,
        "collection_name": COLLECTIONS[short],
        "selected_hits": int(len(y)),
        "features": list(FEATURE_LABELS),
        "conditions": list(CONDITION_LABELS),
        "feature_order_from_raw": list(FEATURE_ORDER),
        "split": {
            "method": "reference row-level sensor-stratified 80/20 split",
            "implementation": "sklearn.model_selection.train_test_split",
            "test_size": 0.2,
            "random_state": split_seed,
            "singleton_policy": "force singleton condition rows into training",
            "forced_singleton_rows": int(len(forced_train)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_is_validation_duplicate": True,
        },
        "singleton_conditions": singleton_conditions,
        "checks": {
            "all_rows_assigned_once": True,
            "training_covers_all_conditions": True,
            "train_validation_disjoint_rows": True,
            "train_unique_source_ids": int(len(np.unique(selected_sources[train]))),
            "validation_unique_source_ids": int(
                len(np.unique(selected_sources[validation]))
            ),
            "shared_train_validation_source_ids": int(
                len(np.intersect1d(
                    np.unique(selected_sources[train]),
                    np.unique(selected_sources[validation]),
                ))
            ),
        },
        "files": files,
    }
    (partial / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n"
    )
    os.replace(partial, destination)
    return split_manifest


def build(
        manifest_path, audit_path, data_root, source_id_root, paper1_root,
        output, split_seed):
    manifest_path = Path(manifest_path).resolve()
    audit_path = Path(audit_path).resolve()
    data_root = Path(data_root).resolve()
    source_id_root = Path(source_id_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")

    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["decision_gate"]["passed_complete_sensor_coverage"]:
        raise ValueError("coverage audit did not pass")
    if Path(audit["source_manifest"]).resolve() != manifest_path:
        raise ValueError("coverage audit refers to a different source manifest")
    selected = selected_ids(manifest)
    local_phi_transform = load_local_phi_transform(paper1_root)

    output.mkdir(parents=True)
    collection_reports = {}
    try:
        for short in COLLECTIONS:
            print(f"Building {short}...", flush=True)
            report = build_collection(
                short,
                data_root / f"{short}_data.npy",
                source_id_root / f"{short}_source_id.npy",
                selected,
                int(audit["collections"][short]["selected_hits"]),
                output / short,
                split_seed,
                local_phi_transform,
            )
            collection_reports[short] = {
                "selected_hits": report["selected_hits"],
                **report["split"],
            }
            print(
                f"  train={report['split']['train_rows']:,}, "
                f"validation={report['split']['validation_rows']:,}, "
                f"forced singletons={report['split']['forced_singleton_rows']:,}",
                flush=True,
            )
    except Exception:
        (output / "INCOMPLETE").write_text(
            "Dataset construction failed; do not use this directory.\n"
        )
        raise

    root_manifest = {
        "schema_version": 1,
        "kind": "norm42_10pct_reference_format_tabddpm_datasets",
        "paper1_commit": PAPER1_COMMIT,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "coverage_audit": str(audit_path),
        "coverage_audit_sha256": sha256(audit_path),
        "data_root": str(data_root),
        "source_id_root": str(source_id_root),
        "split_seed": split_seed,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "collections": collection_reports,
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(root_manifest, indent=2) + "\n"
    )
    print(f"Prepared datasets -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-id-root", required=True)
    parser.add_argument("--paper1-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    build(
        args.manifest,
        args.audit,
        args.data_root,
        args.source_id_root,
        args.paper1_root,
        args.output,
        args.split_seed,
    )


if __name__ == "__main__":
    main()
