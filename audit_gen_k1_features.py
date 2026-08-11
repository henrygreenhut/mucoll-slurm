#!/usr/bin/env python3

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import libtest_common as lc
from audit_gen_k1_phi import (
    SOURCE_LABEL,
    build_model,
    pad,
    source_config,
    test_definitions,
)
from pfn_libtest_train import UnitSampler


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-label", default=SOURCE_LABEL)
    parser.add_argument("--outdir", default="pfn_results")
    parser.add_argument("--label", default="pm_n420_k1_feature_audit_v1")
    parser.add_argument("--test-units", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def interventions(names):
    modes = ["baseline"]
    modes.extend("shuffle:{}".format(name) for name in names)
    modes.extend(("shuffle:phi_pair", "shuffle:pdg_group", "phi_plus_pi_over_2"))
    return modes


def transform(features, names, mode, rng):
    if mode == "baseline":
        return features

    changed = [values.copy() for values in features]
    if mode.startswith("shuffle:"):
        group = mode.split(":", 1)[1]
        if group == "phi_pair":
            columns = [names.index("cosphi"), names.index("sinphi")]
        elif group == "pdg_group":
            columns = [
                index for index, name in enumerate(names)
                if name.startswith("pdg_")
            ]
        else:
            columns = [names.index(group)]
        lengths = [len(values) for values in features]
        pooled = np.concatenate([values[:, columns] for values in features])
        shuffled = pooled[rng.permutation(len(pooled))]
        start = 0
        for values, length in zip(changed, lengths):
            values[:, columns] = shuffled[start:start + length]
            start += length
        return changed

    if mode == "phi_plus_pi_over_2":
        cos_index = names.index("cosphi")
        sin_index = names.index("sinphi")
        for original, values in zip(features, changed):
            values[:, cos_index] = -original[:, sin_index]
            values[:, sin_index] = original[:, cos_index]
        return changed

    raise ValueError("unknown intervention: {}".format(mode))


def selected_test_definitions(source_dir, config, stores, count):
    all_definitions = test_definitions(source_dir, config, stores)
    available = int(config["eval_point_units"])
    if count > available:
        raise ValueError(
            "requested {} test inputs per class; source run has {}".format(
                count, available
            )
        )
    class_a = all_definitions[:count]
    class_b = all_definitions[available:available + count]
    return [definition for pair in zip(class_a, class_b) for definition in pair]


def save_checkpoint(path, labels, scores, mode_names, completed):
    temporary = path.with_suffix(".partial.npz")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            labels=labels,
            scores=scores,
            modes=np.asarray(mode_names),
            completed=np.asarray(completed, dtype=np.int64),
        )
    os.replace(temporary, path)


def load_checkpoint(path, labels, mode_names):
    scores = np.full((len(mode_names), len(labels)), np.nan, dtype=np.float32)
    completed = 0
    if not path.is_file():
        return scores, completed
    with np.load(path) as saved:
        if not np.array_equal(saved["labels"], labels):
            raise ValueError("saved audit labels do not match this evaluation")
        if saved["modes"].tolist() != mode_names:
            raise ValueError("saved audit interventions do not match this evaluation")
        scores = saved["scores"]
        completed = int(saved["completed"])
    return scores, completed


def evaluate(model, definitions, samplers, mean, std, names, mode_names,
             seed, batch_size, checkpoint):
    labels = np.asarray([class_id for class_id, _ in definitions], dtype=np.int32)
    scores, completed = load_checkpoint(checkpoint, labels, mode_names)
    for start in range(completed, len(definitions), batch_size):
        chunk = definitions[start:start + batch_size]
        physical = []
        for class_id, positions in chunk:
            raw = samplers[class_id].store.file_arrays(positions)
            physical.append(lc.build_features(raw, samplers[class_id].feature_set))

        for mode_index, mode in enumerate(mode_names):
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, mode_index, start])
            )
            transformed = transform(physical, names, mode, rng)
            changed = [(values - mean) / std for values in transformed]
            predictions = model.predict_on_batch(pad(changed))
            scores[mode_index, start:start + len(chunk)] = np.asarray(
                predictions
            )[:, 1]

        completed = start + len(chunk)
        save_checkpoint(checkpoint, labels, scores, mode_names, completed)
        print(
            "evaluated {}/{} test inputs across {} interventions".format(
                completed, len(definitions), len(mode_names)
            ),
            flush=True,
        )
    return labels, scores


def save_plot(path, rows):
    feature_rows = [row for row in rows if row["kind"] == "individual"]
    labels = [row["feature"] for row in feature_rows]
    values = [row["auc"] for row in feature_rows]
    figure, axis = plt.subplots(figsize=(8.0, 6.2))
    positions = np.arange(len(labels))
    axis.barh(positions, values, color="#0072B2")
    axis.axvline(0.5, color="0.5", linestyle="--", linewidth=1)
    axis.axvline(rows[0]["auc"], color="#D55E00", linewidth=1.5,
                 label="Unmodified AUC")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Test AUC after within-input feature shuffle")
    axis.set_title("GEN N=420 frozen-PFN feature audit")
    axis.set_xlim(0.0, 1.0)
    axis.grid(axis="x", alpha=0.2, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    args = arguments()
    source_dir = Path(args.outdir) / args.source_label
    output = Path(args.outdir) / args.label
    output.mkdir(parents=True, exist_ok=True)
    config = source_config(source_dir)
    if int(config["clone_factor"]) != 1:
        raise SystemExit("this audit requires native K=1 versus rotated K=1")
    if config["features"] != "expanded":
        raise SystemExit("this audit expects the 13-feature expanded input")

    stores = [lc.Store(config["norm1_store"]), lc.Store(config["norm42_store"])]
    definitions = selected_test_definitions(
        source_dir, config, stores, args.test_units
    )
    files = (
        int(config["n_files"]),
        int(config["n_files"]) // int(config["clone_factor"]),
    )
    samplers = [
        UnitSampler(
            store,
            {"test": np.arange(store.n_files)},
            count,
            config["features"],
            config.get("exclude_muons_above_gev", 0.0),
        )
        for store, count in zip(stores, files)
    ]
    mean, std, latent_scale = lc.load_norm_stats(source_dir / "norm_stats.json")
    names = lc.feature_names(config["features"])
    model = build_model(config, len(names), latent_scale)
    model.load_weights(str(source_dir / "best.weights.h5"))
    batch_size = args.batch_size or int(config["batch_size"])
    mode_names = interventions(names)

    labels, scores = evaluate(
        model,
        definitions,
        samplers,
        mean,
        std,
        names,
        mode_names,
        args.seed,
        batch_size,
        output / "partial_scores.npz",
    )
    baseline_auc = lc.auc_score(labels, scores[0])
    if args.test_units == int(config["eval_point_units"]):
        reported_auc = float(
            json.load(open(source_dir / "point_summary.json"))["auc"]
        )
        if abs(baseline_auc - reported_auc) > 1e-5:
            raise RuntimeError(
                "baseline AUC {:.8f} does not reproduce {:.8f}".format(
                    baseline_auc, reported_auc
                )
            )
    else:
        reported_auc = None

    rows = []
    for mode, mode_scores in zip(mode_names, scores):
        if mode == "baseline":
            kind = "baseline"
            feature = "unmodified"
        elif mode == "phi_plus_pi_over_2":
            kind = "rotation"
            feature = "phi + pi/2"
        elif mode in ("shuffle:phi_pair", "shuffle:pdg_group"):
            kind = "grouped"
            feature = mode.split(":", 1)[1]
        else:
            kind = "individual"
            feature = mode.split(":", 1)[1]
        auc = lc.auc_score(labels, mode_scores)
        rows.append({
            "intervention": mode,
            "kind": kind,
            "feature": feature,
            "auc": auc,
            "auc_change": auc - baseline_auc,
        })

    with (output / "feature_audit_auc.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output / "feature_audit_summary.json").open("w") as stream:
        json.dump(
            {
                "source_label": args.source_label,
                "checkpoint": str(source_dir / "best.weights.h5"),
                "test_units_per_class": args.test_units,
                "baseline_auc": baseline_auc,
                "reported_auc": reported_auc,
                "rows": rows,
                "shuffle_definition": (
                    "particle permutation across balanced native/rotated "
                    "test batches, preserving each construction's multiplicity"
                ),
                "phi_rotation": "coherent +pi/2 for every particle",
            },
            stream,
            indent=2,
        )
    save_plot(output / "feature_permutation_auc", rows)
    print("baseline AUC {:.6f}".format(baseline_auc))
    for row in rows[1:]:
        print(
            "{}: AUC {:.6f} ({:+.6f})".format(
                row["intervention"], row["auc"], row["auc_change"]
            )
        )
    print("audit complete -> {}".format(output), flush=True)


if __name__ == "__main__":
    main()
