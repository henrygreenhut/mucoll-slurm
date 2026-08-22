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
from pfn_libtest_train import F_SIZES, PHI_SIZES, UnitSampler, class_layout


SOURCE_LABEL = (
    "pm_n420_k1_vs_k1_synthetic_scaled_lr1e-4_decay80_mseed1_v1"
)
MODES = ("baseline", "global_rotation", "shuffle_phi", "uniform_phi")
MODE_LABELS = {
    "baseline": "Original",
    "global_rotation": "Global rotation",
    "shuffle_phi": r"Within-construction $\phi$ shuffle",
    "uniform_phi": r"Independent uniform $\phi$",
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-label", default=SOURCE_LABEL)
    parser.add_argument("--outdir", default="pfn_results")
    parser.add_argument("--label", default="pm_n420_k1_phi_audit_v1")
    parser.add_argument("--test-units", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def config_value(config, key, default):
    value = config.get(key, default)
    return default if value is None else value


def source_config(source_dir):
    path = source_dir / "config.json"
    if path.is_file():
        return load_json(path)
    return load_json(source_dir / "auc_summary.json")["config"]


def build_model(config, n_features, latent_scale):
    phi_sizes = tuple(config_value(config, "phi_sizes", PHI_SIZES))
    f_sizes = tuple(config_value(config, "f_sizes", F_SIZES))
    arch = config_value(config, "arch", "local")
    if arch == "energyflow":
        if latent_scale == 1.0:
            return lc.build_pfn_energyflow(
                n_features, phi_sizes=phi_sizes, f_sizes=f_sizes
            )
        return lc.build_pfn_energyflow_scaled(
            n_features, latent_scale, phi_sizes=phi_sizes, f_sizes=f_sizes
        )
    return lc.build_pfn(
        n_features, latent_scale, phi_sizes=phi_sizes, f_sizes=f_sizes
    )


def test_definitions(source_dir, config, stores):
    n_files = int(config["n_files"])
    clone_factor = int(config["clone_factor"])
    store_a, store_b, files_a, files_b = class_layout(
        stores[0], stores[1], n_files, clone_factor
    )
    common, positions_a, positions_b = lc.common_positions(store_a, store_b)
    split = lc.load_or_create_cycle_split(
        str(source_dir / "source_split.npz"), common,
        tuple(config["split_fracs"]), int(config["data_seed"])
    )
    test = lc.cycle_split_positions(common, split)["test"]
    pools = (positions_a[test], positions_b[test])
    files_per_unit = (files_a, files_b)
    rng = np.random.default_rng(int(config["data_seed"]) + 2026)
    count = int(config["eval_point_units"])
    definitions = []
    for class_id, (pool, files) in enumerate(zip(pools, files_per_unit)):
        definitions.extend([
            (class_id, rng.choice(pool, size=files, replace=False))
            for _ in range(count)
        ])
    return definitions


def transform_phi(features, names, mode, rng):
    if mode == "baseline":
        return features
    cos_index = names.index("cosphi")
    sin_index = names.index("sinphi")
    changed = features.copy()
    cosine = features[:, cos_index].copy()
    sine = features[:, sin_index].copy()

    if mode == "global_rotation":
        angle = rng.uniform(0.0, 2.0 * np.pi)
        changed[:, cos_index] = cosine * np.cos(angle) - sine * np.sin(angle)
        changed[:, sin_index] = sine * np.cos(angle) + cosine * np.sin(angle)
    elif mode == "shuffle_phi":
        order = rng.permutation(len(features))
        changed[:, cos_index] = cosine[order]
        changed[:, sin_index] = sine[order]
    elif mode == "uniform_phi":
        angle = rng.uniform(-np.pi, np.pi, size=len(features))
        changed[:, cos_index] = np.cos(angle)
        changed[:, sin_index] = np.sin(angle)
    else:
        raise ValueError("unknown phi intervention: {}".format(mode))
    return changed


def fourier_metrics(features, names, maximum=6):
    cosine = features[:, names.index("cosphi")]
    sine = features[:, names.index("sinphi")]
    phi = np.arctan2(sine, cosine)
    metrics = {"particles": len(phi)}
    for harmonic in range(1, maximum + 1):
        c = float(np.mean(np.cos(harmonic * phi)))
        s = float(np.mean(np.sin(harmonic * phi)))
        metrics["cos{}".format(harmonic)] = c
        metrics["sin{}".format(harmonic)] = s
        metrics["r{}".format(harmonic)] = float(np.hypot(c, s))
    return metrics


def pad(features):
    maximum = max(len(values) for values in features)
    batch = np.zeros(
        (len(features), maximum, features[0].shape[1]), dtype=np.float32
    )
    for index, values in enumerate(features):
        batch[index, :len(values)] = values
    return batch


def evaluate_mode(model, definitions, samplers, mean, std, names, mode,
                  seed, batch_size, collect_metrics=False):
    labels = []
    scores = []
    metrics = []
    for start in range(0, len(definitions), batch_size):
        chunk = definitions[start:start + batch_size]
        features = []
        for offset, (class_id, positions) in enumerate(chunk):
            raw = samplers[class_id].store.file_arrays(positions)
            physical = lc.build_features(raw, samplers[class_id].feature_set)
            if collect_metrics:
                row = fourier_metrics(physical, names)
                row["class"] = class_id
                row["construction"] = (
                    start + offset - class_id * (len(definitions) // 2)
                )
                metrics.append(row)
            rng = np.random.default_rng(
                np.random.SeedSequence([
                    seed, MODES.index(mode), class_id, start + offset
                ])
            )
            changed = transform_phi(physical, names, mode, rng)
            features.append((changed - mean) / std)
            labels.append(class_id)
        prediction = model.predict_on_batch(pad(features))
        scores.extend(np.asarray(prediction)[:, 1].tolist())
        done = min(start + batch_size, len(definitions))
        if done % 100 == 0 or done == len(definitions):
            print("  {}: {}/{}".format(mode, done, len(definitions)), flush=True)
    return np.asarray(labels), np.asarray(scores), metrics


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_correlation(first, second):
    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def save_plot(figure, output):
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_interventions(results, output):
    labels = [MODE_LABELS[mode] for mode in MODES]
    aucs = [results[mode] for mode in MODES]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(labels, aucs, color=["#0072B2", "#56B4E9", "#E69F00", "#D55E00"])
    axis.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    axis.set_ylabel("Test AUC")
    axis.set_title("Frozen GEN K=1 PFN: azimuthal interventions")
    axis.set_ylim(0.0, 1.0)
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save_plot(figure, output / "phi_intervention_auc")


def plot_fourier_auc(rows, output):
    harmonics = np.arange(1, 7)
    width = 0.24
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    for offset, (prefix, label, color) in enumerate((
        ("cos", r"$\langle\cos(n\phi)\rangle$", "#0072B2"),
        ("sin", r"$\langle\sin(n\phi)\rangle$", "#D55E00"),
        ("r", r"Harmonic magnitude $R_n$", "#009E73"),
    )):
        values = [row["auc"] for row in rows if row["metric"] == prefix]
        axis.bar(
            harmonics + (offset - 1) * width, values, width,
            label=label, color=color
        )
    axis.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    axis.set_xticks(harmonics)
    axis.set_xlabel("Azimuthal harmonic n")
    axis.set_ylabel("Single-observable test AUC")
    axis.set_title("GEN N=420 azimuthal-moment separation")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    save_plot(figure, output / "fourier_moment_auc")


def plot_score_correlations(rows, output):
    harmonics = np.arange(1, 7)
    width = 0.24
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    for offset, (prefix, label, color) in enumerate((
        ("cos", r"$\langle\cos(n\phi)\rangle$", "#0072B2"),
        ("sin", r"$\langle\sin(n\phi)\rangle$", "#D55E00"),
        ("r", r"Harmonic magnitude $R_n$", "#009E73"),
    )):
        values = [
            row["pearson"] for row in rows
            if row["sample"] == "all" and row["metric"] == prefix
        ]
        axis.bar(
            harmonics + (offset - 1) * width, values, width,
            label=label, color=color
        )
    axis.axhline(0.0, color="0.5", linewidth=1)
    axis.set_xticks(harmonics)
    axis.set_xlabel("Azimuthal harmonic n")
    axis.set_ylabel("Pearson correlation with PFN score")
    axis.set_title("GEN N=420 PFN score versus azimuthal moments")
    axis.set_ylim(-1.0, 1.0)
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    save_plot(figure, output / "fourier_score_correlation")


def main():
    args = arguments()
    source_dir = Path(args.outdir) / args.source_label
    output = Path(args.outdir) / args.label
    output.mkdir(parents=True, exist_ok=True)
    config = source_config(source_dir)
    if config["clone_factor"] != 1:
        raise SystemExit("the phi audit requires native K=1 vs synthetic K=1")
    if config["features"] != "expanded":
        raise SystemExit("the phi audit expects the expanded feature set")
    if args.test_units != int(config["eval_point_units"]):
        raise SystemExit(
            "use {} test units to reproduce the reported test set exactly"
            .format(config["eval_point_units"])
        )

    stores = [lc.Store(config["norm1_store"]), lc.Store(config["norm42_store"])]
    definitions = test_definitions(source_dir, config, stores)
    files = (config["n_files"], config["n_files"] // config["clone_factor"])
    samplers = [
        UnitSampler(
            store, {"test": np.arange(store.n_files)}, count,
            config["features"],
            config.get("exclude_muons_above_gev", 0.0),
            config.get("exclude_muons", False),
            config.get("exclude_photons", False),
        )
        for store, count in zip(stores, files)
    ]
    mean, std, latent_scale = lc.load_norm_stats(
        str(source_dir / "norm_stats.json")
    )
    names = lc.feature_names(config["features"])
    model = build_model(config, len(mean), latent_scale)
    model.load_weights(str(source_dir / "best.weights.h5"))
    batch_size = args.batch_size or int(config["batch_size"])

    intervention_rows = []
    baseline_labels = None
    baseline_scores = None
    event_metrics = None
    for mode in MODES:
        print("evaluating {}".format(mode), flush=True)
        labels, scores, metrics = evaluate_mode(
            model, definitions, samplers, mean, std, names, mode,
            args.seed, batch_size, collect_metrics=(mode == "baseline")
        )
        auc = lc.auc_score(labels, scores)
        intervention_rows.append({"intervention": mode, "auc": auc})
        print("  {} AUC {:.6f}".format(mode, auc), flush=True)
        if mode == "baseline":
            baseline_labels = labels
            baseline_scores = scores
            event_metrics = metrics

    reported = load_json(source_dir / "point_summary.json")["auc"]
    baseline_auc = intervention_rows[0]["auc"]
    if abs(baseline_auc - reported) > 1e-5:
        raise RuntimeError(
            "baseline AUC {:.8f} does not reproduce reported AUC {:.8f}"
            .format(baseline_auc, reported)
        )

    for row, score in zip(event_metrics, baseline_scores):
        row["score"] = float(score)
    metric_fields = ["class", "construction", "particles", "score"]
    for harmonic in range(1, 7):
        metric_fields.extend([
            "cos{}".format(harmonic), "sin{}".format(harmonic),
            "r{}".format(harmonic)
        ])
    write_rows(output / "event_fourier_metrics.csv", metric_fields, event_metrics)
    write_rows(
        output / "phi_intervention_auc.csv", ["intervention", "auc"],
        intervention_rows
    )

    fourier_auc_rows = []
    correlation_rows = []
    for prefix in ("cos", "sin", "r"):
        for harmonic in range(1, 7):
            key = "{}{}".format(prefix, harmonic)
            values = np.asarray([row[key] for row in event_metrics])
            fourier_auc_rows.append({
                "metric": prefix, "harmonic": harmonic,
                "auc": lc.auc_score(baseline_labels, values),
            })
            for sample, mask in (
                ("all", np.ones(len(values), dtype=bool)),
                ("native", baseline_labels == 0),
                ("synthetic", baseline_labels == 1),
            ):
                correlation_rows.append({
                    "sample": sample,
                    "metric": prefix,
                    "harmonic": harmonic,
                    "pearson": safe_correlation(
                        values[mask], baseline_scores[mask]
                    ),
                })
    write_rows(
        output / "fourier_moment_auc.csv", ["metric", "harmonic", "auc"],
        fourier_auc_rows
    )
    write_rows(
        output / "fourier_score_correlation.csv",
        ["sample", "metric", "harmonic", "pearson"], correlation_rows
    )

    metadata = {
        "source_label": args.source_label,
        "source_checkpoint": str(source_dir / "best.weights.h5"),
        "test_units_per_class": args.test_units,
        "test_definition_seed": int(config["data_seed"]) + 2026,
        "intervention_seed": args.seed,
        "baseline_auc": baseline_auc,
        "reported_auc": reported,
        "interventions": {
            row["intervention"]: row["auc"] for row in intervention_rows
        },
    }
    with (output / "audit_summary.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    plot_interventions(
        {row["intervention"]: row["auc"] for row in intervention_rows},
        output
    )
    plot_fourier_auc(fourier_auc_rows, output)
    plot_score_correlations(correlation_rows, output)
    print("audit complete -> {}".format(output), flush=True)


if __name__ == "__main__":
    main()
