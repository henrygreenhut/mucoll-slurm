#!/usr/bin/env python3

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


N_VALUES = (420, 840, 1260)
COLORS = {"U": "#0072B2", "R": "#D55E00", "null_b": "0.45"}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("plots/reco_signal_audit"))
    parser.add_argument("--weights", type=Path)
    return parser.parse_args()


def load_store(directory, n, class_name):
    path = directory / "n{}_{}_test.h5".format(n, class_name)
    with h5py.File(path, "r") as store:
        raw = store["particles"][:]
        mask = raw[:, :, 0] > 0
        return {
            "raw": raw,
            "n_pfo": store["n_particles"][:],
            "n_track": store["n_tracks"][:],
            "n_cluster": store["n_clusters"][:],
            "n_charged": ((np.abs(raw[:, :, 5]) > 0.1) & mask).sum(axis=1),
            "sum_pt": raw[:, :, 0].sum(axis=1),
            "sum_energy": raw[:, :, 3].sum(axis=1),
        }


def load_scores(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["true_label"]) for row in rows])
    scores = np.asarray([float(row["score"]) for row in rows])
    files = [row["source_file"] for row in rows]
    return labels, scores, files


def save(figure, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / "{}.{}".format(name, suffix), dpi=200, bbox_inches="tight")
    plt.close(figure)


def event_distributions(data, output_dir):
    observables = (
        ("n_pfo", "PFOs per event"),
        ("n_charged", "Charged PFOs per event"),
        ("sum_energy", "Total PFO energy [GeV]"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(11.5, 8.5))
    for column, n in enumerate(N_VALUES):
        for row, (key, label) in enumerate(observables):
            axis = axes[row, column]
            values = np.concatenate([data[n][name][key] for name in ("U", "R", "null_b")])
            high = np.percentile(values, 99.5)
            bins = np.linspace(0, high, 42)
            for name, text, style in (
                ("U", "Unique", "-"),
                ("R", "42× reuse", "-"),
                ("null_b", "Null unique", "--"),
            ):
                axis.hist(
                    data[n][name][key], bins=bins, density=True,
                    histtype="step", linewidth=1.7, linestyle=style,
                    color=COLORS[name], label=text,
                )
            if row == 0:
                axis.set_title("N={}".format(n))
            if column == 0:
                axis.set_ylabel("Density")
            axis.set_xlabel(label)
            axis.grid(axis="y", alpha=0.2, linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    save(figure, output_dir, "reco_event_observables_vs_n")


def scalar_auc_plot(data, score_dir, output_dir):
    series = {
        "PFN": [],
        "PFO count": [],
        "Charged-PFO count": [],
        "Total PFO pT": [],
        "Total PFO energy": [],
        "Null PFN": [],
    }
    rows = []
    for n in N_VALUES:
        labels = np.r_[np.zeros(800), np.ones(800)]
        y, scores, _ = load_scores(score_dir / "n{}_U_vs_R_test_scores.csv".format(n))
        series["PFN"].append(roc_auc_score(y, scores))
        for name, key in (
            ("PFO count", "n_pfo"),
            ("Charged-PFO count", "n_charged"),
            ("Total PFO pT", "sum_pt"),
            ("Total PFO energy", "sum_energy"),
        ):
            values = np.r_[data[n]["U"][key], data[n]["R"][key]]
            series[name].append(roc_auc_score(labels, values))
        y_null, scores_null, _ = load_scores(score_dir / "n{}_null_test_scores.csv".format(n))
        series["Null PFN"].append(roc_auc_score(y_null, scores_null))
        rows.append([n] + [series[name][-1] for name in series])

    with (output_dir / "reco_pfo_scalar_auc_vs_n.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["N"] + list(series))
        writer.writerows(rows)

    colors = ("#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9")
    figure, axis = plt.subplots(figsize=(6.5, 4.6))
    for (name, values), color in zip(list(series.items())[:-1], colors):
        axis.plot(N_VALUES, values, marker="o", linewidth=2, label=name, color=color)
    axis.plot(N_VALUES, series["Null PFN"], marker="o", linewidth=1.5,
              linestyle="--", label="Null PFN", color="0.45")
    axis.axhline(0.5, color="0.55", linewidth=1, linestyle="--")
    axis.set_xticks(N_VALUES)
    axis.set_xlabel("N")
    axis.set_ylabel("Test AUC")
    axis.set_title("PFN and single-observable PFO separation")
    axis.set_ylim(0.47, 0.85)
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=9, ncol=2)
    save(figure, output_dir, "reco_pfo_scalar_auc_vs_n")


def matched_indices(first, second, keys, rng):
    groups = []
    for values in (first, second):
        grouped = defaultdict(list)
        for index in range(len(values[0])):
            key = tuple(int(values[column][index]) for column in keys)
            grouped[key].append(index)
        groups.append(grouped)
    selected = ([], [])
    for key in sorted(set(groups[0]) & set(groups[1])):
        left = np.asarray(groups[0][key])
        right = np.asarray(groups[1][key])
        rng.shuffle(left)
        rng.shuffle(right)
        count = min(len(left), len(right))
        selected[0].extend(left[:count])
        selected[1].extend(right[:count])
    return tuple(np.asarray(values, dtype=int) for values in selected)


def multiplicity_matched_plot(data, score_dir, output_dir):
    full = []
    pfo_matched = []
    both_matched = []
    rows = []
    rng = np.random.default_rng(20260731)
    for n in N_VALUES:
        labels, scores, _ = load_scores(score_dir / "n{}_U_vs_R_test_scores.csv".format(n))
        full.append(roc_auc_score(labels, scores))
        values = []
        for class_name in ("U", "R"):
            values.append((data[n][class_name]["n_pfo"], data[n][class_name]["n_charged"]))
        row = [n, full[-1]]
        for keys, output in (((0,), pfo_matched), ((0, 1), both_matched)):
            left, right = matched_indices(values[0], values[1], keys, rng)
            matched_scores = np.r_[scores[left], scores[800 + right]]
            matched_labels = np.r_[np.zeros(len(left)), np.ones(len(right))]
            output.append(roc_auc_score(matched_labels, matched_scores))
            row.extend((output[-1], len(left)))
        rows.append(row)

    with (output_dir / "reco_multiplicity_matched_auc.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("N", "full_auc", "pfo_count_matched_auc", "pfo_count_matched_events_per_class",
                         "pfo_and_charged_matched_auc", "pfo_and_charged_matched_events_per_class"))
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.plot(N_VALUES, full, marker="o", linewidth=2, label="All test events", color="#0072B2")
    axis.plot(N_VALUES, pfo_matched, marker="o", linewidth=2,
              label="Matched PFO count", color="#009E73")
    axis.plot(N_VALUES, both_matched, marker="o", linewidth=2,
              label="Matched PFO and charged-PFO counts", color="#E69F00")
    axis.axhline(0.5, color="0.55", linewidth=1, linestyle="--")
    axis.set_xticks(N_VALUES)
    axis.set_xlabel("N")
    axis.set_ylabel("Test AUC")
    axis.set_title("PFN separation after multiplicity matching")
    axis.set_ylim(0.48, 0.87)
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=9)
    save(figure, output_dir, "reco_multiplicity_matched_auc")


def job_means(labels, scores, files):
    grouped = {0: defaultdict(list), 1: defaultdict(list)}
    for label, score, source in zip(labels, scores, files):
        job = int(re.search(r"job_(\d+)", source).group(1))
        grouped[int(label)][job].append(score)
    jobs = sorted(set(grouped[0]) & set(grouped[1]))
    first = np.asarray([np.mean(grouped[0][job]) for job in jobs])
    second = np.asarray([np.mean(grouped[1][job]) for job in jobs])
    return jobs, first, second


def paired_job_bootstrap(labels, scores, files, repetitions=2000):
    jobs, _, _ = job_means(labels, scores, files)
    grouped = {0: defaultdict(list), 1: defaultdict(list)}
    for label, score, source in zip(labels, scores, files):
        job = int(re.search(r"job_(\d+)", source).group(1))
        grouped[int(label)][job].append(score)
    rng = np.random.default_rng(20260727)
    aucs = []
    for _ in range(repetitions):
        chosen = rng.choice(jobs, len(jobs), replace=True)
        a = np.concatenate([grouped[0][job] for job in chosen])
        b = np.concatenate([grouped[1][job] for job in chosen])
        aucs.append(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]))
    return np.percentile(aucs, [2.5, 97.5])


def confirmation_plots(input_dir, output_dir):
    confirmation = input_dir / "confirmation"
    comparisons = (
        ("Main", input_dir / "n1260_U_vs_R_test_scores.csv", confirmation / "test2000_scores.csv"),
        ("Null", input_dir / "n1260_null_test_scores.csv", confirmation / "null_" / "test2000_scores.csv"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    summary_rows = []
    for axis, (title, original_path, extended_path) in zip(axes, comparisons):
        original = load_scores(original_path)
        extended = load_scores(extended_path)
        for index, (cohort, values) in enumerate((("800/class", original), ("2,000/class", extended))):
            labels, scores, files = values
            auc = roc_auc_score(labels, scores)
            low, high = paired_job_bootstrap(labels, scores, files)
            axis.errorbar(index, auc, yerr=[[auc - low], [high - auc]], fmt="o",
                          markersize=7, capsize=4, color=COLORS["R"] if title == "Main" else "0.45")
            paired_jobs = len(set(files)) // 2
            summary_rows.append((title, cohort, auc, low, high, paired_jobs))
        axis.axhline(0.5, color="0.55", linewidth=1, linestyle="--")
        axis.set_xticks((0, 1), ("800/class", "2,000/class"))
        axis.set_title(title)
        axis.set_ylabel("Test AUC")
        axis.set_ylim(0.47, 0.84)
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("N=1260 frozen-PFN evaluation")
    with (output_dir / "reco_n1260_test_size_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("comparison", "events_per_class", "auc", "job_bootstrap_low", "job_bootstrap_high", "paired_reco_jobs"))
        writer.writerows(summary_rows)
    save(figure, output_dir, "reco_n1260_test_size_comparison")

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for axis, (title, _, path) in zip(axes, comparisons):
        labels, scores, files = load_scores(path)
        jobs, first, second = job_means(labels, scores, files)
        limits = (min(first.min(), second.min()) - 0.02, max(first.max(), second.max()) + 0.02)
        axis.scatter(first, second, s=28, color=COLORS["R"] if title == "Main" else "0.45")
        axis.plot(limits, limits, color="0.55", linestyle="--", linewidth=1)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_xlabel("Mean class-A score in job")
        axis.set_ylabel("Mean class-B score in job")
        axis.set_title("{}: {} paired jobs".format(title, len(jobs)))
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("N=1260 PFN scores by reconstruction job")
    save(figure, output_dir, "reco_n1260_score_by_job")


def permutation_plot(input_dir, output_dir, weights, data):
    import train_reco_libtest_pfn as trainer
    from reco_libtest_features import pfn_features

    raw = [data[1260][name]["raw"] for name in ("U", "R")]
    width = max(array.shape[1] for array in raw)
    x = np.concatenate([
        pfn_features(trainer.pad_width(array, width)) for array in raw
    ])
    labels = np.r_[np.zeros(len(raw[0])), np.ones(len(raw[1]))]
    mask = np.any(x != 0, axis=2)
    rows, particles = np.where(mask)
    model = trainer.get_pfn(7, "stabilized_dropout", 125)
    model = trainer.underlying_model(model)
    model.load_weights(str(weights))

    def evaluate(array):
        scores = model.predict(array, batch_size=32, verbose=0)[:, 1]
        return roc_auc_score(labels, scores)

    groups = (
        ("None", ()),
        ("log pT", (0,)),
        ("eta", (1,)),
        ("phi", (2, 3)),
        ("log energy", (4,)),
        ("charge", (5, 6)),
    )
    rng = np.random.default_rng(713)
    results = []
    for name, columns in groups:
        repetitions = 1 if not columns else 5
        values = []
        for _ in range(repetitions):
            changed = x.copy()
            if columns:
                shuffled = changed[mask][:, columns].copy()
                rng.shuffle(shuffled, axis=0)
                for index, column in enumerate(columns):
                    changed[rows, particles, column] = shuffled[:, index]
            values.append(evaluate(changed))
        results.append((name, np.mean(values), np.std(values)))

    with (output_dir / "reco_n1260_permutation_importance.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("shuffled_features", "mean_auc", "std_auc"))
        writer.writerows(results)

    figure, axis = plt.subplots(figsize=(6.5, 4.3))
    names = [row[0] for row in results]
    values = [row[1] for row in results]
    errors = [row[2] for row in results]
    axis.bar(names, values, yerr=errors, capsize=3, color="#0072B2")
    axis.axhline(0.5, color="0.55", linewidth=1, linestyle="--")
    axis.set_ylabel("Test AUC after permutation")
    axis.set_title("N=1260 PFO-feature permutation test")
    axis.set_ylim(0.48, 0.85)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, output_dir, "reco_n1260_permutation_importance")

    split_results = []
    for split in ("train", "val", "test"):
        if split == "test":
            split_x = x
            split_labels = labels
        else:
            arrays = []
            for class_name in ("U", "R"):
                path = input_dir / "n1260_{}_{}.h5".format(class_name, split)
                with h5py.File(path, "r") as store:
                    arrays.append(store["particles"][:])
            split_width = max(array.shape[1] for array in arrays)
            split_x = np.concatenate([
                pfn_features(trainer.pad_width(array, split_width))
                for array in arrays
            ])
            split_labels = np.r_[
                np.zeros(len(arrays[0])), np.ones(len(arrays[1]))
            ]
        split_scores = model.predict(split_x, batch_size=32, verbose=0)[:, 1]
        split_results.append((
            split,
            roc_auc_score(split_labels, split_scores),
            log_loss(split_labels, split_scores),
            len(split_labels) // 2,
        ))

    with (output_dir / "reco_n1260_split_performance.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "auc", "loss", "events_per_class"))
        writer.writerows(split_results)

    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.1))
    names = [row[0].capitalize() for row in split_results]
    axes[0].bar(names, [row[1] for row in split_results], color="#0072B2")
    axes[0].axhline(0.5, color="0.55", linewidth=1, linestyle="--")
    axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0.48, 0.85)
    axes[1].bar(names, [row[2] for row in split_results], color="#D55E00")
    axes[1].axhline(np.log(2.0), color="0.55", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Loss")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("N=1260 frozen-PFN performance by source split")
    save(figure, output_dir, "reco_n1260_split_performance")


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "serif"
    data = {
        n: {name: load_store(args.input_dir, n, name) for name in ("U", "R", "null_b")}
        for n in N_VALUES
    }
    event_distributions(data, args.output_dir)
    scalar_auc_plot(data, args.input_dir, args.output_dir)
    multiplicity_matched_plot(data, args.input_dir, args.output_dir)
    confirmation_plots(args.input_dir, args.output_dir)
    if args.weights:
        permutation_plot(args.input_dir, args.output_dir, args.weights, data)


if __name__ == "__main__":
    main()
