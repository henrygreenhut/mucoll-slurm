#!/usr/bin/env python3

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "pfn_results"
OUTPUT = ROOT / "plots" / "gen_n420_k1_audit"
ORIGINAL = "pm_n420_k1_vs_k1_synthetic_scaled_lr1e-4_decay80_mseed1_v1"
ORIGINAL_NULL = ORIGINAL + "_null"
LARGE_EVAL = "pm_n420_k1_vs_k1_synthetic_eval5000_cycleboot200_v1"
PHI_AUDIT = "pm_n420_k1_phi_audit_v1"
BASELINE = {
    1: ORIGINAL,
    2: "pm_n420_k1_vs_k1_synthetic_expanded_scaled_lr1e-4_decay80_mseed2_audit_v1",
    3: "pm_n420_k1_vs_k1_synthetic_expanded_scaled_lr1e-4_decay80_mseed3_audit_v1",
}
NO_PHI = {
    seed: "pm_n420_k1_vs_k1_synthetic_no_phi_scaled_lr1e-4_decay80_mseed{}_audit_v1".format(seed)
    for seed in (1, 2, 3)
}
NO_PHI_NULL = (
    "pm_n420_k1_vs_k1_synthetic_no_phi_scaled_lr1e-4_decay80_"
    "mseed1_audit_v1_null"
)
COLORS = {"baseline": "#0072B2", "no_phi": "#D55E00"}


def load_json(path):
    if not path.is_file():
        raise SystemExit("missing result: {}".format(path))
    with path.open() as handle:
        return json.load(handle)


def load_history(label):
    path = RESULTS / label / "history.csv"
    if not path.is_file():
        raise SystemExit("missing history: {}".format(path))
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in ("epoch", "val_auc", "val_loss")
    }


def save(figure, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "{}.pdf".format(name), bbox_inches="tight")
    figure.savefig(OUTPUT / "{}.png".format(name), dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_seed_auc(summaries, nulls):
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for position, (key, label) in enumerate((
        ("baseline", r"Expanded features"),
        ("no_phi", r"Expanded without $\phi$"),
    )):
        values = np.asarray([
            summaries[key][seed]["test_auc"] for seed in (1, 2, 3)
        ])
        offsets = np.asarray([-0.08, 0.0, 0.08])
        axis.scatter(
            position + offsets, values, s=55, color=COLORS[key],
            label=label, zorder=3
        )
        axis.hlines(
            np.mean(values), position - 0.18, position + 0.18,
            color=COLORS[key], linewidth=2.4
        )
    axis.scatter(
        [2 - 0.05, 2 + 0.05],
        [nulls["baseline"]["test_auc"], nulls["no_phi"]["test_auc"]],
        marker="D", s=48,
        color=[COLORS["baseline"], COLORS["no_phi"]], zorder=3
    )
    axis.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    axis.set_xticks((0, 1, 2), ("With φ", "Without φ", "Same-class null"))
    axis.set_ylabel("Test AUC")
    axis.set_title("GEN N=420 native K=1 vs synthetic K=1")
    axis.set_ylim(0.45, 0.9)
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save(figure, "feature_ablation_seed_auc")


def plot_histories():
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
    for column, (key, labels, title) in enumerate((
        ("baseline", BASELINE, "With φ"),
        ("no_phi", NO_PHI, "Without φ"),
    )):
        for seed, label in labels.items():
            history = load_history(label)
            epoch = history["epoch"] + 1
            axes[0, column].plot(
                epoch, history["val_auc"], linewidth=1.5,
                label="seed {}".format(seed)
            )
            axes[1, column].plot(
                epoch, history["val_loss"], linewidth=1.5,
                label="seed {}".format(seed)
            )
        axes[0, column].axhline(0.5, color="0.55", linestyle="--", linewidth=1)
        axes[1, column].axhline(
            np.log(2.0), color="0.55", linestyle="--", linewidth=1
        )
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel("Validation AUC")
        axes[1, column].set_ylabel("Validation loss")
        axes[1, column].set_xlabel("Epoch")
        for axis in axes[:, column]:
            axis.grid(alpha=0.2, linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0, column].legend(frameon=False)
    figure.suptitle("GEN N=420 K=1 rotation-control training")
    figure.tight_layout()
    save(figure, "feature_ablation_training")


def plot_evaluation_size(original, large):
    values = [original["test_auc"], large["test_auc"]]
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.scatter(
        (0, 1), values, s=70, color="#0072B2", zorder=3,
        label="Point estimate"
    )
    interval = large["bootstrap_ci95"]
    axis.vlines(
        1, interval[0], interval[1], color="#0072B2", linewidth=1.5,
        label="95% paired-cycle bootstrap interval"
    )
    axis.hlines(interval, 0.94, 1.06, color="#0072B2", linewidth=1.5)
    axis.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    axis.set_xticks((0, 1), ("300/class", "5,000/class"))
    axis.set_ylabel("Test AUC")
    axis.set_title("GEN N=420 K=1 frozen-PFN evaluation")
    axis.set_ylim(0.45, 0.9)
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    save(figure, "test_size_and_cycle_bootstrap")


def copy_phi_audit_plots():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = RESULTS / PHI_AUDIT
    for stem in (
        "phi_intervention_auc", "fourier_moment_auc",
        "fourier_score_correlation",
    ):
        for suffix in (".pdf", ".png"):
            path = source / "{}{}".format(stem, suffix)
            if not path.is_file():
                raise SystemExit("missing phi-audit plot: {}".format(path))
            shutil.copy2(path, OUTPUT / path.name)


def main():
    summaries = {
        "baseline": {
            seed: load_json(RESULTS / label / "auc_summary.json")
            for seed, label in BASELINE.items()
        },
        "no_phi": {
            seed: load_json(RESULTS / label / "auc_summary.json")
            for seed, label in NO_PHI.items()
        },
    }
    nulls = {
        "baseline": load_json(RESULTS / ORIGINAL_NULL / "auc_summary.json"),
        "no_phi": load_json(RESULTS / NO_PHI_NULL / "auc_summary.json"),
    }
    original = summaries["baseline"][1]
    large = load_json(RESULTS / LARGE_EVAL / "auc_summary.json")

    plot_seed_auc(summaries, nulls)
    plot_histories()
    plot_evaluation_size(original, large)
    copy_phi_audit_plots()

    rows = []
    for variant, values in summaries.items():
        for seed, summary in values.items():
            rows.append((variant, seed, summary["test_auc"]))
    rows.extend([
        ("baseline_null", 1, nulls["baseline"]["test_auc"]),
        ("no_phi_null", 1, nulls["no_phi"]["test_auc"]),
    ])
    with (OUTPUT / "feature_ablation_seed_auc.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("features", "model_seed", "test_auc"))
        writer.writerows(rows)
    print("plots -> {}".format(OUTPUT))


if __name__ == "__main__":
    main()
