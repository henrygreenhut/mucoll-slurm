#!/usr/bin/env python3
"""Create the plots used by the chronological GEN experiment survey.

These plots are deliberately generated from the saved history/config/result
files.  They group runs by the question they answered; they do not treat
resubmitted wall-clock windows as independent experiments.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "pfn_results"
DEFAULT_OUT = ROOT / "plots" / "gen_experiment_survey"


def read_json(path):
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def read_history(label):
    path = RESULTS / label / "history.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = {}
    for key in rows[0]:
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                values.append(np.nan)
        columns[key] = np.asarray(values)
    return columns


def result_auc(label, fallback=None):
    directory = RESULTS / label
    for name, key in (("auc_summary.json", "test_auc"),
                      ("point_summary.json", "auc")):
        data = read_json(directory / name)
        if key in data:
            return float(data[key])
    return fallback


def label_with_auc(text, label, fallback=None):
    auc = result_auc(label, fallback)
    return text if auc is None else f"{text} (test AUC {auc:.3f})"


def finish(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_histories(specs, title, path, show_val_loss=False, loss_log=False):
    """Two-panel training-loss/validation-AUC comparison."""
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for spec in specs:
        history = read_history(spec["run"])
        epoch = history["epoch"]
        color = spec.get("color")
        line, = ax_loss.plot(
            epoch, history["train_loss"], lw=2,
            label=spec["label"], color=color, ls=spec.get("ls", "-"))
        if show_val_loss and "val_loss" in history:
            ax_loss.plot(
                epoch, history["val_loss"], lw=1.7, color=line.get_color(),
                ls="--", alpha=0.9)
        ax_auc.plot(
            epoch, history["val_auc"], lw=2,
            label=spec["label"], color=line.get_color(),
            ls=spec.get("ls", "-"))

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    if loss_log:
        ax_loss.set_yscale("log")
    ax_loss.grid(alpha=0.25)
    ax_auc.axhline(0.5, color="0.45", lw=1.2, ls="--")
    ax_auc.set_xlabel("epoch")
    ax_auc.set_ylabel("validation AUC")
    ax_auc.set_ylim(0.35, 1.02)
    ax_auc.grid(alpha=0.25)
    ax_auc.legend(frameon=False, fontsize=9, loc="best")
    if show_val_loss:
        ax_loss.text(
            0.02, 0.98, "solid: train   dashed: validation",
            transform=ax_loss.transAxes, ha="left", va="top", fontsize=9)
    fig.suptitle(title, fontsize=15)
    finish(fig, path)


def plot_n42_baseline(out):
    specs = [
        {"run": "A0_n42_scaled_clean",
         "label": label_with_auc("scaled sum", "A0_n42_scaled_clean")},
        {"run": "A0_n42_paper_rawsum",
         "label": label_with_auc("raw sum", "A0_n42_paper_rawsum")},
        {"run": "A0_n42_null_shared_v2",
         "label": label_with_auc("scaled null", "A0_n42_null_shared_v2")},
    ]
    plot_histories(specs, "Initial GEN feasibility test at N=42",
                   out / "01_n42_feasibility.pdf")


def plot_size_sweeps(out):
    scaled = [
        {"run": "A0_n42_scaled_clean", "label": "N=42"},
        {"run": "A0_n126_scaled", "label": "N=126"},
        {"run": "A0_n210_scaled_disjoint", "label": "N=210"},
        {"run": "A0_n420_scaled_disjoint", "label": "N=420"},
    ]
    plot_histories(scaled, "Original scaled-sum PFN: sub-crossing-size sweep",
                   out / "02_scaled_size_sweep.pdf")

    raw = [
        {"run": "A0_n42_paper_rawsum", "label": "N=42"},
        {"run": "A0_n210_rawsum_disjoint", "label": "N=210"},
        {"run": "A0_n420_rawsum_disjoint", "label": "N=420"},
    ]
    plot_histories(raw, "Original raw-sum PFN: sub-crossing-size sweep",
                   out / "03_raw_size_sweep.pdf", loss_log=True)


def plot_n420_original(out):
    specs = [
        {"run": "A0_n420_scaled_disjoint",
         "label": label_with_auc(
             "scaled sum", "EVAL_n420_paired_overlap",
             result_auc("A0_n420_scaled_disjoint"))},
        {"run": "gen_n420_scaled_20260717",
         "label": label_with_auc(
             "scaled repeat", "gen_n420_scaled_20260717")},
        {"run": "A0_n420_rawsum_disjoint",
         "label": label_with_auc(
             "raw sum", "EVAL_n420_rawsum_paired_overlap",
             result_auc("A0_n420_rawsum_disjoint"))},
        {"run": "A0_n420_null_shared",
         "label": label_with_auc("scaled null", "A0_n420_null_shared")},
    ]
    plot_histories(specs, "First N=420 result and collapse check",
                   out / "04_first_n420_result.pdf", loss_log=True)


def plot_capacity(out):
    specs = [
        {"run": "halfphi_raw_small",
         "label": "raw: Φ=(50,50,64), F=(100,100,100), batch 8"},
        {"run": "halfphi_raw_large",
         "label": "raw: Φ=(100,100,128), F=(200,200,200), batch 4"},
        {"run": "halfphi_scaled_large",
         "label": "scaled: Φ=(100,100,128), F=(200,200,200), batch 4"},
        {"run": "halfphi_null_large",
         "label": "null: Φ=(100,100,128), F=(200,200,200), batch 4"},
    ]
    plot_histories(specs, "N=420 capacity and pooling smoke test",
                   out / "05_n420_capacity_scan.pdf",
                   show_val_loss=True, loss_log=True)


def plot_hardware_envelope(out):
    """Derived view of the measured memory and int32 capacity constraints."""
    batches = np.asarray([1, 2, 4, 8])
    int32_max = float(2**31 - 1)
    typical_n420 = 1_255_800
    observed_large = 2_070_000

    fig, (ax_index, ax_memory) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for width, marker, color in ((256, "o", "C0"), (128, "s", "C1")):
        safe_n = int32_max / (batches * width)
        ax_index.plot(
            batches, safe_n / 1e6, marker=marker, lw=2, color=color,
            label=rf"widest $\Phi$ layer = {width}")
    ax_index.axhline(
        typical_n420 / 1e6, color="0.25", ls="--", lw=1.5,
        label="typical N=420 event (~1.26M)")
    ax_index.axhline(
        observed_large / 1e6, color="0.5", ls=":", lw=1.5,
        label="large N=420 draw observed in scan")
    ax_index.set_xticks(batches)
    ax_index.set_xlabel("batch size")
    ax_index.set_ylabel("maximum particles/event before int32 limit [million]")
    ax_index.set_yscale("log", base=2)
    ax_index.grid(alpha=0.25)
    ax_index.legend(frameon=False, fontsize=8)

    # The benchmark measured about 3.8 kB of peak GPU memory per padded
    # particle slot for the original wide nine-input PFN. This is an
    # empirical rule of thumb, not a prediction for the half-width model.
    estimated_gb = batches * typical_n420 * 3800 / 1024**3
    ax_memory.plot(
        batches, estimated_gb, marker="o", lw=2, color="C0",
        label="wide PFN, measured ~3.8 kB/slot")
    ax_memory.axhline(40, color="0.35", ls="--", lw=1.5, label="40 GB A100")
    ax_memory.axhline(80, color="0.6", ls=":", lw=1.5, label="80 GB A100")
    ax_memory.scatter(
        [8], [estimated_gb[-1]], marker="x", s=90, lw=2.5, color="C3",
        label="batch 8: int32 abort despite memory headroom")
    ax_memory.set_xticks(batches)
    ax_memory.set_xlabel("batch size")
    ax_memory.set_ylabel("approximate peak GPU memory [GB]")
    ax_memory.set_ylim(0, 85)
    ax_memory.grid(alpha=0.25)
    ax_memory.legend(frameon=False, fontsize=8)

    fig.suptitle("N=420 PFN hardware-capacity diagnosis", fontsize=15)
    finish(fig, out / "05a_n420_hardware_envelope.pdf")


def plot_energyflow_backend(out):
    specs = []
    for seed in (1, 2, 3):
        specs.append({
            "run": f"oscar_n42_energyflow_pfn_seed{seed}",
            "label": f"official PFN raw, seed {seed}",
            "color": f"C{seed - 1}",
        })
    for seed in (1, 2, 3):
        specs.append({
            "run": f"oscar_n42_energyflow_efn_seed{seed}",
            "label": f"EnergyFlow scaled, seed {seed}",
            "color": f"C{seed - 1}",
            "ls": "--",
        })
    plot_histories(specs, "N=42 EnergyFlow implementation and seed check",
                   out / "06_energyflow_seed_check.pdf", loss_log=True)


def plot_stabilization(out):
    specs = [
        {"run": "oscar_n420_halfphi_raw_seed1",
         "label": label_with_auc(
             "raw, fixed LR", "oscar_n420_halfphi_raw_seed1")},
        {"run": "oscar_n420_halfphi_raw_seed1_w1_c0",
         "label": label_with_auc(
             "raw, 1-epoch warmup", "oscar_n420_halfphi_raw_seed1_w1_c0")},
        {"run": "oscar_n420_halfphi_scaled_seed1",
         "label": label_with_auc(
             "scaled, fixed LR", "oscar_n420_halfphi_scaled_seed1")},
        {"run": "oscar_n420_halfphi_scaled_seed1_w1_c0",
         "label": label_with_auc(
             "scaled, 1-epoch warmup",
             "oscar_n420_halfphi_scaled_seed1_w1_c0")},
    ]
    plot_histories(specs, "N=420 pooling and warmup comparison",
                   out / "07_n420_pooling_warmup.pdf",
                   show_val_loss=True, loss_log=True)

    diagnostics = [
        {"run": "diag_warmup_clip_n420_clip",
         "label": "raw, clipnorm 1"},
        {"run": "diag_warmup_clip_n420_warmup",
         "label": "raw, warmup, no clipping"},
        {"run": "diag_warmup_clip_n420_warmup_efn",
         "label": "scaled, warmup, no clipping"},
    ]
    plot_histories(diagnostics, "N=420 clipping diagnostic",
                   out / "08_n420_clipping_diagnostic.pdf",
                   show_val_loss=True, loss_log=True)


def plot_final_grid(out):
    specs = [
        {"run": "n420_recipe_bs4_expanded_scaled_lr1e-4_mseed1_pointonly",
         "label": label_with_auc(
             r"scaled, LR $10^{-4}$",
             "n420_recipe_bs4_expanded_scaled_lr1e-4_mseed1_pointonly")},
        {"run": "n420_recipe_bs4_expanded_scaled_lr3e-4_mseed1_pointonly",
         "label": label_with_auc(
             r"scaled, LR $3\times10^{-4}$",
             "n420_recipe_bs4_expanded_scaled_lr3e-4_mseed1_pointonly")},
        {"run": "n420_recipe_bs4_expanded_raw_lr1e-4_mseed1_pointonly",
         "label": label_with_auc(
             r"raw, LR $10^{-4}$",
             "n420_recipe_bs4_expanded_raw_lr1e-4_mseed1_pointonly")},
    ]
    plot_histories(specs, "Stabilized N=420 recipe comparison",
                   out / "09_n420_final_recipe_grid.pdf",
                   show_val_loss=True, loss_log=True)

    checks = [
        {"run": "n420_recipe_bs4_expanded_scaled_lr1e-4_mseed1_pointonly",
         "label": label_with_auc(
             "main", "n420_recipe_bs4_expanded_scaled_lr1e-4_mseed1_pointonly")},
        {"run": "n420_recipe_bs4_expanded_scaled_lr1e-4_mseed1_repro1_pointonly",
         "label": label_with_auc(
             "repeat",
             "n420_recipe_bs4_expanded_scaled_lr1e-4_mseed1_repro1_pointonly")},
        {"run": "n420_recipe_bs4_expanded_scaled_lr1e-4_mseed1_null_pointonly",
         "label": label_with_auc(
             "null",
             "n420_recipe_bs4_expanded_scaled_lr1e-4_mseed1_null_pointonly")},
    ]
    plot_histories(checks, "Final N=420 reproducibility and null checks",
                   out / "10_n420_final_checks.pdf", show_val_loss=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.outdir.resolve()
    plot_n42_baseline(out)
    plot_size_sweeps(out)
    plot_n420_original(out)
    plot_hardware_envelope(out)
    plot_capacity(out)
    plot_energyflow_backend(out)
    plot_stabilization(out)
    plot_final_grid(out)
    print(f"wrote 11 plots to {out}")


if __name__ == "__main__":
    main()
