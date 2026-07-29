#!/usr/bin/env python3
"""Make presentation-style loss overlays for completed production studies."""

import argparse
import csv
from pathlib import Path

from pfn_libtest_plot_overlay import load_test_auc, make_plot


RUNS = (
    # N=420 architecture development.
    ("pfn_results/oscar_n420_halfphi_raw_seed1",
     "GEN N=420 — raw sum, fixed LR",
     "gen_n420_development/gen_n420_raw_fixed_lr.pdf"),
    ("pfn_results/oscar_n420_halfphi_raw_seed1_w1_c0",
     "GEN N=420 — raw sum with warmup",
     "gen_n420_development/gen_n420_raw_warmup.pdf"),
    ("pfn_results/oscar_n420_halfphi_scaled_seed1",
     "GEN N=420 — scaled sum, fixed LR",
     "gen_n420_development/gen_n420_scaled_fixed_lr.pdf"),
    ("pfn_results/oscar_n420_halfphi_scaled_seed1_w1_c0",
     "GEN N=420 — scaled sum with warmup",
     "gen_n420_development/gen_n420_scaled_warmup.pdf"),

    # Final N=420 architecture and reproducibility studies.
    ("pfn_results/n420_recipe_bs4_expanded_scaled_lr1e-4_mseed1_pointonly",
     "GEN-Level PFN at N=420",
     "gen_n420_recipe/gen_n420_scaled_lr1e-4.pdf"),
    ("pfn_results/n420_recipe_bs4_expanded_scaled_lr1e-4_mseed1_repro1_pointonly",
     "GEN N=420 — scaled sum, independent repeat",
     "gen_n420_recipe/gen_n420_scaled_lr1e-4_repeat.pdf"),
    ("pfn_results/n420_recipe_bs4_expanded_scaled_lr3e-4_mseed1_pointonly",
     "GEN N=420 — scaled sum, LR $3\\times10^{-4}$",
     "gen_n420_recipe/gen_n420_scaled_lr3e-4.pdf"),
    ("pfn_results/n420_recipe_bs4_expanded_raw_lr1e-4_mseed1_pointonly",
     "GEN N=420 — raw sum, LR $10^{-4}$",
     "gen_n420_recipe/gen_n420_raw_lr1e-4.pdf"),
    ("pfn_results/n420_recipe_bs4_expanded_scaled_lr1e-4_mseed1_null_pointonly",
     "GEN N=420 — scaled-sum null",
     "gen_n420_recipe/gen_n420_scaled_lr1e-4_null.pdf"),

    # Synthetic mother-level reuse scan.
    ("variable_k_results/n420_k1_vs_k10_scaled_lr1e-4_mseed1",
     "GEN N=420 — 1× versus 10× reuse",
     "gen_variable_k/gen_n420_k1_vs_k10.pdf"),
    ("variable_k_results/n420_k10_vs_k42_scaled_lr1e-4_mseed1",
     "GEN N=420 — 10× versus 42× reuse",
     "gen_variable_k/gen_n420_k10_vs_k42.pdf"),
    ("variable_k_results/n420_k10_vs_k42_scaled_lr1e-4_mseed1_null",
     "GEN N=420 — 10× versus 42× null",
     "gen_variable_k/gen_n420_k10_vs_k42_null.pdf"),
    ("variable_k_results/n420_k1_vs_k10_scaled_lr1e-4_mseed1_null",
     "GEN N=420 — 1× versus 10× null",
     "gen_variable_k/gen_n420_k1_vs_k10_null.pdf"),
    ("variable_k_results/n420_k1_vs_k5_scaled_lr1e-4_mseed1",
     "GEN N=420 — 1× versus 5× reuse",
     "gen_variable_k/gen_n420_k1_vs_k5.pdf"),
    ("variable_k_results/n420_k1_vs_k5_scaled_lr1e-4_mseed1_null",
     "GEN N=420 — 1× versus 5× null",
     "gen_variable_k/gen_n420_k1_vs_k5_null.pdf"),
    ("variable_k_results/n420_k1_vs_k5_scaled_lr1e-4_mseed1_min80",
     "GEN N=420 — 1× versus 5× reuse, minimum 80 epochs",
     "gen_variable_k/gen_n420_k1_vs_k5_min80.pdf"),
    ("variable_k_results/n420_k1_vs_k5_scaled_lr1e-4_mseed1_null_min80",
     "GEN N=420 — 1× versus 5× null, minimum 80 epochs",
     "gen_variable_k/gen_n420_k1_vs_k5_null_min80.pdf"),
    ("variable_k_results/n420_k5_vs_k10_scaled_lr1e-4_mseed1",
     "GEN N=420 — 5× versus 10× reuse",
     "gen_variable_k/gen_n420_k5_vs_k10.pdf"),
    ("variable_k_results/n420_k5_vs_k10_scaled_lr1e-4_mseed1_null",
     "GEN N=420 — 5× versus 10× null",
     "gen_variable_k/gen_n420_k5_vs_k10_null.pdf"),

    # Reconstructed-PFO study.
    ("reco_pfn_results/reco_n420_simple_U_vs_R",
     "RECO N=420 — baseline PFN", "reco_n420/reco_n420_baseline.pdf"),
    ("reco_pfn_results/reco_n420_simple_null",
     "RECO N=420 — baseline null", "reco_n420/reco_n420_baseline_null.pdf"),
    ("reco_pfn_results/reco_n420_stabilized_U_vs_R",
     "RECO N=420 — stabilized PFN", "reco_n420/reco_n420_stabilized.pdf"),
    ("reco_pfn_results/reco_n420_stabilized_null",
     "RECO N=420 — stabilized null", "reco_n420/reco_n420_stabilized_null.pdf"),
    ("reco_pfn_results/reco_n420_stabilized_dropout_U_vs_R",
     "RECO N=420 — stabilized PFN with dropout",
     "reco_n420/reco_n420_stabilized_dropout.pdf"),
    ("reco_pfn_results/reco_n420_stabilized_dropout_null",
     "RECO N=420 — stabilized dropout null",
     "reco_n420/reco_n420_stabilized_dropout_null.pdf"),

    # Direct-log PFO preprocessing. Separate names preserve the earlier
    # clipped/scaled-feature results and plots for comparison.
    ("reco_pfn_results/reco_n420_directlog_baseline_U_vs_R",
     "RECO N=420 — direct-log baseline PFN",
     "reco_n420_directlog/reco_n420_directlog_baseline.pdf"),
    ("reco_pfn_results/reco_n420_directlog_baseline_null",
     "RECO N=420 — direct-log baseline null",
     "reco_n420_directlog/reco_n420_directlog_baseline_null.pdf"),
    ("reco_pfn_results/reco_n420_directlog_stabilized_U_vs_R",
     "RECO N=420 — direct-log stabilized PFN",
     "reco_n420_directlog/reco_n420_directlog_stabilized.pdf"),
    ("reco_pfn_results/reco_n420_directlog_stabilized_null",
     "RECO N=420 — direct-log stabilized null",
     "reco_n420_directlog/reco_n420_directlog_stabilized_null.pdf"),
    ("reco_pfn_results/reco_n420_directlog_stabilized_dropout_U_vs_R",
     "RECO N=420 — direct-log stabilized PFN with dropout",
     "reco_n420_directlog/reco_n420_directlog_stabilized_dropout.pdf"),
    ("reco_pfn_results/reco_n420_directlog_stabilized_dropout_null",
     "RECO N=420 — direct-log stabilized dropout null",
     "reco_n420_directlog/reco_n420_directlog_stabilized_dropout_null.pdf"),

    # Track-fixed direct-log results are scientifically distinct from the
    # legacy direct-log models above, which used the old simple stores.
    ("reco_pfn_results/"
     "reco_n420_trackfix_directlog_stabilized_dropout_U_vs_R",
     "RECO N=420 — track-fixed stabilized PFN with dropout",
     "reco_n420_trackfix_directlog/"
     "reco_n420_trackfix_directlog_stabilized_dropout.pdf"),
    ("reco_pfn_results/"
     "reco_n420_trackfix_directlog_stabilized_dropout_null",
     "RECO N=420 — track-fixed stabilized dropout null",
     "reco_n420_trackfix_directlog/"
     "reco_n420_trackfix_directlog_stabilized_dropout_null.pdf"),
    ("reco_pfn_results/"
     "reco_n420_trackfix_directlog_minimal6_stabilized_dropout_U_vs_R",
     "RECO N=420 — track-fixed minimal-six PFN",
     "reco_n420_trackfix_directlog_minimal6/"
     "reco_n420_trackfix_directlog_minimal6_stabilized_dropout.pdf"),
    ("reco_pfn_results/"
     "reco_n420_trackfix_directlog_minimal6_stabilized_dropout_null",
     "RECO N=420 — track-fixed minimal-six null",
     "reco_n420_trackfix_directlog_minimal6/"
     "reco_n420_trackfix_directlog_minimal6_stabilized_dropout_null.pdf"),
    ("reco_pfn_results/"
     "reco_n420_trackfix_directlog_charged7_stabilized_dropout_U_vs_R",
     "RECO N=420 — track-fixed charged-flag PFN",
     "reco_n420_trackfix_directlog_charged7/"
     "reco_n420_trackfix_directlog_charged7_stabilized_dropout.pdf"),
    ("reco_pfn_results/"
     "reco_n420_trackfix_directlog_charged7_stabilized_dropout_null",
     "RECO N=420 — track-fixed charged-flag null",
     "reco_n420_trackfix_directlog_charged7/"
     "reco_n420_trackfix_directlog_charged7_stabilized_dropout_null.pdf"),
    ("reco_pfn_results/"
     "reco_n420_trackfix_val2000_directlog_charged7_"
     "stabilized_dropout_U_vs_R",
     "RECO N=420 — charged-flag PFN, 2,000 validation events",
     "reco_n420_trackfix_validation/"
     "reco_n420_trackfix_charged7_val2000.pdf"),
    ("reco_pfn_results/"
     "reco_n420_trackfix_val2000_directlog_charged7_"
     "stabilized_dropout_null",
     "RECO N=420 — charged-flag null, 2,000 validation events",
     "reco_n420_trackfix_validation/"
     "reco_n420_trackfix_charged7_val2000_null.pdf"),
    ("reco_pfn_results/"
     "reco_n420_trackfix_val25_directlog_charged7_"
     "stabilized_dropout_U_vs_R",
     "RECO N=420 — charged-flag PFN, 25% validation cycles",
     "reco_n420_trackfix_validation/"
     "reco_n420_trackfix_charged7_val25.pdf"),
    ("reco_pfn_results/"
     "reco_n420_trackfix_val25_directlog_charged7_"
     "stabilized_dropout_null",
     "RECO N=420 — charged-flag null, 25% validation cycles",
     "reco_n420_trackfix_validation/"
     "reco_n420_trackfix_charged7_val25_null.pdf"),
    ("reco_pfn_results/"
     "reco_n840_trackfix_val25_directlog_charged7_"
     "stabilized_dropout_U_vs_R",
     "RECO N=840 — charged-flag PFN, 25% validation cycles",
     "reco_n840_trackfix/"
     "reco_n840_trackfix_charged7_val25.pdf"),
    ("reco_pfn_results/"
     "reco_n840_trackfix_val25_directlog_charged7_"
     "stabilized_dropout_null",
     "RECO N=840 — charged-flag null, 25% validation cycles",
     "reco_n840_trackfix/"
     "reco_n840_trackfix_charged7_val25_null.pdf"),
    ("reco_pfn_results/"
     "reco_n1260_trackfix_val25_directlog_charged7_"
     "stabilized_dropout_U_vs_R",
     "RECO N=1260 — charged-flag PFN, 25% validation cycles",
     "reco_n1260_trackfix/"
     "reco_n1260_trackfix_charged7_val25.pdf"),
    ("reco_pfn_results/"
     "reco_n1260_trackfix_val25_directlog_charged7_"
     "stabilized_dropout_null",
     "RECO N=1260 — charged-flag null, 25% validation cycles",
     "reco_n1260_trackfix/"
     "reco_n1260_trackfix_charged7_val25_null.pdf"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        help=(
            "only make plots whose first output-directory component matches "
            "this value, e.g. reco_n420_directlog"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path("plots/training_overlays")
    made = 0
    for rundir, title, relative_output in RUNS:
        study = Path(relative_output).parts[0]
        if args.study and study != args.study:
            continue
        path = Path(rundir)
        history = path / "history.csv"
        if not history.is_file() or load_test_auc(path) is None:
            print("skip incomplete -> {}".format(path))
            continue
        with history.open() as handle:
            columns = next(csv.reader(handle))
        if "val_loss" not in columns:
            print("skip: validation loss was not recorded -> {}".format(path))
            continue
        make_plot(path, title, output_root / relative_output)
        made += 1
    print("{} completed overlays -> {}".format(made, output_root))


if __name__ == "__main__":
    main()
