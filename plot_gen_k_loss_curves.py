#!/usr/bin/env python3

from pathlib import Path

from pfn_libtest_plot_overlay import make_plot


ROOT = Path(__file__).resolve().parent

RUNS = {
    5: ROOT / "pfn_results" /
       "n420_fixed_k1_vs_k5_scaled_lr1e-4_decay80_mseed1_pointonly",
    7: ROOT / "pfn_results" /
       "n420_fixed_k1_vs_k7_scaled_lr1e-4_decay80_mseed1_pointonly",
    10: ROOT / "variable_k_results" /
        "n420_k1_vs_k10_scaled_lr1e-4_mseed1",
    21: ROOT / "variable_k_results" /
        "n420_k1_vs_k21_scaled_lr1e-4_mseed1_min80",
    42: ROOT / "pfn_results" /
        "n420_recipe_bs4_expanded_scaled_lr1e-4_mseed1_pointonly",
}


def main():
    output = ROOT / "plots" / "training_overlays" / "gen_k_scan"
    for k, run in RUNS.items():
        if not (run / "history.csv").is_file():
            raise SystemExit("missing history: {}".format(run / "history.csv"))
        make_plot(
            str(run),
            "GEN-Level PFN at N=420: K=1 vs K={}".format(k),
            str(output / "gen_n420_k1_vs_k{}_loss.pdf".format(k)),
        )
        make_plot(
            str(run),
            "GEN-Level PFN at N=420: K=1 vs K={}".format(k),
            str(output / "gen_n420_k1_vs_k{}_loss.png".format(k)),
        )


if __name__ == "__main__":
    main()
