#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS = Path("reco_pfn_results")
PLOTS = Path("plots")
N_VALUES = (420, 840, 1260)
LABEL = "reco_n{}_trackfix_val25_directlog_charged7_stabilized_dropout_{}"


def load(n, comparison):
    path = RESULTS / LABEL.format(n, comparison) / "summary.json"
    with path.open() as handle:
        summary = json.load(handle)
    return summary["results"]["test"]["auc"]


def main():
    main_auc = [load(n, "U_vs_R") for n in N_VALUES]

    PLOTS.mkdir(exist_ok=True)

    with (PLOTS / "reco_auc_vs_n.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("N", "unique_vs_reused_auc"))
        writer.writerows(zip(N_VALUES, main_auc))

    plt.rcParams["font.family"] = "serif"
    fig, axis = plt.subplots(figsize=(6.2, 4.4))
    axis.plot(
        N_VALUES,
        main_auc,
        marker="o",
        markersize=7,
        linewidth=2,
        color="#0072B2",
        label="Unique vs 42× reuse",
    )
    axis.axhline(0.5, color="0.45", linewidth=1, linestyle="--")
    axis.set_xticks(N_VALUES)
    axis.set_xlabel("N")
    axis.set_ylabel("Test AUC")
    axis.set_title("RECO PFN AUC versus BIB sample size")
    axis.set_ylim(0.48, 0.85)
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)

    fig.tight_layout()
    for suffix in ("pdf", "png"):
        output = PLOTS / "reco_auc_vs_n.{}".format(suffix)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print("plot -> {}".format(output))
    plt.close(fig)


if __name__ == "__main__":
    main()
