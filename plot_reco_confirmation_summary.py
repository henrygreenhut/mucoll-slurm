#!/usr/bin/env python3
"""Plot original and confirmation AUCs for a RECO classifier and its null."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MAIN_LABEL = "reco_n420_trackfix_directlog_stabilized_dropout_U_vs_R"
NULL_LABEL = "reco_n420_trackfix_directlog_stabilized_dropout_null"
CONFIRM_MAIN_LABEL = (
    "reco_n420_trackfix_directlog_stabilized_dropout_confirmation_U_vs_R"
)
CONFIRM_NULL_LABEL = (
    "reco_n420_trackfix_directlog_stabilized_dropout_confirmation_null"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="reco_pfn_results")
    parser.add_argument(
        "--out",
        default=(
            "plots/reco_n420_trackfix_directlog/"
            "reco_n420_trackfix_auc_confirmation.pdf"
        ),
    )
    return parser.parse_args()


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def main():
    args = parse_args()
    root = Path(args.results_root)

    original = [
        read_json(root / MAIN_LABEL / "summary.json"),
        read_json(root / NULL_LABEL / "summary.json"),
    ]
    confirmation = [
        read_json(
            root / CONFIRM_MAIN_LABEL / "confirmation_summary.json"
        ),
        read_json(
            root / CONFIRM_NULL_LABEL / "confirmation_summary.json"
        ),
    ]

    original_auc = np.asarray(
        [payload["results"]["test"]["auc"] for payload in original]
    )
    confirmation_auc = np.asarray(
        [payload["results"]["auc"] for payload in confirmation]
    )
    intervals = np.asarray(
        [
            payload["uncertainty"]["auc_95_percentile_interval"]
            for payload in confirmation
        ]
    )
    confirmation_error = np.vstack(
        [confirmation_auc - intervals[:, 0], intervals[:, 1] - confirmation_auc]
    )

    original_events = original[0]["results"]["test"]["events"] // 2
    confirmation_events = confirmation[0]["events_per_class"]

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    x = np.arange(2)
    offset = 0.12
    ax.plot(
        x - offset,
        original_auc,
        "o",
        color="#0072B2",
        ms=7,
        label="original test ({:,}/class)".format(original_events),
    )
    ax.errorbar(
        x + offset,
        confirmation_auc,
        yerr=confirmation_error,
        fmt="o",
        color="#D55E00",
        ms=7,
        capsize=4,
        lw=1.5,
        label="confirmation ({:,}/class)".format(confirmation_events),
    )
    ax.axhline(0.5, color="#777777", lw=1, ls="--")
    ax.set_xticks(x, ["unique vs reused", "matched null"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(0.47, 0.68)
    ax.set_ylabel("test AUC")
    ax.set_title("RECO N=420 track-fixed PFN")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print("chart -> {}".format(output))


if __name__ == "__main__":
    main()
