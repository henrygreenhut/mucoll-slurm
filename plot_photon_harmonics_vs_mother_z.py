#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_quad(value):
    try:
        label, lower, upper = value.split(":")
        lower = float(lower)
        upper = float(upper)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected LABEL:Z_MIN_M:Z_MAX_M") from error
    if upper <= lower:
        raise argparse.ArgumentTypeError("quadrupole maximum must exceed minimum")
    return label, lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument(
        "--quad",
        action="append",
        default=[],
        type=parse_quad,
        metavar="LABEL:Z_MIN_M:Z_MAX_M",
    )
    args = parser.parse_args()

    source = np.load(args.input)
    labels = source["labels"].tolist()
    quantities = (
        ("c2", r"$C_2=\langle\cos(2\phi)\rangle$"),
        ("s2", r"$S_2=\langle\sin(2\phi)\rangle$"),
        ("a2", r"$A_2=\sqrt{C_2^2+S_2^2}$"),
    )

    plt.rcParams["font.family"] = "serif"
    figure, axes = plt.subplots(
        len(quantities),
        len(labels),
        figsize=(6.4 * len(labels), 9.0),
        sharex=True,
        squeeze=False,
    )

    for column, label in enumerate(labels):
        z = source[f"{label}_z_center_mm"] / 1000.0
        for row, (name, y_label) in enumerate(quantities):
            axis = axes[row, column]
            values = source[f"{label}_{name}"]
            low = source[f"{label}_ci68_low"][row]
            high = source[f"{label}_ci68_high"][row]
            show_legend = row == 0 and column == 0

            axis.plot(
                z,
                values,
                color="#0072B2",
                lw=1.4,
                label="Point estimate" if show_legend else None,
            )
            axis.fill_between(
                z,
                low,
                high,
                color="#0072B2",
                alpha=0.22,
                linewidth=0,
                label="68% source-cycle bootstrap" if show_legend else None,
            )
            if name != "a2":
                axis.axhline(0.0, color="0.35", lw=0.8)
            for quad_label, lower, upper in args.quad:
                axis.axvspan(lower, upper, color="0.75", alpha=0.25, lw=0)
                if row == 0:
                    axis.text(
                        0.5 * (lower + upper),
                        0.96,
                        quad_label,
                        transform=axis.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=9,
                    )

            axis.set_ylabel(y_label)
            axis.grid(axis="y", alpha=0.2, linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
            if row == 0:
                axis.set_title(label)
                if show_legend:
                    axis.legend(frameon=False, fontsize=9)
            if row == len(quantities) - 1:
                axis.set_xlabel(r"Mother-muon decay $z$ [m]")

    figure.suptitle("BIB photon second harmonic versus mother-muon decay position")
    figure.tight_layout()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
