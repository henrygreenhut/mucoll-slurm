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
        raise argparse.ArgumentTypeError(
            "expected LABEL:DISTANCE_MIN_M:DISTANCE_MAX_M"
        ) from error
    if upper <= lower:
        raise argparse.ArgumentTypeError("quadrupole maximum must exceed minimum")
    return label, lower, upper


def load_muplus(source):
    prefix = "MUPLUS_" if "MUPLUS_c2" in source else ""
    return {
        name: source[prefix + name]
        for name in ("z_center_mm", "photon_counts", "c2", "s2", "a2")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--minimum-photons", type=int, default=10_000)
    parser.add_argument("--maximum-distance-m", type=float)
    parser.add_argument(
        "--quad",
        action="append",
        default=[],
        type=parse_quad,
        metavar="LABEL:DISTANCE_MIN_M:DISTANCE_MAX_M",
    )
    args = parser.parse_args()

    source = np.load(args.input)
    data = load_muplus(source)
    distance = np.abs(data["z_center_mm"] / 1000.0)
    selected = data["photon_counts"] >= args.minimum_photons
    if args.maximum_distance_m is not None:
        selected &= distance <= args.maximum_distance_m
    order = np.argsort(distance[selected])
    distance = distance[selected][order]

    quantities = (
        ("c2", r"$C_2$", r"$C_2$"),
        ("s2", r"$S_2$", r"$S_2$"),
        ("a2", r"$A_2$", r"$A_2$"),
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "serif"

    for name, y_label, symbol in quantities:
        values = data[name][selected][order]
        figure, axis = plt.subplots(figsize=(8.0, 5.0))
        axis.plot(distance, values, color="#0072B2", linewidth=1.5)
        if name != "a2":
            axis.axhline(0.0, color="0.35", linewidth=0.8)
        for label, lower, upper in args.quad:
            axis.axvspan(lower, upper, color="0.8", alpha=0.35, linewidth=0)
            axis.text(
                0.5 * (lower + upper),
                0.96,
                label,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
            )

        axis.set_xlabel(r"$|z|$ of mother-muon decay [m]")
        axis.set_ylabel(y_label)
        axis.set_title(f"BIB photon {symbol} versus mother-muon decay position")
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()

        output = Path(f"{args.output_prefix}_{name}")
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        figure.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
