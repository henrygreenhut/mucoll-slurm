#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
BLACK = "#222222"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    parser.add_argument("histograms")
    parser.add_argument("output")
    return parser.parse_args()


def asymmetric_errors(rows, key, interval="95"):
    values = np.array([row[key]["vertical_modulation_percent"] for row in rows])
    bounds = [row[key]["vertical_modulation_intervals"] for row in rows]
    low = np.array([item["low" + interval] for item in bounds])
    high = np.array([item["high" + interval] for item in bounds])
    return values, np.vstack((values - low, high - values))


def summary_plot(data, output):
    rows = data["energy_bins"]
    labels = [row["energy_bin_MeV"] for row in rows]
    positions = np.arange(len(rows))
    unrotated, unrotated_error = asymmetric_errors(rows, "unrotated")
    rotated, rotated_error = asymmetric_errors(rows, "rotated")

    difference = np.array([
        row["paired_difference"]["vertical_modulation_percent"] for row in rows
    ])
    difference_bounds = [row["paired_difference"]["intervals"] for row in rows]
    difference_low = np.array([item["low95"] for item in difference_bounds])
    difference_high = np.array([item["high95"] for item in difference_bounds])
    difference_error = np.vstack((
        difference - difference_low,
        difference_high - difference,
    ))

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.3),
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1], "hspace": 0.08},
    )
    top, bottom = axes

    top.errorbar(
        positions - 0.09,
        unrotated,
        yerr=unrotated_error,
        marker="o",
        markersize=6,
        linewidth=1.8,
        capsize=3,
        color=BLUE,
        label="Unrotated",
    )
    top.errorbar(
        positions + 0.09,
        rotated,
        yerr=rotated_error,
        marker="o",
        markersize=6,
        linewidth=1.8,
        capsize=3,
        color=ORANGE,
        label="Rotated once",
    )
    top.axhline(0.0, color="0.45", linestyle="--", linewidth=1)
    top.set_ylabel(r"Vertical-axis anisotropy $-2C_2$ [percent]", fontsize=15)
    top.set_title(r"BIB photon second-harmonic anisotropy versus energy", fontsize=20, pad=15)
    top.legend(
        frameon=False,
        fontsize=13,
        title="95% source-cycle bootstrap intervals",
        title_fontsize=11,
    )

    bottom.errorbar(
        positions,
        difference,
        yerr=difference_error,
        marker="o",
        markersize=6,
        linewidth=1.8,
        capsize=3,
        color=BLACK,
    )
    bottom.axhline(0.0, color="0.45", linestyle="--", linewidth=1)
    bottom.set_ylabel("Unrotated - rotated\n[percent]", fontsize=14)
    bottom.set_xlabel("Photon energy [MeV]", fontsize=15)
    bottom.set_xticks(positions, labels)

    for axis in axes:
        axis.tick_params(labelsize=12)
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(output / "photon_vertical_modulation_vs_energy.pdf", bbox_inches="tight")
    figure.savefig(output / "photon_vertical_modulation_vs_energy.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def harmonic_map(data, histograms, output):
    rows = data["energy_bins"]
    labels = [row["energy_bin_MeV"] for row in rows]
    phi = np.linspace(-np.pi, np.pi, 361)
    c2 = histograms["fine_c2_unrotated"]
    s2 = histograms["fine_s2_unrotated"]
    modulation = 200.0 * (
        c2[:, None] * np.cos(2.0 * phi)[None, :]
        + s2[:, None] * np.sin(2.0 * phi)[None, :]
    )
    limit = np.max(np.abs(modulation))

    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    image = axis.imshow(
        modulation,
        origin="lower",
        aspect="auto",
        extent=(-np.pi, np.pi, -0.5, len(labels) - 0.5),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    axis.set_xlabel(r"Photon momentum $\phi$ [rad]", fontsize=15)
    axis.set_ylabel("Photon energy [MeV]", fontsize=15)
    axis.set_title(
        "Second-harmonic component of unrotated BIB photons",
        fontsize=19,
        pad=14,
    )
    axis.tick_params(labelsize=12)
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Modulation relative to uniform [percent]", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    figure.tight_layout()
    figure.savefig(output / "photon_second_harmonic_energy_phi_map.pdf", bbox_inches="tight")
    figure.savefig(
        output / "photon_second_harmonic_energy_phi_map.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def distribution_plots(histograms, output):
    phi_edges = histograms["phi_edges"]
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    labels = histograms["coarse_labels"]
    hist_unrotated = histograms["hist_unrotated"]
    hist_rotated = histograms["hist_rotated"]
    residual_low = histograms["residual_low"]
    residual_high = histograms["residual_high"]

    fraction_unrotated = hist_unrotated / hist_unrotated.sum(axis=1, keepdims=True)
    fraction_rotated = hist_rotated / hist_rotated.sum(axis=1, keepdims=True)
    residual = np.divide(
        100.0 * (fraction_unrotated - fraction_rotated),
        fraction_rotated,
        out=np.full_like(fraction_rotated, np.nan),
        where=fraction_rotated > 0,
    )

    top_minimum = min(fraction_unrotated.min(), fraction_rotated.min())
    top_maximum = max(fraction_unrotated.max(), fraction_rotated.max())
    top_padding = 0.07 * (top_maximum - top_minimum)
    bottom_limit = 1.08 * max(
        abs(np.nanmin(residual_low)),
        abs(np.nanmax(residual_high)),
    )

    names = ("lt_0p5", "0p5_to_2", "2_to_5", "ge_5")
    for index, (name, label) in enumerate(zip(names, labels)):
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(10.5, 7.7),
            sharex=True,
            gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.08},
        )
        top, bottom = axes
        top.stairs(
            fraction_unrotated[index],
            phi_edges,
            baseline=None,
            linewidth=1.8,
            color=BLUE,
            label="Unrotated",
        )
        top.stairs(
            fraction_rotated[index],
            phi_edges,
            baseline=None,
            linewidth=1.8,
            color=ORANGE,
            label="Rotated once",
        )
        top.set_ylim(top_minimum - top_padding, top_maximum + top_padding)
        top.set_ylabel("Fraction of photons per bin", fontsize=14)
        top.set_title(
            fr"BIB photon momentum $\phi$: {label} MeV",
            fontsize=19,
            pad=15,
        )
        top.legend(frameon=False, fontsize=12)

        bottom.fill_between(
            phi_centers,
            residual_low[index],
            residual_high[index],
            color="0.75",
            alpha=0.55,
            linewidth=0,
            label="68% cycle bootstrap",
        )
        bottom.step(
            phi_centers,
            residual[index],
            where="mid",
            color=BLACK,
            linewidth=1.5,
            label="Difference",
        )
        delta_c2 = (
            histograms["coarse_c2_unrotated"][index]
            - histograms["coarse_c2_rotated"][index]
        )
        delta_s2 = (
            histograms["coarse_s2_unrotated"][index]
            - histograms["coarse_s2_rotated"][index]
        )
        harmonic = 200.0 * (
            delta_c2 * np.cos(2.0 * phi_centers)
            + delta_s2 * np.sin(2.0 * phi_centers)
        )
        bottom.plot(
            phi_centers,
            harmonic,
            color="#009E73",
            linewidth=2,
            label="Second harmonic",
        )
        bottom.axhline(0.0, color="0.45", linestyle="--", linewidth=1)
        bottom.set_ylim(-bottom_limit, bottom_limit)
        bottom.set_ylabel("Relative difference\n[percent]", fontsize=13)
        bottom.set_xlabel(r"Photon momentum $\phi$ [rad]", fontsize=15)
        bottom.legend(frameon=False, fontsize=10, ncol=3, loc="upper center")

        bottom.set_xlim(-np.pi, np.pi)
        bottom.set_xticks(
            [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
        )
        for axis in axes:
            axis.tick_params(labelsize=11)
            axis.grid(alpha=0.20, linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)

        figure.tight_layout()
        figure.savefig(output / f"photon_phi_unrotated_vs_rotated_{name}.pdf", bbox_inches="tight")
        figure.savefig(output / f"photon_phi_unrotated_vs_rotated_{name}.png", dpi=200, bbox_inches="tight")
        plt.close(figure)


def broad_distribution_plots(histograms, output):
    source_edges = histograms["phi_edges"]
    phi_edges = source_edges[::2]
    hist_unrotated = np.stack((
        histograms["hist_unrotated"][:2].sum(axis=0),
        histograms["hist_unrotated"][2:].sum(axis=0),
    )).reshape(2, 32, 2).sum(axis=2)
    hist_rotated = np.stack((
        histograms["hist_rotated"][:2].sum(axis=0),
        histograms["hist_rotated"][2:].sum(axis=0),
    )).reshape(2, 32, 2).sum(axis=2)

    fraction_unrotated = hist_unrotated / hist_unrotated.sum(axis=1, keepdims=True)
    fraction_rotated = hist_rotated / hist_rotated.sum(axis=1, keepdims=True)
    photon_counts = hist_unrotated.sum(axis=1)
    names = ("0_to_2", "ge_2")
    labels = ("0–2 MeV", r"$\geq 2$ MeV")
    for index, (name, label) in enumerate(zip(names, labels)):
        if name == "0_to_2":
            unrotated = hist_unrotated[index]
            rotated = hist_rotated[index]
            ylabel = "Photon count"
        else:
            unrotated = fraction_unrotated[index]
            rotated = fraction_rotated[index]
            ylabel = "Fraction of photons"

        figure, axis = plt.subplots(figsize=(10.5, 7.15))
        axis.stairs(
            unrotated,
            phi_edges,
            baseline=None,
            linewidth=3,
            color=BLUE,
            label="native",
        )
        axis.stairs(
            rotated,
            phi_edges,
            baseline=None,
            linewidth=3,
            color=ORANGE,
            label="rotated",
        )
        axis.set_ylabel(ylabel, fontsize=18)
        axis.set_xlabel(r"$\phi$ [rad]", fontsize=18)
        axis.set_title(
            fr"Photon momentum $\phi$ ({label}; {photon_counts[index]:,} photons)",
            fontsize=20,
            pad=28,
        )
        axis.legend(frameon=False, fontsize=14, loc="upper left")
        axis.set_xlim(-np.pi, np.pi)
        axis.set_xticks(
            [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
        )
        axis.tick_params(labelsize=16)

        figure.tight_layout()
        figure.savefig(
            output / f"photon_phi_unrotated_vs_rotated_{name}.pdf",
            bbox_inches="tight",
        )
        figure.savefig(
            output / f"photon_phi_unrotated_vs_rotated_{name}.png",
            dpi=100,
            bbox_inches="tight",
        )
        plt.close(figure)


def main():
    args = arguments()
    with open(args.summary) as source:
        data = json.load(source)
    histograms = np.load(args.histograms)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "serif"
    summary_plot(data, output)
    harmonic_map(data, histograms, output)
    distribution_plots(histograms, output)
    broad_distribution_plots(histograms, output)


if __name__ == "__main__":
    main()
