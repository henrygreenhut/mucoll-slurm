#!/usr/bin/env python3
"""Plot reconstructed-track features for the direct and reservoir comparisons."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = {
    "direct_sim": ("Direct norm42 SIM", "black", "-"),
    "direct_count": ("Direct COUNT", "#D55E00", "-"),
    "reservoir_sim": ("Reservoir empirical SIM", "#0072B2", "--"),
    "reservoir_count": ("Reservoir COUNT", "#009E73", "--"),
}
COMPARISONS = {
    "direct": ("Direct norm42: SIM vs COUNT", ("direct_sim", "direct_count")),
    "reservoir": ("Reservoir: empirical SIM vs COUNT",
                  ("reservoir_sim", "reservoir_count")),
    "all": ("Direct and reservoir comparison", tuple(ARMS)),
}
FEATURE_LABELS = {
    "pt": r"$p_T$ [GeV]",
    "eta": r"$\eta$",
    "phi": r"$\phi$",
    "d0": r"$d_0$ [mm]",
    "z0": r"$z_0$ [mm]",
    "charge": "Charge sign",
    "chi2_ndf": r"$\chi^2/\mathrm{ndf}$",
    "n_hits": "Tracker hits / track",
    "multiplicity": "Reconstructed tracks",
}


def finite(data, arm, feature):
    values = np.asarray(data[f"{arm}__{feature}"], dtype=float)
    return values[np.isfinite(values)]


def central_range(series, symmetric=False, positive=False):
    combined = np.concatenate(series)
    low, high = np.quantile(combined, [0.005, 0.995])
    if symmetric:
        extent = max(abs(low), abs(high))
        return -extent, extent
    if positive:
        return max(0.0, low), high
    return low, high


def clipped(values, low, high):
    """Accumulate underflow and overflow in the visible edge bins."""
    return np.clip(values, np.nextafter(low, high), np.nextafter(high, low))


def shape_panel(axis, data, feature, xlabel, bins, limits=None, log_x=False,
                arms=tuple(ARMS)):
    for arm in arms:
        label, color, linestyle = ARMS[arm]
        values = finite(data, arm, feature)
        shown = clipped(values, *limits) if limits else values
        axis.hist(
            shown,
            bins=bins,
            weights=np.full(len(shown), 1.0 / len(shown)),
            histtype="step",
            linewidth=1.7,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    if limits:
        axis.set_xlim(*limits)
    if log_x:
        axis.set_xscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Fraction of tracks / bin")
    axis.grid(alpha=0.2)


def multiplicity_panel(axis, data, arms=tuple(ARMS)):
    event = np.arange(10)
    for arm in arms:
        label, color, linestyle = ARMS[arm]
        multiplicity = data[f"{arm}__multiplicity"]
        axis.plot(event, multiplicity, marker="o", markersize=5, linewidth=1.6,
                  color=color, linestyle=linestyle, label=label)
    axis.set_xlabel("Event index")
    axis.set_ylabel("Reconstructed tracks")
    axis.set_xticks(event)
    axis.grid(alpha=0.2)


def feature_specs(data):
    series = lambda feature: [finite(data, arm, feature) for arm in ARMS]
    pt_limits = central_range(series("pt"), positive=True)
    eta_limits = central_range(series("eta"))
    d0_limits = central_range(series("d0"), symmetric=True)
    z0_limits = central_range(series("z0"), symmetric=True)
    chi2_limits = central_range(series("chi2_ndf"), positive=True)
    hit_limits = (int(min(x.min() for x in series("n_hits"))),
                  int(max(x.max() for x in series("n_hits"))))
    return {
        "pt": (np.geomspace(*pt_limits, 35), pt_limits, True),
        "eta": (35, eta_limits, False),
        "phi": (np.linspace(-np.pi, np.pi, 37), (-np.pi, np.pi), False),
        "d0": (35, d0_limits, False),
        "z0": (35, z0_limits, False),
        "charge": (np.array([-1.5, -0.5, 0.5, 1.5]), (-1.5, 1.5), False),
        "chi2_ndf": (35, chi2_limits, False),
        "n_hits": (np.arange(hit_limits[0] - 0.5, hit_limits[1] + 1.5),
                   None, False),
    }


def save_separate_plots(data, output_dir, specs):
    for comparison, (comparison_title, arms) in COMPARISONS.items():
        destination = output_dir / comparison
        destination.mkdir(parents=True, exist_ok=True)
        for feature in FEATURE_LABELS:
            fig, axis = plt.subplots(figsize=(7.2, 5.4))
            if feature == "multiplicity":
                multiplicity_panel(axis, data, arms)
                note = "Event-level multiplicity is not normalized."
            else:
                bins, limits, log_x = specs[feature]
                shape_panel(axis, data, feature, FEATURE_LABELS[feature], bins,
                            limits, log_x, arms)
                note = "Each category is normalized separately."
                if limits and feature not in ("phi", "charge"):
                    note += " Central 99% shown; tails are in the edge bins."
            axis.set_title(f"{comparison_title}: {FEATURE_LABELS[feature]}")
            handles, labels = axis.get_legend_handles_labels()
            axis.legend(handles, labels, frameon=False)
            fig.text(0.5, 0.015, note, ha="center", fontsize=9)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            fig.savefig(destination / f"{feature}.png", dpi=220)
            fig.savefig(destination / f"{feature}.pdf")
            plt.close(fig)


def make_plot(input_path, output_prefix):
    data = np.load(input_path, allow_pickle=False)
    specs = feature_specs(data)

    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11.5))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.075, top=0.88,
                        wspace=0.24, hspace=0.31)
    panel_features = tuple(FEATURE_LABELS)
    for axis, feature in zip(axes.flat, panel_features):
        if feature == "multiplicity":
            multiplicity_panel(axis, data)
        else:
            bins, limits, log_x = specs[feature]
            shape_panel(axis, data, feature, FEATURE_LABELS[feature], bins,
                        limits, log_x)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=4, frameon=False)
    fig.suptitle("COUNT tracker study: reconstructed-track features",
                 fontsize=15, y=0.985)
    fig.text(
        0.5, 0.015,
        "Shape panels are normalized separately. Central 99% is displayed for "
        "unbounded features; tails are accumulated in the edge bins.",
        ha="center", fontsize=9,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)
    save_separate_plots(data, output_prefix.parent /
                        f"{output_prefix.name}_separate", specs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Extracted four-arm NPZ")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    make_plot(Path(args.input), Path(args.output_prefix))


if __name__ == "__main__":
    main()
