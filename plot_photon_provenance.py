import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"


def read_rows(path):
    with path.open() as handle:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(handle)]


def labels(edges):
    return [f"<{high:.4g}" if np.isneginf(low) else
            f"≥{low:.4g}" if np.isposinf(high) else
            "0.024\n(surface)" if high - low < 1e-6 and abs(low - .024) < 1e-6 else
            f"{low:.4g}–{high:.4g}"
            for low, high in zip(edges[:-1], edges[1:])]


def save(figure, output, name):
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def draw_harmonics(rows, tick_labels, xlabel, title, output, name):
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(10, 5))
    for coordinate, shift, color in (("momentum", -.12, BLUE), ("position", .12, ORANGE)):
        key = coordinate + "_c2"
        mean = 100 * np.array([row[key] for row in rows])
        error = 196 * np.array([row[key + "_se"] for row in rows])
        axis.errorbar(x + shift, mean, yerr=error, fmt="o", ms=4,
                      capsize=2, color=color, label=coordinate.capitalize())
    axis.axhline(0, color=".45", ls="--", lw=1)
    axis.set_xticks(x, tick_labels, rotation=40, ha="right")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"$C_2$ [%]")
    axis.set_title(title)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, output, name)


def draw_map(data, name, xlabel, ylabel, output):
    total = data[name + "_totals"]
    count = total[:, :, 0]
    x_keep = np.flatnonzero(count.sum(axis=0) > 0)
    y_keep = np.flatnonzero(count.sum(axis=1) > 0)
    xlabels = np.array(labels(data[name + "_edges1"]))[x_keep]
    ylabels = np.array(labels(data[name + "_edges0"]))[y_keep]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, column, title in zip(axes, (0, 1, 3),
                                   ("Photon count", r"Momentum $C_2$", r"Position $C_2$")):
        if column == 0:
            values = np.ma.masked_less(count, 1)
            norm = LogNorm(vmin=1, vmax=count.max())
            color = "viridis"
        else:
            values = np.ma.masked_where(count < 10000,
                np.divide(total[:, :, column], count,
                          out=np.zeros_like(count), where=count > 0))
            limit = max(float(np.max(np.abs(values))), .001)
            norm = TwoSlopeNorm(0, vmin=-limit, vmax=limit)
            color = "RdBu_r"
        values = values[np.ix_(y_keep, x_keep)]
        image = axis.imshow(values, origin="lower", aspect="auto", cmap=color, norm=norm)
        axis.set_xticks(np.arange(len(xlabels)), xlabels, rotation=55, ha="right")
        axis.set_yticks(np.arange(len(ylabels)), ylabels)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=.045, pad=.03)
    save(figure, output, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    output = directory / "plots"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    data = np.load(directory / "aggregates.npz")
    summary = json.loads((directory / "summary.json").read_text())
    for name, xlabel, title in (
        ("energy", "Photon energy [MeV]", "BIB photon anisotropy by energy"),
        ("entry_z", "Photon entry z [m]", "BIB photon anisotropy by detector-entry position"),
        ("mother_z", "Original beam-muon decay z [m]", "BIB photon anisotropy by decay position"),
        ("time", "Recorded FLUKA time [ns]", "BIB photon anisotropy by time"),
        ("entry_fine", "Photon entry z [m]", "BIB photon anisotropy near the IP"),
        ("radius", "Photon entry radius [m]", "BIB photon anisotropy by entry radius"),
    ):
        rows = read_rows(directory / f"{name}.csv")
        ticks = labels(data[name + "_edges0"])
        keep = [index for index, row in enumerate(rows) if row["photons"] >= 10000]
        draw_harmonics([rows[index] for index in keep], [ticks[index] for index in keep],
                       xlabel, title, output, name + "_c2")

    for name, xlabel, ylabel in (
        ("mother_entry", "Photon entry z [m]", "Original beam-muon decay z [m]"),
        ("energy_entry", "Photon entry z [m]", "Photon energy [MeV]"),
        ("energy_time", "Recorded FLUKA time [ns]", "Photon energy [MeV]"),
        ("energy_mother", "Original beam-muon decay z [m]", "Photon energy [MeV]"),
        ("entry_radius", "Photon entry radius [m]", "Photon entry z [m]"),
        ("mother_entry_below_2MeV", "Photon entry z [m]", "Original beam-muon decay z [m]"),
        ("mother_entry_above_2MeV", "Photon entry z [m]", "Original beam-muon decay z [m]"),
    ):
        draw_map(data, name, xlabel, ylabel, output)

    for name, xlabel in (("entry_z", "Photon entry z [m]"),
                         ("entry_fine", "Photon entry z [m]"),
                         ("radius", "Photon entry radius [m]"),
                         ("energy", "Photon energy [MeV]")):
        statistics = data[name + "_cycles"]
        counts = statistics[:, :, 0].sum(axis=0)
        keep = np.flatnonzero(counts >= 10000)
        tick_labels = np.array(labels(data[name + "_edges0"]))[keep]
        figure, axis = plt.subplots(figsize=(10, 5))
        for column, shift, color, label in ((1, -.18, BLUE, "Momentum"),
                                            (3, .18, ORANGE, "Position")):
            contribution = statistics[:, :, column].sum(axis=0) / counts.sum()
            residual = statistics[:, :, column] - (
                statistics[:, :, 0].sum(axis=1)[:, None] * contribution)
            error = np.sqrt((residual ** 2).sum(axis=0)
                            * len(statistics) / (len(statistics) - 1)) / counts.sum()
            axis.bar(np.arange(len(keep)) + shift, 100 * contribution[keep],
                      width=.36, yerr=196 * error[keep], color=color,
                      label=label, capsize=2)
        axis.axhline(0, color=".45", ls="--", lw=1)
        axis.set_xticks(np.arange(len(keep)), tick_labels, rotation=40, ha="right")
        axis.set_ylabel(r"Contribution to total $C_2$ [%]")
        axis.set_xlabel(xlabel)
        axis.set_title("Contributions to BIB photon anisotropy")
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
        save(figure, output, name + "_contributions")

    rows = list(summary["subsets"].values())
    ticks = ["First half", "Second half", "Even IDs", "Odd IDs"] + [str(n) for n in range(1, 11)]
    draw_harmonics(rows, ticks, "Source-cycle subset / consecutive block",
                   "BIB photon anisotropy across source cycles", output, "source_subsets")

    phi_edges = data["phi_edges"]
    phi_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    phi_labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    count = data["delta_histogram"]
    axes[0].stairs(count / count.sum(), phi_edges, color=BLUE, lw=1.5)
    axes[0].set_xlabel(r"Momentum $\phi$ − position $\phi$ [rad]")
    axes[0].set_ylabel("Fraction of photons per bin")
    axes[0].set_title("Position–direction alignment")
    counts = data["position_momentum_histogram"]
    image = axes[1].pcolormesh(phi_edges, phi_edges, counts.T,
                              norm=LogNorm(vmin=1, vmax=counts.max()), cmap="viridis")
    axes[1].set_xlabel(r"Position $\phi$ [rad]")
    axes[1].set_ylabel(r"Momentum $\phi$ [rad]")
    axes[1].set_title("BIB photon angles")
    figure.colorbar(image, ax=axes[1], label="Photon count")
    for axis in axes:
        axis.set_xticks(phi_ticks, phi_labels)
    axes[1].set_yticks(phi_ticks, phi_labels)
    save(figure, output, "position_momentum_alignment")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.stairs(data["energy_spectrum_counts"], data["energy_spectrum_edges"],
                color=BLUE, lw=1)
    for energy, label in ((.477, "477 keV"), (.511, "511 keV")):
        axis.axvline(energy, color=".4", lw=.8, ls="--")
        axis.text(energy, .98, label, transform=axis.get_xaxis_transform(),
                  rotation=90, va="top", ha="right")
    axis.set_xlabel("Photon energy [MeV]")
    axis.set_ylabel("Photon count / 0.5 keV")
    axis.set_title("BIB photon energy spectrum")
    axis.grid(axis="y", alpha=.2)
    save(figure, output, "low_energy_spectrum")


if __name__ == "__main__":
    main()
