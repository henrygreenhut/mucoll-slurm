import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROFILES = {
    "photon_c2_vs_position_z_simple_1m": np.arange(-6000, 6001, 1000),
    "photon_c2_vs_position_z_near_ip_50mm": np.arange(-500, 501, 50),
    "photon_c2_vs_position_z_0_to_20cm_10mm": np.arange(0, 201, 10),
}


def summarize(z_mm, phi, edges, full_count):
    counts = np.histogram(z_mm, bins=edges)[0]
    cosine = np.histogram(z_mm, bins=edges, weights=np.cos(2 * phi))[0]
    c2 = np.divide(cosine, counts, out=np.full_like(cosine, np.nan), where=counts > 0)
    return counts, cosine, c2, cosine / full_count


def calculate(bank_path, output):
    import h5py

    with h5py.File(bank_path, "r") as bank:
        selected = np.flatnonzero(bank["particles/pdg"][:] == 22)
        z = bank["particles/vz"][:][selected].astype(float)
        x = bank["particles/px"][:][selected].astype(float)
        y = bank["particles/py"][:][selected].astype(float)
        source_filenames = bank["filenames"].asstr()[:].tolist()
        source_cycles = len(bank["cycle_ids"])
    if not np.all(np.isfinite(z) & np.isfinite(x) & np.isfinite(y) & ((x != 0) | (y != 0))):
        raise ValueError("Invalid photon entry z or momentum phi")
    phi = np.arctan2(y, x)
    full_count = len(z)
    for name, edges in PROFILES.items():
        counts, cosine, c2, contribution = summarize(z, phi, edges, full_count)
        inside = (z >= edges[0]) & (z <= edges[-1])
        assert counts.sum() == np.count_nonzero(inside)
        np.testing.assert_allclose(cosine.sum(), np.cos(2 * phi[inside]).sum(), atol=1e-8)
        np.savez_compressed(output / f"{name}.npz", edges_m=edges / 1000,
                             counts=counts, cosine=cosine, c2=c2,
                             contribution=contribution)
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["position_z_low_m", "position_z_high_m", "photons",
                             "cosine_sum", "c2", "contribution"])
            writer.writerows(zip(edges[:-1] / 1000, edges[1:] / 1000,
                                  counts, cosine, c2, contribution))
        metadata = {
            "bank": str(bank_path), "source_filenames": source_filenames,
            "source_cycles": source_cycles, "particle_selection": "pdg == 22",
            "coordinate": "Signed photon entry/handoff z, GEN particles/vz; not mother decay z or photon creation z",
            "phi": "Momentum atan2(py, px); not position phi",
            "bin_width_m": float((edges[1] - edges[0]) / 1000),
            "range_m": [float(edges[0] / 1000), float(edges[-1] / 1000)],
            "bin_merging": False, "bootstrap": False, "optical_overlay": False,
            "minimum_count_cut": None, "full_bank_photons": full_count,
            "displayed_photons": int(counts.sum()),
            "contribution_normalization": "Full-bank photon count, including photons outside this plot's range",
            "displayed_c2": float(cosine.sum() / counts.sum()),
            "displayed_contribution": float(contribution.sum()),
            "full_bank_c2": float(np.cos(2 * phi).mean()),
        }
        (output / f"{name}.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        print(name, "photons:", int(counts.sum()), "C2:", metadata["displayed_c2"], flush=True)


def plot_profile(output, name):
    from plot_photon_mother_z_summary import draw_simple_panels

    with np.load(output / f"{name}.npz") as data:
        edges = data["edges_m"]
        title = "BIB photon anisotropy by photon z position"
        xlabel = "Photon z position at detector entry [m]"
        if edges[-1] - edges[0] <= 1:
            width_cm = 100 * (edges[1] - edges[0])
            title += f"\n{width_cm:g} cm bins"
        if name == "photon_c2_vs_position_z_0_to_20cm_10mm":
            edges = 100 * edges
            xlabel = "Photon z position at detector entry [cm]"
        draw_simple_panels(edges, data["counts"], data["c2"],
                           data["contribution"], title, xlabel, output / name)


def plot(output):
    for name in PROFILES:
        plot_profile(output, name)
    plot_shares(output)


def plot_shares(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    full_name = "photon_c2_vs_position_z_simple_1m"
    zoom_name = "photon_c2_vs_position_z_near_ip_50mm"
    with np.load(output / f"{full_name}.npz") as full:
        full_count = int(full["counts"].sum())
        full_cosine = float(full["cosine"].sum())
    with np.load(output / f"{zoom_name}.npz") as zoom:
        edges = zoom["edges_m"]
        counts = zoom["counts"]
        cosine = zoom["cosine"]
    metadata = json.loads((output / f"{full_name}.json").read_text())
    assert full_count == metadata["full_bank_photons"]
    np.testing.assert_allclose(full_cosine / full_count, metadata["full_bank_c2"])
    if abs(full_cosine) < 1e-12:
        raise ValueError("A zero net C2 has no defined contribution shares")
    photon_fraction = counts / full_count
    c2_share = cosine / full_cosine
    highlighted = np.isclose(edges[:-1], .1) & np.isclose(edges[1:], .15)
    assert np.count_nonzero(highlighted) == 1
    index = np.flatnonzero(highlighted)[0]
    stem = "photon_position_z_photon_and_c2_shares"
    with (output / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position_z_low_m", "position_z_high_m", "photons",
                         "cosine_sum", "fraction_of_all_photons", "share_of_full_bank_net_c2"])
        writer.writerows(zip(edges[:-1], edges[1:], counts, cosine,
                              photon_fraction, c2_share))
    settings = {
        "source_profiles": [full_name, zoom_name],
        "coordinate": "Signed photon entry/handoff z; NOT original beam-muon decay z",
        "phi": "Momentum atan2(py, px), NOT position phi",
        "bin_width_cm": 5, "range_cm": [-50, 50],
        "full_bank_photons": full_count, "full_bank_cosine_sum": full_cosine,
        "photon_fraction": "Bin photon count divided by ALL bank photons",
        "c2_share": "Bin sum(cos(2 phi)) divided by FULL BANK sum(cos(2 phi)); signed, not absolute",
        "highlight": {"range_cm": [10, 15], "photons": int(counts[index]),
                      "photon_fraction": float(photon_fraction[index]),
                      "net_c2_share": float(c2_share[index])},
        "displayed_photon_fraction": float(photon_fraction.sum()),
        "displayed_net_c2_share": float(c2_share.sum()),
        "uncertainty": "Point estimates only; no error bars. Signed shares can be negative or exceed 100% when contributions cancel. Not a fraction of PFN importance or a causal fraction.",
    }
    (output / f"{stem}.json").write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    figure, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    edges_cm = 100 * edges
    centers = (edges_cm[:-1] + edges_cm[1:]) / 2
    colors = np.where(highlighted, "#D55E00", "#0072B2")
    for axis, fraction, label in zip(
        axes, (photon_fraction, c2_share),
        ("Fraction of all photons [%]", r"Share of net $C_2$ [%]"),
    ):
        values = 100 * fraction
        axis.bar(centers, values, width=np.diff(edges_cm), color=colors,
                  edgecolor="white", linewidth=.6, zorder=2)
        axis.axhline(0, color=".4", linewidth=.8)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=.2, linewidth=.5, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.annotate(f"{values[index]:.1f}%", (centers[index], values[index]),
                       xytext=(0, 5), textcoords="offset points", ha="center",
                       va="bottom", color="#D55E00", fontweight="bold")
        axis.set_ylim(min(0, values.min()) - 4, values.max() * 1.25)
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("BIB photon anisotropy by entry z")
    axes[0].legend(handles=[Patch(facecolor="#D55E00", label="+10–15 cm")],
                    frameon=False, loc="upper left")
    axes[-1].set_xlabel("Photon z position at detector entry [cm]")
    axes[-1].set_xlim(edges_cm[0], edges_cm[-1])
    axes[-1].set_xticks(np.arange(-50, 51, 10))
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.no_plots and args.plot_only:
        parser.error("--no-plots and --plot-only cannot be combined")
    if not args.plot_only:
        if args.bank is None:
            parser.error("--bank is required for recalculation")
        args.output.mkdir(parents=True, exist_ok=False)
        calculate(args.bank, args.output)
    if not args.no_plots:
        plot(args.output)


if __name__ == "__main__":
    main()
