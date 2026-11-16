import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_photon_second_harmonic_axis import configure_phi_axis


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "phi_anisotropy_out/photon_provenance/aggregates.npz"
OUTPUT = ROOT / "phi_anisotropy_out/photon_entry_xy/photon_entry_phi_polar"


def main():
    with np.load(SOURCE) as data:
        edges = data["phi_edges"]
        counts = data["position_momentum_histogram"].sum(axis=1)
        totals = data["cycle_totals"].sum(axis=0)
        cycles = len(data["cycle_ids"])
    np.testing.assert_allclose(np.diff(edges), 2 * np.pi / len(counts))
    assert counts.sum() == totals[0]
    centers = (edges[:-1] + edges[1:]) / 2
    relative = counts / counts.mean()
    c2, s2 = totals[[3, 4]] / totals[0]
    phase = (.5 * np.arctan2(s2, c2)) % np.pi
    phi = np.linspace(-np.pi, np.pi, 721)
    lower, upper = .98, 1.02
    assert relative.min() >= lower and relative.max() <= upper

    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    figure, axis = plt.subplots(figsize=(7.4, 7.4), subplot_kw={"projection": "polar"})
    configure_phi_axis(axis)
    axis.set_thetagrids([0, 90, 180, 270],
                        [r"$0$ (+x)", r"$\pi/2$ (+y)", r"$\pi$ (−x)", r"$-\pi/2$ (−y)"])
    axis.get_xticklabels()[0].set_horizontalalignment("left")
    axis.get_xticklabels()[2].set_horizontalalignment("right")
    axis.plot(np.r_[centers, centers[0] + 2 * np.pi], np.r_[relative, relative[0]],
              "o-", color="#0072B2", lw=1, ms=3, label="Photons")
    axis.plot(phi, np.ones_like(phi), color=".5", ls=":", lw=1, label="Uniform")
    axis.set_ylim(lower, upper)
    axis.set_yticks([.99, 1., 1.01], ["0.99", "1.00", "1.01"])
    axis.set_rlabel_position(35)
    axis.grid(alpha=.2)
    axis.spines["polar"].set_alpha(.3)
    axis.set_title(f"BIB photon position anisotropy\n{int(totals[0]):,} photons", pad=25)
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2,
                 frameon=False, fontsize=10)
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT.with_suffix("." + extension), dpi=220, bbox_inches="tight")
    plt.close(figure)
    with OUTPUT.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phi_low_rad", "phi_high_rad", "photons", "relative_density"])
        writer.writerows(zip(edges[:-1], edges[1:], counts.astype(int), relative))
    metadata = {
        "source": str(SOURCE), "source_cycles": cycles, "photons": int(totals[0]),
        "sample": "Full unrotated MUPLUS source library, not one N=420 construction",
        "angle": "Photon entry-position atan2(vy, vx), NOT momentum phi",
        "histogram": "Position marginal of position_momentum_histogram; equal 7.5-degree bins",
        "radius": "Photon count per angular bin divided by mean angular-bin count; not position radius",
        "c2": float(c2), "s2": float(s2),
        "second_harmonic": "1 + 2*C2*cos(2*phi) + 2*S2*sin(2*phi); reconstructed from unbinned moments, not a fitted full distribution",
        "preferred_axis_degrees": float(np.degrees(phase)),
        "opposite_axis_degrees": float(np.degrees(phase) + 180),
        "fractional_second_harmonic_modulation": float(2 * np.hypot(c2, s2)),
        "second_harmonic_overlay": False, "preferred_axis_arrows": False,
        "radial_display_limits": [lower, upper], "uncertainty": "Not shown",
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
