#!/usr/bin/env python3
"""Plot paired SIM/COUNT fitted-track observables from version-2 stores."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from count_tracker_track_features import RAW, RAW_FEATURES


FEATURES = {
    "pt": (r"$p_T$ [GeV]", True),
    "eta": (r"$\eta$", False),
    "phi": (r"$\phi$ [rad]", False),
    "d0": (r"$d_0$ [mm]", False),
    "z0": (r"$z_0$ [mm]", False),
}
COLORS = {"SIM": "#0072B2", "COUNT": "#D55E00"}


def load_store(store_dir, construction, sample, split):
    prefix = Path(store_dir) / f"{construction}_{sample}_{split}"
    manifest = json.loads(prefix.with_suffix(".json").read_text())
    expected = {"schema_version": 2, "construction": construction,
                "sample": sample, "split": split}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{prefix}: expected {key}={value!r}")
    if tuple(manifest["features"]) != RAW_FEATURES:
        raise ValueError(f"{prefix}: unexpected track features")
    with np.load(prefix.with_suffix(".npz")) as data:
        tracks, counts = data["tracks"], data["n_tracks"]
    mask = np.arange(tracks.shape[1])[None, :] < counts[:, None]
    return tracks[mask], counts, manifest


def limits(values, feature):
    combined = np.concatenate(values)
    if feature == "phi":
        return -np.pi, np.pi
    low, high = np.quantile(combined, (0.005, 0.995))
    if feature in ("d0", "z0"):
        extent = max(abs(low), abs(high))
        return -extent, extent
    if feature == "pt":
        low = max(low, np.nextafter(0.0, 1.0))
    return float(low), float(high)


def plot(store_dir, construction, split, output_dir, title):
    loaded = {sample: load_store(store_dir, construction, sample, split)
              for sample in ("SIM", "COUNT")}
    sim_ids = [event["event_id"] for event in loaded["SIM"][2]["events"]]
    count_ids = [event["event_id"] for event in loaded["COUNT"][2]["events"]]
    if sim_ids != count_ids:
        raise ValueError("SIM and COUNT stores contain different paired events")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into nonempty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"construction": construction, "split": split, "events": len(sim_ids),
               "track_counts": {sample: int(loaded[sample][1].sum())
                                for sample in loaded}}
    for feature, (label, log_x) in FEATURES.items():
        values = {sample: loaded[sample][0][:, RAW[feature]].astype(float)
                  for sample in loaded}
        low, high = limits(list(values.values()), feature)
        bins = (np.geomspace(low, high, 36) if log_x
                else np.linspace(low, high, 36))
        fig, axis = plt.subplots(figsize=(7.2, 5.4))
        for sample in ("SIM", "COUNT"):
            shown = np.clip(values[sample], np.nextafter(low, high), np.nextafter(high, low))
            axis.hist(shown, bins=bins, weights=np.full(len(shown), 1.0 / len(shown)),
                      histtype="step", linewidth=2, color=COLORS[sample],
                      label=f"{sample} ({len(shown)} tracks)")
        if log_x:
            axis.set_xscale("log")
        axis.set(xlabel=label, ylabel="Fraction of tracks / bin",
                 title=f"{title}: {label}")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        fig.text(0.5, 0.015,
                 "Each arm is normalized separately; central 99% shown with tails in edge bins.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(output_dir / f"{feature}.png", dpi=220)
        fig.savefig(output_dir / f"{feature}.pdf")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.0, 5.4))
    x = np.arange(len(sim_ids))
    for sample in ("SIM", "COUNT"):
        axis.plot(x, loaded[sample][1], marker="o", linewidth=1.7,
                  color=COLORS[sample], label=sample)
    axis.set(xlabel="Paired event index", ylabel="Reconstructed SiTracks",
             title=f"{title}: track multiplicity")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "multiplicity.png", dpi=220)
    fig.savefig(output_dir / "multiplicity.pdf")
    plt.close(fig)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--construction", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Matched norm1 mother-split: SIM vs COUNT")
    args = parser.parse_args()
    print(json.dumps(plot(args.store_dir, args.construction, args.split,
                          args.output_dir, args.title), indent=2))


if __name__ == "__main__":
    main()
