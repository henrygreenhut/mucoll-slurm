#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


SAMPLES = {
    "legacy": ("Legacy inclusive norm42", "#D55E00"),
    "split_bh": ("Separately sampled muon-producing BIB", "#0072B2"),
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-dir", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--expected-events", type=int, default=2000)
    return parser.parse_args()


def read_pfo_counts(directory, expected_events):
    paths = sorted(Path(directory).glob("job_*/reco_output_*.edm4hep.root"))
    if not paths:
        raise SystemExit("no RECO files in {}".format(directory))

    counts = []
    for path in paths:
        with uproot.open(path) as root:
            particles = root["events"]["PandoraPFOs/PandoraPFOs.PDG"].array()
        counts.append(ak.to_numpy(ak.num(particles, axis=1)))

    counts = np.concatenate(counts).astype(int)
    if len(counts) != expected_events:
        raise SystemExit(
            "{} contains {} events; expected {}".format(
                directory, len(counts), expected_events
            )
        )
    return counts


def draw(data, samples, output, bins):
    figure, axis = plt.subplots(figsize=(6.4, 4.5))
    for sample in samples:
        label, color = SAMPLES[sample]
        values = data[sample]
        axis.hist(
            values,
            bins=bins,
            weights=np.full(len(values), 1.0 / len(values)),
            histtype="step",
            linewidth=1.8,
            color=color,
            label=label,
        )

    axis.set_xlabel("PFOs per event")
    axis.set_ylabel("Fraction of events")
    axis.set_yscale("log")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", frameon=False)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    args = arguments()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = {
        "legacy": read_pfo_counts(args.legacy_dir, args.expected_events),
        "split_bh": read_pfo_counts(args.split_dir, args.expected_events),
    }
    high = max(int(np.max(values)) for values in data.values())
    bins = np.arange(-0.5, high + 1.5)

    draw(data, ("legacy", "split_bh"), outdir / "reco_n420_legacy_vs_split_bh_pfos", bins)
    draw(data, ("legacy",), outdir / "reco_n420_legacy_pfos", bins)
    draw(data, ("split_bh",), outdir / "reco_n420_split_bh_pfos", bins)

    with (outdir / "reco_n420_pfo_counts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample", "event", "pfos"))
        for sample, values in data.items():
            writer.writerows((sample, index, value) for index, value in enumerate(values))

    for sample, values in data.items():
        print(
            sample,
            "events=", len(values),
            "mean=", round(float(np.mean(values)), 3),
            "std=", round(float(np.std(values, ddof=1)), 3),
            "min=", int(np.min(values)),
            "max=", int(np.max(values)),
        )
    print("plots ->", outdir)


if __name__ == "__main__":
    main()
