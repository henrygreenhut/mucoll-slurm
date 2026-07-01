#!/usr/bin/env python3

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score

from bdt_train import FEATURE_NAMES, sample_features
from ml_common import load_h5, write_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_a")
    parser.add_argument("sample_b")
    parser.add_argument("label")
    return parser.parse_args()


def finite(values):
    return np.asarray(values, dtype=np.float64)[np.isfinite(values)]


def ks_statistic(a, b):
    a = np.sort(finite(a))
    b = np.sort(finite(b))
    if len(a) == 0 or len(b) == 0:
        return 0.0
    values = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / len(a)
    cdf_b = np.searchsorted(b, values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def feature_auc(y, values):
    values = finite(values)
    if len(values) != len(y) or len(np.unique(values)) < 2:
        return 0.5
    return float(roc_auc_score(y, values))


def summary_row(name, a, b, y):
    values = np.concatenate([a, b])
    auc = feature_auc(y, values)
    separation = max(auc, 1.0 - auc)
    direction = "class_b_higher" if auc >= 0.5 else "class_a_higher"

    return {
        "feature": name,
        "auc_class_b_high": auc,
        "separation": separation,
        "direction": direction,
        "ks": ks_statistic(a, b),
        "class_a_mean": float(np.mean(a)),
        "class_b_mean": float(np.mean(b)),
        "class_a_median": float(np.median(a)),
        "class_b_median": float(np.median(b)),
        "class_a_std": float(np.std(a)),
        "class_b_std": float(np.std(b)),
        "class_a_p05": float(np.percentile(a, 5)),
        "class_b_p05": float(np.percentile(b, 5)),
        "class_a_p95": float(np.percentile(a, 95)),
        "class_b_p95": float(np.percentile(b, 95)),
    }


def bins_for(a, b):
    values = finite(np.concatenate([a, b]))
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        return np.asarray([lo - 0.5, hi + 0.5])
    return np.linspace(lo, hi, 31)


def save_distributions(path, x_a, x_b, rows, class_a, class_b, label):
    plt.rcParams["figure.figsize"] = (5, 4)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.autolayout"] = True

    with PdfPages(path) as pdf:
        for row in rows:
            i = FEATURE_NAMES.index(row["feature"])
            a = x_a[:, i]
            b = x_b[:, i]
            plt.figure()
            bins = bins_for(a, b)
            plt.hist(a, bins=bins, histtype="step", density=True, label=class_a)
            plt.hist(b, bins=bins, histtype="step", density=True, label=class_b)
            plt.xlabel(row["feature"])
            plt.ylabel("density")
            plt.title(
                f"{label}: {row['feature']}\n"
                f"sep={row['separation']:.3f}, ks={row['ks']:.3f}"
            )
            plt.legend(frameon=False)
            pdf.savefig()
            plt.close()


def event_rows(x_a, x_b, meta_a, meta_b, class_a, class_b):
    rows = []
    for label, class_name, x, meta in [
        (0, class_a, x_a, meta_a),
        (1, class_b, x_b, meta_b),
    ]:
        for i, row_meta in enumerate(meta):
            row = {
                "label": label,
                "class_name": class_name,
                "event_index": row_meta["event_index"],
                "source_file": row_meta["source_file"],
                "source_event": row_meta["source_event"],
            }
            for j, name in enumerate(FEATURE_NAMES):
                row[name] = float(x[i, j])
            rows.append(row)
    return rows


def main():
    args = parse_args()
    outdir = os.path.join("feature_diagnostics", args.label)
    os.makedirs(outdir, exist_ok=True)

    print("Loading samples")
    a, meta_a, class_a = load_h5(args.sample_a)
    b, meta_b, class_b = load_h5(args.sample_b)

    n = min(len(a), len(b))
    x_a = sample_features(a[:n])
    x_b = sample_features(b[:n])
    y = np.asarray([0] * n + [1] * n, dtype=np.int32)

    rows = [
        summary_row(name, x_a[:, i], x_b[:, i], y)
        for i, name in enumerate(FEATURE_NAMES)
    ]
    rows = sorted(rows, key=lambda row: row["separation"], reverse=True)

    summary_path = os.path.join(outdir, f"feature_summary_{args.label}.csv")
    events_path = os.path.join(outdir, f"event_features_{args.label}.csv")
    plots_path = os.path.join(outdir, f"feature_distributions_{args.label}.pdf")

    write_rows(summary_path, list(rows[0]), rows)
    write_rows(
        events_path,
        ["label", "class_name", "event_index", "source_file", "source_event"] + FEATURE_NAMES,
        event_rows(x_a, x_b, meta_a[:n], meta_b[:n], class_a, class_b),
    )
    save_distributions(plots_path, x_a, x_b, rows, class_a, class_b, args.label)

    print(f"Summary -> {summary_path}")
    print(f"Event features -> {events_path}")
    print(f"Distributions -> {plots_path}")


if __name__ == "__main__":
    main()
