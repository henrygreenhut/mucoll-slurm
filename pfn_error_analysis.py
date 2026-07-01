#!/usr/bin/env python3

import argparse
import csv
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from ml_common import write_rows


FEATURES = [
    "n_pfos",
    "sum_pt",
    "sum_energy",
    "leading_pt",
    "leading_energy",
    "subleading_pt",
    "third_pt",
    "mean_pt",
    "std_pt",
    "mean_energy",
    "std_energy",
    "top3_frac",
    "top5_frac",
    "n_pt_gt_20",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("features")
    parser.add_argument("label")
    parser.add_argument("--outdir", default="pfn_error_analysis")
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def key(row, class_field):
    return row[class_field], int(row["event_index"])


def to_float(row, name, default=np.nan):
    try:
        return float(row[name])
    except Exception:
        return default


def join_rows(predictions, features):
    feature_map = {key(row, "class_name"): row for row in features}
    rows = []
    for row in predictions:
        joined = dict(row)
        joined.update(feature_map.get(key(row, "true_class"), {}))
        joined["correct"] = int(joined["correct"])
        joined["score_class_b"] = to_float(joined, "score_class_b")
        joined["confidence"] = to_float(joined, "confidence")
        joined["margin"] = abs(joined["score_class_b"] - 0.5)
        rows.append(joined)
    return rows


def confusion_rows(rows):
    counts = Counter(
        (row["split"], row["true_class"], row["predicted_class"], row["correct"])
        for row in rows
    )
    out = []
    for (split, true_class, pred_class, correct), count in sorted(counts.items()):
        out.append({
            "split": split,
            "true_class": true_class,
            "predicted_class": pred_class,
            "correct": correct,
            "count": count,
        })
    return out


def accuracy_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["true_class"])].append(row)
        groups[(row["split"], "all")].append(row)
        groups[("all", row["true_class"])].append(row)
        groups[("all", "all")].append(row)

    out = []
    for (split, true_class), group in sorted(groups.items()):
        out.append({
            "split": split,
            "true_class": true_class,
            "n": len(group),
            "correct": sum(row["correct"] for row in group),
            "accuracy": sum(row["correct"] for row in group) / len(group),
            "mean_score_class_b": float(np.mean([row["score_class_b"] for row in group])),
        })
    return out


def feature_rows(rows):
    out = []
    for split in ["test", "val", "train", "all"]:
        split_rows = rows if split == "all" else [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        correct = [row for row in split_rows if row["correct"]]
        wrong = [row for row in split_rows if not row["correct"]]
        for name in FEATURES:
            c = np.asarray([to_float(row, name) for row in correct], dtype=np.float64)
            w = np.asarray([to_float(row, name) for row in wrong], dtype=np.float64)
            c = c[np.isfinite(c)]
            w = w[np.isfinite(w)]
            if len(c) == 0 or len(w) == 0:
                continue
            out.append({
                "split": split,
                "feature": name,
                "correct_mean": float(np.mean(c)),
                "wrong_mean": float(np.mean(w)),
                "wrong_minus_correct": float(np.mean(w) - np.mean(c)),
                "correct_median": float(np.median(c)),
                "wrong_median": float(np.median(w)),
                "correct_p05": float(np.percentile(c, 5)),
                "wrong_p05": float(np.percentile(w, 5)),
                "correct_p95": float(np.percentile(c, 95)),
                "wrong_p95": float(np.percentile(w, 95)),
            })
    return sorted(
        out,
        key=lambda row: (row["split"] != "test", -abs(row["wrong_minus_correct"])),
    )


def selected_fields(rows):
    fields = [
        "split",
        "true_class",
        "predicted_class",
        "correct",
        "score_class_a",
        "score_class_b",
        "confidence",
        "margin",
        "event_index",
        "source_file",
        "source_event",
    ]
    return fields + [name for name in FEATURES if name in rows[0]]


def keep_fields(rows, fields):
    return [{field: row.get(field, "") for field in fields} for row in rows]


def save_plots(path, rows, label):
    plt.rcParams["figure.figsize"] = (5, 4)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.autolayout"] = True

    with PdfPages(path) as pdf:
        test = [row for row in rows if row["split"] == "test"]
        for subset_name, subset in [("all", rows), ("test", test)]:
            if not subset:
                continue
            plt.figure()
            for cls in sorted({row["true_class"] for row in subset}):
                values = [row["score_class_b"] for row in subset if row["true_class"] == cls]
                plt.hist(values, bins=np.linspace(0, 1, 31), histtype="step", label=cls)
            plt.xlabel("PFN score for class_b")
            plt.ylabel("events")
            plt.title(f"{label}: PFN scores ({subset_name})")
            plt.legend(frameon=False)
            pdf.savefig()
            plt.close()

        for name in ["leading_pt", "std_pt", "sum_pt", "sum_energy", "n_pfos"]:
            if name not in rows[0]:
                continue
            values = np.asarray([to_float(row, name) for row in rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            bins = np.linspace(float(np.min(values)), float(np.max(values)), 31)
            plt.figure()
            for correct, style in [(1, "-"), (0, "--")]:
                vals = [
                    to_float(row, name)
                    for row in rows
                    if row["split"] == "test" and row["correct"] == correct
                ]
                vals = np.asarray(vals, dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    plt.hist(
                        vals,
                        bins=bins,
                        histtype="step",
                        density=True,
                        linestyle=style,
                        label="right" if correct else "wrong",
                    )
            plt.xlabel(name)
            plt.ylabel("density")
            plt.title(f"{label}: {name} for test right/wrong")
            plt.legend(frameon=False)
            pdf.savefig()
            plt.close()


def main():
    args = parse_args()
    outdir = os.path.join(args.outdir, args.label)
    os.makedirs(outdir, exist_ok=True)

    rows = join_rows(read_csv(args.predictions), read_csv(args.features))
    fields = selected_fields(rows)
    wrong = sorted(
        [row for row in rows if not row["correct"]],
        key=lambda row: row["confidence"],
        reverse=True,
    )
    right = sorted(
        [row for row in rows if row["correct"]],
        key=lambda row: row["confidence"],
        reverse=True,
    )
    feature_summary = feature_rows(rows)

    write_rows(
        os.path.join(outdir, f"confusion_{args.label}.csv"),
        ["split", "true_class", "predicted_class", "correct", "count"],
        confusion_rows(rows),
    )
    write_rows(
        os.path.join(outdir, f"accuracy_{args.label}.csv"),
        ["split", "true_class", "n", "correct", "accuracy", "mean_score_class_b"],
        accuracy_rows(rows),
    )
    if feature_summary:
        write_rows(
            os.path.join(outdir, f"feature_right_wrong_{args.label}.csv"),
            list(feature_summary[0]),
            feature_summary,
        )
    write_rows(os.path.join(outdir, f"wrong_events_{args.label}.csv"), fields, keep_fields(wrong, fields))
    write_rows(os.path.join(outdir, f"right_events_{args.label}.csv"), fields, keep_fields(right, fields))
    save_plots(os.path.join(outdir, f"right_wrong_plots_{args.label}.pdf"), rows, args.label)

    test = [row for row in rows if row["split"] == "test"]
    wrong_test = [row for row in test if not row["correct"]]
    print(f"Events: {len(rows)}")
    print(f"Test events: {len(test)}")
    print(f"Wrong test events: {len(wrong_test)}")
    print(f"Test accuracy: {1 - len(wrong_test) / len(test):.3f}")
    print(f"Output -> {outdir}")


if __name__ == "__main__":
    main()
