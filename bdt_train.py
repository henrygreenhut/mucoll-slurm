#!/usr/bin/env python3

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from ml_common import PFO, load_h5, real_pfos, split_by_class, write_rows


N_ESTIMATORS = 200
LEARNING_RATE = 0.05
MAX_DEPTH = 3
SUBSAMPLE = 0.8
SEED = 1


FEATURE_NAMES = [
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
    "eta_mean",
    "eta_std",
    "abs_eta_mean",
    "central_pt_frac",
    "forward_pt_frac",
    "phi_resultant",
    "n_charged",
    "charged_pt_frac",
    "mean_abs_charge",
    "n_pfo_types",
    "leading_type",
    "n_pt_gt_1",
    "n_pt_gt_10",
    "n_pt_gt_20",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_a")
    parser.add_argument("sample_b")
    parser.add_argument("label")
    return parser.parse_args()


def frac(num, den):
    return float(num / den) if den > 0 else 0.0


def weighted_mean_std(values, weights):
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0, 0.0
    mean = float(np.sum(weights * values) / total)
    var = float(np.sum(weights * (values - mean) ** 2) / total)
    return mean, float(np.sqrt(max(var, 0.0)))


def leading_sum(pt, n):
    return float(np.sum(pt[:n])) if len(pt) else 0.0


def features(event):
    pfos = real_pfos(event)
    if len(pfos) == 0:
        return [0.0] * len(FEATURE_NAMES)

    pt = np.asarray(pfos[:, PFO["pt"]], dtype=np.float64)
    eta = np.asarray(pfos[:, PFO["eta"]], dtype=np.float64)
    phi = np.asarray(pfos[:, PFO["phi"]], dtype=np.float64)
    energy = np.asarray(pfos[:, PFO["energy"]], dtype=np.float64)
    charge = np.asarray(pfos[:, PFO["charge"]], dtype=np.float64)
    pfo_type = np.asarray(pfos[:, PFO["type"]], dtype=np.float64)
    order = np.argsort(pt)[::-1]
    pt = pt[order]
    eta = eta[order]
    phi = phi[order]
    energy = energy[order]
    charge = charge[order]
    pfo_type = pfo_type[order]

    total_pt = float(np.sum(pt))
    total_energy = float(np.sum(energy))
    eta_mean, eta_std = weighted_mean_std(eta, pt)
    abs_eta_mean, _ = weighted_mean_std(np.abs(eta), pt)
    cos_mean = frac(float(np.sum(pt * np.cos(phi))), total_pt)
    sin_mean = frac(float(np.sum(pt * np.sin(phi))), total_pt)
    phi_resultant = float(np.sqrt(cos_mean * cos_mean + sin_mean * sin_mean))
    charged = np.abs(charge) > 0

    return [
        float(len(pt)),
        total_pt,
        total_energy,
        float(pt[0]),
        float(energy[0]),
        float(pt[1]) if len(pt) > 1 else 0.0,
        float(pt[2]) if len(pt) > 2 else 0.0,
        float(np.mean(pt)),
        float(np.std(pt)),
        float(np.mean(energy)),
        float(np.std(energy)),
        frac(leading_sum(pt, 3), total_pt),
        frac(leading_sum(pt, 5), total_pt),
        eta_mean,
        eta_std,
        abs_eta_mean,
        frac(float(np.sum(pt[np.abs(eta) < 1.5])), total_pt),
        frac(float(np.sum(pt[np.abs(eta) > 2.0])), total_pt),
        phi_resultant,
        float(np.sum(charged)),
        frac(float(np.sum(pt[charged])), total_pt),
        float(np.mean(np.abs(charge))),
        float(len(np.unique(pfo_type))),
        float(pfo_type[0]),
        float(np.sum(pt > 1.0)),
        float(np.sum(pt > 10.0)),
        float(np.sum(pt > 20.0)),
    ]


def sample_features(particles):
    return np.asarray([features(event) for event in particles], dtype=np.float32)


def save_roc(path, y_true, scores, auc, class_a, class_b, label):
    fp, tp, _ = roc_curve(y_true, scores)
    plt.rcParams["figure.figsize"] = (4, 4)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.autolayout"] = True
    plt.figure()
    plt.plot(tp, 1 - fp, "-", color="black", label=f"BDT (AUC={auc:.3f})")
    plt.plot([0, 1], [1, 0], "--", color="gray", label="Random")
    plt.xlabel(f"{class_b} efficiency")
    plt.ylabel(f"{class_a} rejection")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower left", frameon=False)
    plt.title(label)
    plt.savefig(path)
    plt.close()


def prediction_rows(meta, y, split, scores, x, class_a, class_b):
    rows = []
    for i, row_meta in enumerate(meta):
        score_b = float(scores[i])
        score_a = 1.0 - score_b
        pred = int(score_b >= 0.5)
        row = {
            **row_meta,
            "true_label": int(y[i]),
            "true_class": class_b if y[i] else class_a,
            "split": split[i],
            "class_a_name": class_a,
            "class_b_name": class_b,
            "score_class_a": score_a,
            "score_class_b": score_b,
            "predicted_label": pred,
            "predicted_class": class_b if pred else class_a,
            "confidence": max(score_a, score_b),
            "correct": int(pred == y[i]),
        }
        for j, name in enumerate(FEATURE_NAMES):
            row[name] = float(x[i, j])
        rows.append(row)
    return rows


def main():
    args = parse_args()
    outdir = os.path.join("bdt_results", args.label)
    os.makedirs(outdir, exist_ok=True)

    print("Loading samples")
    a, meta_a, class_a = load_h5(args.sample_a)
    b, meta_b, class_b = load_h5(args.sample_b)

    n = min(len(a), len(b))
    x = np.concatenate([sample_features(a[:n]), sample_features(b[:n])], axis=0)
    y = np.asarray([0] * n + [1] * n, dtype=np.int32)
    meta = meta_a[:n] + meta_b[:n]

    train_idx, val_idx, test_idx = split_by_class(y)
    fit_idx = np.concatenate([train_idx, val_idx])
    split = np.full(len(y), "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"

    model = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        random_state=SEED,
    )
    model.fit(x[fit_idx], y[fit_idx])

    scores = model.predict_proba(x)[:, 1]
    auc = roc_auc_score(y[test_idx], scores[test_idx])
    print(f"\nBDT AUC: {auc:.4f}\n")

    feature_rows = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(
            zip(FEATURE_NAMES, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    rows = prediction_rows(meta, y, split, scores, x, class_a, class_b)

    meta_fields = [
        "sample_path", "event_index", "source_file", "source_event",
        "true_label", "true_class", "split", "class_a_name", "class_b_name",
        "score_class_a", "score_class_b", "predicted_label", "predicted_class",
        "confidence", "correct", "n_particles", "sum_pt", "leading_pt",
        "sum_energy", "leading_energy", "event_hash",
    ]
    fields = meta_fields + [name for name in FEATURE_NAMES if name not in meta_fields]

    predictions_path = os.path.join(outdir, f"predictions_by_event_{args.label}.csv")
    roc_path = os.path.join(outdir, f"roc_{args.label}.pdf")

    write_rows(predictions_path, fields, rows)
    write_rows(
        os.path.join(outdir, f"predictions_test_{args.label}.csv"),
        fields,
        [row for row in rows if row["split"] == "test"],
    )
    write_rows(
        os.path.join(outdir, f"feature_importance_{args.label}.csv"),
        ["feature", "importance"],
        feature_rows,
    )
    save_roc(roc_path, y[test_idx], scores[test_idx], auc, class_a, class_b, args.label)

    with open(os.path.join(outdir, "auc_summary.csv"), "w", newline="") as handle:
        fields = ["label", "class_a", "class_b", "events_per_class", "auc", "roc_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "label": args.label,
            "class_a": class_a,
            "class_b": class_b,
            "events_per_class": n,
            "auc": auc,
            "roc_path": roc_path,
        })

    print(f"ROC -> {roc_path}")
    print(f"Predictions -> {predictions_path}")
    print(f"Feature importance -> {outdir}/feature_importance_{args.label}.csv")


if __name__ == "__main__":
    main()
