#!/usr/bin/env python3

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from ml_common import fit_slots, load_h5, split_by_class, write_rows


PHI_SIZES = (200, 200, 256)
F_SIZES = (200, 200, 200)
EPOCHS = 150
BATCH_SIZE = 16
PATIENCE = 25
MIN_DELTA = 0.0001


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_a")
    parser.add_argument("sample_b")
    parser.add_argument("label")
    return parser.parse_args()


def one_hot(y):
    out = np.zeros((len(y), 2), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def set_seed(seed=1):
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def get_pfn():
    try:
        from energyflow.archs.efn import PFN
    except ImportError:
        from energyflow.archs import PFN
    return PFN


def callbacks(weights_path):
    try:
        from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    return [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            weights_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]


def write_history(path, history):
    keys = list(history.history.keys())
    rows = []
    for epoch in range(len(history.history[keys[0]])):
        row = {"epoch": epoch + 1}
        for key in keys:
            row[key] = history.history[key][epoch]
        rows.append(row)
    write_rows(path, ["epoch"] + keys, rows)


def save_roc(path, y_true, score_b, auc, class_a, class_b, label):
    fp, tp, _ = roc_curve(y_true, score_b)
    plt.rcParams["figure.figsize"] = (4, 4)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.autolayout"] = True
    plt.figure()
    plt.plot(tp, 1 - fp, "-", color="black", label=f"PFN (AUC={auc:.3f})")
    plt.plot([0, 1], [1, 0], "--", color="gray", label="Random")
    plt.xlabel(f"{class_b} efficiency")
    plt.ylabel(f"{class_a} rejection")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower left", frameon=False)
    plt.title(label)
    plt.savefig(path)
    plt.close()


def prediction_rows(meta, y, split, preds, class_a, class_b):
    rows = []
    for i, row_meta in enumerate(meta):
        score_a = float(preds[i, 0])
        score_b = float(preds[i, 1])
        pred = int(score_b >= score_a)
        rows.append({
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
        })
    return rows


def main():
    args = parse_args()
    outdir = os.path.join("pfn_results", args.label)
    os.makedirs(outdir, exist_ok=True)
    set_seed()

    print("Loading samples")
    x_a, meta_a, class_a = load_h5(args.sample_a)
    x_b, meta_b, class_b = load_h5(args.sample_b)

    n = min(len(x_a), len(x_b))
    x_a, x_b = fit_slots(x_a[:n], x_b[:n])
    x = np.concatenate([x_a, x_b], axis=0)
    y = np.asarray([0] * n + [1] * n, dtype=np.int32)
    y_hot = one_hot(y)
    meta = meta_a[:n] + meta_b[:n]

    train_idx, val_idx, test_idx = split_by_class(y)
    split = np.full(len(y), "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"

    print(f"Using {n} events per class")
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"Phi={PHI_SIZES}, F={F_SIZES}")

    PFN = get_pfn()
    pfn = PFN(input_dim=x.shape[-1], Phi_sizes=PHI_SIZES, F_sizes=F_SIZES)
    weights_path = os.path.join(outdir, f"best_pfn_{args.label}.weights.h5")

    history = pfn.fit(
        x[train_idx],
        y_hot[train_idx],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x[val_idx], y_hot[val_idx]),
        verbose=1,
        callbacks=callbacks(weights_path),
    )

    if os.path.isfile(weights_path):
        pfn.model.load_weights(weights_path)

    preds = pfn.predict(x, batch_size=BATCH_SIZE)
    auc = roc_auc_score(y[test_idx], preds[test_idx, 1])
    print(f"\nPFN AUC: {auc:.4f}\n")

    fields = [
        "sample_path", "event_index", "source_file", "source_event",
        "true_label", "true_class", "split", "class_a_name", "class_b_name",
        "score_class_a", "score_class_b", "predicted_label", "predicted_class",
        "confidence", "correct", "n_particles", "sum_pt", "leading_pt",
        "sum_energy", "leading_energy", "event_hash",
    ]
    rows = prediction_rows(meta, y, split, preds, class_a, class_b)
    predictions_path = os.path.join(outdir, f"predictions_by_event_{args.label}.csv")
    roc_path = os.path.join(outdir, f"roc_{args.label}.pdf")

    write_history(os.path.join(outdir, f"history_{args.label}.csv"), history)
    write_rows(predictions_path, fields, rows)
    write_rows(
        os.path.join(outdir, f"predictions_test_{args.label}.csv"),
        fields,
        [row for row in rows if row["split"] == "test"],
    )
    save_roc(
        roc_path,
        y[test_idx],
        preds[test_idx, 1],
        auc,
        class_a,
        class_b,
        args.label,
    )

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


if __name__ == "__main__":
    main()
