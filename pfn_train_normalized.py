#!/usr/bin/env python3

import argparse
import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from ml_common import PFO, fit_slots, load_h5, split_by_class, write_rows
from pfn_train import (
    BATCH_SIZE,
    EPOCHS,
    F_SIZES,
    PHI_SIZES,
    callbacks,
    get_pfn,
    one_hot,
    prediction_rows,
    save_roc,
    set_seed,
    write_history,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_a")
    parser.add_argument("sample_b")
    parser.add_argument("label")
    return parser.parse_args()


def normalized_inputs(x):
    out = np.zeros_like(x, dtype=np.float32)

    for i, event in enumerate(x):
        pt = event[:, PFO["pt"]]
        mask = pt > 0
        if not np.any(mask):
            continue

        pt = event[mask, PFO["pt"]].astype(np.float64)
        eta = event[mask, PFO["eta"]].astype(np.float64)
        phi = event[mask, PFO["phi"]].astype(np.float64)
        energy = np.maximum(event[mask, PFO["energy"]].astype(np.float64), 0.0)
        mass = np.abs(event[mask, PFO["mass"]].astype(np.float64))
        charge = event[mask, PFO["charge"]].astype(np.float64)
        pfo_type = event[mask, PFO["type"]].astype(np.float64)

        sum_pt = np.sum(pt)
        sum_energy = np.sum(energy)

        out[i, mask, 0] = pt / sum_pt if sum_pt > 0 else 0.0
        out[i, mask, 1] = np.log1p(pt) / 6.0
        out[i, mask, 2] = energy / sum_energy if sum_energy > 0 else 0.0
        out[i, mask, 3] = np.log1p(energy) / 6.0
        out[i, mask, 4] = np.clip(eta / 5.0, -2.0, 2.0)
        out[i, mask, 5] = np.sin(phi)
        out[i, mask, 6] = np.cos(phi)
        out[i, mask, 7] = np.clip(charge, -3.0, 3.0) / 3.0
        out[i, mask, 8] = np.log1p(mass) / 6.0
        out[i, mask, 9] = np.clip(pfo_type / 1000.0, -5.0, 5.0)

    return out


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
    x = normalized_inputs(np.concatenate([x_a, x_b], axis=0))
    y = np.asarray([0] * n + [1] * n, dtype=np.int32)
    y_hot = one_hot(y)
    meta = meta_a[:n] + meta_b[:n]

    train_idx, val_idx, test_idx = split_by_class(y)
    split = np.full(len(y), "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"

    print(f"Using {n} events per class")
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print("Preprocessing: pt fractions, log scales, angular sin/cos")
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
    print(f"\nPFN normalized AUC: {auc:.4f}\n")

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
        fields = [
            "label", "class_a", "class_b", "events_per_class",
            "preprocessing", "auc", "roc_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "label": args.label,
            "class_a": class_a,
            "class_b": class_b,
            "events_per_class": n,
            "preprocessing": "normalized",
            "auc": auc,
            "roc_path": roc_path,
        })

    print(f"ROC -> {roc_path}")
    print(f"Predictions -> {predictions_path}")


if __name__ == "__main__":
    main()
