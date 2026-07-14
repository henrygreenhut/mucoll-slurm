#!/usr/bin/env python3
"""Train a PFN to distinguish unique-mother (norm1) from cloned-mother
(norm42-RandomRot) GEN pseudo-crossings.

Units are whole variable-length particle sets (no capping, no random
subsampling): unique = --n-files norm1 cycles, reuse = n_files/42 norm42
cycles, same decay statistics. Splits are by cycle (train/val/test =
60/15/25 of the paired cycles); within a split both classes use the same
cycles, so mother identity carries no label information.

Designed for 30-minute Perlmutter debug-queue windows: checkpoints every
epoch and exits cleanly at --max-minutes; resubmit with the same --label to
resume. Test evaluation (disjoint blocked units + unit bootstrap) runs
automatically once training finishes.

Needs only numpy, h5py, tensorflow (no sklearn, no energyflow); the PFN
architecture (Phi=(200,200,256), masked scaled sum, F=(200,200,200)) is
built in plain Keras in libtest_common.build_pfn. On Perlmutter GPU nodes:
`module load tensorflow`.

Examples:
    python pfn_libtest_train.py --label A0_n42
    python pfn_libtest_train.py --label A1_nophi_n42 --drop-phi
    python pfn_libtest_train.py --label null_n42 --null-test
    python pfn_libtest_train.py --label shuffle_n42 --shuffle-labels
"""

import argparse
import csv
import json
import os
import time

import numpy as np

import libtest_common as lc

PHI_SIZES = (200, 200, 256)
F_SIZES = (200, 200, 200)


def parse_args():
    scratch = os.environ.get("PSCRATCH", ".")
    store_dir = os.path.join(scratch, "mucoll/libtest/stores")
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm1-store", default=os.path.join(store_dir, "gen_norm1_MUPLUS.h5"))
    parser.add_argument("--norm42-store", default=os.path.join(store_dir, "gen_norm42_MUPLUS.h5"))
    parser.add_argument("--label", required=True, help="run name; also resume key")
    parser.add_argument("--outdir", default="pfn_results")
    parser.add_argument("--n-files", type=int, default=42,
                        help="norm1 files per unit (must be multiple of clone factor)")
    parser.add_argument("--clone-factor", type=int, default=42)
    parser.add_argument("--units-per-epoch", type=int, default=2000, help="per class")
    parser.add_argument("--val-units", type=int, default=300, help="per class, fixed")
    parser.add_argument(
        "--overlap-test-units", type=int, default=0, metavar="N",
        help="exploratory test mode: draw N random units per class from the "
             "held-out cycle pool; units may overlap each other, so no "
             "independent-unit bootstrap uncertainty is reported (default 0 "
             "uses disjoint blocked test units)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--split-fracs", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
        default=(0.60, 0.15, 0.25),
        help="cycle-level train/validation/test fractions (default: "
             "0.60 0.15 0.25; use 0.50 0.25 0.25 for large null units)")
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="checkpoint and exit after this wall time (0 = off)")
    parser.add_argument("--features", default="paper", choices=["paper", "bib"],
                        help="paper = PFN-ID inputs per arXiv:1810.05165 "
                             "(log pT, theta, cos/sin phi, PDG one-hot); "
                             "bib = + asinh time/vertex (BIB-literature tier)")
    parser.add_argument("--drop-phi", action="store_true",
                        help="ablation A1: remove cos/sin phi features")
    parser.add_argument("--null-test", action="store_true",
                        help="norm1-vs-norm1 control (expect AUC 0.5)")
    parser.add_argument(
        "--null-partition", choices=["shared", "random-halves", "halves"],
        default="shared",
        help="null source pools: 'shared' draws both labels from the same "
             "cycle pool (recommended; removes cycle identity as a label "
             "feature); 'random-halves' uses disjoint randomly interleaved "
             "pools for independent test units; 'halves' preserves the "
             "legacy contiguous-half control")
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="randomize training labels (expect val AUC 0.5)")
    parser.add_argument("--e-min", type=float, default=0.0, help="energy cut [GeV]")
    parser.add_argument("--t-abs-max", type=float, default=0.0, help="|t| cut")
    parser.add_argument("--norm-stat-units", type=int, default=100,
                        help="units per class used to compute feature mean/std")
    parser.add_argument("--latent-scale", default="auto",
                        help="constant multiplying the summed latent: 'auto' = "
                             "1/median unit multiplicity (default), 'none' = raw "
                             "sum (ablation), or an explicit float")
    return parser.parse_args()


class UnitSampler:
    """Builds (features, label) units for one class from one store."""

    def __init__(self, store, positions_by_split, files_per_unit, args):
        self.store = store
        self.positions = positions_by_split
        self.files_per_unit = files_per_unit
        self.args = args

    def build(self, file_positions, mean, std):
        raw = self.store.file_arrays(file_positions)
        raw = lc.apply_cuts(raw, self.args.e_min, self.args.t_abs_max)
        feats = lc.build_features(raw, feature_set=self.args.features,
                                  drop_phi=self.args.drop_phi)
        return (feats - mean) / std

    def random_unit(self, rng, split):
        return lc.sample_unit_positions(rng, self.positions[split], self.files_per_unit)


def make_batches(unit_defs, samplers, mean, std, batch_size, rng=None):
    """Yield padded (x, y) batches from a list of (class_id, positions)."""
    order = np.arange(len(unit_defs))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [unit_defs[i] for i in order[start:start + batch_size]]
        feats = [samplers[c].build(pos, mean, std) for c, pos in chunk]
        labels = np.asarray([c for c, _ in chunk], dtype=np.int32)
        max_n = max(len(f) for f in feats)
        x = np.zeros((len(feats), max_n, feats[0].shape[1]), dtype=np.float32)
        for i, f in enumerate(feats):
            x[i, : len(f)] = f
        y = np.zeros((len(feats), 2), dtype=np.float32)
        y[np.arange(len(feats)), labels] = 1.0
        yield x, y, labels


def predict_units(model, unit_defs, samplers, mean, std, batch_size):
    scores, labels = [], []
    for x, _, lab in make_batches(unit_defs, samplers, mean, std, batch_size):
        preds = model.predict_on_batch(x)
        scores.extend(np.asarray(preds)[:, 1].tolist())
        labels.extend(lab.tolist())
    return np.asarray(labels), np.asarray(scores)


def load_state(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {"epoch": 0, "best_val_auc": -1.0, "best_epoch": -1, "done": False}


def save_state(path, state):
    with open(path, "w") as f:
        json.dump(state, f, indent=1)


def append_history(path, row):
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    start_time = time.time()
    outdir = os.path.join(args.outdir, args.label)
    os.makedirs(outdir, exist_ok=True)
    state_path = os.path.join(outdir, "state.json")
    last_w = os.path.join(outdir, "last.weights.h5")
    best_w = os.path.join(outdir, "best.weights.h5")
    stats_path = os.path.join(outdir, "norm_stats.json")
    state = load_state(state_path)

    if args.n_files % args.clone_factor != 0:
        raise SystemExit("--n-files must be a multiple of --clone-factor")
    if args.overlap_test_units < 0:
        raise SystemExit("--overlap-test-units must be non-negative")
    if any(frac <= 0 for frac in args.split_fracs):
        raise SystemExit("--split-fracs values must all be positive")
    if not np.isclose(sum(args.split_fracs), 1.0):
        raise SystemExit("--split-fracs values must sum to 1")
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    print(f"[{args.label}] loading stores")
    store1 = lc.Store(args.norm1_store)
    if args.null_test:
        store_b = store1
    else:
        store_b = lc.Store(args.norm42_store)
    common, pos1, pos_b = lc.common_positions(store1, store_b)
    print(f"  paired cycles: {len(common)}"
          f" (norm1 files: {store1.n_files}, classB files: {store_b.n_files})")
    splits = lc.split_indices(len(common), tuple(args.split_fracs))

    if args.null_test:
        if args.null_partition == "shared":
            # Label-independent null: both classes draw independently from
            # the same cycle pool.  Train/val/test remain cycle-disjoint.
            split_a = {k: pos1[v] for k, v in splits.items()}
            split_b = {k: pos1[v] for k, v in splits.items()}
        elif args.null_partition == "random-halves":
            # Independent-source null: randomly interleave the cycles before
            # assigning disjoint halves to the labels.  This preserves blocked
            # disjoint test evaluation without making cycle order a label
            # feature, as the legacy contiguous halves did.
            rng_partition = np.random.default_rng(args.seed + 714210)
            split_a = {}
            split_b = {}
            for key, values in splits.items():
                shuffled = rng_partition.permutation(values)
                midpoint = len(shuffled) // 2
                split_a[key] = pos1[shuffled[:midpoint]]
                split_b[key] = pos1[shuffled[midpoint:]]
        else:
            # Legacy null retained only for reproducing earlier runs.  At
            # large n, contiguous halves can expose cycle-range differences.
            split_a = {k: pos1[v[: len(v) // 2]] for k, v in splits.items()}
            split_b = {k: pos1[v[len(v) // 2:]] for k, v in splits.items()}
        files_b = args.n_files
    else:
        split_a = {k: pos1[v] for k, v in splits.items()}
        split_b = {k: pos_b[v] for k, v in splits.items()}
        files_b = args.n_files // args.clone_factor

    samplers = [
        UnitSampler(store1, split_a, args.n_files, args),   # class 0: unique
        UnitSampler(store_b, split_b, files_b, args),        # class 1: reuse
    ]
    for cls, sampler in enumerate(samplers):
        for split_name, positions in sampler.positions.items():
            if len(positions) < sampler.files_per_unit:
                raise SystemExit(
                    "class {} {} split has {} cycles but a unit requires {}; "
                    "adjust --split-fracs or --n-files".format(
                        cls, split_name, len(positions),
                        sampler.files_per_unit))

    # --- feature normalization + latent scale from train-split units -----
    if os.path.isfile(stats_path):
        mean, std, latent_scale = lc.load_norm_stats(stats_path)
    else:
        rng = np.random.default_rng(args.seed)
        sample_feats = []
        for cls in (0, 1):
            for _ in range(args.norm_stat_units):
                pos = samplers[cls].random_unit(rng, "train")
                raw = samplers[cls].store.file_arrays(pos)
                raw = lc.apply_cuts(raw, args.e_min, args.t_abs_max)
                sample_feats.append(lc.build_features(
                    raw, feature_set=args.features, drop_phi=args.drop_phi))
        mean, std = lc.compute_norm_stats(sample_feats)
        if args.latent_scale == "auto":
            latent_scale = 1.0 / float(np.median([len(f) for f in sample_feats]))
        elif args.latent_scale == "none":
            latent_scale = 1.0
        else:
            latent_scale = float(args.latent_scale)
        lc.save_norm_stats(stats_path, mean, std,
                           lc.feature_names(args.features, args.drop_phi),
                           latent_scale)
        del sample_feats
    n_features = len(mean)
    print(f"  features: {n_features} ('{args.features}'"
          f"{', no-phi' if args.drop_phi else ''})"
          f" | latent scale 1/{1.0 / latent_scale:.0f}")

    # --- fixed validation units ------------------------------------------
    rng_val = np.random.default_rng(args.seed + 999)
    val_defs = [(c, samplers[c].random_unit(rng_val, "val"))
                for c in (0, 1) for _ in range(args.val_units)]

    # --- model -------------------------------------------------------------
    model = lc.build_pfn(n_features, latent_scale,
                         phi_sizes=PHI_SIZES, f_sizes=F_SIZES)
    if state["epoch"] > 0 and os.path.isfile(last_w):
        model.load_weights(last_w)
        print(f"  resumed from epoch {state['epoch']}"
              f" (best val AUC {state['best_val_auc']:.4f})")

    # --- training loop ------------------------------------------------------
    history_path = os.path.join(outdir, "history.csv")
    while not state["done"] and state["epoch"] < args.epochs:
        epoch = state["epoch"]
        rng = np.random.default_rng(args.seed * 100003 + epoch)
        train_defs = [(c, samplers[c].random_unit(rng, "train"))
                      for c in (0, 1) for _ in range(args.units_per_epoch)]
        if args.shuffle_labels:
            classes = rng.integers(0, 2, size=len(train_defs))
            train_defs = [(int(k), pos) for k, (_, pos) in zip(classes, train_defs)]

        t0 = time.time()
        losses = []
        for x, y, _ in make_batches(train_defs, samplers, mean, std,
                                    args.batch_size, rng=rng):
            out = model.train_on_batch(x, y)
            losses.append(float(out[0] if isinstance(out, (list, tuple)) else out))
        train_time = time.time() - t0

        y_val, s_val = predict_units(model, val_defs, samplers, mean, std,
                                     args.batch_size)
        val_auc = lc.auc_score(y_val, s_val)

        state["epoch"] = epoch + 1
        improved = val_auc > state["best_val_auc"] + 1e-4
        if improved:
            state["best_val_auc"] = val_auc
            state["best_epoch"] = epoch
            model.save_weights(best_w)
        model.save_weights(last_w)
        append_history(history_path, {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "val_auc": val_auc, "seconds": round(train_time, 1),
        })
        save_state(state_path, state)
        print(f"epoch {epoch}: loss {np.mean(losses):.4f} | val AUC {val_auc:.4f}"
              f"{' *' if improved else ''} | {train_time:.0f}s", flush=True)

        if epoch - state["best_epoch"] >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs")
            state["done"] = True
            save_state(state_path, state)
        if args.max_minutes > 0 and (time.time() - start_time) / 60 > args.max_minutes:
            print("wall-clock limit reached -- checkpoint saved;"
                  " resubmit with the same --label to resume")
            return

    if state["epoch"] >= args.epochs:
        state["done"] = True
        save_state(state_path, state)

    # --- test evaluation ---------------------------------------------------
    if os.path.isfile(best_w):
        model.load_weights(best_w)
    if args.overlap_test_units:
        test_mode = "overlapping"
        print("test evaluation (overlapping held-out units, best weights; "
              "exploratory AUC)")
        rng_test = np.random.default_rng(args.seed + 2026)
        test_defs = [
            (c, samplers[c].random_unit(rng_test, "test"))
            for c in (0, 1) for _ in range(args.overlap_test_units)
        ]
    else:
        test_mode = "disjoint"
        print("test evaluation (disjoint units, best weights)")
        blocks_a = lc.blocked_unit_positions(split_a["test"], args.n_files)
        positions_b = split_b["test"]
        if args.null_test and args.null_partition == "shared":
            # Both null labels use the same held-out distribution, as they do
            # during training, but a second deterministic blocking avoids
            # comparing every unit with an identical copy of itself.  Blocks
            # are disjoint within each label; source cycles may appear once in
            # each label, matching shared-pool null resampling semantics.
            rng_blocks = np.random.default_rng(args.seed + 2027)
            positions_b = rng_blocks.permutation(positions_b)
            test_mode = "shared-blocked"
        blocks_b = lc.blocked_unit_positions(positions_b, files_b)
        test_defs = [(0, b) for b in blocks_a] + [(1, b) for b in blocks_b]
    y_test, s_test = predict_units(model, test_defs, samplers, mean, std,
                                   args.batch_size)
    auc = lc.auc_score(y_test, s_test)
    score_std = float(np.std(s_test))
    score_range = float(np.ptp(s_test))
    near_constant_scores = score_std < 1e-3
    if near_constant_scores:
        print("WARNING: test scores are nearly constant "
              f"(std={score_std:.3g}, range={score_range:.3g}); "
              "rank AUC may be driven by numerical differences")
    if test_mode == "disjoint":
        boot_mean, boot_std = lc.bootstrap_auc(
            s_test[y_test == 0], s_test[y_test == 1])
        uncertainty_note = "unit bootstrap over mutually disjoint test units"
        print(f"\nTEST AUC = {auc:.4f} +- {boot_std:.4f} (bootstrap over"
              f" {int((y_test == 0).sum())} unique /"
              f" {int((y_test == 1).sum())} reuse units)")
    elif test_mode == "shared-blocked":
        boot_mean, boot_std = None, None
        uncertainty_note = (
            "not estimated: units are disjoint within each null label, but "
            "the labels reuse the same held-out source-cycle pool")
        print(f"\nTEST AUC = {auc:.4f} * ({int((y_test == 0).sum())} /"
              f" {int((y_test == 1).sum())} null units; disjoint within each"
              " label, shared held-out source pool, no bootstrap error)")
    else:
        boot_mean, boot_std = None, None
        uncertainty_note = (
            "not estimated: constructed test units overlap in source cycles; "
            "unit-level bootstrap would understate uncertainty")
        print(f"\nTEST AUC = {auc:.4f} * ({int((y_test == 0).sum())} unique /"
              f" {int((y_test == 1).sum())} reuse units; overlapping held-out"
              " units, exploratory point estimate, no bootstrap error)")

    with open(os.path.join(outdir, "test_scores.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "score", "first_cycle_position", "n_files",
                         "test_mode"])
        for (cls, pos), score in zip(test_defs, s_test):
            writer.writerow([cls, f"{score:.6f}", int(pos[0]), len(pos), test_mode])
    with open(os.path.join(outdir, "auc_summary.json"), "w") as f:
        json.dump({
            "label": args.label, "test_auc": auc, "bootstrap_std": boot_std,
            "bootstrap_mean": boot_mean, "best_val_auc": state["best_val_auc"],
            "best_epoch": state["best_epoch"], "epochs_run": state["epoch"],
            "n_test_units": len(test_defs), "test_mode": test_mode,
            "test_units_mutually_disjoint": test_mode == "disjoint",
            "test_score_std": score_std, "test_score_range": score_range,
            "near_constant_test_scores": near_constant_scores,
            "uncertainty_note": uncertainty_note, "config": vars(args),
        }, f, indent=1)
    print(f"outputs -> {outdir}")


if __name__ == "__main__":
    main()
