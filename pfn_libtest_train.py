#!/usr/bin/env python3
"""Train a PFN to distinguish unique-mother (norm1) from cloned-mother
(norm42-RandomRot) GEN pseudo-crossings.

Units are whole variable-length particle sets (no capping, no random
subsampling): unique = --n-files norm1 cycles, reuse = n_files/42 norm42
cycles, same decay statistics. Splits are by cycle (train/val/test =
50/25/25 of the paired cycles); within a split both classes use the same
cycles, so mother identity carries no label information.

Designed for resumable Perlmutter windows: model and Adam optimizer state are
checkpointed every epoch, and the process exits cleanly at --max-minutes.
Resubmit with the same --label to resume. Test evaluation (disjoint blocked
units + unit bootstrap) runs automatically once training finishes.

Needs only numpy, h5py, tensorflow (no sklearn, no energyflow); the PFN
architecture (Phi=(200,200,256), masked scaled sum, F=(200,200,200)) is
built in plain Keras in libtest_common.build_pfn. On Perlmutter GPU nodes:
`module load tensorflow`.

Fixed analysis choices are declared below. Command-line options are reserved
for event size, compute budget, random seeds, and the scaled/raw sum test.
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
CLONE_FACTOR = 42
SOURCE_SPLIT = (0.50, 0.25, 0.25)
NORM_STAT_UNITS = 100


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
    parser.add_argument("--min-epochs", type=int, default=0,
                        help="do not apply early stopping before this many "
                             "epochs have completed")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int,
                        help="TensorFlow initialization seed (default: --seed)")
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="checkpoint and exit after this wall time (0 = off)")
    parser.add_argument("--null-test", action="store_true",
                        help="norm1-vs-norm1 control (expect AUC 0.5)")
    parser.add_argument("--latent-scale", default="auto",
                        help="constant multiplying the summed latent: 'auto' = "
                             "1/median unit multiplicity (default), 'none' = raw "
                             "sum (ablation), or an explicit float")
    args = parser.parse_args()
    # Persist fixed scientific choices in every result config.
    args.clone_factor = CLONE_FACTOR
    args.split_fracs = SOURCE_SPLIT
    args.norm_stat_units = NORM_STAT_UNITS
    args.null_partition = "shared"
    return args


class UnitSampler:
    """Builds (features, label) units for one class from one store."""

    def __init__(self, store, positions_by_split, files_per_unit):
        self.store = store
        self.positions = positions_by_split
        self.files_per_unit = files_per_unit

    def build(self, file_positions, mean, std):
        raw = self.store.file_arrays(file_positions)
        feats = lc.build_features(raw)
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
    if args.model_seed is None:
        args.model_seed = args.seed
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.model_seed)
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
    if args.min_epochs < 0 or args.min_epochs > args.epochs:
        raise SystemExit("--min-epochs must be between 0 and --epochs")
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
        # Label-independent null: both classes independently sample the same
        # source pool. Train/validation/test remain cycle-disjoint.
        split_a = {k: pos1[v] for k, v in splits.items()}
        split_b = {k: pos1[v] for k, v in splits.items()}
        files_b = args.n_files
    else:
        split_a = {k: pos1[v] for k, v in splits.items()}
        split_b = {k: pos_b[v] for k, v in splits.items()}
        files_b = args.n_files // args.clone_factor

    samplers = [
        UnitSampler(store1, split_a, args.n_files),   # class 0: unique
        UnitSampler(store_b, split_b, files_b),       # class 1: reuse
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
                sample_feats.append(lc.build_features(raw))
        mean, std = lc.compute_norm_stats(sample_feats)
        if args.latent_scale == "auto":
            latent_scale = 1.0 / float(np.median([len(f) for f in sample_feats]))
        elif args.latent_scale == "none":
            latent_scale = 1.0
        else:
            latent_scale = float(args.latent_scale)
        lc.save_norm_stats(stats_path, mean, std, lc.FEATURE_NAMES, latent_scale)
        del sample_feats
    n_features = len(mean)
    print(f"  features: {n_features} | latent scale 1/{1.0 / latent_scale:.0f}")

    # --- fixed validation units ------------------------------------------
    rng_val = np.random.default_rng(args.seed + 999)
    val_defs = [(c, samplers[c].random_unit(rng_val, "val"))
                for c in (0, 1) for _ in range(args.val_units)]

    # --- model -------------------------------------------------------------
    model = lc.build_pfn(n_features, latent_scale,
                         phi_sizes=PHI_SIZES, f_sizes=F_SIZES)
    # Materialize Adam slot variables before restoring so its moments and
    # iteration counter are included, rather than silently resetting at each
    # Slurm window.
    if hasattr(model.optimizer, "build"):
        model.optimizer.build(model.trainable_variables)
    checkpoint_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_best_auc = tf.Variable(-1.0, dtype=tf.float64, trainable=False)
    checkpoint_best_epoch = tf.Variable(-1, dtype=tf.int64, trainable=False)
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=model.optimizer, epoch=checkpoint_epoch,
        best_val_auc=checkpoint_best_auc, best_epoch=checkpoint_best_epoch)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, os.path.join(outdir, "resume_checkpoint"), max_to_keep=1)
    if checkpoint_manager.latest_checkpoint:
        status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
        status.assert_existing_objects_matched()
        state["epoch"] = int(checkpoint_epoch.numpy())
        state["best_val_auc"] = float(checkpoint_best_auc.numpy())
        state["best_epoch"] = int(checkpoint_best_epoch.numpy())
        print(f"  resumed model + Adam from epoch {state['epoch']}"
              f" (best val AUC {state['best_val_auc']:.4f})")
    elif state["epoch"] > 0 and os.path.isfile(last_w):
        # Backward compatibility for pre-fix runs. New labels immediately use
        # the full TensorFlow checkpoint above.
        model.load_weights(last_w)
        print(f"  resumed legacy weights from epoch {state['epoch']}"
              f" (best val AUC {state['best_val_auc']:.4f}); optimizer state"
              " was unavailable")

    # --- training loop ------------------------------------------------------
    history_path = os.path.join(outdir, "history.csv")
    while not state["done"] and state["epoch"] < args.epochs:
        epoch = state["epoch"]
        rng = np.random.default_rng(args.seed * 100003 + epoch)
        train_defs = [(c, samplers[c].random_unit(rng, "train"))
                      for c in (0, 1) for _ in range(args.units_per_epoch)]
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
        checkpoint_epoch.assign(state["epoch"])
        checkpoint_best_auc.assign(state["best_val_auc"])
        checkpoint_best_epoch.assign(state["best_epoch"])
        checkpoint_manager.save(checkpoint_number=state["epoch"])
        append_history(history_path, {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "val_auc": val_auc, "seconds": round(train_time, 1),
        })
        save_state(state_path, state)
        print(f"epoch {epoch}: loss {np.mean(losses):.4f} | val AUC {val_auc:.4f}"
              f"{' *' if improved else ''} | {train_time:.0f}s", flush=True)

        if lc.should_early_stop(state, args.patience, args.min_epochs):
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
