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


def parse_size_list(text):
    sizes = tuple(int(v) for v in text.split(","))
    if any(s <= 0 for s in sizes):
        raise argparse.ArgumentTypeError("layer sizes must be positive")
    return sizes


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
    parser.add_argument("--epochs", type=int, default=None,
                        help="max epochs (default: 200, or 40 for --null-test "
                             "-- a null has nothing to converge to, so it "
                             "doesn't need the same budget as a signal run)")
    parser.add_argument("--patience", type=int, default=None,
                        help="epochs without improvement before stopping "
                             "(default: 15, or 8 for --null-test)")
    parser.add_argument("--min-delta", type=float, default=None,
                        help="minimum val AUC gain over the running best to "
                             "count as improvement (default: 1e-4, or 0.02 "
                             "for --null-test). A null's val AUC has ~0.02-"
                             "0.03 sampling noise at --val-units 300/class "
                             "(SE = sqrt((2n+1)/(12n^2)) under AUC=0.5); too "
                             "small a threshold lets pure noise repeatedly "
                             "look like a new best, resetting patience and "
                             "running to the epoch cap chasing nothing -- "
                             "observed burning a full 80-epoch cap (~11 GPU-h) "
                             "on a null at n=420")
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
    parser.add_argument("--arch", default="local", choices=["local", "energyflow"],
                        help="'energyflow' uses energyflow.archs.PFN directly "
                             "(textbook raw sum; requires --latent-scale none); "
                             "'local' is the equivalence-checked Keras build "
                             "with the optional latent scale")
    parser.add_argument("--jit", action="store_true",
                        help="compile with XLA JIT (model.compile(jit_compile="
                             "True)). Experimental: may sidestep the TF/XLA "
                             "int32 overflow bug (different codegen path than "
                             "the legacy GPU kernels that hit it), but our "
                             "particle count N varies every batch, so watch "
                             "per-epoch seconds for recompilation overhead")
    parser.add_argument("--eval-point-units", type=int, default=300,
                        help="overlapping held-out events per class for the "
                             "primary (automatic) test AUC")
    parser.add_argument("--eval-bootstrap-reps", type=int, default=200,
                        help="paired-cycle bootstrap replicates (resumable; "
                             "each one regenerates events from resampled "
                             "cycles and reruns the model, so this is the "
                             "dominant cost of evaluation -- e.g. 25-50 for "
                             "a quick separation check, 0 or 1 to skip the "
                             "bootstrap entirely and report only the point "
                             "estimate test_auc with no uncertainty)")
    parser.add_argument("--eval-bootstrap-units", type=int, default=100,
                        help="regenerated events per class per bootstrap pool")
    parser.add_argument("--phi-sizes", type=parse_size_list,
                        default=PHI_SIZES,
                        help=f"comma-separated Phi (per-particle) layer "
                             f"widths, e.g. 100,100,128 for a small network "
                             f"(default: {PHI_SIZES[0]},{PHI_SIZES[1]},"
                             f"{PHI_SIZES[2]})")
    parser.add_argument("--f-sizes", type=parse_size_list,
                        default=F_SIZES,
                        help=f"comma-separated F (event-level) layer widths "
                             f"(default: {F_SIZES[0]},{F_SIZES[1]},{F_SIZES[2]})")
    parser.add_argument("--split-fracs", type=float, nargs=3,
                        default=SOURCE_SPLIT, metavar=("TRAIN", "VAL", "TEST"),
                        help=f"cycle-level train/val/test fractions, must "
                             f"sum to 1 (default: {SOURCE_SPLIT[0]} "
                             f"{SOURCE_SPLIT[1]} {SOURCE_SPLIT[2]})")
    args = parser.parse_args()
    if abs(sum(args.split_fracs) - 1.0) > 1e-6:
        raise SystemExit(f"--split-fracs must sum to 1, got {args.split_fracs}")
    # Persist fixed scientific choices in every result config.
    args.clone_factor = CLONE_FACTOR
    args.split_fracs = tuple(args.split_fracs)
    args.norm_stat_units = NORM_STAT_UNITS
    args.null_partition = "shared"
    # Null-aware defaults: a null has no real ceiling to converge to, so it
    # doesn't warrant the signal run's budget, and its val AUC noise floor
    # (~0.02-0.03 at --val-units 300/class) means a tiny min-delta just
    # chases fluctuations instead of detecting genuine improvement.
    if args.epochs is None:
        args.epochs = 40 if args.null_test else 200
    if args.patience is None:
        args.patience = 8 if args.null_test else 15
    if args.min_delta is None:
        args.min_delta = 0.02 if args.null_test else 1e-4
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


def binary_cross_entropy(labels, scores):
    """Mean two-class cross entropy from PFN class-1 probabilities."""
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.where(labels == 1, scores, 1.0 - scores)
    return float(-np.mean(np.log(probabilities)))


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
    fieldnames = list(row)
    if exists:
        # Old runs did not record val_loss. Preserve their column layout if a
        # user resumes one, instead of silently shifting CSV columns.
        with open(path, newline="") as f:
            fieldnames = next(csv.reader(f))
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
    if args.arch == "energyflow":
        if latent_scale == 1.0:
            model = lc.build_pfn_energyflow(n_features,
                                            phi_sizes=args.phi_sizes,
                                            f_sizes=args.f_sizes,
                                            jit_compile=args.jit)
        else:
            # energyflow.archs.EFN's actual weighted-aggregation graph with
            # z_i = latent_scale (real particles) / 0 (padding), verified
            # bitwise-equivalent to the local scaled build by
            # pfn_arch_equivalence_check.py -- official-package provenance
            # for the scaled variant too, not a local reimplementation.
            model = lc.build_pfn_energyflow_scaled(
                n_features, latent_scale,
                phi_sizes=args.phi_sizes, f_sizes=args.f_sizes,
                jit_compile=args.jit)
    else:
        model = lc.build_pfn(n_features, latent_scale,
                             phi_sizes=args.phi_sizes, f_sizes=args.f_sizes,
                             jit_compile=args.jit)
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
        val_loss = binary_cross_entropy(y_val, s_val)

        state["epoch"] = epoch + 1
        improved = val_auc > state["best_val_auc"] + args.min_delta
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
            "val_loss": val_loss, "val_auc": val_auc,
            "seconds": round(train_time, 1),
        })
        save_state(state_path, state)
        print(f"epoch {epoch}: loss {np.mean(losses):.4f} | val loss {val_loss:.4f}"
              f" | val AUC {val_auc:.4f}"
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

    # --- evaluation --------------------------------------------------------
    # Primary protocol (automatic): overlapping held-out events with a
    # paired-cycle bootstrap -- the cluster bootstrap over the true
    # independent objects (source cycles). A quick disjoint blocked
    # evaluation is retained as a secondary cross-check. Resumable across
    # windows; skipped entirely once the summary exists.
    summary_path = os.path.join(outdir, "auc_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            prior = json.load(f)
        if prior.get("test_mode") == "overlapping-paired-cycle-bootstrap":
            print(f"evaluation already complete -> {summary_path}")
            return

    if os.path.isfile(best_w):
        model.load_weights(best_w)
    pool_a = split_a["test"]
    pool_b = split_b["test"]
    files_per_unit = (args.n_files, files_b)

    # 1) secondary: disjoint blocked cross-check (cheap, recomputed on resume)
    blocks_a = lc.blocked_unit_positions(pool_a, args.n_files)
    positions_b = pool_b
    if args.null_test:
        rng_blocks = np.random.default_rng(args.seed + 2027)
        positions_b = rng_blocks.permutation(positions_b)
    blocks_b = lc.blocked_unit_positions(positions_b, files_b)
    disjoint_defs = [(0, b) for b in blocks_a] + [(1, b) for b in blocks_b]
    y_dj, s_dj = predict_units(model, disjoint_defs, samplers, mean, std,
                               args.batch_size)
    disjoint_auc = lc.auc_score(y_dj, s_dj)
    if args.null_test:
        disjoint_std = None
    else:
        _, disjoint_std = lc.bootstrap_auc(s_dj[y_dj == 0], s_dj[y_dj == 1])
    print(f"disjoint cross-check: AUC {disjoint_auc:.4f}"
          + (f" +- {disjoint_std:.4f}" if disjoint_std is not None else ""))

    # 2) primary: overlapping point estimate (cached once computed)
    point_path = os.path.join(outdir, "point_summary.json")
    if os.path.isfile(point_path):
        with open(point_path) as f:
            point = json.load(f)
    else:
        print(f"primary evaluation: {args.eval_point_units} overlapping"
              " events/class from the held-out cycle pool")
        rng_pt = np.random.default_rng(args.seed + 2026)
        point_defs = [(c, rng_pt.choice(pools, size=files_per_unit[c],
                                        replace=False))
                      for c, pools in ((0, pool_a), (1, pool_b))
                      for _ in range(args.eval_point_units)]
        y_pt, s_pt = predict_units(model, point_defs, samplers, mean, std,
                                   args.batch_size)
        point = {"auc": lc.auc_score(y_pt, s_pt),
                 "score_std": float(np.std(s_pt)),
                 "score_range": float(np.ptp(s_pt))}
        with open(os.path.join(outdir, "test_scores.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "score", "test_mode"])
            for (cls, _), score in zip(point_defs, s_pt):
                writer.writerow([cls, f"{score:.6g}", "overlapping"])
        with open(point_path, "w") as f:
            json.dump(point, f, indent=1)
        print(f"point AUC = {point['auc']:.6f}")
    if point["score_std"] < 1e-3:
        print("WARNING: test scores are nearly constant "
              f"(std={point['score_std']:.3g}); rank AUC may be driven by "
              "numerical noise")

    # 3) primary uncertainty: paired-cycle bootstrap, resumable via CSV
    boot_path = os.path.join(outdir, "paired_cycle_bootstrap.csv")
    values = []
    if os.path.isfile(boot_path):
        with open(boot_path, newline="") as f:
            values = [float(row["auc"]) for row in csv.DictReader(f)]
    n_test_cycles = len(pool_a)
    for rep in range(len(values), args.eval_bootstrap_reps):
        rng = np.random.default_rng(args.seed + 1000003 * (rep + 1))
        slots = rng.integers(0, n_test_cycles, size=n_test_cycles)
        bpool_a, bpool_b = pool_a[slots], pool_b[slots]
        boot_defs = [(c, rng.choice(bpool, size=files_per_unit[c],
                                    replace=False))
                     for c, bpool in ((0, bpool_a), (1, bpool_b))
                     for _ in range(args.eval_bootstrap_units)]
        y_b, s_b = predict_units(model, boot_defs, samplers, mean, std,
                                 args.batch_size)
        rep_auc = lc.auc_score(y_b, s_b)
        exists = os.path.isfile(boot_path)
        with open(boot_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["replicate", "auc"])
            writer.writerow([rep, f"{rep_auc:.12g}"])
        values.append(rep_auc)
        print(f"bootstrap {rep + 1}/{args.eval_bootstrap_reps}:"
              f" AUC {rep_auc:.6f}", flush=True)
        if (args.max_minutes > 0
                and (time.time() - start_time) / 60 > args.max_minutes):
            print("wall-clock limit reached -- bootstrap checkpointed;"
                  " resubmit with the same --label to continue")
            return

    values = np.asarray(values, dtype=np.float64)
    have_bootstrap = len(values) > 1
    with open(summary_path, "w") as f:
        json.dump({
            "label": args.label,
            "test_auc": point["auc"],
            "bootstrap_mean": float(np.mean(values)) if have_bootstrap else None,
            "bootstrap_std": float(np.std(values, ddof=1)) if have_bootstrap else None,
            "bootstrap_ci68": (np.percentile(values, [16, 84]).tolist()
                               if have_bootstrap else None),
            "bootstrap_ci95": (np.percentile(values, [2.5, 97.5]).tolist()
                               if have_bootstrap else None),
            "test_mode": ("overlapping-paired-cycle-bootstrap" if have_bootstrap
                         else "overlapping-point-estimate-only"),
            "test_units_mutually_disjoint": False,
            "n_test_units": 2 * args.eval_point_units,
            "test_score_std": point["score_std"],
            "test_score_range": point["score_range"],
            "near_constant_test_scores": point["score_std"] < 1e-3,
            "disjoint_check": {"auc": disjoint_auc, "bootstrap_std": disjoint_std,
                               "n_units": len(disjoint_defs)},
            "best_val_auc": state["best_val_auc"],
            "best_epoch": state["best_epoch"], "epochs_run": state["epoch"],
            "uncertainty_note": (
                "two-level nonparametric bootstrap over matched held-out "
                "cycle pairs; events regenerated per pool; frozen classifier"
                if have_bootstrap else
                "no bootstrap requested (--eval-bootstrap-reps <= 1): point "
                "estimate only, no calibrated uncertainty on test_auc"),
            "config": vars(args),
        }, f, indent=1)
    if have_bootstrap:
        print(f"\nTEST AUC = {point['auc']:.4f}"
              f" (paired-cycle bootstrap SD {np.std(values, ddof=1):.4f},"
              f" 95% CI [{np.percentile(values, 2.5):.4f},"
              f" {np.percentile(values, 97.5):.4f}])"
              f" | disjoint cross-check {disjoint_auc:.4f}")
    else:
        print(f"\nTEST AUC = {point['auc']:.4f} (point estimate only, no"
              " bootstrap requested) | disjoint cross-check"
              f" {disjoint_auc:.4f}")
    print(f"outputs -> {outdir}")


if __name__ == "__main__":
    main()
