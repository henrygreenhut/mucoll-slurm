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

Needs numpy, h5py, tensorflow; --arch energyflow additionally needs the
energyflow package + tf_keras. The PFN architecture (Phi=(200,200,256),
masked scaled sum, F=(200,200,200) by default) is built in plain Keras in
libtest_common.build_pfn, or via the real energyflow.archs.PFN/EFN classes
-- see --arch below. On Perlmutter GPU nodes: `module load tensorflow`.

Command-line flags are grouped below by what they control, roughly in the
order a reader would want to reason about them:
    data/unit construction   --norm1-store .. --n-files, --split-fracs
    validation & stopping    --val-units .. --min-delta-sigma
    reproducibility          --seed, --model-seed
    runtime/resume           --max-minutes
    what the model sees      --null-test, --features
    architecture             --latent-scale, --arch, --jit, --phi-sizes,
                             --f-sizes
    training dynamics        --lr .. --f-l2 (warmup/clipping/dropout/L2 --
                             all off by default, reproducing the original
                             fixed-lr/unregularized behavior exactly)
    test evaluation          --eval-point-units .. --eval-bootstrap-units
Every flag defaults to the original, already-validated behavior; nothing
here changes what a bare `pfn_libtest_train.py --label X` run does.
"""

import argparse
import csv
import json
import math
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
    # --- data / unit construction -------------------------------------
    parser.add_argument("--norm1-store", default=os.path.join(store_dir, "gen_norm1_MUPLUS.h5"))
    parser.add_argument("--norm42-store", default=os.path.join(store_dir, "gen_norm42_MUPLUS.h5"))
    parser.add_argument("--label", required=True, help="run name; also resume key")
    parser.add_argument("--outdir", default="pfn_results")
    parser.add_argument("--n-files", type=int, default=42,
                        help="norm1 files per unit (must be multiple of clone factor)")
    parser.add_argument("--units-per-epoch", type=int, default=2000, help="per class")
    # --- validation & early stopping ----------------------------------
    # A fresh random draw of --units-per-epoch/class every epoch (not the
    # same batch reused), since the true "dataset" here -- every possible
    # n_files-file combination -- is far too large to enumerate; val_defs
    # below is the opposite: drawn ONCE and held fixed for the whole run,
    # so early-stopping is judged against a stable target.
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
    parser.add_argument("--select-metric", default="auc", choices=["auc", "loss"],
                        help="metric that drives 'improved'/early-stopping/"
                             "best-checkpoint selection. 'auc' (default, the "
                             "original behavior) uses the fixed --min-delta "
                             "threshold above. 'loss' uses a self-"
                             "calibrating criterion instead: improved iff "
                             "val_loss drops by more than --min-delta-sigma "
                             "standard errors of the per-unit val loss (SEM "
                             "= std/sqrt(2*val_units)) -- automatically "
                             "compatible with whatever --val-units is "
                             "chosen, unlike a hand-picked --min-delta which "
                             "needs re-deriving every time val_units changes")
    parser.add_argument("--min-delta-sigma", type=float, default=1.0,
                        help="only used by --select-metric loss: required "
                             "val_loss improvement, in standard errors of "
                             "the per-unit val loss, to count as a new best "
                             "(default 1.0)")
    # --- reproducibility -----------------------------------------------
    # --seed controls weight init AND every data-sampling RNG (which units
    # get drawn each epoch, which units land in the fixed val set, which
    # test/bootstrap units get evaluated) -- a genuinely different run, not
    # just a different starting point. --model-seed lets you decouple init
    # from data sampling if you ever need to isolate which one a result
    # depends on (not yet exercised in practice: every sweep so far has
    # varied both together via --seed alone).
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int,
                        help="TensorFlow initialization seed (default: --seed)")
    # --- runtime / resume ------------------------------------------------
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="checkpoint and exit after this wall time (0 = off)")
    # --- what the model sees --------------------------------------------
    parser.add_argument("--null-test", action="store_true",
                        help="norm1-vs-norm1 control (expect AUC 0.5)")
    parser.add_argument("--features", default="paper",
                        choices=list(lc.FEATURE_SETS),
                        help="'paper' = momentum direction/magnitude + PDG "
                             "one-hot only (arXiv:1810.05165 recipe, adapted "
                             "for BIB). 'expanded' = paper + log energy, "
                             "asinh time/vertex-z/vertex-radius, and charge "
                             "-- maximum GEN-level truth sensitivity, not "
                             "meant to be realistic (reco-level features "
                             "would be smeared); see FEATURE_SETS in "
                             "libtest_common.py")
    # --- architecture ----------------------------------------------------
    # latent_scale=none (raw, unnormalized sum) is the one most prone to
    # the training collapse this project has spent a lot of time
    # characterizing: the pooled latent's magnitude scales with however
    # many particles are in a unit, which can push early activations/
    # gradients to extremes. See --warmup-epochs/--clipnorm below, and
    # arXiv:2206.11925 (Set Norm) on why normalizing sum-pooled Deep-Sets-
    # style architectures isn't just "add BatchNorm" -- naive normalization
    # here can also destroy real signal, so it isn't done reflexively.
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
                        help="compile both the model and Adam optimizer updates "
                             "with XLA JIT. By default both are explicitly "
                             "disabled. Experimental: may sidestep the TF/XLA "
                             "int32 overflow bug (different codegen path than "
                             "the legacy GPU kernels that hit it), but our "
                             "particle count N varies every batch, so watch "
                             "per-epoch seconds for recompilation overhead")
    # --- test evaluation ---------------------------------------------
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
    # --- architecture, continued (layer widths) -----------------------
    # Widest Phi layer matters for the TF/XLA int32 kernel-launch overflow
    # bug: batch_size * N * widest_Phi_width must stay under 2^31. Halving
    # Phi width (200,200,256 -> 100,100,128) doubles the safe N ceiling --
    # this is why the n420-scale runs use a halved network, not a
    # capacity/accuracy choice.
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
    # --- training dynamics ----------------------------------------------
    # All off by default (reproduces the original fixed-lr, unregularized
    # Adam exactly). Warmup and clipping are complementary responses to
    # the same raw-sum instability, not alternatives to each other: warmup
    # keeps EARLY steps small while Adam's moment estimates are still
    # noisy (the documented mechanism behind why Adam needs warmup at all
    # -- arXiv:1908.03265 RAdam, arXiv:1910.04209); clipping bounds the
    # worst case for ANY step, early or late, if a single batch's gradient
    # is still huge despite that.
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Adam base/target learning rate (default 0.001, "
                             "unchanged from the original hardcoded value)")
    parser.add_argument("--warmup-epochs", type=float, default=0.0,
                        help="linear LR warmup from 0 to --lr over this many "
                             "epochs' worth of gradient steps, then held "
                             "constant (0 = off, the original fixed-lr "
                             "behavior). Epochs, not a raw step count, so it "
                             "stays correct automatically if --units-per-"
                             "epoch/--batch-size change -- the actual step "
                             "count (steps/epoch = ceil(2*units_per_epoch/"
                             "batch_size), matching make_batches' chunking) "
                             "is computed once args are known, logged, and "
                             "recorded in config.json for exact reproduction. "
                             "Targets the raw-sum instability specifically: "
                             "with --latent-scale none, early loss/gradients "
                             "can be enormous (observed 70,000+ at n420) "
                             "while Adam's moment estimates are still noisy")
    parser.add_argument("--clipnorm", type=float, default=0.0,
                        help="clip each gradient's global norm to this value "
                             "(0 = off, the original unclipped behavior). "
                             "Complementary to --warmup-epochs, not "
                             "redundant: bounds the worst case if a single "
                             "batch's gradient is still huge despite warmup")
    parser.add_argument("--latent-dropout", type=float, default=0.0,
                        help="dropout on the pooled per-event latent vector, "
                             "post-sum (0 = off). Maps directly to "
                             "energyflow's own latent_dropout hyperparameter "
                             "for --arch energyflow")
    parser.add_argument("--f-dropout", type=float, default=0.0,
                        help="dropout on the F (event-level) dense layers "
                             "(0 = off). Maps to energyflow's F_dropouts")
    parser.add_argument("--phi-l2", type=float, default=0.0,
                        help="L2 regularization strength on the Phi "
                             "(per-particle) dense layers (0 = off). Maps "
                             "to energyflow's Phi_l2_regs")
    parser.add_argument("--f-l2", type=float, default=0.0,
                        help="L2 regularization strength on the F "
                             "(event-level) dense layers (0 = off). Maps "
                             "to energyflow's F_l2_regs")
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

    def __init__(self, store, positions_by_split, files_per_unit, feature_set="paper"):
        self.store = store
        self.positions = positions_by_split
        self.files_per_unit = files_per_unit
        self.feature_set = feature_set

    def build(self, file_positions, mean, std):
        raw = self.store.file_arrays(file_positions)
        feats = lc.build_features(raw, feature_set=self.feature_set)
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


def per_unit_cross_entropy(labels, scores):
    """Per-unit two-class cross entropy from PFN class-1 probabilities."""
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.where(labels == 1, scores, 1.0 - scores)
    return -np.log(probabilities)


def binary_cross_entropy(labels, scores):
    """Mean two-class cross entropy from PFN class-1 probabilities.

    Thin wrapper kept for test_libtest_training.py's coverage of the mean-
    loss formula; main()'s training loop calls per_unit_cross_entropy
    directly instead, since it also needs the per-unit values (not just
    the mean) to compute val_loss_sem for --select-metric loss.
    """
    return float(np.mean(per_unit_cross_entropy(labels, scores)))


def load_state(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {"epoch": 0, "best_val_auc": -1.0, "best_val_loss": float("inf"),
            "best_epoch": -1, "done": False}


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
    # Steps/epoch matches make_batches' own chunking exactly: it shuffles
    # the 2*units_per_epoch (both classes) unit list, then walks it in
    # batch_size chunks -- ceil, since a final partial batch still counts
    # as one training step. Resolving --warmup-epochs to an exact step
    # count here (once, from this run's own config) means no one -- not
    # the user, not whoever reads the command later -- has to hand-compute
    # or guess it; it's also recorded in config.json below for exact
    # reproduction even if --units-per-epoch/--batch-size defaults change
    # later.
    steps_per_epoch = math.ceil(2 * args.units_per_epoch / args.batch_size)
    args.warmup_steps = round(args.warmup_epochs * steps_per_epoch)
    if args.warmup_epochs > 0:
        print(f"  warmup: {args.warmup_epochs} epoch(s) = "
              f"{args.warmup_steps} steps ({steps_per_epoch} steps/epoch)")
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
        UnitSampler(store1, split_a, args.n_files, args.features),   # class 0: unique
        UnitSampler(store_b, split_b, files_b, args.features),       # class 1: reuse
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
    expected_names = lc.feature_names(args.features)
    if os.path.isfile(stats_path):
        mean, std, latent_scale = lc.load_norm_stats(stats_path)
        with open(stats_path) as f:
            cached_names = json.load(f).get("names")
        if cached_names != expected_names:
            raise SystemExit(
                f"{stats_path} was computed for feature set {cached_names}, "
                f"but --features {args.features!r} expects {expected_names}; "
                "this label was likely started with a different --features "
                "value. Use a new --label for a different feature set.")
    else:
        rng = np.random.default_rng(args.seed)
        sample_feats = []
        for cls in (0, 1):
            for _ in range(args.norm_stat_units):
                pos = samplers[cls].random_unit(rng, "train")
                raw = samplers[cls].store.file_arrays(pos)
                sample_feats.append(lc.build_features(raw, feature_set=args.features))
        mean, std = lc.compute_norm_stats(sample_feats)
        if args.latent_scale == "auto":
            latent_scale = 1.0 / float(np.median([len(f) for f in sample_feats]))
        elif args.latent_scale == "none":
            latent_scale = 1.0
        else:
            latent_scale = float(args.latent_scale)
        lc.save_norm_stats(stats_path, mean, std, expected_names, latent_scale)
        del sample_feats
    n_features = len(mean)
    print(f"  features: {n_features} ('{args.features}')"
          f" | latent scale 1/{1.0 / latent_scale:.0f}")

    # --- fixed validation units ------------------------------------------
    rng_val = np.random.default_rng(args.seed + 999)
    val_defs = [(c, samplers[c].random_unit(rng_val, "val"))
                for c in (0, 1) for _ in range(args.val_units)]

    # --- model -------------------------------------------------------------
    train_kwargs = dict(
        lr=args.lr, warmup_steps=args.warmup_steps, clipnorm=args.clipnorm,
        latent_dropout=args.latent_dropout, f_dropouts=args.f_dropout,
        phi_l2=args.phi_l2, f_l2=args.f_l2)
    if args.arch == "energyflow":
        if latent_scale == 1.0:
            model = lc.build_pfn_energyflow(n_features,
                                            phi_sizes=args.phi_sizes,
                                            f_sizes=args.f_sizes,
                                            jit_compile=args.jit,
                                            **train_kwargs)
        else:
            # energyflow.archs.EFN's actual weighted-aggregation graph with
            # z_i = latent_scale (real particles) / 0 (padding), verified
            # bitwise-equivalent to the local scaled build by
            # pfn_arch_equivalence_check.py -- official-package provenance
            # for the scaled variant too, not a local reimplementation.
            model = lc.build_pfn_energyflow_scaled(
                n_features, latent_scale,
                phi_sizes=args.phi_sizes, f_sizes=args.f_sizes,
                jit_compile=args.jit, **train_kwargs)
    else:
        model = lc.build_pfn(n_features, latent_scale,
                             phi_sizes=args.phi_sizes, f_sizes=args.f_sizes,
                             jit_compile=args.jit, **train_kwargs)
    print("  XLA JIT: model requested {} | optimizer effective {}"
          .format(args.jit,
                  bool(getattr(model.optimizer, "jit_compile", False))))
    # Materialize Adam slot variables before restoring so its moments and
    # iteration counter are included, rather than silently resetting at each
    # Slurm window.
    if hasattr(model.optimizer, "build"):
        model.optimizer.build(model.trainable_variables)
    # Both best_val_auc and best_val_loss are tracked regardless of which
    # one --select-metric actually uses to decide "improved" -- so a run's
    # summary always shows what the other metric was doing too, even
    # though only one of them drove checkpoint selection.
    checkpoint_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_best_auc = tf.Variable(-1.0, dtype=tf.float64, trainable=False)
    checkpoint_best_loss = tf.Variable(float("inf"), dtype=tf.float64, trainable=False)
    checkpoint_best_epoch = tf.Variable(-1, dtype=tf.int64, trainable=False)
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=model.optimizer, epoch=checkpoint_epoch,
        best_val_auc=checkpoint_best_auc, best_val_loss=checkpoint_best_loss,
        best_epoch=checkpoint_best_epoch)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, os.path.join(outdir, "resume_checkpoint"), max_to_keep=1)
    if checkpoint_manager.latest_checkpoint:
        status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
        # expect_partial(), not assert_existing_objects_matched(): the
        # latter turns out to hard-fail (not silently tolerate, despite
        # its docstring suggesting otherwise) whenever the CURRENT Python
        # object graph has a trackable with no match in an OLDER
        # checkpoint file -- exactly what happens whenever a new
        # checkpointed variable (like best_val_loss, added for
        # --select-metric loss) gets added and someone resumes a run
        # whose checkpoint predates it. expect_partial() is TF's own
        # idiom for "the object graph evolved since this was saved, and
        # that's fine" -- confirmed safe here because the restored values
        # are printed immediately below, so a genuinely bad mismatch
        # (wrong epoch count, nonsensical AUC) would be visible, not
        # silently swallowed.
        status.expect_partial()
        state["epoch"] = int(checkpoint_epoch.numpy())
        state["best_val_auc"] = float(checkpoint_best_auc.numpy())
        state["best_val_loss"] = float(checkpoint_best_loss.numpy())
        state["best_epoch"] = int(checkpoint_best_epoch.numpy())
        print(f"  resumed model + Adam from epoch {state['epoch']}"
              f" (best val AUC {state['best_val_auc']:.4f},"
              f" best val loss {state['best_val_loss']:.4f})")
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
        per_unit_losses = per_unit_cross_entropy(y_val, s_val)
        val_loss = float(np.mean(per_unit_losses))
        val_loss_sem = float(np.std(per_unit_losses, ddof=1)
                             / np.sqrt(len(per_unit_losses)))

        state["epoch"] = epoch + 1
        # "loss": noise-relative and self-calibrating (compares the drop to
        # this epoch's own SEM, so it stays correctly calibrated whatever
        # --val-units is). "auc": the original fixed-threshold rule --
        # kept only for old-configuration comparability, since --min-delta
        # is known to sit below val_auc's actual sampling noise floor at
        # small --val-units (see --min-delta's help text for the formula).
        if args.select_metric == "loss":
            improved = val_loss < state["best_val_loss"] - args.min_delta_sigma * val_loss_sem
        else:
            improved = val_auc > state["best_val_auc"] + args.min_delta
        if improved:
            state["best_val_auc"] = val_auc
            state["best_val_loss"] = val_loss
            state["best_epoch"] = epoch
            model.save_weights(best_w)
        model.save_weights(last_w)
        checkpoint_epoch.assign(state["epoch"])
        checkpoint_best_auc.assign(state["best_val_auc"])
        checkpoint_best_loss.assign(state["best_val_loss"])
        checkpoint_best_epoch.assign(state["best_epoch"])
        checkpoint_manager.save(checkpoint_number=state["epoch"])
        append_history(history_path, {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "val_loss": val_loss, "val_loss_sem": val_loss_sem, "val_auc": val_auc,
            "seconds": round(train_time, 1),
        })
        save_state(state_path, state)
        print(f"epoch {epoch}: loss {np.mean(losses):.4f} | val loss {val_loss:.4f}"
              f" (SEM {val_loss_sem:.4f}) | val AUC {val_auc:.4f}"
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
            "select_metric": args.select_metric,
            "best_val_auc": state["best_val_auc"],
            "best_val_loss": state["best_val_loss"],
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
