#!/usr/bin/env python3
"""Train a PFN to distinguish unique-mother (norm1) from cloned-mother
(norm42-RandomRot) GEN pseudo-crossings.

Units are whole variable-length particle sets (no capping, no random
subsampling): unique = --n-files norm1 cycles, reuse = n_files/42 norm42
cycles, same decay statistics. A saved, shuffled 50/25/25 split of paired
source cycles prevents train/validation/test mother leakage; within a split
both classes use the same cycles, so mother identity carries no label
information.

Designed for resumable Perlmutter windows: model and Adam optimizer state are
checkpointed every epoch, and the process exits cleanly at --max-minutes.
Resubmit an identical scientific configuration with the same --label to
resume. Test evaluation runs automatically unless --skip-evaluation is set.

Needs numpy, h5py, tensorflow; --arch energyflow additionally needs the
energyflow package + tf_keras. The PFN architecture (Phi=(200,200,256),
masked scaled sum, F=(200,200,200) by default) is built in plain Keras in
libtest_common.build_pfn, or via the real energyflow.archs.PFN/EFN classes
-- see --arch below. On Perlmutter GPU nodes: `module load tensorflow`.

Command-line flags are grouped below by what they control, roughly in the
order a reader would want to reason about them:
    data/unit construction   --norm1-store .. --n-files, --split-fracs
    validation & stopping    --val-units .. --min-delta-sigma
    reproducibility          --data-seed, --model-seed
    runtime/resume           --max-minutes
    what the model sees      --null-test, --features
    architecture             --latent-scale, --arch, --jit, --phi-sizes,
                             --f-sizes
    training dynamics        --lr .. --f-l2 (warmup/clipping/dropout/L2 --
                             all off by default, reproducing the original
                             fixed-lr/unregularized behavior exactly)
    test evaluation          --eval-point-units .. --eval-bootstrap-units
Training batches are class-balanced; the default batch size 4 means two
independently sampled unique events and two independently sampled reused
events per optimizer step.
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
CONFIG_SCHEMA_VERSION = 2
RUNTIME_CONFIG_KEYS = {"max_minutes", "progress_every"}


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
    parser.add_argument("--batch-size", type=int, default=4)
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
    parser.add_argument("--data-seed", "--seed", dest="data_seed", type=int,
                        default=1701,
                        help="seed for source split and all event sampling; "
                             "--seed is retained as an alias")
    parser.add_argument("--model-seed", type=int,
                        help="independent TensorFlow initialization seed "
                             "(default: --data-seed, for legacy commands)")
    # --- runtime / resume ------------------------------------------------
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="checkpoint and exit after this wall time (0 = off)")
    parser.add_argument("--progress-every", type=int, default=25,
                        help="print train/validation progress every N batches")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="stop after training (for short smoke tests)")
    # --- what the model sees --------------------------------------------
    parser.add_argument("--null-test", action="store_true",
                        help="norm1-vs-norm1 control (expect AUC 0.5)")
    parser.add_argument("--features", default="paper",
                        choices=list(lc.FEATURE_SETS),
                        help="'paper' = momentum direction/magnitude + PDG "
                             "one-hot only (arXiv:1810.05165 recipe, adapted "
                             "for BIB). 'expanded' = paper + log energy, "
                             "asinh time/vertex-z/vertex-radius "
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
    parser.add_argument("--decay-epochs", type=float, default=0.0,
                        help="after warmup, cosine-decay --lr to --min-lr "
                             "over this many epochs (0 keeps LR constant)")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="final learning rate for cosine decay")
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
    if args.model_seed is None:
        args.model_seed = args.data_seed
    if abs(sum(args.split_fracs) - 1.0) > 1e-6:
        raise SystemExit(f"--split-fracs must sum to 1, got {args.split_fracs}")
    # Persist fixed scientific choices in every result config.
    args.clone_factor = CLONE_FACTOR
    args.split_fracs = tuple(args.split_fracs)
    args.norm_stat_units = NORM_STAT_UNITS
    args.null_partition = "shared"
    args.config_schema_version = CONFIG_SCHEMA_VERSION
    args.balanced_batches = True
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


def _padded_batch(chunk, samplers, mean, std):
    """Materialize and pad one batch of (class_id, source positions)."""
    feats = [samplers[c].build(pos, mean, std) for c, pos in chunk]
    labels = np.asarray([c for c, _ in chunk], dtype=np.int32)
    max_n = max(len(f) for f in feats)
    x = np.zeros((len(feats), max_n, feats[0].shape[1]), dtype=np.float32)
    for i, f in enumerate(feats):
        x[i, : len(f)] = f
    y = np.zeros((len(feats), 2), dtype=np.float32)
    y[np.arange(len(feats)), labels] = 1.0
    return x, y, labels


def make_batches(unit_defs, samplers, mean, std, batch_size, rng=None):
    """Yield ordinary padded batches, used for prediction/evaluation."""
    order = np.arange(len(unit_defs))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [unit_defs[i] for i in order[start:start + batch_size]]
        yield _padded_batch(chunk, samplers, mean, std)


def balanced_chunks(class_defs, batch_size, rng):
    """Return shuffled batches containing exactly half of each class."""
    if batch_size < 2 or batch_size % 2:
        raise ValueError("balanced training requires an even --batch-size >= 2")
    if len(class_defs) != 2 or len(class_defs[0]) != len(class_defs[1]):
        raise ValueError("balanced training requires two equally sized classes")
    half = batch_size // 2
    if len(class_defs[0]) % half:
        raise ValueError("--units-per-epoch must be divisible by batch-size/2")
    orders = [rng.permutation(len(defs)) for defs in class_defs]
    chunks = []
    for start in range(0, len(class_defs[0]), half):
        chunk = ([class_defs[0][i] for i in orders[0][start:start + half]]
                 + [class_defs[1][i] for i in orders[1][start:start + half]])
        rng.shuffle(chunk)
        chunks.append(chunk)
    return chunks


def make_balanced_batches(class_defs, samplers, mean, std, batch_size, rng):
    for chunk in balanced_chunks(class_defs, batch_size, rng):
        yield _padded_batch(chunk, samplers, mean, std)


def predict_units(model, unit_defs, samplers, mean, std, batch_size,
                  progress_every=0, progress_label="validation"):
    scores, labels = [], []
    n_batches = math.ceil(len(unit_defs) / batch_size)
    for step, (x, _, lab) in enumerate(
            make_batches(unit_defs, samplers, mean, std, batch_size), 1):
        preds = model.predict_on_batch(x)
        scores.extend(np.asarray(preds)[:, 1].tolist())
        labels.extend(lab.tolist())
        if progress_every and (step == 1 or step % progress_every == 0
                               or step == n_batches):
            print(f"  {progress_label} batch {step}/{n_batches}", flush=True)
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


def initial_state(select_metric):
    return {
        "epoch": 0,
        "max_val_auc": -1.0,
        "max_val_auc_epoch": -1,
        "min_val_loss": float("inf"),
        "min_val_loss_epoch": -1,
        "best_metric_value": (-1.0 if select_metric == "auc" else float("inf")),
        "best_epoch": -1,
        "done": False,
    }


def load_state(path, select_metric):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return initial_state(select_metric)


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


def write_or_validate_config(path, config):
    """Create an immutable scientific config; permit runtime-only changes."""
    config = json.loads(json.dumps(config))
    if not os.path.isfile(path):
        with open(path, "w") as f:
            json.dump(config, f, indent=1, sort_keys=True)
        return
    with open(path) as f:
        saved = json.load(f)
    if saved.get("config_schema_version") != CONFIG_SCHEMA_VERSION:
        raise SystemExit(
            f"{path} is a legacy/incompatible run. Use a new --label.")
    mismatches = []
    for key in sorted(set(saved) | set(config)):
        if key in RUNTIME_CONFIG_KEYS:
            continue
        if saved.get(key) != config.get(key):
            mismatches.append(
                f"  {key}: saved={saved.get(key)!r}, requested={config.get(key)!r}")
    if mismatches:
        raise SystemExit(
            "Refusing to resume with a changed scientific configuration.\n"
            + "\n".join(mismatches) + "\nUse a new --label.")


def load_or_create_validation_units(path, val_cycles, files_per_class,
                                    n_units, seed):
    """Persist the exact fixed validation source-cycle definitions."""
    val_cycles = np.asarray(val_cycles, dtype=np.int64)
    if os.path.isfile(path):
        with np.load(path) as payload:
            rows = [np.asarray(payload[f"class{c}"], dtype=np.int64)
                    for c in (0, 1)]
    else:
        rng = np.random.default_rng(seed)
        rows = [
            np.stack([rng.choice(val_cycles, size=n_files, replace=False)
                      for _ in range(n_units)])
            for n_files in files_per_class
        ]
        np.savez(path, class0=rows[0], class1=rows[1])
    for cls, (array, n_files) in enumerate(zip(rows, files_per_class)):
        if array.shape != (n_units, n_files):
            raise ValueError(
                f"saved validation class{cls} has shape {array.shape}, "
                f"expected {(n_units, n_files)}")
        if not np.all(np.isin(array, val_cycles)):
            raise ValueError("saved validation units use cycles outside val split")
        if any(len(np.unique(row)) != n_files for row in array):
            raise ValueError("saved validation unit repeats a source cycle")
    return rows


def validation_defs_from_cycles(rows, stores):
    defs = []
    for cls, (array, store) in enumerate(zip(rows, stores)):
        positions = np.searchsorted(store.cycle_ids, array)
        if not np.array_equal(store.cycle_ids[positions], array):
            raise ValueError("validation cycle is absent from its source store")
        defs.extend((cls, row) for row in positions)
    return defs


def update_validation_state(state, val_auc, val_loss, val_loss_sem,
                            select_metric, min_delta, min_delta_sigma, epoch):
    """Track both extrema; separately decide whether selected metric improved."""
    if val_auc > state["max_val_auc"]:
        state["max_val_auc"] = val_auc
        state["max_val_auc_epoch"] = epoch
    if val_loss < state["min_val_loss"]:
        state["min_val_loss"] = val_loss
        state["min_val_loss_epoch"] = epoch
    if select_metric == "loss":
        improved = (val_loss
                    < state["best_metric_value"] - min_delta_sigma * val_loss_sem)
        candidate = val_loss
    else:
        improved = val_auc > state["best_metric_value"] + min_delta
        candidate = val_auc
    if improved:
        state["best_metric_value"] = candidate
        state["best_epoch"] = epoch
    return improved


def current_learning_rate(model):
    schedule = getattr(model.optimizer, "_learning_rate",
                       model.optimizer.learning_rate)
    value = schedule(model.optimizer.iterations) if callable(schedule) else schedule
    return float(np.asarray(value.numpy() if hasattr(value, "numpy") else value))


def main():
    args = parse_args()
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.model_seed)
    start_time = time.time()
    outdir = os.path.join(args.outdir, args.label)
    os.makedirs(outdir, exist_ok=True)
    state_path = os.path.join(outdir, "state.json")
    last_w = os.path.join(outdir, "last.weights.h5")
    best_w = os.path.join(outdir, "best.weights.h5")
    stats_path = os.path.join(outdir, "norm_stats.json")

    if args.n_files % args.clone_factor != 0:
        raise SystemExit("--n-files must be a multiple of --clone-factor")
    if args.overlap_test_units < 0:
        raise SystemExit("--overlap-test-units must be non-negative")
    if args.min_epochs < 0 or args.min_epochs > args.epochs:
        raise SystemExit("--min-epochs must be between 0 and --epochs")
    if args.batch_size < 2 or args.batch_size % 2:
        raise SystemExit("--batch-size must be even and at least 2")
    if args.units_per_epoch % (args.batch_size // 2):
        raise SystemExit(
            "--units-per-epoch must be divisible by --batch-size/2")
    if args.decay_epochs < 0 or args.warmup_epochs < 0:
        raise SystemExit("--warmup-epochs and --decay-epochs must be non-negative")
    if args.decay_epochs > 0 and not 0.0 <= args.min_lr <= args.lr:
        raise SystemExit("--min-lr must be between zero and --lr")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
    steps_per_epoch = 2 * args.units_per_epoch // args.batch_size
    args.warmup_steps = round(args.warmup_epochs * steps_per_epoch)
    args.decay_steps = round(args.decay_epochs * steps_per_epoch)
    args.steps_per_epoch = steps_per_epoch
    if args.warmup_epochs > 0:
        print(f"  warmup: {args.warmup_epochs} epoch(s) = "
              f"{args.warmup_steps} steps ({steps_per_epoch} steps/epoch)")
    if args.decay_epochs > 0:
        print(f"  cosine decay: {args.decay_epochs} epoch(s) = "
              f"{args.decay_steps} steps, ending at {args.min_lr:g}")
    write_or_validate_config(os.path.join(outdir, "config.json"), vars(args))
    state = load_state(state_path, args.select_metric)

    print(f"[{args.label}] loading stores")
    store1 = lc.Store(args.norm1_store)
    if args.null_test:
        store_b = store1
    else:
        store_b = lc.Store(args.norm42_store)
    common, pos1, pos_b = lc.common_positions(store1, store_b)
    print(f"  paired cycles: {len(common)}"
          f" (norm1 files: {store1.n_files}, classB files: {store_b.n_files})")
    cycle_split = lc.load_or_create_cycle_split(
        os.path.join(outdir, "source_split.npz"), common,
        tuple(args.split_fracs), args.data_seed)
    splits = lc.cycle_split_positions(common, cycle_split)

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
        rng = np.random.default_rng(args.data_seed)
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
    val_rows = load_or_create_validation_units(
        os.path.join(outdir, "validation_units.npz"),
        cycle_split["val"], (args.n_files, files_b),
        args.val_units, args.data_seed + 999)
    val_defs = validation_defs_from_cycles(
        val_rows, (store1, store_b if not args.null_test else store1))

    # --- model -------------------------------------------------------------
    train_kwargs = dict(
        lr=args.lr, warmup_steps=args.warmup_steps, clipnorm=args.clipnorm,
        decay_steps=args.decay_steps, min_lr=args.min_lr,
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
    print("  XLA JIT: requested {} | model effective {} | optimizer effective {}"
          .format(args.jit, getattr(model, "_jit_compile", None),
                  bool(getattr(model.optimizer, "jit_compile", False))))
    # Materialize Adam slot variables before restoring so its moments and
    # iteration counter are included, rather than silently resetting at each
    # Slurm window.
    if hasattr(model.optimizer, "build"):
        model.optimizer.build(model.trainable_variables)
    checkpoint_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_max_auc = tf.Variable(-1.0, dtype=tf.float64, trainable=False)
    checkpoint_max_auc_epoch = tf.Variable(-1, dtype=tf.int64, trainable=False)
    checkpoint_min_loss = tf.Variable(float("inf"), dtype=tf.float64,
                                      trainable=False)
    checkpoint_min_loss_epoch = tf.Variable(-1, dtype=tf.int64, trainable=False)
    checkpoint_best_metric = tf.Variable(
        -1.0 if args.select_metric == "auc" else float("inf"),
        dtype=tf.float64, trainable=False)
    checkpoint_best_epoch = tf.Variable(-1, dtype=tf.int64, trainable=False)
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=model.optimizer, epoch=checkpoint_epoch,
        max_val_auc=checkpoint_max_auc,
        max_val_auc_epoch=checkpoint_max_auc_epoch,
        min_val_loss=checkpoint_min_loss,
        min_val_loss_epoch=checkpoint_min_loss_epoch,
        best_metric_value=checkpoint_best_metric,
        best_epoch=checkpoint_best_epoch)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, os.path.join(outdir, "resume_checkpoint"), max_to_keep=1)
    if checkpoint_manager.latest_checkpoint:
        status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
        try:
            status.assert_consumed()
        except (AssertionError, ValueError) as exc:
            raise SystemExit(
                "Checkpoint does not exactly match this model, optimizer, "
                "and state schema. Refusing a partial restore; use a new "
                f"--label.\n{exc}")
        state["epoch"] = int(checkpoint_epoch.numpy())
        state["max_val_auc"] = float(checkpoint_max_auc.numpy())
        state["max_val_auc_epoch"] = int(checkpoint_max_auc_epoch.numpy())
        state["min_val_loss"] = float(checkpoint_min_loss.numpy())
        state["min_val_loss_epoch"] = int(checkpoint_min_loss_epoch.numpy())
        state["best_metric_value"] = float(checkpoint_best_metric.numpy())
        state["best_epoch"] = int(checkpoint_best_epoch.numpy())
        print(f"  resumed model + Adam from epoch {state['epoch']}"
              f" (max val AUC {state['max_val_auc']:.4f},"
              f" min val loss {state['min_val_loss']:.4f})")
    elif state["epoch"] > 0:
        raise SystemExit(
            "state.json reports a resumed run but no full TensorFlow "
            "checkpoint exists; refusing to restore weights without Adam state")

    # --- training loop ------------------------------------------------------
    history_path = os.path.join(outdir, "history.csv")
    while not state["done"] and state["epoch"] < args.epochs:
        epoch = state["epoch"]
        rng = np.random.default_rng(args.data_seed * 100003 + epoch)
        train_defs = [
            [(c, samplers[c].random_unit(rng, "train"))
             for _ in range(args.units_per_epoch)]
            for c in (0, 1)
        ]
        lr_value = current_learning_rate(model)
        t0 = time.time()
        losses = []
        for step, (x, y, _) in enumerate(
                make_balanced_batches(train_defs, samplers, mean, std,
                                      args.batch_size, rng), 1):
            out = model.train_on_batch(x, y)
            losses.append(float(out[0] if isinstance(out, (list, tuple)) else out))
            if (args.progress_every
                    and (step == 1 or step % args.progress_every == 0
                         or step == steps_per_epoch)):
                print(f"  train batch {step}/{steps_per_epoch}: "
                      f"mean loss {np.mean(losses):.4f}, "
                      f"{time.time() - t0:.0f}s", flush=True)
        train_time = time.time() - t0

        val_t0 = time.time()
        y_val, s_val = predict_units(model, val_defs, samplers, mean, std,
                                     args.batch_size, args.progress_every)
        val_time = time.time() - val_t0
        val_auc = lc.auc_score(y_val, s_val)
        per_unit_losses = per_unit_cross_entropy(y_val, s_val)
        val_loss = float(np.mean(per_unit_losses))
        val_loss_sem = float(np.std(per_unit_losses, ddof=1)
                             / np.sqrt(len(per_unit_losses)))

        state["epoch"] = epoch + 1
        improved = update_validation_state(
            state, val_auc, val_loss, val_loss_sem, args.select_metric,
            args.min_delta, args.min_delta_sigma, epoch)
        if improved:
            model.save_weights(best_w)
        model.save_weights(last_w)
        checkpoint_epoch.assign(state["epoch"])
        checkpoint_max_auc.assign(state["max_val_auc"])
        checkpoint_max_auc_epoch.assign(state["max_val_auc_epoch"])
        checkpoint_min_loss.assign(state["min_val_loss"])
        checkpoint_min_loss_epoch.assign(state["min_val_loss_epoch"])
        checkpoint_best_metric.assign(state["best_metric_value"])
        checkpoint_best_epoch.assign(state["best_epoch"])
        checkpoint_manager.save(checkpoint_number=state["epoch"])
        append_history(history_path, {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "val_loss": val_loss, "val_loss_sem": val_loss_sem, "val_auc": val_auc,
            "learning_rate": lr_value,
            "train_seconds": round(train_time, 1),
            "val_seconds": round(val_time, 1),
            "seconds": round(train_time + val_time, 1),
        })
        save_state(state_path, state)
        print(f"epoch {epoch}: loss {np.mean(losses):.4f} | val loss {val_loss:.4f}"
              f" (SEM {val_loss_sem:.4f}) | val AUC {val_auc:.4f}"
              f"{' *' if improved else ''} | lr {lr_value:.3g}"
              f" | train {train_time:.0f}s + val {val_time:.0f}s", flush=True)

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

    if args.skip_evaluation:
        print("training complete; held-out test evaluation skipped by request")
        return

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
        rng_blocks = np.random.default_rng(args.data_seed + 2027)
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
        rng_pt = np.random.default_rng(args.data_seed + 2026)
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
        rng = np.random.default_rng(args.data_seed + 1000003 * (rep + 1))
        slots = rng.integers(0, n_test_cycles, size=n_test_cycles)
        weights = np.bincount(slots, minlength=n_test_cycles).astype(float)
        if np.count_nonzero(weights) < max(files_per_unit):
            raise RuntimeError(
                "bootstrap replicate has too few distinct cycles for one unit")
        probability = weights / weights.sum()
        boot_defs = [
            (c, pool[rng.choice(n_test_cycles, size=files_per_unit[c],
                                replace=False, p=probability)])
            for c, pool in ((0, pool_a), (1, pool_b))
            for _ in range(args.eval_bootstrap_units)
        ]
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
            "best_val_auc": state["max_val_auc"],
            "best_val_auc_epoch": state["max_val_auc_epoch"],
            "best_val_loss": state["min_val_loss"],
            "best_val_loss_epoch": state["min_val_loss_epoch"],
            "selected_best_metric_value": state["best_metric_value"],
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
