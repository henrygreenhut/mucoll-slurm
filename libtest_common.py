"""Shared helpers for the norm1-vs-norm42 library-discrimination test.

A "store" is an HDF5 file produced by gen_libtest_make_store.py holding all
GEN particles of one (library, polarity), file by file:

    particles/px,py,pz,E,t,vx,vy,vz   float32, flat over all files
    particles/pdg                     int32
    offsets                           int64, len nfiles+1
    cycle_ids                         int64, len nfiles (sorted ascending)
    filenames                         str, len nfiles

A "unit" (pseudo-crossing) is a list of file positions into one store:
    unique class = n_files norm1 cycles
    reuse  class = n_files / clone_factor norm42 cycles
Both represent the same decay statistics; splits are by cycle so no mother
appears in more than one of train/val/test.

Sections in this file, roughly in the order data flows through them:
    data I/O       Store, common_positions, split_indices, sample/blocked_
                   unit_positions
    features       FEATURE_SETS, build_features, compute_norm_stats,
                   {save,load,validate}_norm_stats
    model           make_optimizer, build_pfn / build_pfn_energyflow[_scaled]
    training math   should_early_stop, auc_score, bootstrap_auc
"""

import json
import os
import re

import h5py
import numpy as np

# Every field Store will try to load, if present -- see Store.__init__ for
# why "if present" (not "must have").
RAW_KEYS = ["px", "py", "pz", "E", "t", "vx", "vy", "vz", "pdg", "charge"]


def assign_cycle_ids(paths):
    """Cycle id per file: the integer token in the basename that VARIES
    across the directory (constant tokens like version tags are skipped).

    Why "most distinct", not just "the last token": filenames can carry
    several integers (a format-version tag, a date, the actual cycle
    number, ...), and the cycle number is whichever one is different in
    every file -- picking the token position with the most distinct values
    across the batch finds it automatically instead of hand-parsing a
    naming convention per BIB library.

    Returns (ids, token_pos_from_end, n_distinct). Callers should require
    n_distinct == len(paths); anything less means the naming convention is
    ambiguous and needs a hand-picked rule.
    """
    tokens = [re.findall(r"\d+", os.path.basename(p)) for p in paths]
    n_tok = min(len(t) for t in tokens)
    if n_tok == 0:
        raise ValueError("filenames contain no integer tokens")
    best_pos, best_distinct = 1, 0
    for pos in range(1, n_tok + 1):
        distinct = len({t[-pos] for t in tokens})
        if distinct > best_distinct:
            best_pos, best_distinct = pos, distinct
    ids = [int(t[-best_pos]) for t in tokens]
    return ids, best_pos, best_distinct

# Feature tiers:
#   paper    = PFN-ID inputs (pT, angle, ID) per arXiv:1810.05165, adapted
#              for BIB: theta instead of rapidity (forward particles),
#              absolute angles via cos/sin phi (no jet axis to center on),
#              log pT (six-decade spectrum). Momentum direction/magnitude
#              + coarse particle type only.
#   expanded = paper + every other cheap truth-level field sitting in the
#              GEN stores: log energy, asinh-compressed time/vertex-z/
#              vertex-radius (the BIB discriminants of arXiv:2105.09116 /
#              2203.06773 -- a pure z-axis rotation preserves time and
#              vertex exactly, same as |p|/theta, so these carry more of
#              the same exact-duplicate reuse signature, not a different
#              kind of signal). Deliberately NOT trying to be "realistic"
#              (reconstructed quantities would be smeared, these are MC
#              truth) -- this tier's job is to establish how separable
#              norm1-vs-norm42 is with maximum GEN-level sensitivity; the
#              reco-level classifier is what answers whether any of this
#              actually survives detector resolution.
#
#   Deliberately excludes "charge": it's the one field not already in the
#   original stores (E/t/vx/vy/vz were there from the start), so it's the
#   sole reason a store rebuild (_v2, +15% size/RAM) would be needed --
#   and its marginal information is small (PDG one-hot on |pdg| already
#   fixes charge for photons/neutrons; charge only adds the e/mu particle-
#   vs-antiparticle sign on top, nothing like the exact-invariant vz/t
#   signal above). Not worth a ~4.5GB RAM/store-rebuild cost for that --
#   confirmed the hard way: oscar_n420_full (job 4228090) thrashed at its
#   memory ceiling using the charge-including _v2 stores where every
#   earlier, charge-free n420 run fit comfortably in the same --mem.
#   build_features still supports "charge" as a column (see `columns`
#   below) for anyone who wants to opt into it explicitly later.
PDG_ONEHOT = ["pdg_gamma", "pdg_n", "pdg_e", "pdg_mu", "pdg_other"]
FEATURE_SETS = {
    "paper": ["logpt", "theta", "cosphi", "sinphi"] + PDG_ONEHOT,
    "expanded": ["logpt", "theta", "cosphi", "sinphi", "loge",
                 "asinh_t", "asinh_vz", "asinh_vr"] + PDG_ONEHOT,
}
# Back-compat for callers that predate feature tiers (pfn_variable_reuse_train.py,
# bib_example_unit_feature_plots.py) and always want the "paper" set.
FEATURE_NAMES = FEATURE_SETS["paper"]


class Store:
    """In-RAM view of a particle store.

    Loads whatever of RAW_KEYS the file actually has -- older stores built
    before a field was added (e.g. "charge") simply won't have it, and
    that's fine as long as no requested feature set needs it. A feature
    set that DOES need a missing field fails loudly inside build_features
    (KeyError on raw[...]), not silently here.
    """

    def __init__(self, path):
        self.path = path
        with h5py.File(path, "r") as f:
            self.offsets = f["offsets"][:]
            self.cycle_ids = f["cycle_ids"][:]
            available = set(f["particles"].keys())
            self.raw = {k: f["particles"][k][:] for k in RAW_KEYS if k in available}
        self.n_files = len(self.cycle_ids)

    def file_arrays(self, positions):
        """Concatenated raw arrays for the given file positions."""
        segs = [(self.offsets[p], self.offsets[p + 1]) for p in positions]
        out = {}
        for key, arr in self.raw.items():
            out[key] = np.concatenate([arr[a:b] for a, b in segs])
        return out


def common_positions(store_a, store_b):
    """Positions (per store) of the cycles present in both stores, sorted."""
    common = np.intersect1d(store_a.cycle_ids, store_b.cycle_ids)
    pos_a = np.searchsorted(store_a.cycle_ids, common)
    pos_b = np.searchsorted(store_b.cycle_ids, common)
    return common, pos_a, pos_b


def split_indices(n_common, fracs=(0.60, 0.15, 0.25)):
    """Split [0, n_common) into train/val/test index arrays by cycle order."""
    n_train = int(round(n_common * fracs[0]))
    n_val = int(round(n_common * fracs[1]))
    idx = np.arange(n_common)
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


def feature_names(feature_set="paper"):
    return list(FEATURE_SETS[feature_set])


def build_features(raw, feature_set="paper"):
    """(N, F) float32 feature array from raw particle arrays.

    raw must contain every field feature_names(feature_set) needs -- for
    "expanded" that includes "charge", which older stores built before it
    was added won't have (Store loads whatever RAW_KEYS the file actually
    has); requesting "expanded" against such a store fails here with a
    plain KeyError on raw["charge"], not silently.
    """
    px, py = raw["px"], raw["py"]
    pt = np.hypot(px, py)
    phi = np.arctan2(py, px)

    apdg = np.abs(raw["pdg"])
    onehot = {name: np.zeros(len(px), dtype=np.float32) for name in PDG_ONEHOT}
    onehot["pdg_gamma"][apdg == 22] = 1.0
    onehot["pdg_n"][apdg == 2112] = 1.0
    onehot["pdg_e"][apdg == 11] = 1.0
    onehot["pdg_mu"][apdg == 13] = 1.0
    assigned = sum(onehot[n] for n in PDG_ONEHOT[:4])
    onehot["pdg_other"][assigned == 0] = 1.0

    columns = {
        "logpt": lambda: np.log10(np.maximum(pt, 1e-9)),
        "theta": lambda: np.arctan2(pt, raw["pz"]),
        "cosphi": lambda: np.cos(phi),
        "sinphi": lambda: np.sin(phi),
        "loge": lambda: np.log10(np.maximum(raw["E"], 1e-9)),
        "asinh_t": lambda: np.arcsinh(raw["t"]),
        "asinh_vz": lambda: np.arcsinh(raw["vz"]),
        "asinh_vr": lambda: np.arcsinh(np.hypot(raw["vx"], raw["vy"])),
        "charge": lambda: raw["charge"],
    }
    cols = []
    for name in feature_names(feature_set):
        if name in PDG_ONEHOT:
            cols.append(onehot[name])
        else:
            cols.append(columns[name]())
    return np.column_stack(cols).astype(np.float32)


def compute_norm_stats(feature_arrays):
    """Per-feature mean/std with streaming float64 accumulation.

    The GEN studies can use O(10^8) float32 particle rows to estimate these
    statistics.  Reducing a two-dimensional float32 array along axis zero
    eventually stops incrementing one-hot counts at 2**24, corrupting both
    the mean and variance.  Merge per-array float64 moments instead; this is
    also much less memory-hungry than concatenating all normalization units.
    """
    count = 0
    mean = None
    m2 = None
    n_features = None
    for array in feature_arrays:
        values = np.asarray(array)
        if values.ndim != 2:
            raise ValueError("normalization arrays must have shape (N, F)")
        if n_features is None:
            n_features = values.shape[1]
            mean = np.zeros(n_features, dtype=np.float64)
            m2 = np.zeros(n_features, dtype=np.float64)
        elif values.shape[1] != n_features:
            raise ValueError("normalization arrays have inconsistent feature counts")
        chunk_count = len(values)
        if chunk_count == 0:
            continue
        chunk_mean = np.mean(values, axis=0, dtype=np.float64)
        chunk_var = np.var(values, axis=0, dtype=np.float64)
        new_count = count + chunk_count
        delta = chunk_mean - mean
        mean += delta * (float(chunk_count) / float(new_count))
        m2 += (chunk_var * chunk_count
               + delta * delta * count * chunk_count / float(new_count))
        count = new_count
    if count == 0:
        raise ValueError("cannot compute normalization from zero particles")
    std = np.sqrt(np.maximum(m2 / float(count), 0.0))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def validate_norm_stats(mean, std, names=None):
    """Reject non-finite or internally inconsistent cached statistics."""
    mean = np.asarray(mean)
    std = np.asarray(std)
    if mean.ndim != 1 or std.shape != mean.shape:
        raise ValueError("normalization mean/std shapes do not match")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("normalization statistics contain non-finite values")
    if np.any(std <= 0):
        raise ValueError("normalization standard deviations must be positive")
    if names is not None:
        names = list(names)
        if len(names) != len(mean):
            raise ValueError("normalization feature names do not match mean/std")
        if all(name in names for name in PDG_ONEHOT):
            onehot_total = float(np.sum(
                [mean[names.index(name)] for name in PDG_ONEHOT],
                dtype=np.float64))
            if not np.isclose(onehot_total, 1.0, rtol=0.0, atol=1e-4):
                raise ValueError(
                    "invalid PDG one-hot normalization means: sum is {:.8g}, "
                    "expected 1; cached statistics may have float32 reduction "
                    "overflow".format(onehot_total))


def save_norm_stats(path, mean, std, names, latent_scale):
    validate_norm_stats(mean, std, names)
    with open(path, "w") as f:
        json.dump({"names": names, "mean": mean.tolist(), "std": std.tolist(),
                   "latent_scale": latent_scale}, f, indent=1)


def load_norm_stats(path):
    with open(path) as f:
        d = json.load(f)
    mean = np.asarray(d["mean"], np.float32)
    std = np.asarray(d["std"], np.float32)
    validate_norm_stats(mean, std, d.get("names"))
    return mean, std, float(d["latent_scale"])


def should_early_stop(state, patience, min_epochs, metric_epoch_key="best_epoch"):
    """Whether patience is exhausted after respecting an epoch floor."""
    if state["epoch"] < min_epochs:
        return False
    return state["epoch"] - 1 - state[metric_epoch_key] >= patience


def _broadcast(value, n):
    """A single float applied to all n layers, or an already-per-layer list."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * n


def make_optimizer(optimizers_module, schedules_module, lr, warmup_steps,
                   clipnorm, jit_compile=False):
    """Adam, optionally with a linear-warmup learning-rate schedule and/or
    gradient-norm clipping -- both off (warmup_steps=0, clipnorm=0) by
    default, reproducing the original fixed-lr/unclipped behavior exactly.

    Motivation (raw-sum instability, the subject of the whole reuse-
    pressure investigation): with --latent-scale none the pooled latent's
    magnitude scales with however many particles are in a unit, so early
    train_loss/gradients can be enormous (observed: 70,000+ at n420).
    Warmup keeps early Adam steps small while its moment estimates are
    still noisy (the mechanism behind why Adam needs warmup at all --
    arXiv:1908.03265, arXiv:1910.04209); clipping bounds the worst case if
    a single batch's gradient is still huge despite that. Complementary,
    not redundant.

    Uses the library's own PolynomialDecay for the warmup ramp rather than
    a hand-rolled schedule: with initial_learning_rate=0, end_learning_rate
    =lr, power=1 (linear) it computes lr * step/warmup_steps -- and
    PolynomialDecay clips step at decay_steps by default (cycle=False), so
    past warmup_steps it just holds flat at lr indefinitely. Exactly
    "linear warmup then constant" with no custom class.

    jit_compile is explicit because tf_keras 2.15 conditionally defaults
    Adam's optimizer-level JIT to True when a GPU is visible.  Leaving that
    implicit made --jit false disable model JIT while optimizer updates still
    used XLA.  The one flag now controls both compilation paths.

    schedules_module/optimizers_module let callers pass either
    tensorflow.keras's or tf_keras's namespace -- the two Keras
    implementations in play across build_pfn/build_pfn_energyflow*
    (see build_pfn_energyflow_scaled's docstring for why that split
    exists) aren't guaranteed cross-compatible, so the schedule is built
    against whichever namespace will actually consume it.
    """
    learning_rate = lr
    if warmup_steps and warmup_steps > 0:
        learning_rate = schedules_module.PolynomialDecay(
            initial_learning_rate=0.0, decay_steps=warmup_steps,
            end_learning_rate=lr, power=1.0)
    kwargs = {"jit_compile": bool(jit_compile)}
    if clipnorm and clipnorm > 0:
        kwargs["clipnorm"] = clipnorm
    return optimizers_module.Adam(learning_rate=learning_rate, **kwargs)


def build_pfn_energyflow(input_dim, phi_sizes=(200, 200, 256),
                         f_sizes=(200, 200, 200), jit_compile=False,
                         lr=0.001, warmup_steps=0, clipnorm=0.0,
                         latent_dropout=0.0, f_dropouts=0.0,
                         phi_l2=0.0, f_l2=0.0):
    """The textbook PFN straight from the energyflow package (raw sum).

    Identical computation to build_pfn with latent_scale=1 (verified by
    pfn_arch_equivalence_check.py) when warmup_steps/clipnorm/dropout/l2
    are all at their off defaults; use this when package provenance is
    preferred and the raw-sum optimization dynamics are acceptable.

    latent_dropout/f_dropouts/phi_l2/f_l2 map directly onto energyflow's
    own latent_dropout/F_dropouts/Phi_l2_regs/F_l2_regs constructor
    hyperparameters (confirmed from source: both only ever apply to the
    POOLED per-event vector and the F network, never to the per-particle
    Phi network pre-pooling -- Phi only supports L2, not dropout -- so
    there's no set-structure/shared-mask subtlety to worry about here).
    """
    try:
        from energyflow.archs.efn import PFN
    except ImportError:
        try:
            from energyflow.archs import PFN
        except ImportError:
            raise SystemExit(
                "energyflow is not installed in this environment; "
                "`pip install --user energyflow` or use --arch local")
    import tf_keras
    opt = make_optimizer(tf_keras.optimizers, tf_keras.optimizers.schedules,
                         lr, warmup_steps, clipnorm, jit_compile)
    model = PFN(input_dim=input_dim, Phi_sizes=phi_sizes, F_sizes=f_sizes,
               optimizer=opt, latent_dropout=latent_dropout,
               F_dropouts=f_dropouts, Phi_l2_regs=phi_l2, F_l2_regs=f_l2).model
    if jit_compile:
        # PFN() already compiled this model with its own optimizer/loss;
        # recompile with the SAME optimizer/loss, only adding XLA JIT, so
        # training behavior is unchanged apart from the compilation path.
        # (model.metrics isn't reused here -- it includes internal trackers
        # like the loss metric itself, unsafe to pass back into metrics=.)
        model.compile(optimizer=model.optimizer, loss=model.loss,
                      metrics=["acc"], jit_compile=True)
    return model


def build_pfn_energyflow_scaled(input_dim, latent_scale,
                                phi_sizes=(200, 200, 256),
                                f_sizes=(200, 200, 200), jit_compile=False,
                                lr=0.001, warmup_steps=0, clipnorm=0.0,
                                latent_dropout=0.0, f_dropouts=0.0,
                                phi_l2=0.0, f_l2=0.0):
    """The scaled-sum PFN using energyflow.archs.EFN's actual aggregation
    graph, not a local reimplementation.

    EFN computes F(sum_i z_i * Phi(p_i)) for an arbitrary per-particle
    weight z_i (its intended use is z_i = an IRC-safe energy fraction; we
    are not using it that way, so this is officially a "weighted PFN built
    on EFN's architecture", not a physical EFN). Setting z_i = latent_scale
    for every real particle and 0 for padding gives
    latent_scale * sum_i Phi(p_i) -- exactly build_pfn's scaled sum.
    Verified bitwise-identical (0.0 max diff, weight-transplant, variable-
    length padded batches) to build_pfn(latent_scale=...) by
    pfn_arch_equivalence_check.py, at the dropout/l2/warmup/clipnorm off
    defaults.

    Requires tf_keras explicitly (not tensorflow.keras/Keras 3): EFN's
    .model is a tf_keras Functional model, and wrapping it inside a Keras-3
    functional graph fails with a KerasTensor-incompatibility error. The
    returned model is plain tf_keras throughout, safe to use with
    train_on_batch/predict_on_batch/get_weights/set_weights as usual.

    latent_dropout/f_dropouts/phi_l2/f_l2: passed to the INNER EFN(...)
    constructor (they shape its internal graph); EFN's own default
    optimizer/compile is irrelevant and discarded, since the OUTER wrapper
    model built here gets its own fresh compile() with our optimizer.
    """
    import tensorflow as tf
    import tf_keras

    try:
        from energyflow.archs import EFN
    except ImportError:
        raise SystemExit(
            "energyflow is not installed in this environment; "
            "`pip install --user energyflow` or use --arch local")

    efn_model = EFN(input_dim=input_dim, Phi_sizes=phi_sizes, F_sizes=f_sizes,
                    latent_dropout=latent_dropout, F_dropouts=f_dropouts,
                    Phi_l2_regs=phi_l2, F_l2_regs=f_l2).model
    inp = tf_keras.layers.Input(shape=(None, input_dim), name="particles")
    z = tf_keras.layers.Lambda(
        lambda x: tf.cast(tf.reduce_any(tf.not_equal(x, 0.0), axis=-1),
                          tf.float32) * latent_scale,
        name="scaled_mask")(inp)
    out = efn_model([z, inp])
    model = tf_keras.Model(inp, out, name="efn_scaled_wrapped")
    opt = make_optimizer(tf_keras.optimizers, tf_keras.optimizers.schedules,
                         lr, warmup_steps, clipnorm, jit_compile)
    model.compile(optimizer=opt, loss="categorical_crossentropy",
                  metrics=["acc"], jit_compile=jit_compile)
    return model


def build_pfn(input_dim, latent_scale, phi_sizes=(200, 200, 256),
              f_sizes=(200, 200, 200), lr=0.001, n_classes=2,
              jit_compile=False, warmup_steps=0, clipnorm=0.0,
              latent_dropout=0.0, f_dropouts=0.0, phi_l2=0.0, f_l2=0.0):
    """PFN (per-particle Phi MLP -> masked sum -> F MLP) in plain Keras.

    Zero-padded particles (all features exactly 0) are masked out. The
    latent sum is multiplied by the constant latent_scale (typically
    1/median particles-per-unit) so the F network sees O(1) inputs at any
    unit size. A constant scale is class-blind and linear, so relative
    multiplicity information is fully preserved.

    latent_dropout/f_dropouts/phi_l2/f_l2 mirror
    build_pfn_energyflow{,_scaled}'s equivalent hyperparameters: dropout
    only ever applies post-pooling (on the summed latent vector and in F),
    never inside Phi pre-pooling, matching energyflow's own placement
    (confirmed from source) -- keeps the two --arch choices comparable
    under the same flags.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, optimizers, regularizers

    def l2_kwargs(strength):
        """kernel_/bias_regularizer kwargs for Dense(...), or {} if off."""
        if strength <= 0:
            return {}
        reg = regularizers.l2(strength)
        return {"kernel_regularizer": reg, "bias_regularizer": reg}

    inp = layers.Input(shape=(None, input_dim), name="particles")
    mask = layers.Lambda(
        lambda x: tf.cast(tf.reduce_any(tf.not_equal(x, 0.0), axis=-1), tf.float32),
        name="mask")(inp)
    h = inp
    for i, width in enumerate(phi_sizes):
        h = layers.Dense(width, activation="relu", name=f"phi_{i}",
                         **l2_kwargs(phi_l2))(h)
    summed = layers.Lambda(
        lambda t: tf.reduce_sum(t[0] * t[1][..., None], axis=1) * latent_scale,
        name="scaled_sum")([h, mask])
    g = summed
    if latent_dropout > 0:
        g = layers.Dropout(latent_dropout, name="latent_dropout")(g)
    f_dropout_list = _broadcast(f_dropouts, len(f_sizes))
    for i, width in enumerate(f_sizes):
        g = layers.Dense(width, activation="relu", name=f"f_{i}",
                         **l2_kwargs(f_l2))(g)
        if f_dropout_list[i] > 0:
            g = layers.Dropout(f_dropout_list[i], name=f"f_{i}_dropout")(g)
    out = layers.Dense(n_classes, activation="softmax", name="output")(g)
    model = Model(inp, out)
    opt = make_optimizer(optimizers, tf.keras.optimizers.schedules,
                         lr, warmup_steps, clipnorm, jit_compile)
    model.compile(optimizer=opt, loss="categorical_crossentropy",
                  metrics=["acc"], jit_compile=jit_compile)
    return model


def sample_unit_positions(rng, split_positions, n_files):
    """Random distinct file positions for one unit."""
    return rng.choice(split_positions, size=n_files, replace=False)


def blocked_unit_positions(split_positions, n_files):
    """Disjoint consecutive blocks of n_files positions (for test eval)."""
    n_blocks = len(split_positions) // n_files
    return [split_positions[i * n_files:(i + 1) * n_files] for i in range(n_blocks)]


def auc_score(y_true, scores):
    """ROC AUC via the Mann-Whitney U / Wilcoxon rank-sum statistic (ties
    handled by average ranks), not the ROC-curve integral -- no sklearn.

    Mann-Whitney U and the ROC AUC are the same number by a standard
    theorem: AUC = P(a random positive score > a random negative score),
    which is exactly what the (properly tie-adjusted) rank-sum computes.
    Deep-dived and verified correct earlier in this project against
    brute-force pairwise comparison and sklearn cross-validation -- the
    surprisingly-high-AUC-despite-bad-loss result that motivated that
    check was real (ranking and calibration are different things, see
    --select-metric in pfn_libtest_train.py), not a bug here.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_true).astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc(scores_class0, scores_class1, n_boot=1000, seed=7):
    """AUC uncertainty by resampling disjoint units within each class.

    Treats every unit as an independent draw -- fine for the disjoint
    blocked cross-check (pfn_libtest_train.py's secondary evaluation),
    where that's actually true by construction. NOT used for the primary
    evaluation, where units can share source cycles across "different"
    draws (overlapping-by-design, for statistical power) -- that one needs
    the more careful cycle-level paired bootstrap in pfn_libtest_train.py,
    which resamples the true independent objects (cycles) rather than the
    derived, non-independent units built from them.
    """
    rng = np.random.default_rng(seed)
    s0 = np.asarray(scores_class0)
    s1 = np.asarray(scores_class1)
    vals = []
    for _ in range(n_boot):
        b0 = rng.choice(s0, size=len(s0), replace=True)
        b1 = rng.choice(s1, size=len(s1), replace=True)
        y = np.concatenate([np.zeros(len(b0)), np.ones(len(b1))])
        vals.append(auc_score(y, np.concatenate([b0, b1])))
    return float(np.mean(vals)), float(np.std(vals))
