#!/usr/bin/env python3
"""Initialization probe for the PFN latent-sum scale, on real units.

Builds the identical PFN twice (same weights, same seed): once with the
latent sum multiplied by 1/median-unit-multiplicity, once with the raw sum
(scale = 1, i.e. the textbook/energyflow PFN). Feeds both the same batch of
real units from the stores and prints, for each:

    - magnitude of the latent (post-sum) components
    - the softmax outputs and how many are saturated
    - the initial loss vs ln(2) = 0.693 (an untrained 2-class net should
      start at random guessing)
    - the gradient norm of the first Phi layer

No training happens; this is a ~2-minute, deterministic demonstration of
why the scale is needed at BIB multiplicities (~65k particles/unit) even
though jet-scale PFNs (~100 particles) never needed it. The scale constant
is absorbable into the first F-layer weights (W -> sW is the identical
function), so the two models span the same function space -- the probe
shows the raw-sum one merely *starts* saturated.

Run on Perlmutter after the stores exist (GPU node or login node, ~min):

    module load tensorflow
    python pfn_latent_scale_check.py
"""

import argparse
import os

import numpy as np

import libtest_common as lc


def parse_args():
    scratch = os.environ.get("PSCRATCH", ".")
    store_dir = os.path.join(scratch, "mucoll/libtest/stores")
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm1-store", default=os.path.join(store_dir, "gen_norm1_MUPLUS.h5"))
    parser.add_argument("--norm42-store", default=os.path.join(store_dir, "gen_norm42_MUPLUS.h5"))
    parser.add_argument("--n-files", type=int, default=42)
    parser.add_argument("--clone-factor", type=int, default=42)
    parser.add_argument("--units", type=int, default=4, help="units per class in the probe batch")
    parser.add_argument("--seed", type=int, default=3)
    return parser.parse_args()


def build_batch(args):
    store1 = lc.Store(args.norm1_store)
    store42 = lc.Store(args.norm42_store)
    common, pos1, pos42 = lc.common_positions(store1, store42)
    splits = lc.split_indices(len(common))
    rng = np.random.default_rng(args.seed + 2)

    feats = []
    for store, positions, n_files in [
        (store1, pos1[splits["train"]], args.n_files),
        (store42, pos42[splits["train"]], args.n_files // args.clone_factor),
    ]:
        for _ in range(args.units):
            pos = lc.sample_unit_positions(rng, positions, n_files)
            feats.append(lc.build_features(store.file_arrays(pos)))
    mean, std = lc.compute_norm_stats(feats)
    feats = [(f - mean) / std for f in feats]

    max_n = max(len(f) for f in feats)
    x = np.zeros((len(feats), max_n, feats[0].shape[1]), dtype=np.float32)
    for i, f in enumerate(feats):
        x[i, : len(f)] = f
    y = np.zeros((len(feats), 2), dtype=np.float32)
    y[: args.units, 0] = 1.0
    y[args.units:, 1] = 1.0
    median_n = float(np.median([len(f) for f in feats]))
    return x, y, median_n


def probe(name, scale, x, y, seed):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = lc.build_pfn(x.shape[-1], scale)

    latent = tf.keras.Model(model.input, model.get_layer("scaled_sum").output)(x).numpy()
    probs = model(x).numpy()
    with tf.GradientTape() as tape:
        p = model(tf.constant(x), training=True)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.constant(y), p))
    grads = tape.gradient(loss, model.trainable_variables)
    phi0 = next(g for g, v in zip(grads, model.trainable_variables)
                if "phi_0" in getattr(v, "path", getattr(v, "name", "")))

    saturated = np.mean((probs[:, 1] > 0.999) | (probs[:, 1] < 0.001))
    print(f"\n=== {name} (scale = {scale:.3e}) ===")
    print(f"  |latent component| median / max : "
          f"{np.median(np.abs(latent)):.3f} / {np.abs(latent).max():.3f}")
    print(f"  softmax outputs (class-1 score) : "
          + " ".join(f"{v:.3f}" for v in probs[:, 1]))
    print(f"  outputs saturated (>0.999/<0.001): {100 * saturated:.0f}%")
    print(f"  initial loss                    : {float(loss):.4f}"
          f"   (random guessing = {np.log(2):.4f})")
    print(f"  grad norm, first Phi layer      : {float(tf.norm(phi0)):.3e}")


def main():
    args = parse_args()
    x, y, median_n = build_batch(args)
    print(f"probe batch: {x.shape[0]} real units (n={args.n_files}),"
          f" padded to {x.shape[1]} particles, median unit N = {median_n:.0f}")
    print("identical weights in both models (same seed); only the constant"
          " multiplying the latent sum differs")
    probe("WITH latent scale  (1/median N)", 1.0 / median_n, x, y, args.seed)
    probe("WITHOUT latent scale (raw sum) ", 1.0, x, y, args.seed)
    print("\nExpected pattern: the raw-sum model starts with saturated outputs,"
          "\nloss far above ln 2, and gradient norms orders of magnitude larger"
          "\n-- i.e. it begins training from a numerically pathological state.")


if __name__ == "__main__":
    main()
