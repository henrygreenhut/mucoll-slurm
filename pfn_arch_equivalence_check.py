#!/usr/bin/env python3
"""Certify the local PFN build against energyflow.archs.PFN.

Builds both models with identical layer sizes, copies the weights from the
energyflow model into the local build (latent scale = 1), and compares
outputs on random variable-length, zero-padded batches. Agreement at
float32 precision certifies that the local implementation IS the package
architecture plus (optionally) one documented scale constant.

    module load tensorflow          # + pip install --user energyflow
    python pfn_arch_equivalence_check.py
"""

import argparse

import numpy as np

import libtest_common as lc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, default=9)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-particles", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ef_model = lc.build_pfn_energyflow(args.input_dim)
    local = lc.build_pfn(args.input_dim, latent_scale=1.0)
    if len(ef_model.get_weights()) != len(local.get_weights()):
        raise SystemExit("weight-list lengths differ -- architectures diverge")
    local.set_weights(ef_model.get_weights())

    worst = 0.0
    for b in range(args.batches):
        n = int(rng.integers(50, args.max_particles))
        x = rng.standard_normal((args.batch_size, n, args.input_dim)).astype(np.float32)
        # zero-pad a random tail per event to exercise the masking path
        for i in range(args.batch_size):
            keep = int(rng.integers(10, n))
            x[i, keep:] = 0.0
        out_ef = np.asarray(ef_model.predict_on_batch(x))
        out_local = np.asarray(local.predict_on_batch(x))
        diff = float(np.max(np.abs(out_ef - out_local)))
        worst = max(worst, diff)
        print(f"batch {b}: N={n:5d} | max |softmax diff| = {diff:.3e}")

    print(f"\nworst-case disagreement: {worst:.3e}")
    if worst < 1e-5:
        print("PASS: local build reproduces energyflow.archs.PFN at float32"
              " precision (identical weights, identical masking and sum).")
    else:
        print("FAIL: outputs differ beyond float32 tolerance -- do not treat"
              " the local build as package-equivalent until resolved.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
