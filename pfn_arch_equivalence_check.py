#!/usr/bin/env python3
"""Certify the local PFN builds against the official energyflow package.

Two checks, each building both a local and an energyflow-package model with
identical layer sizes, copying weights from the package model into the
local build, and comparing outputs on random variable-length, zero-padded
batches. Agreement at float32 precision certifies the local implementation
IS the package computation, not an independent reimplementation:

  1. Raw sum: local build_pfn(latent_scale=1) vs energyflow.archs.PFN.
  2. Scaled sum: local build_pfn(latent_scale=C) vs energyflow.archs.EFN
     with per-particle weight z_i = C (real particles) / 0 (padding), which
     reduces EFN's F(sum_i z_i*Phi(p_i)) to exactly C*sum_i Phi(p_i). This
     is not a physical EFN (z_i isn't an IRC-safe energy fraction) -- it's
     EFN's actual weighted-aggregation graph repurposed for our fixed scale,
     giving the scaled variant the same official-package provenance as the
     raw-sum one.

    module load tensorflow          # + pip install --user energyflow tf_keras
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
    parser.add_argument("--latent-scale", type=float, default=1.0 / 50000.0)
    parser.add_argument("--seed", type=int, default=3)
    return parser.parse_args()


def compare(name, ref_model, local_model, input_dim, batches, batch_size,
           max_particles, seed):
    rng = np.random.default_rng(seed)
    if len(ref_model.get_weights()) != len(local_model.get_weights()):
        print(f"{name}: FAIL -- weight-list lengths differ, architectures diverge")
        return False
    local_model.set_weights(ref_model.get_weights())

    worst = 0.0
    for b in range(batches):
        n = int(rng.integers(50, max_particles))
        x = rng.standard_normal((batch_size, n, input_dim)).astype(np.float32)
        for i in range(batch_size):
            keep = int(rng.integers(10, n))
            x[i, keep:] = 0.0
        out_ref = np.asarray(ref_model.predict_on_batch(x))
        out_local = np.asarray(local_model.predict_on_batch(x))
        diff = float(np.max(np.abs(out_ref - out_local)))
        worst = max(worst, diff)
        print(f"  batch {b}: N={n:5d} | max |softmax diff| = {diff:.3e}")

    print(f"  worst-case disagreement: {worst:.3e}")
    passed = worst < 1e-5
    print(f"{name}: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    args = parse_args()
    results = {}

    print("=== 1. raw sum: build_pfn(latent_scale=1) vs energyflow.archs.PFN ===")
    ef_model = lc.build_pfn_energyflow(args.input_dim)
    local = lc.build_pfn(args.input_dim, latent_scale=1.0)
    results["raw"] = compare("raw sum", ef_model, local, args.input_dim,
                             args.batches, args.batch_size,
                             args.max_particles, args.seed)

    print(f"\n=== 2. scaled sum (latent_scale={args.latent_scale:.3e}): "
          "build_pfn vs energyflow.archs.EFN (z_i weighted) ===")
    efn_wrapped = lc.build_pfn_energyflow_scaled(args.input_dim, args.latent_scale)
    local_scaled = lc.build_pfn(args.input_dim, latent_scale=args.latent_scale)
    results["scaled"] = compare("scaled sum", efn_wrapped, local_scaled,
                                args.input_dim, args.batches, args.batch_size,
                                args.max_particles, args.seed + 1)

    print()
    if all(results.values()):
        print("PASS: both local builds reproduce the official energyflow "
              "package at float32 precision.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"FAIL: {failed} disagree beyond tolerance -- do not treat "
              "as package-equivalent until resolved.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
