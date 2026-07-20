#!/usr/bin/env python3
"""Measure how many particles per event the PFN can handle on this machine.

Builds the actual local PFN (libtest_common.build_pfn; Phi=(200,200,256),
F=(200,200,200)) and times training steps while sweeping the
particles-per-event N and batch size, until it runs out of memory.

Restored from git history (commit d447536; deleted in bd689bf) and updated
for the current pipeline: input_dim default corrected to 9 (the current
FEATURE_NAMES length), n-list/batch-sizes defaulted to the actual n-sweep
particle counts (n=42/126/210/420 files) and the batch sizes bracketing the
observed n=420 --batch-size 1 configuration.

Batch sizes within one N are tried ASCENDING (smallest first), stopping at
the first SKIP or OOM -- not descending. A hard GPU abort (SIGABRT, seen for
a sufficiently oversized allocation attempt) kills the process outright and
is not a catchable Python exception; descending from the largest size means
a crash there would lose every smaller, likely-successful size that was
never reached. Ascending guarantees whatever succeeded before a crash is
already on record, illuminating the true "starts working here" boundary
rather than just "didn't crash at the top of the range". stdout is written
unbuffered for the same reason: a crash must not lose already-printed,
not-yet-flushed results.

Two run modes:

  1. Default: fixed-shape synthetic sweep. Phi is applied densely to every
     particle SLOT (real or zero-padded) before the mask is used, only at
     the final sum -- so memory and per-step FLOPs are shape-determined,
     not value-determined, and synthetic random data reused across many
     timed steps should match real training closely for both. Cheap and
     data-independent: the fast way to find the OOM ceiling at any N,
     including hypothetical ones no store has been built for yet.

  2. --real-store PATH: builds a FRESH real batch from an actual GEN store
     (libtest_common.Store + build_features) for every single timed step --
     new random files, so the particle count varies batch to batch exactly
     as it does in real training (no synthetic approximation of that
     variability). This is the direct, no-proxy answer to "how fast/costly
     is actual training": real feature values, real shape variability, and
     whatever retracing cost real variable-shape batches actually incur,
     all in one number, comparable against mode 1's fixed-shape figure at
     the same N to see the combined real-world gap.

Run on a Perlmutter GPU node (needs `module load tensorflow`):

    python pfn_capacity_benchmark.py
    python pfn_capacity_benchmark.py \
        --real-store $PSCRATCH/mucoll/libtest/stores/gen_norm1_MUPLUS.h5

Reports ms/step, particles/s, peak GPU memory, projected seconds/epoch, and
the host-RAM cost of a dense (all-in-RAM padded array) dataset at each N.
"""

import argparse
import functools
import time

import numpy as np

print = functools.partial(print, flush=True)  # a crash must not lose buffered output


PHI_SIZES = (200, 200, 256)
F_SIZES = (200, 200, 200)
# n=42/126/210/420 files, at the measured ~2990 particles/norm1 file
# (Layer-0 check: 125,575 particles / 42 files).
N_SWEEP_DEFAULT = "125000,375000,625000,1255800"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, default=9,
                        help="features per particle (current FEATURE_NAMES: 9)")
    parser.add_argument("--n-list", default=N_SWEEP_DEFAULT,
                        help="comma-separated particles-per-event values")
    parser.add_argument("--batch-sizes", default="1,2,4,8",
                        help="comma-separated batch sizes; always tried "
                             "ascending regardless of the order given here, "
                             "stopping at the first SKIP or OOM")
    parser.add_argument("--steps", type=int, default=10,
                        help="timed steps per configuration (after 3 warmup)")
    parser.add_argument("--events-per-epoch", type=int, default=40000,
                        help="events/epoch used for the projected epoch time")
    parser.add_argument("--real-store", default=None,
                        help="path to a GEN store (gen_libtest_make_store.py "
                             "output); if given, benchmarks a freshly-sampled "
                             "real batch (new random files, naturally varying "
                             "N) at every timed step instead of synthetic data")
    parser.add_argument("--real-seed", type=int, default=1)
    parser.add_argument("--max-gb", type=float, default=0.0,
                        help="skip (without attempting) any batch*N combo "
                             "whose estimated training memory exceeds this "
                             "(0 = off, default). A sufficiently oversized "
                             "allocation attempt can trigger a hard CUDA/"
                             "cuDNN abort (SIGABRT) that no Python except "
                             "clause can catch and that kills the process --"
                             " observed once for batch=8 at N~1.26M on a "
                             "40GB A100. Off by default because ascending "
                             "order + flush=True + the launch script's "
                             "--kill-on-bad-exit=0 already contain that: a "
                             "crash only loses the untested sizes above it, "
                             "every smaller result (incl. its real-data "
                             "check) is on disk first, and sibling ranks are "
                             "unaffected. Set a cap only if even that residual "
                             "cost (a few wasted seconds, one lost rank) is "
                             "unacceptable for a given sweep.")
    return parser.parse_args()


def estimate_training_gb(n_particles, batch_size, bytes_per_particle=4300):
    """Per-particle training-memory rule of thumb, used only to SKIP combos
    before attempting them -- not a substitute for the measured peakGPU MB
    column, which is exact for whatever was actually attempted.

    4300 B/particle calibrated from real measured peakGPU across n=42/126/
    210/420 (job 56213836, A100-SXM4-40GB): every point converged on ~3800
    B/particle to within ~1%; 4300 keeps a ~13% margin. The original 8192
    figure was a pre-measurement guess and was ~2.1x too conservative --
    it caused n210 batch=8 and n420 batch=4 to be skipped (never attempted,
    not crashed) when they likely would have fit."""
    return batch_size * n_particles * bytes_per_particle / 1024**3


def get_model(input_dim):
    from libtest_common import build_pfn
    return build_pfn(input_dim, latent_scale=1e-3,
                     phi_sizes=PHI_SIZES, f_sizes=F_SIZES)


def gpu_peak_mb():
    import tensorflow as tf
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return info["peak"] / 1024**2
    except Exception:
        return None


def reset_gpu_stats():
    import tensorflow as tf
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


def dense_dataset_gb(n_events, n_particles, input_dim):
    return n_events * n_particles * input_dim * 4 / 1024**3


def make_synthetic_batch(n_particles, batch_size, input_dim):
    x = np.random.randn(batch_size, n_particles, input_dim).astype(np.float32)
    y = np.zeros((batch_size, 2), dtype=np.float32)
    y[: batch_size // 2, 0] = 1.0
    y[batch_size // 2:, 1] = 1.0
    return x, y


def bench_fixed_shape(model, n_particles, batch_size, input_dim, steps):
    """Fixed N for every step; lets TF cache one compute graph. Returns ms/step."""
    x, y = make_synthetic_batch(n_particles, batch_size, input_dim)
    for _ in range(3):
        model.train_on_batch(x, y)
    start = time.perf_counter()
    for _ in range(steps):
        model.train_on_batch(x, y)
    return (time.perf_counter() - start) / steps * 1000.0


def make_real_batch(store, rng, n_target, batch_size, input_dim):
    """One batch of freshly-sampled real events (~n_target particles each)."""
    import libtest_common as lc

    n_files_per_event = max(1, round(n_target / 2990))
    n_files_per_event = min(n_files_per_event, store.n_files)
    all_positions = np.arange(store.n_files)
    feats = []
    for _ in range(batch_size):
        pos = rng.choice(all_positions, size=n_files_per_event, replace=False)
        raw = store.file_arrays(pos)
        feats.append(lc.build_features(raw))
    max_n = max(len(f) for f in feats)
    x = np.zeros((batch_size, max_n, input_dim), dtype=np.float32)
    for i, f in enumerate(feats):
        x[i, : len(f)] = f
    y = np.zeros((batch_size, 2), dtype=np.float32)
    y[: batch_size // 2, 0] = 1.0
    y[batch_size // 2:, 1] = 1.0
    return x, y, max_n


def bench_real(model, store, n_target, batch_size, input_dim, steps, seed):
    """A FRESH real batch every step -- particle count varies batch to batch
    exactly as in real training. Returns (ms/step, achieved_n per step)."""
    rng = np.random.default_rng(seed)
    x, y, _ = make_real_batch(store, rng, n_target, batch_size, input_dim)
    model.train_on_batch(x, y)  # one warmup call to init CUDA/cuDNN kernels

    achieved_ns = []
    start = time.perf_counter()
    for _ in range(steps):
        x, y, n = make_real_batch(store, rng, n_target, batch_size, input_dim)
        model.train_on_batch(x, y)
        achieved_ns.append(n)
    ms = (time.perf_counter() - start) / steps * 1000.0
    return ms, achieved_ns


def main():
    args = parse_args()
    n_list = [int(v) for v in args.n_list.split(",")]
    batch_sizes = [int(v) for v in args.batch_sizes.split(",")]

    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}; GPUs: {[g.name for g in gpus] or 'none (CPU)'}")
    print(f"PFN Phi={PHI_SIZES} F={F_SIZES}, input_dim={args.input_dim}")
    print(f"Projected epoch = {args.events_per_epoch} events\n")

    model = get_model(args.input_dim)
    store = None
    if args.real_store:
        from libtest_common import Store
        store = Store(args.real_store)
        print(f"real store: {store.n_files} files loaded\n")

    header = (f"{'N':>8} {'batch':>6} {'status':>7} {'ms/step':>9} "
              f"{'Mparticles/s':>13} {'peakGPU MB':>11} {'s/epoch':>9} {'denseRAM GB':>12}")
    print(header)
    print("-" * len(header))

    oom_errors = (tf.errors.ResourceExhaustedError, tf.errors.InternalError, MemoryError)
    for n_particles in n_list:
        largest_ok = None
        # Ascending: whatever succeeds is on record BEFORE we risk a size
        # that might hard-crash the whole process. batch*N grows monotonically
        # with batch (fixed N), so stopping at the first SKIP/OOM is exact --
        # nothing larger in this sorted list would do better. The real-data
        # check runs after EVERY success (not just the final one) so a later
        # crash never erases an already-earned measurement.
        for batch_size in sorted(set(batch_sizes)):
            est_gb = estimate_training_gb(n_particles, batch_size)
            if args.max_gb > 0 and est_gb > args.max_gb:
                print(f"{n_particles:>8} {batch_size:>6} {'SKIP':>7}"
                      f"  est. ~{est_gb:.0f} GB > --max-gb {args.max_gb:g} GB"
                      " -- not attempted (risk of an uncatchable GPU abort)")
                break
            reset_gpu_stats()
            try:
                ms = bench_fixed_shape(model, n_particles, batch_size,
                                       args.input_dim, args.steps)
            except oom_errors:
                print(f"{n_particles:>8} {batch_size:>6} {'OOM':>7}")
                break
            largest_ok = batch_size
            mpps = batch_size * n_particles / ms / 1000.0
            sec_epoch = args.events_per_epoch / batch_size * ms / 1000.0
            peak = gpu_peak_mb()
            peak_s = f"{peak:11.0f}" if peak is not None else f"{'-':>11}"
            dense = dense_dataset_gb(args.events_per_epoch, n_particles, args.input_dim)
            print(f"{n_particles:>8} {batch_size:>6} {'ok':>7} {ms:9.1f} "
                  f"{mpps:13.2f} {peak_s} {sec_epoch:9.0f} {dense:12.1f}")

            if store is not None:
                reset_gpu_stats()
                try:
                    ms_real, ns = bench_real(model, store, n_particles, batch_size,
                                             args.input_dim, args.steps, args.real_seed)
                except oom_errors:
                    print(f"  real-data check: OOM at batch={batch_size}"
                          " (synthetic fit; real shape variance pushed it over)")
                    continue
                peak = gpu_peak_mb()
                peak_s = f"{peak:.0f}" if peak is not None else "-"
                print(f"  real, freshly-sampled each step: N {min(ns)}-{max(ns)}"
                      f" (target {n_particles}) batch={batch_size} ->"
                      f" {ms_real:.1f} ms/step (fixed-shape synthetic was"
                      f" {ms:.1f} ms/step, {ms_real/ms - 1:+.1%}) peakGPU {peak_s} MB")
        if largest_ok is None:
            print(f"\nStopping: N={n_particles} does not fit at any batch size"
                  " (not even the smallest tested).")
            break

    print("\nNotes:")
    print(" - 'denseRAM GB' = host RAM if the dataset were one padded array as in")
    print("   pfn_train.py/fit_slots. Above ~0.5x machine RAM, switch to ragged")
    print("   HDF5 + per-batch padding generator regardless of GPU capacity.")
    print(" - Largest 'ok' N x batch is the per-event particle budget on this machine.")
    if store is not None:
        print(" - real-data row: every step samples fresh files, so N varies "
              "batch to batch as in actual training; a large gap vs the "
              "fixed-shape synthetic row reflects real value effects AND/OR "
              "per-step shape-change (retracing) cost, combined.")


if __name__ == "__main__":
    main()
