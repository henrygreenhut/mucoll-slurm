#!/usr/bin/env python3
"""Measure how many particles per event the PFN can handle on this machine.

Builds the same energyflow PFN as pfn_train.py (Phi=(200,200,256),
F=(200,200,200)) and times training steps on synthetic data while sweeping
the particles-per-event N and batch size, until it runs out of memory.

Run on any machine (laptop CPU or Perlmutter GPU node):

    python pfn_capacity_benchmark.py
    python pfn_capacity_benchmark.py --n-list 20000,50000,100000,200000 \
        --batch-sizes 16,8,4 --events-per-epoch 40000

Reports ms/step, particles/s, peak GPU memory, projected seconds/epoch, and
the host-RAM cost of a dense pfn_train.py-style dataset at each N (the
current pipeline materializes the full padded array in RAM, which is the
real bottleneck long before the model is).
"""

import argparse
import time

import numpy as np


PHI_SIZES = (200, 200, 256)
F_SIZES = (200, 200, 200)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, default=8,
                        help="features per particle")
    parser.add_argument("--n-list", default="1000,2000,5000,10000,20000,50000,100000,200000",
                        help="comma-separated particles-per-event values")
    parser.add_argument("--batch-sizes", default="16,8,4",
                        help="comma-separated batch sizes (tried in order per N)")
    parser.add_argument("--steps", type=int, default=10,
                        help="timed steps per configuration (after 3 warmup)")
    parser.add_argument("--events-per-epoch", type=int, default=40000,
                        help="events/epoch used for the projected epoch time")
    return parser.parse_args()


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


def bench_one(model, n_particles, batch_size, input_dim, steps):
    """Return ms/step or raise on OOM."""
    x = np.random.randn(batch_size, n_particles, input_dim).astype(np.float32)
    y = np.zeros((batch_size, 2), dtype=np.float32)
    y[: batch_size // 2, 0] = 1.0
    y[batch_size // 2:, 1] = 1.0

    for _ in range(3):
        model.train_on_batch(x, y)
    start = time.perf_counter()
    for _ in range(steps):
        model.train_on_batch(x, y)
    return (time.perf_counter() - start) / steps * 1000.0


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

    header = (f"{'N':>8} {'batch':>6} {'status':>7} {'ms/step':>9} "
              f"{'Mparticles/s':>13} {'peakGPU MB':>11} {'s/epoch':>9} {'denseRAM GB':>12}")
    print(header)
    print("-" * len(header))

    oom_errors = (tf.errors.ResourceExhaustedError, tf.errors.InternalError, MemoryError)
    for n_particles in n_list:
        fit_any = False
        for batch_size in batch_sizes:
            reset_gpu_stats()
            try:
                ms = bench_one(model, n_particles, batch_size, args.input_dim, args.steps)
            except oom_errors:
                print(f"{n_particles:>8} {batch_size:>6} {'OOM':>7}")
                continue
            fit_any = True
            mpps = batch_size * n_particles / ms / 1000.0
            sec_epoch = args.events_per_epoch / batch_size * ms / 1000.0
            peak = gpu_peak_mb()
            peak_s = f"{peak:11.0f}" if peak is not None else f"{'-':>11}"
            dense = dense_dataset_gb(args.events_per_epoch, n_particles, args.input_dim)
            print(f"{n_particles:>8} {batch_size:>6} {'ok':>7} {ms:9.1f} "
                  f"{mpps:13.2f} {peak_s} {sec_epoch:9.0f} {dense:12.1f}")
            break  # largest fitting batch size is enough per N
        if not fit_any:
            print(f"\nStopping: N={n_particles} does not fit at any batch size.")
            break

    print("\nNotes:")
    print(" - 'denseRAM GB' = host RAM if the dataset were one padded array as in")
    print("   pfn_train.py/fit_slots. Above ~0.5x machine RAM, switch to ragged")
    print("   HDF5 + per-batch padding generator regardless of GPU capacity.")
    print(" - Largest 'ok' N x batch is the per-event particle budget on this machine.")


if __name__ == "__main__":
    main()
