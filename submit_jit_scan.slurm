#!/bin/bash
#SBATCH -J jit_scan
#SBATCH -q debug
#SBATCH -C gpu&hbm80g
#SBATCH -A m5197_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -t 00:29:00
#SBATCH --output=jit_scan_%x_%j.out
#SBATCH --error=jit_scan_%x_%j.err

# Re-run of the arch_scan debug screen (same 4 configs, same cost: one
# exclusive debug node, billed per-node regardless of GPU utilization) with
# --jit added to every rank. 3 of 4 ranks (raw_large, raw_small,
# scaled_large) crashed on the TF/XLA int32 overflow bug in the original
# arch_scan run (job 56224614); jit_null_large is an unaffected control,
# included to confirm --jit doesn't change/break the one config that
# already worked. See jit_scan_task.sh for the four configs.
#
# Two things to check in the output, not just "did it crash":
#   1. Did the crash go away (all 4 ranks produce history.csv rows)?
#   2. Per-epoch "seconds" column -- N varies every batch, and XLA compiles
#      kernels keyed to shape, so watch for recompilation blowing up
#      per-epoch time even if the crash is gone.

set -e
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

export N_FILES=${N_FILES:-420}
export UNITS_PER_EPOCH=${UNITS_PER_EPOCH:-500}
export MAX_MINUTES=${MAX_MINUTES:-22}

# --kill-on-bad-exit=0: one rank crashing must not take the other three,
# still-informative ranks down with it.
srun --cpu-bind=cores --kill-on-bad-exit=0 bash jit_scan_task.sh || \
    echo "one or more ranks exited non-zero; check individual .out files" \
         "before trusting the whole set"

echo ""
echo "=== per-rank full history (epoch, train_loss, val_loss, val_auc, seconds) ==="
for label in jit_raw_large jit_raw_small jit_scaled_large jit_null_large; do
    hist="pfn_results/${label}/history.csv"
    if [ -f "$hist" ]; then
        echo "-- $label --"
        cat "$hist"
    else
        echo "-- $label -- NO history.csv (did not complete even one epoch)"
    fi
done
