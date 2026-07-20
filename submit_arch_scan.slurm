#!/bin/bash
#SBATCH -J arch_scan
#SBATCH -q debug
#SBATCH -C gpu&hbm80g
#SBATCH -A m5197_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -t 00:29:00
#SBATCH --output=arch_scan_%x_%j.out
#SBATCH --error=arch_scan_%x_%j.err

# N=420 architecture screen: 4 configs, one per GPU, one exclusive debug
# node (billed per-node regardless of GPU utilization, same cost as running
# one). Each rank checkpoints and exits at MAX_MINUTES (default 22, leaving
# buffer under the 29-min job limit for store loading + TF/CUDA startup).
# This is a coarse triage, not a final ranking: healthy vs. chaotic vs.
# collapsed within a handful of epochs, not converged AUCs. See
# arch_scan_task.sh for the four configs.
#
# Each rank writes its own arch_scan_<label>_<jobid>.out; after the job:
#   cat arch_scan_scan_*_<jobid>.out | less
#   for f in pfn_results/scan_*/history.csv; do echo "== $f"; cat "$f"; done

set -e
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

export N_FILES=${N_FILES:-420}
export UNITS_PER_EPOCH=${UNITS_PER_EPOCH:-500}
export MAX_MINUTES=${MAX_MINUTES:-22}

# --kill-on-bad-exit=0: one rank crashing (e.g. the known collapse-to-
# constant path, or an int32-overflow-style abort) must not take the other
# three, still-informative ranks down with it.
srun --cpu-bind=cores --kill-on-bad-exit=0 bash arch_scan_task.sh || \
    echo "one or more ranks exited non-zero; check individual .out files" \
         "before trusting the whole set"

echo ""
echo "=== per-rank summary (last history.csv line each) ==="
for label in scan_raw_large scan_raw_small scan_scaled_large scan_null_large; do
    hist="pfn_results/${label}/history.csv"
    if [ -f "$hist" ]; then
        echo "-- $label --"
        head -1 "$hist"
        tail -1 "$hist"
    else
        echo "-- $label -- NO history.csv (did not complete even one epoch)"
    fi
done
