#!/bin/bash
#SBATCH -J phi_half_scan
#SBATCH -q debug
#SBATCH -C gpu&hbm80g
#SBATCH -A m5197_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -t 00:29:00
#SBATCH --output=phi_half_scan_%x_%j.out
#SBATCH --error=phi_half_scan_%x_%j.err

# Same 4 configs, same batch sizes, same cost (one exclusive debug node) as
# the original arch_scan (job 56224614), with every Phi network halved and
# --jit dropped (proven a dead end). See phi_half_scan_task.sh for the
# four configs and the reasoning.

set -e
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

export N_FILES=${N_FILES:-420}
export UNITS_PER_EPOCH=${UNITS_PER_EPOCH:-500}
export MAX_MINUTES=${MAX_MINUTES:-22}

# --kill-on-bad-exit=0: one rank crashing must not take the other three,
# still-informative ranks down with it.
srun --cpu-bind=cores --kill-on-bad-exit=0 bash phi_half_scan_task.sh || \
    echo "one or more ranks exited non-zero; check individual .out files" \
         "before trusting the whole set"

echo ""
echo "=== per-rank full history (epoch, train_loss, val_loss, val_auc, seconds) ==="
for label in halfphi_raw_large halfphi_raw_small halfphi_scaled_large halfphi_null_large; do
    hist="pfn_results/${label}/history.csv"
    if [ -f "$hist" ]; then
        echo "-- $label --"
        cat "$hist"
    else
        echo "-- $label -- NO history.csv (did not complete even one epoch)"
    fi
done
