#!/bin/bash
#SBATCH -J phi_half_long
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 07:00:00
#SBATCH --output=phi_half_long_%x_%j.out
#SBATCH --error=phi_half_long_%x_%j.err

# Single-GPU shared-QOS job for ONE config (raw_large or scaled_large),
# matching NERSC's own documented example exactly (-n 1 -c 32
# --gpus-per-task=1: https://docs.nersc.gov/jobs/policy/) rather than
# bundling both configs into one 2-task job. Bundling via
# --ntasks-per-node=2 hit a persistent "requires 32.0 cores per GPU...
# for 1 GPUs" mismatch from the account's gpu_shared_ss11 logical queue,
# even after matching the 32-cores/GPU ratio exactly -- the validator
# doesn't seem to distribute a multi-task GPU request correctly. A
# single task/single GPU job sidesteps that entirely.
#
# Submit twice, once per config -- shared QOS bills only for resources
# actually used, so two separate 1-GPU jobs cost the same as one bundled
# 2-GPU job, just as two job IDs instead of one:
#   sbatch submit_phi_half_long.slurm raw_large
#   sbatch submit_phi_half_long.slurm scaled_large
#
# Resumable: SAME labels as the original debug scan (job 56272947), so
# each resumes from its existing epoch-3 checkpoint. --max-minutes
# leaves a buffer under the 7-hour wall time for store loading + TF/CUDA
# startup + the final checkpoint write; resubmit the same command to
# continue past this window if training hasn't early-stopped or hit the
# 200-epoch cap yet.
#
# Lightweight logging: one line per epoch (~240s/epoch observed in the
# debug scan). Check progress live with:
#   tail -f phi_half_long_<jobid>.out
# or watch history.csv update every epoch:
#   watch -n 30 tail -3 pfn_results/halfphi_raw_large/history.csv

set -e
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

CONFIG=$1
if [ "$CONFIG" != "raw_large" ] && [ "$CONFIG" != "scaled_large" ]; then
    echo "usage: sbatch submit_phi_half_long.slurm {raw_large|scaled_large}"
    exit 1
fi

if [ "$CONFIG" == "raw_large" ]; then
    LABEL=halfphi_raw_large
    LATENT=none
else
    LABEL=halfphi_scaled_large
    LATENT=auto
fi

export N_FILES=${N_FILES:-420}
export UNITS_PER_EPOCH=${UNITS_PER_EPOCH:-500}
export MAX_MINUTES=${MAX_MINUTES:-400}

srun --cpu-bind=cores python -u pfn_libtest_train.py \
    --label "$LABEL" \
    --n-files "$N_FILES" \
    --units-per-epoch "$UNITS_PER_EPOCH" \
    --batch-size 4 \
    --max-minutes "$MAX_MINUTES" \
    --latent-scale "$LATENT" \
    --phi-sizes 100,100,128 \
    --f-sizes 200,200,200 \
    --arch energyflow

echo ""
echo "=== history tail ($LABEL) ==="
hist="pfn_results/${LABEL}/history.csv"
if [ -f "$hist" ]; then
    tail -5 "$hist"
else
    echo "NO history.csv yet"
fi
