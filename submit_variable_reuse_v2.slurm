#!/bin/bash
#SBATCH -J var_reuse_v2
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 03:30:00
#SBATCH --output=var_reuse_v2_%x_%j.out
#SBATCH --error=var_reuse_v2_%x_%j.err

# Rerun of the k=1-vs-k=10 reuse study using RAW (unnormalized) sum --
# --latent-scale none, matching this session's raw_large config, not the
# original collapsed variable_n420_U_vs_R10_scaled run (which used the
# scaled sum). Original run(s) collapsed to a near-frozen constant output
# at batch_size=1 -- see pfn_variable_reuse_train.py history. Same fixes
# validated this session applied here: batch_size=4 instead of 1 (the
# suspected primary driver of the collapse -- gradient noise scales
# ~1/batch_size, and this exact architecture family broke out of the same
# kind of saturated state once off batch=1 in our own raw_large/
# scaled_large runs), plus norm_stat_units bumped 1->100 in the script
# itself (a separate, independent fragility -- latent_scale was being set
# from a single sampled unit's particle count). Halved Phi (100,100,128)
# is also now hardcoded in pfn_variable_reuse_train.py, matching the
# crash-free config from phi_half_scan (job 56272947).
#
# New labels (_v2 suffix) so this does NOT resume from the old collapsed
# run's checkpoint -- that would just reload the already-broken weights.
#
# Single-GPU job, matching the shared-QOS pattern validated for
# phi_half_long after the earlier gres/ratio errors (gpu_shared_ss11
# requires exactly 32 cores/GPU per task, and bundling multiple
# tasks/GPUs into one job broke that validation -- one task, one GPU,
# submit twice).
#
# 3.5h window, shorter than phi_half_long's 7h: this is a diagnostic
# check ("does moving off batch=1 actually prevent the collapse", not a
# full convergence run yet), and the shared queue already has 2 other
# jobs (phi_half_long raw_large/scaled_large) contending for slots before
# tomorrow's 6am maintenance reservation. Extend later if the trend looks
# healthy -- same resumable checkpoint/--max-minutes design as the rest
# of this session's runs.
#
# Submit twice, once per config:
#   sbatch submit_variable_reuse_v2.slurm main
#   sbatch submit_variable_reuse_v2.slurm null

set -e
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

CONFIG=$1
if [ "$CONFIG" != "main" ] && [ "$CONFIG" != "null" ]; then
    echo "usage: sbatch submit_variable_reuse_v2.slurm {main|null}"
    exit 1
fi

if [ "$CONFIG" == "main" ]; then
    LABEL=variable_n420_U_vs_R10_rawsum_v2
    EXTRA=""
else
    LABEL=variable_n420_U_vs_R10_rawsum_null_v2
    EXTRA="--null-test"
fi

MAX_MINUTES=${MAX_MINUTES:-180}

srun --cpu-bind=cores python -u pfn_variable_reuse_train.py \
    --label "$LABEL" \
    --reuse-k 1 10 \
    --mother-equivalents 29400 \
    --rotation-policy baseline-unrotated \
    --units-per-epoch 500 \
    --val-units 300 \
    --test-units 500 \
    --batch-size 4 \
    --epochs 200 \
    --patience 20 \
    --seed 1 \
    --latent-scale none \
    --max-minutes "$MAX_MINUTES" \
    $EXTRA

echo ""
echo "=== history tail ($LABEL) ==="
hist="variable_k_results/${LABEL}/history.csv"
if [ -f "$hist" ]; then
    tail -10 "$hist"
else
    echo "NO history.csv yet"
fi
