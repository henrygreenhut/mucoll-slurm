#!/bin/bash
#SBATCH -J variable_k_bundle
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -t 00:30:00
#SBATCH --output=variable_k_bundle_%j.out
#SBATCH --error=variable_k_bundle_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

COMMON_ARGS="--reuse-k 1 2 3 6 10 14 21 42 --mother-equivalents 29400 --units-per-epoch 20 --val-units 10 --test-units 30 --batch-size 1 --patience 10 --max-minutes 25"

pids=()
srun --exclusive --exact -N 1 -n 1 -c 32 --gpus-per-task=1 --gpu-bind=single:1 \
    python pfn_variable_reuse_train.py \
        --label variable_k_scaled_allrandom \
        --rotation-policy all-random --latent-scale auto $COMMON_ARGS &
pids+=("$!")

srun --exclusive --exact -N 1 -n 1 -c 32 --gpus-per-task=1 --gpu-bind=single:1 \
    python pfn_variable_reuse_train.py \
        --label variable_k_raw_allrandom \
        --rotation-policy all-random --latent-scale none $COMMON_ARGS &
pids+=("$!")

srun --exclusive --exact -N 1 -n 1 -c 32 --gpus-per-task=1 --gpu-bind=single:1 \
    python pfn_variable_reuse_train.py \
        --label variable_k_scaled_baseline_unrotated \
        --rotation-policy baseline-unrotated --latent-scale auto $COMMON_ARGS &
pids+=("$!")

srun --exclusive --exact -N 1 -n 1 -c 32 --gpus-per-task=1 --gpu-bind=single:1 \
    python pfn_variable_reuse_train.py \
        --label variable_k_null_allrandom \
        --rotation-policy all-random --latent-scale auto --null-test $COMMON_ARGS &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
