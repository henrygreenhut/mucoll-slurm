#!/bin/bash
#SBATCH -J syn42_validate
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -t 00:30:00
#SBATCH --output=synthetic42_validation_%j.out
#SBATCH --error=synthetic42_validation_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

COMMON_ARGS="--cycles-per-unit 10 --clone-factor 42 --units-per-epoch 20 --val-units 10 --test-units 30 --batch-size 1 --patience 10 --max-minutes 25 --seed 1 --features rotation"

pids=()
for model_seed in 1 2 3; do
    srun --exclusive --exact -N 1 -n 1 -c 32 --gpus-per-task=1 --gpu-bind=single:1 \
        python pfn_synthetic42_validate.py \
            --label "synthetic42_vs_original_s${model_seed}" \
            --model-seed "$model_seed" --latent-scale auto $COMMON_ARGS &
    pids+=("$!")
done

srun --exclusive --exact -N 1 -n 1 -c 32 --gpus-per-task=1 --gpu-bind=single:1 \
    python pfn_synthetic42_validate.py \
        --label synthetic42_null_s1 --model-seed 1 \
        --latent-scale auto --null-test $COMMON_ARGS &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
