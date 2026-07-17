#!/bin/bash
#SBATCH -J variable_k_pfn
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH --output=variable_k_train_%x_%j.out
#SBATCH --error=variable_k_train_%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

LABEL=${LABEL:?set LABEL to a unique result name}
TRAIN_ARGS=${TRAIN_ARGS:-}

python pfn_variable_reuse_train.py \
    --label "$LABEL" \
    --max-minutes 25 \
    $TRAIN_ARGS
