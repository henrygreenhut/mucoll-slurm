#!/bin/bash
# Sample the train, validation, and test parents with one GPU task per split.
#SBATCH --job-name=count_disjoint_sample
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --exclude=gpu3005
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"; : "${MODEL_ROOT:?}"; : "${PAPER1_ROOT:?}"
SPLITS=(train val test)
SPLIT="${SPLITS[${SLURM_ARRAY_TASK_ID:?submit --array=0-2}]}"
POINT="$SCAN_ROOT/n1666"

module load "${CONDA_MODULE:-miniforge3/25.3.0-3-a6hh}"
eval "$(conda shell.bash hook)"
conda activate "${GENBIB_ENV:-genbib}"
export PYTHONNOUSERSITE=1
cd "$CT_DIR"

python count_tracker_sample.py \
    --conditions "$POINT/conditions" --split "$SPLIT" \
    --model-root "$MODEL_ROOT" --paper1-root "$PAPER1_ROOT" \
    --seed-base "${SEED_BASE:-12345}" \
    --max-rounds "${MAX_ROUNDS:-100}" \
    --unfilled-policy "${UNFILLED_POLICY:-drop}" \
    --inference-cache data --num-shards 1 --shard-index 0 \
    --output "$POINT/count_samples"
