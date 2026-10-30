#!/bin/bash
# Sample one high-occupancy point, optionally sharded across event-level GPU tasks.
#SBATCH --job-name=count_high_sample
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --exclude=gpu3005
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00

set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"; : "${MODEL_ROOT:?}"; : "${PAPER1_ROOT:?}"
SHARDS="${SHARDS:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?submit --array=0-(SHARDS-1)}"
TAG="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" \
    --scan "$SCAN_ROOT/scan.json" --index 0 --field tag)"
POINT="$SCAN_ROOT/$TAG"

module load "${CONDA_MODULE:-miniforge3/25.3.0-3-a6hh}"
eval "$(conda shell.bash hook)"
conda activate "${GENBIB_ENV:-genbib}"
export PYTHONNOUSERSITE=1
cd "$CT_DIR"

python count_tracker_sample.py \
    --conditions "$POINT/conditions" --split test \
    --model-root "$MODEL_ROOT" --paper1-root "$PAPER1_ROOT" \
    --seed-base "${SEED_BASE:-12345}" \
    --max-rounds "${MAX_ROUNDS:-100}" --unfilled-policy "${UNFILLED_POLICY:-drop}" \
    --inference-cache data --num-shards "$SHARDS" --shard-index "$SHARD_INDEX" \
    --output "$POINT/count_samples"
