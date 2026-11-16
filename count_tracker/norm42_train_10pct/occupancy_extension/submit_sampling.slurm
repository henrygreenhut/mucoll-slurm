#!/usr/bin/env bash
# One or more shards per occupancy point.  A shard keeps each model/collection
# loaded while sampling its assigned events.
#SBATCH --job-name=count10_occ_sample
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --exclude=gpu3005
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${EXT_ROOT:?}"; : "${MODEL_10PCT:?}"; : "${PAPER1_ROOT:?}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?}"
SHARDS_PER_POINT="${SHARDS_PER_POINT:-1}"
(( SHARDS_PER_POINT > 0 )) || { echo "SHARDS_PER_POINT must be positive" >&2; exit 2; }
POINT_INDEX=$((TASK_INDEX / SHARDS_PER_POINT))
SHARD_INDEX=$((TASK_INDEX % SHARDS_PER_POINT))
PLAN="$EXT_ROOT/plan.json"
TAG="$(python3 "$CT_DIR/norm42_train_10pct/occupancy_extension/plan_task.py" \
  --plan "$PLAN" --stage point --index "$POINT_INDEX" --field tag)"
POINT="$EXT_ROOT/$TAG"

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate genbib
export PYTHONNOUSERSITE=1
cd "$CT_DIR"

python count_tracker_sample.py \
  --conditions "$POINT/conditions" --split test \
  --model-root "$MODEL_10PCT" --paper1-root "$PAPER1_ROOT" \
  --seed-base "${SEED_BASE:-12345}" \
  --max-rounds "${MAX_ROUNDS:-100}" --unfilled-policy drop \
  --inference-cache data \
  --num-shards "$SHARDS_PER_POINT" --shard-index "$SHARD_INDEX" \
  --output "$POINT/count_samples"
