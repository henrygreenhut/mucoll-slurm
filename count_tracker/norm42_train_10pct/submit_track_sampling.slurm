#!/usr/bin/env bash
#SBATCH --job-name=count10_track_sample
#SBATCH --partition=gpu
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --exclude=gpu3005
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"; : "${MODEL_FULL:?}"
: "${MODEL_10PCT:?}"; : "${PAPER1_ROOT:?}"

case "${SLURM_ARRAY_TASK_ID:?submit --array=0-1}" in
  0) LABEL=full; MODEL_ROOT="$MODEL_FULL" ;;
  1) LABEL=10pct; MODEL_ROOT="$MODEL_10PCT" ;;
  *) echo "invalid array index" >&2; exit 1 ;;
esac
POINT="$SCAN_ROOT/n0167"

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate genbib
export PYTHONNOUSERSITE=1
cd "$CT_DIR"

echo "sampling arm=$LABEL model_root=$MODEL_ROOT"
python count_tracker_sample.py \
  --conditions "$POINT/conditions" --split test \
  --model-root "$MODEL_ROOT" --paper1-root "$PAPER1_ROOT" \
  --seed-base "${SEED_BASE:-12345}" \
  --max-rounds "${MAX_ROUNDS:-100}" --unfilled-policy drop \
  --inference-cache data --num-shards 1 --shard-index 0 \
  --output "$POINT/count_samples_$LABEL"
