#!/bin/bash
# =============================================================================
#  COUNT sampling as a SLURM GPU array. Each array task samples one interleaved
#  shard of the split's events into a shared, resumable output directory, so
#  N array tasks give an ~N x speedup. Submit with --array sized to the shards:
#
#    mkdir -p "$OUTC/logs"
#    CONDITIONS=$OUTC/norm42/conditions SPLIT=test MODEL_ROOT=$MODELS \
#    OUTPUT=$OUTC/norm42/count_samples CT_DIR=$CT SEED_BASE=12345 \
#      sbatch --array=0-5 -o "$OUTC/logs/sample_%A_%a.out" -e "$OUTC/logs/sample_%A_%a.err" \
#      $CT/submit_count_tracker_sample.slurm
#
#  The number of shards is taken from the array size automatically, so only
#  change --array to use more/fewer GPUs. Safe to resubmit: finished events are
#  skipped. torch cu111 needs an Ampere (sm_86) GPU, so an a5000 is requested.
# =============================================================================
#SBATCH --job-name=count_sample
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --exclude=gpu3005
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
set -euo pipefail

: "${CONDITIONS:?set CONDITIONS to the prepared conditions dir}"
: "${SPLIT:?set SPLIT to train|val|test}"
: "${MODEL_ROOT:?set MODEL_ROOT to the local_phi model directory}"
: "${OUTPUT:?set OUTPUT to the (shared) count_samples dir}"
CT_DIR="${CT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SEED_BASE="${SEED_BASE:-12345}"
CONDA_MODULE="${CONDA_MODULE:-miniforge3/25.3.0-3-a6hh}"
GENBIB_ENV="${GENBIB_ENV:-genbib}"

# Shards come from the array size (override with NUM_SHARDS if your Slurm does
# not export SLURM_ARRAY_TASK_COUNT); a non-array run is a single shard.
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$GENBIB_ENV"
export PYTHONNOUSERSITE=1   # ignore ~/.local; use the env exactly
cd "$CT_DIR"

echo "=== count sample: shard ${SHARD_INDEX}/${NUM_SHARDS}  split=${SPLIT} ==="
echo "conditions=${CONDITIONS}"
echo "output=${OUTPUT}"
python -c "import torch; print('cuda', torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python count_tracker_sample.py \
    --conditions "$CONDITIONS" --split "$SPLIT" --model-root "$MODEL_ROOT" \
    --seed-base "$SEED_BASE" \
    --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" \
    --output "$OUTPUT"
