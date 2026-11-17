#!/usr/bin/env bash
# Resume selected events in one point; existing collection outputs are skipped.
#SBATCH --job-name=count10_occ_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --exclude=gpu3005
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${EXT_ROOT:?}"; : "${MODEL_10PCT:?}"
: "${PAPER1_ROOT:?}"; : "${POINT_TAG:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?}"
PLAN="$EXT_ROOT/plan.json"

readarray -t EVENT_FIELDS < <(python3 - "$PLAN" "$POINT_TAG" "$INDEX" <<'PY'
import json, sys
plan, tag, index = sys.argv[1], sys.argv[2], int(sys.argv[3])
point = next(row for row in json.load(open(plan))["points"] if row["tag"] == tag)
event = point["events"][index]
print(event["event_id"])
print(event["split"])
PY
)
EVENT_ID="${EVENT_FIELDS[0]}"
SPLIT="${EVENT_FIELDS[1]}"
POINT="$EXT_ROOT/$POINT_TAG"

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate genbib
export PYTHONNOUSERSITE=1
cd "$CT_DIR"

python count_tracker_sample.py \
  --conditions "$POINT/conditions" --split "$SPLIT" --event-id "$EVENT_ID" \
  --model-root "$MODEL_10PCT" --paper1-root "$PAPER1_ROOT" \
  --seed-base "${SEED_BASE:-12345}" \
  --max-rounds "${MAX_ROUNDS:-100}" --unfilled-policy drop \
  --inference-cache data --output "$POINT/count_samples"
