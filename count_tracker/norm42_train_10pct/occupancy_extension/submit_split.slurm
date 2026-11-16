#!/usr/bin/env bash
#SBATCH --job-name=count10_occ_split
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${EXT_ROOT:?}"; : "${IMAGE:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?}"; PLAN="$EXT_ROOT/plan.json"
resolve() { python3 "$CT_DIR/norm42_train_10pct/occupancy_extension/plan_task.py" --plan "$PLAN" --stage region --index "$INDEX" --field "$1"; }
TAG="$(resolve tag)"; EVENT_ID="$(resolve event_id)"; MIN="$(resolve theta_min)"; MAX="$(resolve theta_max)"; REGION_TAG="$(resolve region_tag)"
POINT="$EXT_ROOT/$TAG"; DIGI_FILE="$POINT/digi/$EVENT_ID/COUNT/digi/digi_output.edm4hep.root"
OUTPUT_DIR="$POINT/events/$EVENT_ID/COUNT/theta_split/$REGION_TAG"
[ -f "$DIGI_FILE" ] || { echo "missing $DIGI_FILE" >&2; exit 1; }
if [ -f "$OUTPUT_DIR/split_output.edm4hep.root" ]; then echo "$TAG/$EVENT_ID/$REGION_TAG split exists"; exit 0; fi
IMAGE="$IMAGE" DIGI_FILE="$DIGI_FILE" OUTPUT_DIR="$OUTPUT_DIR" \
  THETA_MIN="$MIN" THETA_MAX="$MAX" bash "$CT_DIR/v3_fullbx/run_theta_split_v3.sh"
