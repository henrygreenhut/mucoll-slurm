#!/usr/bin/env bash
# One task per event; tasks belonging to five-region points exit successfully.
#SBATCH --job-name=count10_occ_recou
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${EXT_ROOT:?}"; : "${IMAGE:?}"; : "${BENCHMARK_DIR:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?}"; PLAN="$EXT_ROOT/plan.json"
resolve() { python3 "$CT_DIR/norm42_train_10pct/occupancy_extension/plan_task.py" --plan "$PLAN" --stage event --index "$INDEX" --field "$1"; }
TAG="$(resolve tag)"; EVENT_ID="$(resolve event_id)"; MODE="$(resolve mode)"
if [ "$MODE" != unsplit ]; then echo "$TAG/$EVENT_ID uses five-region CKF"; exit 0; fi
POINT="$EXT_ROOT/$TAG"; DIGI_FILE="$POINT/digi/$EVENT_ID/COUNT/digi/digi_output.edm4hep.root"
OUTPUT_DIR="$POINT/events/$EVENT_ID/COUNT/unsplit_reco"
[ -f "$DIGI_FILE" ] || { echo "missing $DIGI_FILE" >&2; exit 1; }
if [ -f "$OUTPUT_DIR/reco_output.edm4hep.root" ]; then echo "$TAG/$EVENT_ID unsplit reco exists"; exit 0; fi
IMAGE="$IMAGE" BENCHMARK_DIR="$BENCHMARK_DIR" DIGI_FILE="$DIGI_FILE" \
  OUTPUT_DIR="$OUTPUT_DIR" NUM_EVENTS=1 \
  bash "$CT_DIR/v3_fullbx/run_reco_v3_theta.sh" --stage reco
