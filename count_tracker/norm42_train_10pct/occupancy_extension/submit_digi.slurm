#!/usr/bin/env bash
#SBATCH --job-name=count10_occ_digi
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${EXT_ROOT:?}"; : "${IMAGE:?}"; : "${BENCHMARK_DIR:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?}"; PLAN="$EXT_ROOT/plan.json"
resolve() { python3 "$CT_DIR/norm42_train_10pct/occupancy_extension/plan_task.py" --plan "$PLAN" --stage event --index "$INDEX" --field "$1"; }
TAG="$(resolve tag)"; EVENT_ID="$(resolve event_id)"; POINT="$EXT_ROOT/$TAG"
INPUT_FILE="$POINT/inputs/$EVENT_ID/COUNT/input/input.edm4hep.root"
OUTPUT_DIR="$POINT/digi/$EVENT_ID/COUNT/digi"
[ -f "$INPUT_FILE" ] || { echo "missing $INPUT_FILE" >&2; exit 1; }
if [ -f "$OUTPUT_DIR/digi_output.edm4hep.root" ]; then echo "$TAG/$EVENT_ID digi exists"; exit 0; fi
IMAGE="$IMAGE" BENCHMARK_DIR="$BENCHMARK_DIR" INPUT_FILE="$INPUT_FILE" \
  OUTPUT_DIR="$OUTPUT_DIR" NUM_EVENTS=1 \
  bash "$CT_DIR/v3_fullbx/run_reco_v3_theta.sh" --stage digi
