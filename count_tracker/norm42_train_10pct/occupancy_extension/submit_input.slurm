#!/usr/bin/env bash
# One COUNT-only input task per reused SIM event.
#SBATCH --job-name=count10_occ_input
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${EXT_ROOT:?}"; : "${GENBIB_DIR:?}"; : "${GEOMAP:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?}"; PLAN="$EXT_ROOT/plan.json"
resolve() { python3 "$CT_DIR/norm42_train_10pct/occupancy_extension/plan_task.py" --plan "$PLAN" --stage event --index "$INDEX" --field "$1"; }
TAG="$(resolve tag)"; EVENT_ID="$(resolve event_id)"; SPLIT="$(resolve split)"
POINT="$EXT_ROOT/$TAG"; OUTPUT="$POINT/inputs/$EVENT_ID/COUNT"
if [ -f "$OUTPUT/input/input.edm4hep.root" ]; then
  echo "$TAG/$EVENT_ID COUNT input already exists"; exit 0
fi
[ ! -e "$OUTPUT" ] || { echo "refusing incomplete output $OUTPUT" >&2; exit 1; }

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate genbib
export ASSIGN_PYTHON="$(command -v python)"
conda deactivate
conda activate count-hdf
export EXTRACT_PYTHON="$(command -v python)"

WORK="$POINT/input_work/${EVENT_ID}.${SLURM_JOB_ID}.${INDEX}"
[ ! -e "$WORK" ] || { echo "refusing stale work directory $WORK" >&2; exit 1; }
mkdir -p "$WORK" "$(dirname "$OUTPUT")"
export IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
bash "$CT_DIR/v2_11/run_count_tracker_event_v2_11.sh" \
  --conditions "$POINT/conditions" --construction norm42 \
  --split "$SPLIT" --event-id "$EVENT_ID" \
  --count-samples "$POINT/count_samples" --geomap "$GEOMAP" \
  --only COUNT --stop-after input --output "$WORK"
[ -f "$WORK/COUNT/input/input.edm4hep.root" ] || { echo "COUNT input missing" >&2; exit 1; }
mv "$WORK/COUNT" "$OUTPUT"
rmdir "$WORK"
echo "prepared $TAG/$EVENT_ID/COUNT"
