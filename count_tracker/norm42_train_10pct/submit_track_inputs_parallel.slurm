#!/usr/bin/env bash
# One task per event arm. Submit as --array=0-$((3*EVENTS_PER_POINT-1))%6.
# Each task runs the unchanged CellID assignment and v2.11 input writer.
#SBATCH --job-name=count10_track_input
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"; : "${GENBIB_DIR:?}"; : "${GEOMAP:?}"
TASK="${SLURM_ARRAY_TASK_ID:?}"
EVENTS_PER_POINT="${EVENTS_PER_POINT:-1}"
(( TASK < EVENTS_PER_POINT * 3 )) || { echo "task $TASK outside cohort" >&2; exit 1; }
EVENT_INDEX=$((TASK / 3)); ARM_INDEX=$((TASK % 3))
ARMS=(SIM COUNT_FULL COUNT_10PCT)
ARM="${ARMS[$ARM_INDEX]}"
POINT="$SCAN_ROOT/n0167"
EVENT_ID="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" \
  --scan "$SCAN_ROOT/scan.json" --index 0 --field event_id \
  --event-index "$EVENT_INDEX")"
OUTPUT="$POINT/inputs/$EVENT_ID/$ARM"

if [ -f "$OUTPUT/input/input.edm4hep.root" ]; then
  echo "$EVENT_ID/$ARM input already exists"
  exit 0
fi
[ ! -e "$OUTPUT" ] || { echo "refusing incomplete output $OUTPUT" >&2; exit 1; }

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate genbib
export ASSIGN_PYTHON="$(command -v python)"
conda deactivate
conda activate count-hdf
export EXTRACT_PYTHON="$(command -v python)"

WORK_ROOT="$POINT/input_work/$EVENT_ID"
WORK="$WORK_ROOT/${ARM}.${SLURM_JOB_ID}.${TASK}"
[ ! -e "$WORK" ] || { echo "refusing stale work directory $WORK" >&2; exit 1; }
mkdir -p "$WORK" "$(dirname "$OUTPUT")"

export IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
ARGS=(
  --conditions "$POINT/conditions" --construction norm42
  --split test --event-id "$EVENT_ID"
  --geomap "$GEOMAP" --stop-after input --output "$WORK"
)
case "$ARM" in
  SIM)
    ARGS+=(--only SIM)
    NATIVE_ARM=SIM
    ;;
  COUNT_FULL)
    ARGS+=(--only COUNT --count-samples "$POINT/count_samples_full")
    NATIVE_ARM=COUNT
    ;;
  COUNT_10PCT)
    ARGS+=(--only COUNT --count-samples "$POINT/count_samples_10pct")
    NATIVE_ARM=COUNT
    ;;
esac

bash "$CT_DIR/v2_11/run_count_tracker_event_v2_11.sh" "${ARGS[@]}"
[ -f "$WORK/$NATIVE_ARM/input/input.edm4hep.root" ] || {
  echo "input writer did not publish $NATIVE_ARM" >&2
  exit 1
}
mv "$WORK/$NATIVE_ARM" "$OUTPUT"
rmdir "$WORK"
ARM="$ARM" EVENT_ID="$EVENT_ID" OUTPUT="$OUTPUT" python3 -c '
import json, os, pathlib
path = pathlib.Path(os.environ["OUTPUT"]) / "study_arm.json"
path.write_text(json.dumps({"event_id": os.environ["EVENT_ID"], "arm": os.environ["ARM"]}, indent=2) + "\n")
'
echo "prepared $EVENT_ID/$ARM -> $OUTPUT"
