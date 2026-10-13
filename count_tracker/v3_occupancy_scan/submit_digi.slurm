#!/bin/bash
# Digitize 5 events x 2 arms per scan point. Submit --array=0-49%4.
#SBATCH --job-name=count_occ_digi
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"; : "${IMAGE:?}"; : "${BENCHMARK_DIR:?}"
TASK="${SLURM_ARRAY_TASK_ID:?}"
EVENTS_PER_POINT="${EVENTS_PER_POINT:-5}"
PER_POINT=$((EVENTS_PER_POINT * 2))
POINT_INDEX=$((TASK / PER_POINT))
WITHIN=$((TASK % PER_POINT))
EVENT_INDEX=$((WITHIN / 2))
if [ $((WITHIN % 2)) -eq 0 ]; then SAMPLE=SIM; else SAMPLE=COUNT; fi
TAG="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" --scan "$SCAN_ROOT/scan.json" --index "$POINT_INDEX" --field tag)"
EVENT_ID="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" --scan "$SCAN_ROOT/scan.json" --index "$POINT_INDEX" --field event_id --event-index "$EVENT_INDEX")"
POINT="$SCAN_ROOT/$TAG"
INPUT_FILE="$POINT/inputs/$EVENT_ID/$SAMPLE/input/input.edm4hep.root"
OUTPUT_DIR="$POINT/digi/$EVENT_ID/$SAMPLE/digi"
[ -f "$INPUT_FILE" ] || { echo "Missing input: $INPUT_FILE" >&2; exit 1; }
if [ -f "$OUTPUT_DIR/digi_output.edm4hep.root" ]; then
    echo "$TAG/$EVENT_ID/$SAMPLE digitization already exists"
    exit 0
fi

IMAGE="$IMAGE" BENCHMARK_DIR="$BENCHMARK_DIR" INPUT_FILE="$INPUT_FILE" \
    OUTPUT_DIR="$OUTPUT_DIR" NUM_EVENTS=1 \
    bash "$CT_DIR/v3_fullbx/run_reco_v3_theta.sh" --stage digi

cat > "$OUTPUT_DIR/run_metadata.txt" <<EOF
scan_point=$TAG
event_id=$EVENT_ID
sample=$SAMPLE
input=$INPUT_FILE
image=$IMAGE
benchmark=$BENCHMARK_DIR
benchmark_commit=$(git -C "$BENCHMARK_DIR" rev-parse HEAD)
configuration=standard_v3_tracker_digitization
EOF
