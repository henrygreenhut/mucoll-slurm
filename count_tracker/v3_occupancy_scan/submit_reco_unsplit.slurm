#!/bin/bash
# Run one standard v3 CKF over all digitized tracker hits in an event arm.
# Five points x five events x two arms: submit --array=0-49.
# Point ranges are 0-9, 10-19, 20-29, 30-39, and 40-49.
#SBATCH --job-name=count_occ_unsplit
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"; : "${IMAGE:?}"; : "${BENCHMARK_DIR:?}"
TASK="${SLURM_ARRAY_TASK_ID:?}"
EVENTS_PER_POINT="${EVENTS_PER_POINT:-5}"
PER_POINT=$((EVENTS_PER_POINT * 2))
POINT_INDEX=$((TASK / PER_POINT))
WITHIN=$((TASK % PER_POINT))
EVENT_INDEX=$((WITHIN / 2))
if [ $((WITHIN % 2)) -eq 0 ]; then SAMPLE=SIM; else SAMPLE=COUNT; fi

TAG="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" \
    --scan "$SCAN_ROOT/scan.json" --index "$POINT_INDEX" --field tag)"
EVENT_ID="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" \
    --scan "$SCAN_ROOT/scan.json" --index "$POINT_INDEX" \
    --field event_id --event-index "$EVENT_INDEX")"
POINT="$SCAN_ROOT/$TAG"
DIGI_FILE="$POINT/digi/$EVENT_ID/$SAMPLE/digi/digi_output.edm4hep.root"
OUTPUT_DIR="$POINT/events/$EVENT_ID/$SAMPLE/unsplit_reco"

[ -f "$DIGI_FILE" ] || { echo "Missing digitized input: $DIGI_FILE" >&2; exit 1; }
if [ -f "$OUTPUT_DIR/reco_output.edm4hep.root" ]; then
    echo "$TAG/$EVENT_ID/$SAMPLE unsplit reconstruction already exists"
    exit 0
fi

IMAGE="$IMAGE" BENCHMARK_DIR="$BENCHMARK_DIR" DIGI_FILE="$DIGI_FILE" \
    OUTPUT_DIR="$OUTPUT_DIR" NUM_EVENTS=1 \
    bash "$CT_DIR/v3_fullbx/run_reco_v3_theta.sh" --stage reco

cat > "$OUTPUT_DIR/run_metadata.txt" <<EOF
scan_point=$TAG
event_id=$EVENT_ID
sample=$SAMPLE
input=$DIGI_FILE
image=$IMAGE
benchmark=$BENCHMARK_DIR
benchmark_commit=$(git -C "$BENCHMARK_DIR" rev-parse HEAD)
theta_partition=none
ckf_configuration=standard_v3_on_complete_digitized_tracker_event
control=five_region_partition_sensitivity
EOF
