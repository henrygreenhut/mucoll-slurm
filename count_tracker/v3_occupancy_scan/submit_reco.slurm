#!/bin/bash
# Reconstruct 5 points x 5 events x 2 arms x 5 theta regions. Submit --array=0-249%10.
#SBATCH --job-name=count_occ_reco
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"; : "${IMAGE:?}"; : "${BENCHMARK_DIR:?}"
LOWER=(0 30 70 110 150); UPPER=(30 70 110 150 180)
TASK="${SLURM_ARRAY_TASK_ID:?}"
EVENTS_PER_POINT="${EVENTS_PER_POINT:-5}"
PER_EVENT=10; PER_POINT=$((EVENTS_PER_POINT * PER_EVENT))
POINT_INDEX=$((TASK / PER_POINT)); WITHIN_POINT=$((TASK % PER_POINT))
EVENT_INDEX=$((WITHIN_POINT / PER_EVENT)); WITHIN_EVENT=$((WITHIN_POINT % PER_EVENT))
if [ "$WITHIN_EVENT" -lt 5 ]; then SAMPLE=SIM; REGION="$WITHIN_EVENT"; else SAMPLE=COUNT; REGION=$((WITHIN_EVENT - 5)); fi
MIN="${LOWER[$REGION]}"; MAX="${UPPER[$REGION]}"
REGION_TAG="theta_$(printf '%03d' "$MIN")_$(printf '%03d' "$MAX")"
TAG="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" --scan "$SCAN_ROOT/scan.json" --index "$POINT_INDEX" --field tag)"
EVENT_ID="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" --scan "$SCAN_ROOT/scan.json" --index "$POINT_INDEX" --field event_id --event-index "$EVENT_INDEX")"
POINT="$SCAN_ROOT/$TAG"
DIGI_FILE="$POINT/events/$EVENT_ID/$SAMPLE/theta_split/$REGION_TAG/split_output.edm4hep.root"
OUTPUT_DIR="$POINT/events/$EVENT_ID/$SAMPLE/theta_reco/$REGION_TAG"
[ -f "$DIGI_FILE" ] || { echo "Missing theta input: $DIGI_FILE" >&2; exit 1; }
if [ -f "$OUTPUT_DIR/reco_output.edm4hep.root" ]; then
    echo "$TAG/$EVENT_ID/$SAMPLE/$REGION_TAG reconstruction already exists"
    exit 0
fi

IMAGE="$IMAGE" BENCHMARK_DIR="$BENCHMARK_DIR" DIGI_FILE="$DIGI_FILE" \
    OUTPUT_DIR="$OUTPUT_DIR" NUM_EVENTS=1 \
    bash "$CT_DIR/v3_fullbx/run_reco_v3_theta.sh" --stage reco --split-input

cat > "$OUTPUT_DIR/run_metadata.txt" <<EOF
scan_point=$TAG
event_id=$EVENT_ID
sample=$SAMPLE
theta_min_deg=$MIN
theta_max_deg=$MAX
theta_regions=0:30,30:70,70:110,110:150,150:180
input=$DIGI_FILE
image=$IMAGE
benchmark=$BENCHMARK_DIR
benchmark_commit=$(git -C "$BENCHMARK_DIR" rev-parse HEAD)
ckf_configuration=standard_v3_on_prematerialized_theta_split
EOF
