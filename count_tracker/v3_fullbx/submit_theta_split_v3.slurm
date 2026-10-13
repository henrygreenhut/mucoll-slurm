#!/bin/bash
# Materialize five Paper-1 theta regions natively in the v3 EDM4hep model.
# Submit as --array=0-9%2: SIM regions first, then COUNT regions.
#SBATCH --job-name=count_v3_split_native
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?set CT_DIR to count_tracker}"
: "${DIGI_ROOT:?set DIGI_ROOT to the completed v3 digitization event directory}"
: "${EVENT_ROOT:?set EVENT_ROOT to a new native-split output event directory}"
: "${IMAGE:?set the v3 image}"

LOWER=(0 30 70 110 150)
UPPER=(30 70 110 150 180)
TASK="${SLURM_ARRAY_TASK_ID:-}"
[[ "$TASK" =~ ^[0-9]+$ ]] && [ "$TASK" -lt 10 ] || {
    echo "Submit with --array=0-9%2" >&2; exit 1;
}
if [ "$TASK" -lt 5 ]; then
    SAMPLE=SIM
    REGION="$TASK"
else
    SAMPLE=COUNT
    REGION=$((TASK - 5))
fi

MIN="${LOWER[$REGION]}"
MAX="${UPPER[$REGION]}"
TAG="theta_$(printf '%03d' "$MIN")_$(printf '%03d' "$MAX")"
DIGI_FILE="$DIGI_ROOT/$SAMPLE/digi/digi_output.edm4hep.root"
OUTPUT_DIR="$EVENT_ROOT/$SAMPLE/theta_split/$TAG"
[ -f "$DIGI_FILE" ] || { echo "Missing digitized input: $DIGI_FILE" >&2; exit 1; }
if [ -f "$OUTPUT_DIR/split_output.edm4hep.root" ]; then
    echo "$SAMPLE $TAG native v3 split already exists"
    exit 0
fi

IMAGE="$IMAGE" DIGI_FILE="$DIGI_FILE" OUTPUT_DIR="$OUTPUT_DIR" \
    THETA_MIN="$MIN" THETA_MAX="$MAX" \
    bash "$CT_DIR/v3_fullbx/run_theta_split_v3.sh"

cat > "$OUTPUT_DIR/run_metadata.txt" <<EOF
sample=$SAMPLE
theta_min_deg=$MIN
theta_max_deg=$MAX
input=$DIGI_FILE
image=$IMAGE
steering=$CT_DIR/v3_fullbx/split_digi_by_theta_v3.py
steering_sha256=$(sha256sum "$CT_DIR/v3_fullbx/split_digi_by_theta_v3.py" | cut -d' ' -f1)
operation=native_v3_position_theta_filter_with_matching_relations
EOF
