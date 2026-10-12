#!/bin/bash
# Materialize five Paper-1 theta regions for a v3-digitized SIM/COUNT pair.
# Submit as --array=0-9%2 to limit simultaneous reads of the two large inputs.
#SBATCH --job-name=count_v3_split
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?set CT_DIR to count_tracker}"
: "${EVENT_ROOT:?set EVENT_ROOT to the v3 full-BX directory}"
export IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"

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
DIGI_FILE="$EVENT_ROOT/$SAMPLE/digi/digi_output.edm4hep.root"
OUTPUT_DIR="$EVENT_ROOT/$SAMPLE/theta_split/$TAG"
[ -f "$DIGI_FILE" ] || { echo "Missing digitized input: $DIGI_FILE" >&2; exit 1; }
if [ -f "$OUTPUT_DIR/split_output.edm4hep.root" ]; then
    echo "$SAMPLE $TAG split already exists"
    exit 0
fi

IMAGE_V2_11="$IMAGE_V2_11" DIGI_FILE="$DIGI_FILE" OUTPUT_DIR="$OUTPUT_DIR" \
    THETA_MIN="$MIN" THETA_MAX="$MAX" \
    bash "$CT_DIR/v3_fullbx/run_theta_split_v2_11.sh"

cat > "$OUTPUT_DIR/run_metadata.txt" <<EOF
sample=$SAMPLE
theta_min_deg=$MIN
theta_max_deg=$MAX
input=$DIGI_FILE
image=$IMAGE_V2_11
steering=$CT_DIR/v3_fullbx/split_digi_by_theta_v2_11.py
steering_sha256=$(sha256sum "$CT_DIR/v3_fullbx/split_digi_by_theta_v2_11.py" | cut -d' ' -f1)
operation=SplitCollectionByPolarAngle_only
EOF
