#!/bin/bash
# Paper-style five-region CKF for one digitized SIM/COUNT pair.
# Submit as --array=0-9: SIM regions first, then COUNT regions.
#SBATCH --job-name=count_v2_theta
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
set -euo pipefail

: "${CT_DIR:?set CT_DIR to the count_tracker directory}"
: "${EVENT_ROOT:?set EVENT_ROOT to the prepared v2.11 event directory}"
export IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"

LOWER=(0 30 70 110 150)
UPPER=(30 70 110 150 180)
TASK="${SLURM_ARRAY_TASK_ID:-}"
[[ "$TASK" =~ ^[0-9]+$ ]] && [ "$TASK" -lt 10 ] || {
    echo "Submit this script with --array=0-9" >&2; exit 1;
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
OUTPUT_DIR="$EVENT_ROOT/$SAMPLE/theta_reco/$TAG"

[ -f "$DIGI_FILE" ] || { echo "Missing digitized input: $DIGI_FILE" >&2; exit 1; }
[ ! -e "$OUTPUT_DIR/reco_output.edm4hep.root" ] || {
    echo "$SAMPLE $TAG already exists: $OUTPUT_DIR/reco_output.edm4hep.root"
    exit 0
}

IMAGE_V2_11="$IMAGE_V2_11" DIGI_FILE="$DIGI_FILE" OUTPUT_DIR="$OUTPUT_DIR" \
    NUM_EVENTS=1 bash "$CT_DIR/v2_11/run_reco_v2_11.sh" \
    --stage reco --theta-min "$MIN" --theta-max "$MAX"

cat > "$OUTPUT_DIR/run_metadata.txt" <<EOF
sample=$SAMPLE
theta_min_deg=$MIN
theta_max_deg=$MAX
theta_regions=0:30,30:70,70:110,110:150,150:180
input=$DIGI_FILE
image=$IMAGE_V2_11
steering=$CT_DIR/v2_11/tracker_reco_override_v2_11.py
steering_sha256=$(sha256sum "$CT_DIR/v2_11/tracker_reco_override_v2_11.py" | cut -d' ' -f1)
ckf_configuration=established_v2_11_diagnostic_plus_theta_split
EOF
