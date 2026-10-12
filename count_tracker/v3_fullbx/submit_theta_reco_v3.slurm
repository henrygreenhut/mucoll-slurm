#!/bin/bash
# Run standard v3 CKF independently in five paper-style theta regions.
# Submit as an array 0-9: SIM regions first, then COUNT regions.
#SBATCH --job-name=count_v3_theta
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
set -euo pipefail

: "${CT_DIR:?set CT_DIR to count_tracker}"
: "${EVENT_ROOT:?set EVENT_ROOT to the v3 full-BX directory}"
: "${IMAGE:?set the v3 image}"
: "${BENCHMARK_DIR:?set the pinned benchmark checkout}"

LOWER=(0 30 70 110 150)
UPPER=(30 70 110 150 180)
TASK="${SLURM_ARRAY_TASK_ID:-}"
[[ "$TASK" =~ ^[0-9]+$ ]] && [ "$TASK" -lt 10 ] || {
    echo "Submit with --array=0-9" >&2; exit 1;
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
DIGI_FILE="$EVENT_ROOT/$SAMPLE/theta_split/$TAG/split_output.edm4hep.root"
OUTPUT_DIR="$EVENT_ROOT/$SAMPLE/theta_reco/$TAG"
[ -f "$DIGI_FILE" ] || { echo "Missing digitized input: $DIGI_FILE" >&2; exit 1; }
if [ -f "$OUTPUT_DIR/reco_output.edm4hep.root" ]; then
    echo "$SAMPLE $TAG v3 reconstruction already exists"
    exit 0
fi

IMAGE="$IMAGE" BENCHMARK_DIR="$BENCHMARK_DIR" DIGI_FILE="$DIGI_FILE" \
    OUTPUT_DIR="$OUTPUT_DIR" NUM_EVENTS=1 \
    bash "$CT_DIR/v3_fullbx/run_reco_v3_theta.sh" --stage reco --split-input

cat > "$OUTPUT_DIR/run_metadata.txt" <<EOF
sample=$SAMPLE
theta_min_deg=$MIN
theta_max_deg=$MAX
theta_regions=0:30,30:70,70:110,110:150,150:180
input=$DIGI_FILE
image=$IMAGE
benchmark=$BENCHMARK_DIR
benchmark_commit=$(git -C "$BENCHMARK_DIR" rev-parse HEAD)
steering=$CT_DIR/v3_fullbx/tracker_reco_override_v3_theta.py
steering_sha256=$(sha256sum "$CT_DIR/v3_fullbx/tracker_reco_override_v3_theta.py" | cut -d' ' -f1)
ckf_configuration=standard_v3_on_prematerialized_theta_split
EOF
