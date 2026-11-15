#!/usr/bin/env bash
#SBATCH --job-name=count10_track_reco
#SBATCH --partition=batch
#SBATCH --array=0-14%10
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"; : "${IMAGE:?}"; : "${BENCHMARK_DIR:?}"
ARMS=(SIM COUNT_FULL COUNT_10PCT); LOWER=(0 30 70 110 150); UPPER=(30 70 110 150 180)
TASK="${SLURM_ARRAY_TASK_ID:?}"; ARM="${ARMS[$((TASK / 5))]}"; REGION=$((TASK % 5))
MIN="${LOWER[$REGION]}"; MAX="${UPPER[$REGION]}"; REGION_TAG="theta_$(printf '%03d' "$MIN")_$(printf '%03d' "$MAX")"
POINT="$SCAN_ROOT/n0167"
EVENT_ID="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" --scan "$SCAN_ROOT/scan.json" --index 0 --field event_id --event-index 0)"
DIGI_FILE="$POINT/events/$EVENT_ID/$ARM/theta_split/$REGION_TAG/split_output.edm4hep.root"
OUTPUT_DIR="$POINT/events/$EVENT_ID/$ARM/theta_reco/$REGION_TAG"
[ -f "$DIGI_FILE" ] || { echo "missing $DIGI_FILE" >&2; exit 1; }
if [ -f "$OUTPUT_DIR/reco_output.edm4hep.root" ]; then echo "$ARM/$REGION_TAG reco exists"; exit 0; fi
IMAGE="$IMAGE" BENCHMARK_DIR="$BENCHMARK_DIR" DIGI_FILE="$DIGI_FILE" \
  OUTPUT_DIR="$OUTPUT_DIR" NUM_EVENTS=1 \
  bash "$CT_DIR/v3_fullbx/run_reco_v3_theta.sh" --stage reco --split-input
