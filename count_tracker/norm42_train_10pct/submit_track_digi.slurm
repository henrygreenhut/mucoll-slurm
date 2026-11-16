#!/usr/bin/env bash
#SBATCH --job-name=count10_track_digi
#SBATCH --partition=batch
#SBATCH --array=0-2%3
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"; : "${IMAGE:?}"; : "${BENCHMARK_DIR:?}"
ARMS=(SIM COUNT_FULL COUNT_10PCT)
TASK="${SLURM_ARRAY_TASK_ID:?}"
EVENTS_PER_POINT="${EVENTS_PER_POINT:-1}"
(( TASK < EVENTS_PER_POINT * 3 )) || { echo "task $TASK outside cohort" >&2; exit 1; }
EVENT_INDEX=$((TASK / 3)); ARM_INDEX=$((TASK % 3)); ARM="${ARMS[$ARM_INDEX]}"
POINT="$SCAN_ROOT/n0167"
EVENT_ID="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" --scan "$SCAN_ROOT/scan.json" --index 0 --field event_id --event-index "$EVENT_INDEX")"
INPUT_FILE="$POINT/inputs/$EVENT_ID/$ARM/input/input.edm4hep.root"
OUTPUT_DIR="$POINT/digi/$EVENT_ID/$ARM/digi"
[ -f "$INPUT_FILE" ] || { echo "missing $INPUT_FILE" >&2; exit 1; }
if [ -f "$OUTPUT_DIR/digi_output.edm4hep.root" ]; then echo "$ARM digitization exists"; exit 0; fi
IMAGE="$IMAGE" BENCHMARK_DIR="$BENCHMARK_DIR" INPUT_FILE="$INPUT_FILE" \
  OUTPUT_DIR="$OUTPUT_DIR" NUM_EVENTS=1 \
  bash "$CT_DIR/v3_fullbx/run_reco_v3_theta.sh" --stage digi
