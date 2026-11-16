#!/usr/bin/env bash
# One task per event arm for a fresh paired 10%-BX cohort.
#SBATCH --job-name=count10_pair_input
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"; : "${GENBIB_DIR:?}"; : "${GEOMAP:?}"
TASK="${SLURM_ARRAY_TASK_ID:?}"; EVENTS_PER_POINT="${EVENTS_PER_POINT:-5}"
EVENT_INDEX=$((TASK / 2)); ARM_INDEX=$((TASK % 2))
(( EVENT_INDEX < EVENTS_PER_POINT )) || { echo "task outside cohort" >&2; exit 1; }
if [ "$ARM_INDEX" -eq 0 ]; then ARM=SIM; else ARM=COUNT; fi
POINT="$SCAN_ROOT/n0167"
EVENT_ID="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" --scan "$SCAN_ROOT/scan.json" --index 0 --field event_id --event-index "$EVENT_INDEX")"
OUTPUT="$POINT/inputs/$EVENT_ID/$ARM"
if [ -f "$OUTPUT/input/input.edm4hep.root" ]; then echo "$EVENT_ID/$ARM exists"; exit 0; fi
[ ! -e "$OUTPUT" ] || { echo "refusing incomplete output $OUTPUT" >&2; exit 1; }

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate genbib
export ASSIGN_PYTHON="$(command -v python)"
conda deactivate
conda activate count-hdf
export EXTRACT_PYTHON="$(command -v python)"

WORK="$POINT/input_work/${EVENT_ID}.${ARM}.${SLURM_JOB_ID}.${TASK}"
mkdir -p "$WORK" "$(dirname "$OUTPUT")"
export IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
ARGS=(--conditions "$POINT/conditions" --construction norm42 --split test \
  --event-id "$EVENT_ID" --geomap "$GEOMAP" --only "$ARM" \
  --stop-after input --output "$WORK")
[ "$ARM" = SIM ] || ARGS+=(--count-samples "$POINT/count_samples")
bash "$CT_DIR/v2_11/run_count_tracker_event_v2_11.sh" "${ARGS[@]}"
[ -f "$WORK/$ARM/input/input.edm4hep.root" ] || { echo "$ARM input missing" >&2; exit 1; }
mv "$WORK/$ARM" "$OUTPUT"; rmdir "$WORK"
