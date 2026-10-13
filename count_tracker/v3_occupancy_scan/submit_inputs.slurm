#!/bin/bash
# Build BIB-only paired inputs exactly as in the established full-occupancy run.
# Submit 5 points x 5 events as --array=0-24%2.
#SBATCH --job-name=count_occ_input
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"; : "${GENBIB_DIR:?}"; : "${GEOMAP:?}"
TASK="${SLURM_ARRAY_TASK_ID:?}"
EVENTS_PER_POINT="${EVENTS_PER_POINT:-5}"
POINT_INDEX=$((TASK / EVENTS_PER_POINT))
EVENT_INDEX=$((TASK % EVENTS_PER_POINT))
TAG="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" --scan "$SCAN_ROOT/scan.json" --index "$POINT_INDEX" --field tag)"
EVENT_ID="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" --scan "$SCAN_ROOT/scan.json" --index "$POINT_INDEX" --field event_id --event-index "$EVENT_INDEX")"
POINT="$SCAN_ROOT/$TAG"
OUTPUT="$POINT/inputs/$EVENT_ID"

if [ -f "$OUTPUT/SIM/input/input.edm4hep.root" ] && \
   [ -f "$OUTPUT/COUNT/input/input.edm4hep.root" ]; then
    echo "$TAG/$EVENT_ID inputs already exist"
    exit 0
fi
[ -d "$OUTPUT" ] && rm -rf "$OUTPUT"

module load "${CONDA_MODULE:-miniforge3/25.3.0-3-a6hh}"
eval "$(conda shell.bash hook)"
conda activate "${ASSIGN_ENV:-genbib}"
export ASSIGN_PYTHON="$(command -v python)"
conda deactivate
conda activate "${EXTRACT_ENV:-count-hdf}"
export EXTRACT_PYTHON="$(command -v python)"

export IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
bash "$CT_DIR/v2_11/run_count_tracker_event_v2_11.sh" \
    --conditions "$POINT/conditions" --construction norm42 \
    --split test --event-id "$EVENT_ID" \
    --count-samples "$POINT/count_samples" --geomap "$GEOMAP" \
    --stop-after input --output "$OUTPUT"
