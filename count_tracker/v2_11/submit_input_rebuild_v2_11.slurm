#!/bin/bash
#SBATCH --job-name=count_v2_build
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
set -euo pipefail

: "${CT_DIR:?set CT_DIR to the count_tracker directory}"
: "${GENBIB_DIR:?set GENBIB_DIR to the GenBIB-ML checkout}"
: "${CONDITIONS:?set CONDITIONS to the prepared conditions directory}"
: "${CONSTRUCTION:?set CONSTRUCTION to the conditions construction}"
: "${SPLIT:?set SPLIT to train, val, or test}"
: "${EVENT_ID:?set EVENT_ID to the prepared event ID}"
: "${SIGNAL:?set SIGNAL to the neutrino SIM file}"
: "${SIGNAL_ENTRY:?set SIGNAL_ENTRY to the paired signal entry}"
: "${COUNT_SAMPLES:?set COUNT_SAMPLES to the generated arrays root}"
: "${GEOMAP:?set GEOMAP to a dedicated v2.11 geometry map}"
: "${OUTPUT:?set OUTPUT to a new v2.11 event directory}"

CONDA_MODULE="${CONDA_MODULE:-miniforge3/25.3.0-3-a6hh}"
COUNT_ENV="${COUNT_ENV:-count-hdf}"
export IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"

module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$COUNT_ENV"
export ASSIGN_PYTHON="$(command -v python)"

bash "$CT_DIR/v2_11/run_count_tracker_event_v2_11.sh" \
    --conditions "$CONDITIONS" \
    --construction "$CONSTRUCTION" \
    --split "$SPLIT" \
    --event-id "$EVENT_ID" \
    --signal "$SIGNAL" \
    --signal-entry "$SIGNAL_ENTRY" \
    --count-samples "$COUNT_SAMPLES" \
    --geomap "$GEOMAP" \
    --output "$OUTPUT"
