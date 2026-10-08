#!/bin/bash
# Select explicit COUNT-tracker SIM source lists on a CPU node.
# This job writes only the source manifest; it does not read hit branches.
#SBATCH --job-name=count_manifest
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail

: "${CT_DIR:?set CT_DIR to the count_tracker directory}"
: "${POOLS:?set POOLS to the prepared source-pool root}"
: "${CONSTRUCTION:?set CONSTRUCTION to norm1 or norm42}"
: "${COHORT:?set COHORT to the event cohort label}"
: "${SOURCE_SEED:?set SOURCE_SEED}"
: "${TRAIN_EVENTS:?set TRAIN_EVENTS}"
: "${VAL_EVENTS:?set VAL_EVENTS}"
: "${TEST_EVENTS:?set TEST_EVENTS}"
: "${SOURCE_MANIFEST:?set SOURCE_MANIFEST to a new JSON path}"

CONDA_MODULE="${CONDA_MODULE:-miniforge3/25.3.0-3-a6hh}"
CONDA_ENV="${CONDA_ENV:-count-hdf}"
SOURCE_POOL="${SOURCE_POOL:-split}"
NORM42_FULL_BX="${NORM42_FULL_BX:-0}"

case "$NORM42_FULL_BX" in
    0|1) : ;;
    *) echo "NORM42_FULL_BX must be 0 or 1" >&2; exit 1 ;;
esac

module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1

args=(
    --pools "$POOLS"
    --construction "$CONSTRUCTION"
    --source-pool "$SOURCE_POOL"
    --cohort "$COHORT"
    --seed "$SOURCE_SEED"
    --train-events "$TRAIN_EVENTS"
    --val-events "$VAL_EVENTS"
    --test-events "$TEST_EVENTS"
    --output "$SOURCE_MANIFEST"
)
if [ "$NORM42_FULL_BX" -eq 1 ]; then
    args+=(--norm42-full-bx)
fi

python "$CT_DIR/count_tracker_manifest.py" "${args[@]}"
