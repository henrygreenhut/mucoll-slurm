#!/bin/bash
# Prepare a classifier-ready direct norm1 split-mother cohort on one CPU node.
# The writer publishes the conditions directory atomically after all events pass.
#SBATCH --job-name=count_n1_prepare
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
set -euo pipefail

: "${CT_DIR:?}"
: "${SIM_ROOT:?}"
: "${METADATA:?}"
: "${CONDITIONS:?}"
: "${TRAIN_EVENTS:?}"
: "${VAL_EVENTS:?}"
: "${TEST_EVENTS:?}"

CONDA_MODULE="${CONDA_MODULE:-miniforge3/25.3.0-3-a6hh}"
CONDA_ENV="${CONDA_ENV:-count-hdf}"
COHORT="${COHORT:-overnight}"
SEED="${SEED:-260922}"

module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

python "$CT_DIR/count_tracker_mother_direct.py" \
    --sim-root "$SIM_ROOT" \
    --metadata "$METADATA" \
    --cohort "$COHORT" --seed "$SEED" \
    --train-events "$TRAIN_EVENTS" \
    --val-events "$VAL_EVENTS" \
    --test-events "$TEST_EVENTS" \
    --hit-selection raw-time-lt-1e7-ns \
    --output "$CONDITIONS"
