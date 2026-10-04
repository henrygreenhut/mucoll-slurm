#!/bin/bash
# Prepare direct SIM/COUNT sensor conditions from an explicit source manifest.
# The Python writer publishes CONDITIONS atomically only after every event passes.
#SBATCH --job-name=count_conditions
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail

: "${CT_DIR:?}"
: "${MANIFEST:?}"
: "${CONDITIONS:?}"

CONDA_MODULE="${CONDA_MODULE:-miniforge3/25.3.0-3-a6hh}"
CONDA_ENV="${CONDA_ENV:-count-hdf}"
HIT_SELECTION="${HIT_SELECTION:-all-stored}"

module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1

python "$CT_DIR/count_tracker_conditions.py" \
    --manifest "$MANIFEST" \
    --hit-selection "$HIT_SELECTION" \
    --output "$CONDITIONS"
