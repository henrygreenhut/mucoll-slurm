#!/bin/bash
# Prepare one full-library Norm1 event: every complete MUPLUS and MUMINUS
# cycle exactly once, with the model-training raw-time selection.
#SBATCH --job-name=count_n1_fullprep
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
set -euo pipefail

: "${CT_DIR:?}"
: "${SIM_ROOT:?}"
: "${METADATA:?}"
: "${CONDITIONS:?}"

CONDA_MODULE="${CONDA_MODULE:-miniforge3/25.3.0-3-a6hh}"
CONDA_ENV="${CONDA_ENV:-count-hdf}"
COHORT="${COHORT:-fullsample}"

module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

python "$CT_DIR/count_tracker_mother_direct.py" \
    --sim-root "$SIM_ROOT" \
    --metadata "$METADATA" \
    --cohort "$COHORT" \
    --all-cycles \
    --hit-selection raw-time-lt-1e7-ns \
    --output "$CONDITIONS"
