#!/bin/bash
# Build the immutable four-parent source manifest after recovery succeeds.
#SBATCH --job-name=count_disjoint_prepare
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${SHARED_SIM_ROOT:?}"; : "${RECOVERY_ROOT:?}"; : "${SCAN_ROOT:?}"
module load "${CONDA_MODULE:-miniforge3/25.3.0-3-a6hh}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-count-hdf}"
export PYTHONNOUSERSITE=1

python "$CT_DIR/source_disjoint_fullbx/prepare.py" \
    --shared-sim-root "$SHARED_SIM_ROOT" \
    --recovery-muminus-root "$RECOVERY_ROOT/MUMINUS" \
    --seed "${SOURCE_SEED:-12345}" \
    --cohort "${COHORT:-disjoint_fullbx}" \
    --output "$SCAN_ROOT"
