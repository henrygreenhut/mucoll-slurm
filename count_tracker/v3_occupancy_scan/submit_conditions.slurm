#!/bin/bash
#SBATCH --job-name=count_occ_cond
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?submit one task per scan point}"
TAG="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" --scan "$SCAN_ROOT/scan.json" --index "$INDEX" --field tag)"
POINT="$SCAN_ROOT/$TAG"
if [ -f "$POINT/conditions/manifest.json" ]; then
    echo "$TAG conditions already exist"
    exit 0
fi

module load "${CONDA_MODULE:-miniforge3/25.3.0-3-a6hh}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-count-hdf}"
export PYTHONNOUSERSITE=1

python "$CT_DIR/count_tracker_conditions.py" \
    --manifest "$POINT/sources.json" \
    --hit-selection raw-time-lt-1e7-ns \
    --output "$POINT/conditions"
