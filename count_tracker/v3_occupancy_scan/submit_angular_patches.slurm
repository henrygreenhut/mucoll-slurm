#!/bin/bash
# Build angular-patch stores from completed regional reconstruction outputs.
#SBATCH --job-name=count_patch_build
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

: "${CT_DIR:?export CT_DIR or pass it with sbatch --export}"
: "${SCAN_ROOT:?export SCAN_ROOT or pass it with sbatch --export}"
: "${PATCH_ROOT:?export PATCH_ROOT or pass it with sbatch --export}"

POINT="${POINT:-n1666}"
PHI_SECTORS="${PHI_SECTORS:-72}"
INCLUDE_CHI2_NDF="${INCLUDE_CHI2_NDF:-0}"
read -r -a SPLIT_COUNT_VALUES <<< "${SPLIT_COUNTS:-3 1 1}"
if [ "${#SPLIT_COUNT_VALUES[@]}" -ne 3 ]; then
    echo "SPLIT_COUNTS must contain TRAIN VAL TEST counts" >&2
    exit 1
fi

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate count-hdf

args=(
    --scan-root "$SCAN_ROOT"
    --point "$POINT"
    --phi-sectors "$PHI_SECTORS"
    --split-counts "${SPLIT_COUNT_VALUES[@]}"
    --output "$PATCH_ROOT"
)
if [ "$INCLUDE_CHI2_NDF" = 1 ]; then
    args+=(--include-chi2-ndf)
elif [ "$INCLUDE_CHI2_NDF" != 0 ]; then
    echo "INCLUDE_CHI2_NDF must be 0 or 1" >&2
    exit 1
fi

python "$CT_DIR/v3_occupancy_scan/build_angular_patches.py" "${args[@]}"
