#!/usr/bin/env bash
# Rebuild one point's conditions from its immutable source manifest.
#SBATCH --job-name=count10_occ_cond
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${EXT_ROOT:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?}"
HELPER="$CT_DIR/norm42_train_10pct/occupancy_extension/plan_task.py"
TAG="$(python3 "$HELPER" --plan "$EXT_ROOT/plan.json" \
  --stage point --index "$INDEX" --field tag)"
POINT="$EXT_ROOT/$TAG"

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate count-hdf

python "$CT_DIR/count_tracker_conditions.py" \
  --manifest "$POINT/sources.json" \
  --hit-selection raw-time-lt-1e7-ns \
  --output "$POINT/conditions"
