#!/usr/bin/env bash
#SBATCH --job-name=count10_audit
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=node2336

set -euo pipefail

: "${CT_DIR:?export CT_DIR as the count_tracker directory}"
: "${MANIFEST:?export MANIFEST as sources.json}"
: "${DATA_ROOT:?export DATA_ROOT as the Norm42 input-array directory}"
: "${SOURCE_ID_ROOT:?export SOURCE_ID_ROOT as the source-ID sidecar directory}"
: "${AUDIT_OUTPUT:?export AUDIT_OUTPUT as a new output directory}"

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate count-hdf

python "$CT_DIR/norm42_train_10pct/audit_subset.py" \
  --manifest "$MANIFEST" \
  --data-root "$DATA_ROOT" \
  --source-id-root "$SOURCE_ID_ROOT" \
  --output "$AUDIT_OUTPUT"
