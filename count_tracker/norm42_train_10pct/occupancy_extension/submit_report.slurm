#!/usr/bin/env bash
#SBATCH --job-name=count10_occ_report
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
: "${CT_DIR:?}"; : "${EXT_ROOT:?}"
module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate count-hdf
python "$CT_DIR/norm42_train_10pct/occupancy_extension/report.py" \
  --root "$EXT_ROOT" --output "$EXT_ROOT/report.json" --csv "$EXT_ROOT/report.csv"
