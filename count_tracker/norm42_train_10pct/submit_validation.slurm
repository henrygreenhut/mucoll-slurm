#!/usr/bin/env bash
#SBATCH --job-name=count10_valid
#SBATCH --partition=gpu
#SBATCH --array=0-5%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --exclude=gpu3005

set -euo pipefail

: "${CT_DIR:?export CT_DIR as the count_tracker directory}"
: "${MODEL_ROOT:?export MODEL_ROOT as the completed 10pct model root}"
: "${PAPER1_ROOT:?export PAPER1_ROOT as the exact archived Paper1 checkout}"
: "${VALIDATION_OUTPUT:?export VALIDATION_OUTPUT as a new output directory}"

HITS="${HITS:-20000}"
SEED="${SEED:-42}"
MAX_ROUNDS="${MAX_ROUNDS:-100}"
collections=(VBC VEC ITBC ITEC OTBC OTEC)
collection=${collections[$SLURM_ARRAY_TASK_ID]}

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate genbib
export PYTHONNOUSERSITE=1

echo "collection=$collection hits=$HITS seed=$SEED max_rounds=$MAX_ROUNDS"
echo "model_root=$MODEL_ROOT"
echo "paper1_root=$PAPER1_ROOT"
nvidia-smi

python "$CT_DIR/norm42_train_10pct/validate_collection.py" \
  --collection "$collection" \
  --model-root "$MODEL_ROOT" \
  --paper1-root "$PAPER1_ROOT" \
  --hits "$HITS" \
  --seed "$SEED" \
  --max-rounds "$MAX_ROUNDS" \
  --output "$VALIDATION_OUTPUT/$collection"
