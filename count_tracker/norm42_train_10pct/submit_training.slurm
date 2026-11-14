#!/usr/bin/env bash
#SBATCH --job-name=count10_train
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
: "${DATASET_ROOT:?export DATASET_ROOT as the prepared dataset root}"
: "${PAPER1_10PCT:?export PAPER1_10PCT as the exact archived Paper1 checkout}"
: "${MODEL_OUTPUT:?export MODEL_OUTPUT as a new model-root directory}"

collections=(VBC VEC ITBC ITEC OTBC OTEC)
collection=${collections[$SLURM_ARRAY_TASK_ID]}

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate genbib

echo "Collection: $collection"
echo "Dataset root: $DATASET_ROOT"
echo "Model root: $MODEL_OUTPUT"
nvidia-smi

python "$CT_DIR/norm42_train_10pct/train_collection.py" \
  --collection "$collection" \
  --dataset-root "$DATASET_ROOT" \
  --paper1-root "$PAPER1_10PCT" \
  --output-root "$MODEL_OUTPUT"
