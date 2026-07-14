#!/bin/bash
#SBATCH -J libtest_eval
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH --output=libtest_eval_%x_%j.out
#SBATCH --error=libtest_eval_%x_%j.err

# One resumable, fractionally charged paired-cycle-bootstrap evaluation window.
# Env overrides: LABEL (evaluation output label), TRAIN_ARGS (evaluator args)

set -e
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

LABEL=${LABEL:?set LABEL to the new evaluation label}
TRAIN_ARGS=${TRAIN_ARGS:-}

echo "evaluation label=$LABEL args='$TRAIN_ARGS'"
python pfn_libtest_evaluate.py \
    --label "$LABEL" \
    --max-minutes 25 \
    $TRAIN_ARGS
