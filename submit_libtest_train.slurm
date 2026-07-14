#!/bin/bash
#SBATCH -J libtest_train
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH --output=libtest_train_%x_%j.out
#SBATCH --error=libtest_train_%x_%j.err

# One 30-minute, fractionally charged shared-QOS training window; checkpoints
# and resumes via --label, so chain several jobs to complete a training:
#
#   jid=$(sbatch --parsable --export=ALL,LABEL=A0_n42 submit_libtest_train.slurm)
#   for i in 1 2 3 4 5; do
#     jid=$(sbatch --parsable --dependency=afterany:$jid \
#           --export=ALL,LABEL=A0_n42 submit_libtest_train.slurm)
#   done
#
# Needs only the NERSC tensorflow module (no pip installs).
# Env overrides: LABEL (required-ish), TRAIN_ARGS, STORE_DIR, POLARITY

set -e
cd "$SLURM_SUBMIT_DIR"

module load tensorflow

LABEL=${LABEL:-A0_n42}
TRAIN_ARGS=${TRAIN_ARGS:-}
POLARITY=${POLARITY:-MUPLUS}
STORE_DIR=${STORE_DIR:-$PSCRATCH/mucoll/libtest/stores}

echo "label=$LABEL args='$TRAIN_ARGS' polarity=$POLARITY"

python pfn_libtest_train.py \
    --norm1-store "$STORE_DIR/gen_norm1_${POLARITY}.h5" \
    --norm42-store "$STORE_DIR/gen_norm42_${POLARITY}.h5" \
    --label "$LABEL" \
    --max-minutes 25 \
    $TRAIN_ARGS
