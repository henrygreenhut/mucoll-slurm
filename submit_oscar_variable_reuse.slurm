#!/bin/bash
#SBATCH -J variable_reuse
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu3005
#SBATCH -o variable_reuse_%j.out
#SBATCH -e variable_reuse_%j.err

# Usage:
#   sbatch submit_oscar_variable_reuse.slurm \
#       {1v5|5v10|1v10|10v42} {main|null} [train|smoke]
#
# A fresh long-training diagnostic can be requested without changing the
# default production labels:
#   sbatch --export=ALL,RUN_TAG=min80,EPOCHS=120,MIN_EPOCHS=80 \
#       submit_oscar_variable_reuse.slurm 1v5 main

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 {1v5|5v10|1v10|10v42} {main|null} [train|smoke]"
    exit 2
fi
COMPARISON=$1
RUN=$2
MODE=${3:-train}
case "$COMPARISON" in
    1v5) K0=1; K1=5 ;;
    5v10) K0=5; K1=10 ;;
    1v10) K0=1; K1=10 ;;
    10v42) K0=10; K1=42 ;;
    *) echo "first argument must be 1v5, 5v10, 1v10, or 10v42"; exit 2 ;;
esac
case "$RUN" in
    main) NULL_ARG= ;;
    null) NULL_ARG=--null-test ;;
    *) echo "second argument must be main or null"; exit 2 ;;
esac
case "$MODE" in
    train) SMOKE_ARG= ;;
    smoke) SMOKE_ARG=--smoke ;;
    *) echo "third argument must be train or smoke"; exit 2 ;;
esac

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

MOTHER_STORE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores/gen_split_mothers_MUPLUS.h5
if [ ! -f "$MOTHER_STORE" ]; then
    echo "ERROR: missing $MOTHER_STORE"
    exit 1
fi

LABEL="n420_k${K0}_vs_k${K1}_scaled_lr1e-4_mseed1"
[ "$RUN" = null ] && LABEL="${LABEL}_null"
[ "$MODE" = smoke ] && LABEL="${LABEL}_smoke"
RUN_TAG=${RUN_TAG:-}
[ -n "$RUN_TAG" ] && LABEL="${LABEL}_${RUN_TAG}"
EPOCHS=${EPOCHS:-80}
MIN_EPOCHS=${MIN_EPOCHS:-0}

echo "label=$LABEL comparison=$COMPARISON run=$RUN mode=$MODE epochs=$EPOCHS min_epochs=$MIN_EPOCHS"
apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" \
    python -u pfn_variable_reuse_train.py \
        --mother-store "$MOTHER_STORE" \
        --label "$LABEL" \
        --reuse-k "$K0" "$K1" \
        --model-seed 1 \
        --epochs "$EPOCHS" \
        --min-epochs "$MIN_EPOCHS" \
        --max-minutes 1400 \
        $NULL_ARG \
        $SMOKE_ARG
