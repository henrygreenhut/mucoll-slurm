#!/bin/bash
#SBATCH -J k10_k42
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu3005
#SBATCH -o variable_k10_k42_%j.out
#SBATCH -e variable_k10_k42_%j.err

# Usage:
#   sbatch submit_oscar_variable_k10_k42.slurm {main|null} [train|smoke]

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 {main|null} [train|smoke]"
    exit 2
fi
RUN=$1
MODE=${2:-train}
case "$RUN" in
    main) NULL_ARG= ;;
    null) NULL_ARG=--null-test ;;
    *) echo "first argument must be main or null"; exit 2 ;;
esac
case "$MODE" in
    train) SMOKE_ARG= ;;
    smoke) SMOKE_ARG=--smoke ;;
    *) echo "second argument must be train or smoke"; exit 2 ;;
esac

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

MOTHER_STORE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores/gen_split_mothers_MUPLUS.h5
if [ ! -f "$MOTHER_STORE" ]; then
    echo "ERROR: missing $MOTHER_STORE"
    exit 1
fi

LABEL="n420_k10_vs_k42_scaled_lr1e-4_mseed1"
[ "$RUN" = null ] && LABEL="${LABEL}_null"
[ "$MODE" = smoke ] && LABEL="${LABEL}_smoke"

echo "label=$LABEL run=$RUN mode=$MODE"
apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" \
    python -u pfn_variable_reuse_train.py \
        --mother-store "$MOTHER_STORE" \
        --label "$LABEL" \
        --reuse-k 10 42 \
        --model-seed 1 \
        --max-minutes 1400 \
        $NULL_ARG \
        $SMOKE_ARG
