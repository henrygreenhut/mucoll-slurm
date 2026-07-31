#!/bin/bash
#SBATCH -J recok_train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --exclude=gpu3005
#SBATCH -o recok_train_%j.out
#SBATCH -e recok_train_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -ne 2 ]; then
    echo "usage: sbatch $0 {1v5|1v7|1v10|1v21} {main|null}" >&2
    exit 2
fi

COMPARISON=$1
RUN=$2
case "$COMPARISON" in
    1v5) K=5 ;;
    1v7) K=7 ;;
    1v10) K=10 ;;
    1v21) K=21 ;;
    *) echo "unsupported comparison: $COMPARISON" >&2; exit 2 ;;
esac
case "$RUN" in
    main) NULL_ARGS=() ;;
    null) NULL_ARGS=(--permute-labels) ;;
    *) echo "run must be main or null" >&2; exit 2 ;;
esac

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

TRAIN_SITE=/oscar/scratch/$USER/mucoll/envs/reco-train-site-v1
STORE_DIR=/oscar/scratch/$USER/mucoll/libtest/reco_variable_k_n420_stores
DATASET_TAG="variable_n420_k1_vs_k${K}"
LABEL="reco_${DATASET_TAG}_charged7_stabilized_dropout"
[ "$RUN" = null ] && LABEL="${LABEL}_null"

if [ ! -d "$TRAIN_SITE/matplotlib" ]; then
    echo "missing RECO training environment: $TRAIN_SITE" >&2
    exit 1
fi

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"
export PYTHONPATH="$TRAIN_SITE"
mkdir -p "$MPLCONFIGDIR"

apptainer exec --nv --bind /oscar:/oscar "$NGC_TENSORFLOW_CONTAINER" \
    python train_reco_libtest_pfn.py \
        --n-files 420 \
        --store-dir "$STORE_DIR" \
        --class-a k1 \
        --class-b "k${K}" \
        --dataset-tag "$DATASET_TAG" \
        --require-clean-code \
        --require-pfo-track-links \
        --train-events 2000 \
        --val-events 2000 \
        --test-events 2000 \
        --recipe stabilized_dropout \
        --label "$LABEL" \
        "${NULL_ARGS[@]}"
