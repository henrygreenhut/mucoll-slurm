#!/bin/bash
#SBATCH -J gen_size
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu3005
#SBATCH -o gen_size_%j.out
#SBATCH -e gen_size_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -ne 2 ]; then
    echo "usage: sbatch $0 {840|1260} {main|null}" >&2
    exit 2
fi

N_FILES=$1
RUN=$2

case "$N_FILES" in
    840)
        BATCH_SIZE=4
        ACCUMULATION_STEPS=1
        ;;
    1260)
        BATCH_SIZE=2
        ACCUMULATION_STEPS=2
        ;;
    *)
        echo "N must be 840 or 1260" >&2
        exit 2
        ;;
esac

case "$RUN" in
    main)
        NULL_ARGS=()
        ;;
    null)
        NULL_ARGS=(--null-test)
        ;;
    *)
        echo "run must be main or null" >&2
        exit 2
        ;;
esac

LABEL="gen_n${N_FILES}_expanded_scaled_lr1e-4_effbs4_mseed1"
if [ "$RUN" = null ]; then
    LABEL="${LABEL}_null"
fi

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

NORM1_STORE="$HOME/mucoll/stores/gen_norm1_reconstructed_MUPLUS.h5"
NORM42_STORE="/oscar/scratch/$USER/mucoll/stores/gen_norm42_MUPLUS.h5"

for store in "$NORM1_STORE" "$NORM42_STORE"; do
    if [ ! -f "$store" ]; then
        echo "missing store: $store" >&2
        exit 1
    fi
done

echo "label=$LABEL"
echo "N=$N_FILES run=$RUN microbatch=$BATCH_SIZE accumulation=$ACCUMULATION_STEPS effective_batch=4"

apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" \
    python -u pfn_libtest_train.py \
        --label "$LABEL" \
        --outdir pfn_results \
        --norm1-store "$NORM1_STORE" \
        --norm42-store "$NORM42_STORE" \
        --n-files "$N_FILES" \
        --units-per-epoch 500 \
        --val-units 300 \
        --batch-size "$BATCH_SIZE" \
        --gradient-accumulation-steps "$ACCUMULATION_STEPS" \
        --epochs 80 \
        --patience 15 \
        --latent-scale auto \
        --phi-sizes 100,100,128 \
        --f-sizes 200,200,200 \
        --arch energyflow \
        --features expanded \
        --lr 1e-4 \
        --warmup-epochs 1 \
        --decay-epochs 30 \
        --min-lr 1e-6 \
        --clipnorm 0 \
        --data-seed 1701 \
        --model-seed 1 \
        --select-metric loss \
        --min-delta 1e-4 \
        --eval-point-units 300 \
        --eval-bootstrap-reps 0 \
        --skip-disjoint-check \
        --max-minutes 1400 \
        --progress-every 25 \
        "${NULL_ARGS[@]}"
