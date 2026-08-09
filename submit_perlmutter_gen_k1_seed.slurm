#!/bin/bash
#SBATCH -J gen_k1_seed
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 12:00:00
#SBATCH -o perlmutter_gen_k1_seed_%j.out
#SBATCH -e perlmutter_gen_k1_seed_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "usage: $0 {baseline|no-phi} {1|2|3} {main|null} [train|smoke]" >&2
    exit 2
fi

VARIANT=$1
MODEL_SEED=$2
RUN=$3
MODE=${4:-train}

case "$VARIANT" in
    baseline)
        FEATURES=expanded
        FEATURE_LABEL=expanded
        ;;
    no-phi)
        FEATURES=expanded_no_phi
        FEATURE_LABEL=no_phi
        ;;
    *)
        echo "variant must be baseline or no-phi" >&2
        exit 2
        ;;
esac
case "$MODEL_SEED" in
    1|2|3) ;;
    *) echo "model seed must be 1, 2, or 3" >&2; exit 2 ;;
esac
case "$RUN" in
    main|null) ;;
    *) echo "run must be main or null" >&2; exit 2 ;;
esac
case "$MODE" in
    train|smoke) ;;
    *) echo "mode must be train or smoke" >&2; exit 2 ;;
esac

module load tensorflow/2.15.0

STORE_DIR="$PSCRATCH/mucoll/libtest/stores"
NATIVE_STORE="$STORE_DIR/gen_split_mothers_MUPLUS.h5"
SYNTHETIC_STORE="$STORE_DIR/gen_fixed_k1_MUPLUS.h5"
for store in "$NATIVE_STORE" "$SYNTHETIC_STORE"; do
    if [ ! -s "$store" ]; then
        echo "missing store: $store" >&2
        exit 1
    fi
done

LABEL="pm_n420_k1_vs_k1_synthetic_${FEATURE_LABEL}_scaled_lr1e-4_decay80_mseed${MODEL_SEED}_audit_v1"
[ "$RUN" = null ] && LABEL="${LABEL}_null"
NULL_ARGS=()
[ "$RUN" = null ] && NULL_ARGS=(--null-test --null-source reuse)

RUN_ARGS=(
    --epochs 55
    --patience 15
    --min-epochs 55
    --units-per-epoch 500
    --val-units 300
    --eval-point-units 300
    --eval-bootstrap-reps 0
    --max-minutes 690
)
if [ "$RUN" = null ]; then
    RUN_ARGS=(
        --epochs 20
        --patience 8
        --min-epochs 0
        --units-per-epoch 500
        --val-units 300
        --eval-point-units 300
        --eval-bootstrap-reps 0
        --max-minutes 225
    )
fi
if [ "$MODE" = smoke ]; then
    LABEL="${LABEL}_smoke"
    RUN_ARGS=(
        --epochs 2
        --patience 2
        --min-epochs 0
        --units-per-epoch 4
        --val-units 4
        --eval-bootstrap-reps 0
        --max-minutes 25
        --skip-evaluation
    )
fi

echo "label=$LABEL"
echo "variant=$VARIANT features=$FEATURES model_seed=$MODEL_SEED run=$RUN mode=$MODE"

python -u pfn_libtest_train.py \
    --label "$LABEL" \
    --norm1-store "$NATIVE_STORE" \
    --norm42-store "$SYNTHETIC_STORE" \
    --clone-factor 1 \
    --n-files 420 \
    --split-fracs 0.50 0.25 0.25 \
    --batch-size 4 \
    --gradient-accumulation-steps 1 \
    --features "$FEATURES" \
    --arch energyflow \
    --latent-scale auto \
    --phi-sizes 100,100,128 \
    --f-sizes 200,200,200 \
    --lr 1e-4 \
    --warmup-epochs 1 \
    --decay-epochs 80 \
    --min-lr 1e-6 \
    --clipnorm 0 \
    --latent-dropout 0 \
    --f-dropout 0 \
    --phi-l2 0 \
    --f-l2 0 \
    --data-seed 1701 \
    --model-seed "$MODEL_SEED" \
    --select-metric loss \
    --min-delta 1e-4 \
    --min-delta-sigma 1 \
    --skip-disjoint-check \
    --progress-every 25 \
    "${NULL_ARGS[@]}" \
    "${RUN_ARGS[@]}"
