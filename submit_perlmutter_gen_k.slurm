#!/bin/bash
#SBATCH -J gen_k
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -o perlmutter_gen_k_%j.out
#SBATCH -e perlmutter_gen_k_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 {5|42-synthetic|42-production} {main|null} [train|smoke]" >&2
    exit 2
fi

COMPARISON=$1
RUN=$2
MODE=${3:-train}
case "$COMPARISON" in
    5)
        K=5
        PROVENANCE=synthetic
        ;;
    42-synthetic)
        K=42
        PROVENANCE=synthetic
        ;;
    42-production)
        K=42
        PROVENANCE=production
        ;;
    *)
        echo "comparison must be 5, 42-synthetic, or 42-production" >&2
        exit 2
        ;;
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
UNIQUE_STORE="$STORE_DIR/gen_split_mothers_MUPLUS.h5"
if [ "$PROVENANCE" = production ]; then
    REUSE_STORE="$STORE_DIR/gen_norm42_MUPLUS.h5"
else
    REUSE_STORE="$STORE_DIR/gen_fixed_k${K}_MUPLUS.h5"
fi

for STORE in "$UNIQUE_STORE" "$REUSE_STORE"; do
    if [ ! -s "$STORE" ]; then
        echo "missing store: $STORE" >&2
        exit 1
    fi
done

LABEL="pm_n420_k1_vs_k${K}_${PROVENANCE}_scaled_lr1e-4_decay80_mseed1_v1"
[ "$RUN" = null ] && LABEL="${LABEL}_null"
[ "$MODE" = smoke ] && LABEL="${LABEL}_smoke"

COMMAND=(python -u pfn_libtest_train.py
    --label "$LABEL"
    --norm1-store "$UNIQUE_STORE"
    --norm42-store "$REUSE_STORE"
    --clone-factor "$K"
    --n-files 420
    --split-fracs 0.50 0.25 0.25
    --units-per-epoch 500
    --val-units 300
    --batch-size 4
    --gradient-accumulation-steps 1
    --features expanded
    --arch energyflow
    --latent-scale auto
    --phi-sizes 100,100,128
    --f-sizes 200,200,200
    --lr 1e-4
    --warmup-epochs 1
    --decay-epochs 80
    --min-lr 1e-6
    --clipnorm 0
    --latent-dropout 0
    --f-dropout 0
    --phi-l2 0
    --f-l2 0
    --data-seed 1701
    --model-seed 1
    --select-metric loss
    --min-delta 1e-4
    --min-delta-sigma 1
    --eval-point-units 300
    --eval-bootstrap-reps 0
    --skip-disjoint-check
    --progress-every 25
    --max-minutes 2820)

MIN_EPOCHS=0
if [ "$RUN" = main ]; then
    [ "$K" = 5 ] && MIN_EPOCHS=80
    COMMAND+=(--epochs 120 --patience 15 --min-epochs "$MIN_EPOCHS")
else
    COMMAND+=(--null-test --null-source reuse
        --epochs 40 --patience 8 --min-epochs 0)
fi

if [ "$MODE" = smoke ]; then
    MIN_EPOCHS=0
    COMMAND+=(--epochs 2 --patience 2 --min-epochs 0
        --units-per-epoch 4 --val-units 4 --skip-evaluation)
fi

echo "label=$LABEL"
echo "class 0: 420 native, unrotated source cycles"
echo "class 1: $((420 / K)) source cycles with $K fixed rotations/mother"
echo "reused-class provenance: $PROVENANCE ($REUSE_STORE)"
echo "minimum training epochs: $MIN_EPOCHS"
[ "$RUN" = null ] && echo "null: both labels use the fixed k=$K store"

if [ "${DIRECT_RUN:-0}" = 1 ]; then
    "${COMMAND[@]}"
else
    srun --cpu-bind=cores "${COMMAND[@]}"
fi
