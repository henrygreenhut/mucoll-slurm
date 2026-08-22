#!/bin/bash
#SBATCH -J gen_k1_ablation
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu3005
#SBATCH -o gen_k1_ablation_%j.out
#SBATCH -e gen_k1_ablation_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 {no-muons-photons|no-energy-scales|no-particle-id} {main|null} [train|smoke]"
    exit 2
fi

ABLATION=$1
RUN=$2
MODE=${3:-train}

FEATURES=expanded
FILTER_ARGS=()
case "$ABLATION" in
    no-muons-photons)
        FILTER_ARGS=(--exclude-muons --exclude-photons)
        ;;
    no-energy-scales)
        FEATURES=expanded_no_energy_scales
        ;;
    no-particle-id)
        FEATURES=expanded_no_particle_id
        ;;
    *)
        echo "unknown ablation: $ABLATION"
        exit 2
        ;;
esac
case "$RUN" in
    main|null) ;;
    *) echo "run must be main or null"; exit 2 ;;
esac
case "$MODE" in
    train|smoke) ;;
    *) echo "mode must be train or smoke"; exit 2 ;;
esac

module load ngc-tensorflow-container/25.02-tf2-py3-azkb

STORE_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores
UNROTATED_STORE="$STORE_DIR/gen_split_mothers_MUPLUS.h5"
ROTATED_STORE="$STORE_DIR/gen_fixed_k1_MUPLUS.h5"
for store in "$UNROTATED_STORE" "$ROTATED_STORE"; do
    if [ ! -s "$store" ]; then
        echo "missing store: $store"
        exit 1
    fi
done

LABEL="n420_k1_${ABLATION}_scaled_lr1e-4_decay80_mseed1"
NULL_ARGS=()
if [ "$RUN" = null ]; then
    LABEL="${LABEL}_null"
    NULL_ARGS=(--null-test --null-source reuse)
fi
if [ "$MODE" = smoke ]; then
    LABEL="${LABEL}_smoke"
fi

COMMON_ARGS=(
    --label "$LABEL"
    --norm1-store "$UNROTATED_STORE"
    --norm42-store "$ROTATED_STORE"
    --clone-factor 1
    --n-files 420
    --split-fracs 0.50 0.25 0.25
    --batch-size 4
    --features "$FEATURES"
    --arch energyflow
    --latent-scale auto
    --phi-sizes 100,100,128
    --f-sizes 200,200,200
    --lr 1e-4
    --warmup-epochs 1
    --decay-epochs 80
    --min-lr 1e-6
    --clipnorm 0
    --data-seed 1701
    --model-seed 1
    --select-metric loss
    --min-delta 1e-4
    --progress-every 25
)

if [ "$MODE" = smoke ]; then
    RUN_ARGS=(
        --units-per-epoch 4
        --val-units 4
        --epochs 2
        --patience 2
        --skip-evaluation
    )
elif [ "$RUN" = null ]; then
    RUN_ARGS=(
        --units-per-epoch 500
        --val-units 300
        --epochs 20
        --patience 8
        --min-epochs 0
        --eval-point-units 300
        --eval-bootstrap-reps 0
        --skip-disjoint-check
        --max-minutes 1400
    )
else
    RUN_ARGS=(
        --units-per-epoch 500
        --val-units 300
        --epochs 55
        --patience 15
        --min-epochs 55
        --eval-point-units 300
        --eval-bootstrap-reps 0
        --skip-disjoint-check
        --max-minutes 1400
    )
fi

echo "ablation=$ABLATION run=$RUN mode=$MODE label=$LABEL"
echo "features=$FEATURES filters=${FILTER_ARGS[*]:-none}"
apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" \
    python -u pfn_libtest_train.py \
    "${COMMON_ARGS[@]}" \
    "${FILTER_ARGS[@]}" \
    "${NULL_ARGS[@]}" \
    "${RUN_ARGS[@]}"
