#!/bin/bash
#SBATCH -J fixed_reuse_train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu3005
#SBATCH -o fixed_reuse_train_%j.out
#SBATCH -e fixed_reuse_train_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "usage: $0 {1|5|7} {main|null} [train|smoke] [none|all|exclude-muons-above-gev]"
    exit 2
fi

K=$1
RUN=$2
MODE=${3:-train}
MUON_FILTER=${4:-none}
EXCLUDE_MUONS_ABOVE_GEV=0
EXCLUDE_ALL_MUONS=0
case "$MUON_FILTER" in
    none|0) ;;
    all) EXCLUDE_ALL_MUONS=1 ;;
    *) EXCLUDE_MUONS_ABOVE_GEV=$MUON_FILTER ;;
esac
case "$K" in
    1|5|7) ;;
    *) echo "reuse factor must be 1, 5, or 7"; exit 2 ;;
esac
case "$RUN" in
    main) NULL_ARG= ;;
    null) NULL_ARG=--null-test ;;
    *) echo "run must be main or null"; exit 2 ;;
esac
case "$MODE" in
    train|smoke) ;;
    *) echo "mode must be train or smoke"; exit 2 ;;
esac

module load ngc-tensorflow-container/25.02-tf2-py3-azkb

STORE_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores
NORM1_STORE="$STORE_DIR/gen_split_mothers_MUPLUS.h5"
REUSE_STORE="$STORE_DIR/gen_fixed_k${K}_MUPLUS.h5"
for store in "$NORM1_STORE" "$REUSE_STORE"; do
    if [ ! -f "$store" ]; then
        echo "ERROR: missing $store"
        exit 1
    fi
done

LABEL="n420_fixed_k1_vs_k${K}_scaled_lr1e-4_decay80_mseed1_pointonly"
[ "$EXCLUDE_ALL_MUONS" = 1 ] && LABEL="${LABEL}_exclude_all_muons"
[ "$EXCLUDE_ALL_MUONS" = 0 ] && [ "$EXCLUDE_MUONS_ABOVE_GEV" != 0 ] && \
    LABEL="${LABEL}_exclude_muons_gt${EXCLUDE_MUONS_ABOVE_GEV}gev"
[ "$RUN" = null ] && LABEL="${LABEL}_null"
[ "$MODE" = smoke ] && LABEL="${LABEL}_smoke"

COMMON_ARGS=(
    --label "$LABEL"
    --norm1-store "$NORM1_STORE"
    --norm42-store "$REUSE_STORE"
    --clone-factor "$K"
    --n-files 420
    --batch-size 4
    --latent-scale auto
    --phi-sizes 100,100,128
    --f-sizes 200,200,200
    --arch energyflow
    --features expanded
    --exclude-muons-above-gev "$EXCLUDE_MUONS_ABOVE_GEV"
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
[ "$EXCLUDE_ALL_MUONS" = 1 ] && COMMON_ARGS+=(--exclude-muons)
[ "$RUN" = null ] && COMMON_ARGS+=(--null-test --null-source reuse)

if [ "$MODE" = smoke ]; then
    RUN_ARGS=(
        --units-per-epoch 4
        --val-units 4
        --epochs 2
        --patience 2
        --eval-bootstrap-reps 0
        --skip-evaluation
    )
elif [ "$RUN" = null ]; then
    NULL_EPOCHS=40
    NULL_MAX_MINUTES=1400
    if [ "$K" = 1 ]; then
        NULL_EPOCHS=20
        NULL_MAX_MINUTES=225
    fi
    RUN_ARGS=(
        --units-per-epoch 500
        --val-units 300
        --epochs "$NULL_EPOCHS"
        --patience 8
        --min-epochs 0
        --eval-point-units 300
        --eval-bootstrap-reps 0
        --skip-disjoint-check
        --max-minutes "$NULL_MAX_MINUTES"
    )
elif [ "$K" = 1 ]; then
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
else
    RUN_ARGS=(
        --units-per-epoch 500
        --val-units 300
        --epochs 120
        --patience 15
        --min-epochs 80
        --eval-point-units 300
        --eval-bootstrap-reps 0
        --skip-disjoint-check
        --max-minutes 1400
    )
fi

echo "label=$LABEL native_unrotated_class=true class_a=420_cycles class_b=$((420 / K))_fixed_rotated_cycles"
if [ "$EXCLUDE_ALL_MUONS" = 1 ]; then
    echo "muon filter: remove every particle with |PDG|=13"
else
    echo "exclude |PDG|=13 above: $EXCLUDE_MUONS_ABOVE_GEV GeV (0 = disabled)"
fi
apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" \
    python -u pfn_libtest_train.py "${COMMON_ARGS[@]}" "${RUN_ARGS[@]}"
