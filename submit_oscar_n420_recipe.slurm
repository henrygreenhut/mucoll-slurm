#!/bin/bash
#SBATCH -J n420_recipe
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu3005
#SBATCH -o n420_recipe_%j.out
#SBATCH -e n420_recipe_%j.err

# Usage:
#   sbatch submit_oscar_n420_recipe.slurm {scaled|raw} {1e-4|3e-4} MODEL_SEED
#   sbatch -p gpu-debug -t 00:15:00 submit_oscar_n420_recipe.slurm \
#       scaled 1e-4 1 smoke
#
# Every production run has the same shuffled source split and sampled data
# stream (--data-seed 1701). MODEL_SEED changes only network initialization.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

VARIANT=${1:?usage: $0 {scaled|raw} {1e-4|3e-4} MODEL_SEED [train|smoke]}
LR=${2:?usage: $0 {scaled|raw} {1e-4|3e-4} MODEL_SEED [train|smoke]}
MODEL_SEED=${3:?usage: $0 {scaled|raw} {1e-4|3e-4} MODEL_SEED [train|smoke]}
MODE=${4:-train}

case "$VARIANT" in
    scaled) LATENT_SCALE=auto ;;
    raw) LATENT_SCALE=none ;;
    *) echo "VARIANT must be scaled or raw"; exit 2 ;;
esac
case "$LR" in
    1e-4|3e-4) ;;
    *) echo "LR must be 1e-4 or 3e-4"; exit 2 ;;
esac
case "$MODE" in
    train|smoke) ;;
    *) echo "MODE must be train or smoke"; exit 2 ;;
esac

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

NORM1_STORE="$HOME/mucoll/stores/gen_norm1_reconstructed_MUPLUS.h5"
NORM42_STORE="/oscar/scratch/$USER/mucoll/stores/gen_norm42_MUPLUS.h5"
for store in "$NORM1_STORE" "$NORM42_STORE"; do
    if [ ! -f "$store" ]; then
        echo "ERROR: missing store $store"
        exit 1
    fi
done

LR_LABEL=${LR/./p}
LABEL="n420_recipe_bs4_${VARIANT}_lr${LR_LABEL}_mseed${MODEL_SEED}"
COMMON_ARGS=(
    --label "$LABEL"
    --norm1-store "$NORM1_STORE"
    --norm42-store "$NORM42_STORE"
    --n-files 420
    --batch-size 4
    --latent-scale "$LATENT_SCALE"
    --phi-sizes 100,100,128
    --f-sizes 200,200,200
    --arch energyflow
    --features paper
    --lr "$LR"
    --warmup-epochs 1
    --decay-epochs 30
    --min-lr 1e-6
    --clipnorm 0
    --data-seed 1701
    --model-seed "$MODEL_SEED"
    --select-metric loss
    --progress-every 25
)

if [ "$MODE" = smoke ]; then
    LABEL="${LABEL}_smoke"
    COMMON_ARGS[1]="$LABEL"
    RUN_ARGS=(
        --units-per-epoch 4
        --val-units 4
        --epochs 2
        --patience 2
        --eval-bootstrap-reps 1
        --skip-evaluation
    )
else
    RUN_ARGS=(
        --units-per-epoch 500
        --val-units 300
        --epochs 80
        --patience 15
        --eval-point-units 300
        --eval-bootstrap-reps 200
        --eval-bootstrap-units 100
        --max-minutes 1400
    )
fi

echo "label=$LABEL variant=$VARIANT lr=$LR data_seed=1701 model_seed=$MODEL_SEED mode=$MODE"
apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" \
    python -u pfn_libtest_train.py "${COMMON_ARGS[@]}" "${RUN_ARGS[@]}"
