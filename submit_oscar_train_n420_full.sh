#!/bin/bash
#SBATCH -J oscar_train_n420_full
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 24:00:00
#SBATCH -o oscar_train_n420_full_%x_%j.out
#SBATCH -e oscar_train_n420_full_%x_%j.err

# The "everything new, except L2/dropout" n420 preset -- a named config
# (not another parametrized wrapper) since it bundles several axes at
# once deliberately, as a single headline comparison point against
# submit_oscar_train_n420_variant_long.sh's plain warmup+clip test:
#
#   features:      expanded (paper + loge/asinh_t/asinh_vz/asinh_vr/
#                  charge) -- needs the _v2 stores (charge only exists
#                  there); run submit_make_norm42_store.sh MUPLUS and
#                  submit_reconstruct_unrotated.sh MUPLUS first if they
#                  don't exist yet.
#   validation:    --val-units 1000 (vs 300), --select-metric loss (self-
#                  calibrating SEM-based selection instead of the fixed
#                  --min-delta, which is known to sit below val_auc's
#                  noise floor at small val_units).
#   training:      --warmup-epochs 1 --clipnorm 1.0 -- same values as
#                  submit_oscar_train_n420_variant_long.sh's warmup/clip
#                  test, so that job and this one differ by EXACTLY
#                  "features + validation overhaul", isolating what those
#                  two add on top of warmup+clip alone.
#   NOT included:  --latent-dropout/--f-dropout/--phi-l2/--f-l2 all stay
#                  at their 0 (off) defaults -- regularizers for
#                  overfitting, which isn't the problem being diagnosed
#                  here (see chat: the model isn't overfitting, it's
#                  failing to fit anything in the collapsed cases).
#
# Variant defaults to pfn (raw sum): the one Perlmutter data (A0_n420_
# rawsum_disjoint) showed CAN find real signal (0.977 peak) at n420,
# unlike scaled sum which never separated at all in either Perlmutter
# n420 run found. Pass efn explicitly if you want the scaled version
# instead/also.
#
# Submit (variant defaults to pfn if omitted):
#   sbatch submit_oscar_train_n420_full.sh 1
#   sbatch submit_oscar_train_n420_full.sh 1 efn

set -e
cd "$SLURM_SUBMIT_DIR"

SEED=$1
VARIANT=${2:-pfn}
if [ -z "$SEED" ]; then
    echo "usage: sbatch submit_oscar_train_n420_full.sh <seed> [pfn|efn]"
    exit 1
fi
if [ "$VARIANT" != "pfn" ] && [ "$VARIANT" != "efn" ]; then
    echo "usage: sbatch submit_oscar_train_n420_full.sh <seed> [pfn|efn]"
    exit 1
fi

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

NORM1_STORE="$HOME/mucoll/stores/gen_norm1_reconstructed_MUPLUS_v2.h5"
NORM42_STORE="/oscar/scratch/$USER/mucoll/stores/gen_norm42_MUPLUS_v2.h5"

if [ ! -f "$NORM1_STORE" ]; then
    echo "ERROR: _v2 norm1 store not found at $NORM1_STORE"
    echo "  (run: sbatch submit_reconstruct_unrotated.sh MUPLUS)"
    exit 1
fi
if [ ! -f "$NORM42_STORE" ]; then
    echo "ERROR: _v2 norm42 store not found at $NORM42_STORE"
    echo "  (run: sbatch submit_make_norm42_store.sh MUPLUS)"
    exit 1
fi

if [ "$VARIANT" = "pfn" ]; then
    LATENT_SCALE="none"
else
    LATENT_SCALE="auto"
fi
LABEL="oscar_n420_full_${VARIANT}_seed${SEED}"

apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" python -u pfn_libtest_train.py \
    --label "$LABEL" \
    --norm1-store "$NORM1_STORE" \
    --norm42-store "$NORM42_STORE" \
    --n-files 420 \
    --units-per-epoch 500 \
    --batch-size 4 \
    --max-minutes 1400 \
    --latent-scale "$LATENT_SCALE" \
    --phi-sizes 100,100,128 \
    --f-sizes 200,200,200 \
    --arch energyflow \
    --features expanded \
    --val-units 1000 \
    --select-metric loss \
    --warmup-epochs 1 \
    --clipnorm 1.0 \
    --seed "$SEED"

echo ""
echo "=== history ==="
hist="pfn_results/${LABEL}/history.csv"
if [ -f "$hist" ]; then
    cat "$hist"
else
    echo "NO history.csv -- did not complete even one epoch"
fi
