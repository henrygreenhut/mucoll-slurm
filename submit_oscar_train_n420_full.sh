#!/bin/bash
#SBATCH -J oscar_train_n420_full
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu3005
#SBATCH -o oscar_train_n420_full_%x_%j.out
#SBATCH -e oscar_train_n420_full_%x_%j.err

# --exclude=gpu3005: that node has hit CUDA_ERROR_ILLEGAL_ADDRESS on every
# job that's landed there so far (4+ occurrences across different configs/
# labels) -- baked in here rather than relying on remembering the CLI flag
# every time, since forgetting it once already wasted two job slots.

# The "everything new, except L2/dropout" n420 preset -- a named config
# (not another parametrized wrapper) since it bundles several axes at
# once deliberately, as a single headline comparison point against
# submit_oscar_train_n420_variant_long.sh's plain warmup+clip test:
#
#   features:      expanded (paper + loge/asinh_t/asinh_vz/asinh_vr) --
#                  all four fields (E/t/vx/vy/vz) were already in the
#                  ORIGINAL stores from day one, so this uses those, not
#                  the _v2 ones. "expanded" originally also included
#                  charge, the one field that needed the _v2 rebuild --
#                  dropped (see libtest_common.py's FEATURE_SETS comment):
#                  charge's marginal info over the existing PDG one-hot is
#                  small (only the e/mu particle-vs-antiparticle sign),
#                  not worth the _v2 stores' extra ~15% size/RAM -- which
#                  is exactly what thrashed this job (4228090) at its
#                  --mem ceiling the first time, where every charge-free
#                  n420 run before it fit comfortably in the same budget.
#   validation:    --val-units 1000 (vs 300), --select-metric loss (self-
#                  calibrating SEM-based selection instead of the fixed
#                  --min-delta, which is known to sit below val_auc's
#                  noise floor at small val_units).
#   training:      --warmup-epochs 1, --clipnorm defaults to 1.0 (matches
#                  the original seed2 run, for backward compat / resuming
#                  it unchanged) but is now optional $3 -- clipnorm=1.0
#                  turned out to be far too small next to raw-sum's actual
#                  gradient scale (train_loss starts ~100000+ vs ~0.69
#                  baseline): the n420 warmup-only diagnostic (no clip)
#                  climbed cleanly to AUC 0.70 by epoch 4, while seed2
#                  (warmup+clipnorm=1.0) sat flat/declining at AUC ~0.48-
#                  0.50 over the same epochs despite falling loss -- same
#                  collapse signature as n42's clip/both modes. Pass 0 for
#                  warmup-only, or a much larger value (e.g. 1000) once the
#                  n420 clip-threshold diagnostic confirms a working one.
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
# Optional $3/$4: clipnorm override (default 1.0, seed2's original value)
# and max-minutes override (default 1400). clipnorm=1.0 turned out to
# collapse training (see script header) -- pass 0 for warmup-only. Label
# gets a _c<value> suffix when overridden so it doesn't collide with an
# existing clipnorm=1.0 checkpoint under the same seed.
#
# Submit (variant defaults to pfn, clipnorm to 1.0, if omitted):
#   sbatch submit_oscar_train_n420_full.sh 1
#   sbatch submit_oscar_train_n420_full.sh 1 efn
#   sbatch submit_oscar_train_n420_full.sh 3 pfn 0 130   # warmup only, capped, for overnight queueing

set -e
cd "$SLURM_SUBMIT_DIR"

SEED=$1
VARIANT=${2:-pfn}
CLIPNORM=${3:-1.0}
MAX_MINUTES=${4:-1400}
if [ -z "$SEED" ]; then
    echo "usage: sbatch submit_oscar_train_n420_full.sh <seed> [pfn|efn] [clipnorm] [max_minutes]"
    exit 1
fi
if [ "$VARIANT" != "pfn" ] && [ "$VARIANT" != "efn" ]; then
    echo "usage: sbatch submit_oscar_train_n420_full.sh <seed> [pfn|efn] [clipnorm] [max_minutes]"
    exit 1
fi

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

NORM1_STORE="$HOME/mucoll/stores/gen_norm1_reconstructed_MUPLUS.h5"
NORM42_STORE="/oscar/scratch/$USER/mucoll/stores/gen_norm42_MUPLUS.h5"

if [ ! -f "$NORM1_STORE" ]; then
    echo "ERROR: norm1 store not found at $NORM1_STORE"
    exit 1
fi
if [ ! -f "$NORM42_STORE" ]; then
    echo "ERROR: norm42 store not found at $NORM42_STORE"
    exit 1
fi

if [ "$VARIANT" = "pfn" ]; then
    LATENT_SCALE="none"
else
    LATENT_SCALE="auto"
fi
LABEL="oscar_n420_full_${VARIANT}_seed${SEED}"
if [ "$CLIPNORM" != "1.0" ]; then
    LABEL="${LABEL}_c${CLIPNORM}"
fi

apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" python -u pfn_libtest_train.py \
    --label "$LABEL" \
    --norm1-store "$NORM1_STORE" \
    --norm42-store "$NORM42_STORE" \
    --n-files 420 \
    --units-per-epoch 500 \
    --batch-size 4 \
    --max-minutes "$MAX_MINUTES" \
    --latent-scale "$LATENT_SCALE" \
    --phi-sizes 100,100,128 \
    --f-sizes 200,200,200 \
    --arch energyflow \
    --features expanded \
    --val-units 1000 \
    --select-metric loss \
    --warmup-epochs 1 \
    --clipnorm "$CLIPNORM" \
    --seed "$SEED"

echo ""
echo "=== history ==="
hist="pfn_results/${LABEL}/history.csv"
if [ -f "$hist" ]; then
    cat "$hist"
else
    echo "NO history.csv -- did not complete even one epoch"
fi
