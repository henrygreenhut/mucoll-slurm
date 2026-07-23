#!/bin/bash
#SBATCH -J oscar_train_n420_variant_long
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 24:00:00
#SBATCH -o oscar_train_n420_variant_long_%x_%j.out
#SBATCH -e oscar_train_n420_variant_long_%x_%j.err

# Production follow-up to submit_oscar_train_n420_variant.sh's 25-minute
# debug-scale runs (6/6 hit the wall-clock cap after only 6-7 epochs,
# too shallow to tell raw-sum instability apart from scaled-sum stability
# at n420 scale -- the n42 comparison needed 8-15 epochs to show its clean
# split). Same labels, so this resumes each run's existing checkpoint
# rather than starting over.
#
# gpu partition has MaxTime=UNLIMITED (confirmed via `scontrol show
# partition gpu`), so this uses one long job per config instead of
# `submit_phi_half_long.slurm`'s repeated-resubmit pattern on Perlmutter's
# shared QOS -- --max-minutes 1400 (23h20m) leaves a ~40min buffer under
# the 24h walltime for store loading + the final disjoint/bootstrap
# evaluation, letting the training loop run to its own natural stop
# (early stop, patience=15, or the 200-epoch cap) rather than being cut
# off again. At ~150-190s/epoch observed, 200 epochs is ~9-10.5h --
# comfortably inside this window even in the worst case (no early stop).
#
# Only one job runs at a time anyway (QOSMaxCpuPerUserLimit on this
# partition), so submitting all 6 (raw/scaled x seed 1-3) back to back
# just queues them serially -- no downside to submitting all of them now.
#
# Optional --warmup-epochs/--clipnorm (args 3/4, both default 0 = off,
# reproducing the exact original behavior/label if omitted): when set,
# the label gets a _wN_cM suffix rather than resuming the plain
# oscar_n420_halfphi_{raw,scaled}_seed<N> checkpoint from the earlier
# short run. That's required, not just tidy -- this label's checkpoint
# already has ~1500+ optimizer steps in it (6 epochs from the 25-min
# run), so resuming it with warmup newly enabled would read the RESTORED
# iterations count into the warmup schedule, which is already well past
# any sane warmup window -- warmup would silently do nothing.
#
# warmup_epochs, not a raw step count: pfn_libtest_train.py resolves it to
# an exact step count itself from THIS run's own --units-per-epoch/
# --batch-size (logged, and recorded in that run's config.json) -- nobody
# has to hand-compute steps/epoch to pick this number.
#
# Resume per (variant, seed[, warmup_epochs, clipnorm]):
#   sbatch submit_oscar_train_n420_variant_long.sh pfn 1
#   sbatch submit_oscar_train_n420_variant_long.sh pfn 1 1 1.0
#   sbatch submit_oscar_train_n420_variant_long.sh efn 1

set -e
cd "$SLURM_SUBMIT_DIR"

VARIANT=$1
SEED=$2
WARMUP_EPOCHS=${3:-0}
CLIPNORM=${4:-0}
if [ "$VARIANT" != "pfn" ] && [ "$VARIANT" != "efn" ]; then
    echo "usage: sbatch submit_oscar_train_n420_variant_long.sh {pfn|efn} <seed> [warmup_epochs] [clipnorm]"
    exit 1
fi
if [ -z "$SEED" ]; then
    echo "usage: sbatch submit_oscar_train_n420_variant_long.sh {pfn|efn} <seed> [warmup_epochs] [clipnorm]"
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
    echo "ERROR: norm42 store not found at $NORM42_STORE -- has submit_make_norm42_store.sh finished?"
    exit 1
fi

if [ "$VARIANT" = "pfn" ]; then
    LATENT_SCALE="none"
    LABEL="oscar_n420_halfphi_raw_seed${SEED}"
else
    LATENT_SCALE="auto"
    LABEL="oscar_n420_halfphi_scaled_seed${SEED}"
fi
if [ "$WARMUP_EPOCHS" != "0" ] || [ "$CLIPNORM" != "0" ]; then
    LABEL="${LABEL}_w${WARMUP_EPOCHS}_c${CLIPNORM}"
fi

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
    --warmup-epochs "$WARMUP_EPOCHS" \
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
