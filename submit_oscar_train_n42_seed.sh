#!/bin/bash
#SBATCH -J oscar_train_n42_seed
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 00:40:00
#SBATCH -o oscar_train_n42_seed_%x_%j.out
#SBATCH -e oscar_train_n42_seed_%x_%j.err

# Seed sweep on the n42 reference config (submit_oscar_train_n42.sh), to
# check whether that run's epoch-0-then-collapse-to-constant-output result
# was a fluke of this specific seed/hardware/software combo, or a
# consistent property of this config on OSCAR. Ruled out first: feature
# set (current hardcoded FEATURE_NAMES is byte-identical to Perlmutter's
# "paper" set, confirmed via git history) and store pairing (norm1-
# reconstructed and norm42 stores are built from the identical file list,
# so cycle_ids can't misalign) -- so if this ALSO collapses across
# multiple seeds, that points to run-to-run instability in the raw-sum
# aggregation itself (the subject of the whole reuse-pressure investigation)
# rather than an OSCAR-specific data/pipeline bug.
#
# --seed only (not --model-seed): varies both TF weight init AND the
# data-sampling RNGs together, i.e. a genuinely different run, not just a
# different initialization of an otherwise-identical training sequence.
#
# Submit per seed:
#   sbatch submit_oscar_train_n42_seed.sh 2
#   sbatch submit_oscar_train_n42_seed.sh 3

set -e
cd "$SLURM_SUBMIT_DIR"

SEED=$1
if [ -z "$SEED" ]; then
    echo "usage: sbatch submit_oscar_train_n42_seed.sh <seed>"
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

LABEL="oscar_n42_paper_rawsum_seed${SEED}"

apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" python -u pfn_libtest_train.py \
    --label "$LABEL" \
    --norm1-store "$NORM1_STORE" \
    --norm42-store "$NORM42_STORE" \
    --n-files 42 \
    --units-per-epoch 1000 \
    --val-units 300 \
    --batch-size 8 \
    --max-minutes 25 \
    --latent-scale none \
    --phi-sizes 200,200,256 \
    --f-sizes 200,200,200 \
    --arch local \
    --seed "$SEED"

echo ""
echo "=== history ==="
hist="pfn_results/${LABEL}/history.csv"
if [ -f "$hist" ]; then
    cat "$hist"
else
    echo "NO history.csv -- did not complete even one epoch"
fi
