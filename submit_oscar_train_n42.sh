#!/bin/bash
#SBATCH -J oscar_train_n42
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 00:40:00
#SBATCH -o oscar_train_n42_%x_%j.out
#SBATCH -e oscar_train_n42_%x_%j.err

# Reproduction of the TRUE original n_files=42 reference run
# (A0_n42_paper_rawsum, Perlmutter) on OSCAR: n_files=42, batch=8, full
# Phi=(200,200,256)/F=(200,200,200), units_per_epoch=1000, raw sum
# (latent_scale=none), --arch local (that run predates --arch existing,
# so it used the only build available, now called "local").
#
# Caveat, not fixable without restoring old code: that reference run used
# "features": "paper" (a feature-extraction option no longer present in
# the current libtest_common.py, which only implements the "adapted"
# 9-feature set). This matches every other hyperparameter exactly but
# will use the current feature set -- a meaningful comparison at matching
# hyperparameters, not a byte-for-byte replay.
#
# At n_files=42 (vs 420), typical unit size is roughly 10x smaller than
# what drove the int32 overflow bug all session -- comfortably clear of
# that ceiling even at batch=8/width=256, so no halved-Phi workaround
# needed here.
#
# -p gpu (not gpu-debug): running alongside submit_oscar_train.sh
# (n_files=420 config) at the same time -- each store load alone needs
# ~30GB (see that script's --mem comment), so two concurrent jobs both
# drawing from gpu-debug's QOS cap (mem=96G, aggregated across a user's
# concurrent jobs in that QOS) would be uncomfortably tight together.
# Regular gpu partition has its own separate, much larger cap (mem=192G),
# so running one job per partition avoids them competing for the same
# budget entirely.

set -e
cd "$SLURM_SUBMIT_DIR"

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

apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" python -u pfn_libtest_train.py \
    --label oscar_n42_paper_rawsum \
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
    --arch local

echo ""
echo "=== history ==="
hist="pfn_results/oscar_n42_paper_rawsum/history.csv"
if [ -f "$hist" ]; then
    cat "$hist"
else
    echo "NO history.csv -- did not complete even one epoch"
fi
