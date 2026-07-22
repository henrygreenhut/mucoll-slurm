#!/bin/bash
#SBATCH -J oscar_train_n42_variant
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 00:40:00
#SBATCH -o oscar_train_n42_variant_%x_%j.out
#SBATCH -e oscar_train_n42_variant_%x_%j.err

# Extends the n42 seed sweep (submit_oscar_train_n42_seed.sh, --arch local
# only) to the two REAL energyflow-package architectures, to test whether
# the seed-dependent collapse-to-constant-output failure seen with the raw
# sum (seeds 1/2 collapsed, seed 3 converged cleanly like the Perlmutter
# original) is specific to the raw/unnormalized latent sum, or shows up
# regardless of architecture:
#
#   pfn      -> energyflow.archs.PFN direct (build_pfn_energyflow),
#               raw unweighted sum, --latent-scale none. Textbook PFN.
#   efn      -> energyflow.archs.EFN direct (build_pfn_energyflow_scaled),
#               weighted sum with constant per-particle weight
#               z_i = latent_scale (1/median unit multiplicity via
#               --latent-scale auto) -- normalizes the latent vector to
#               O(1) regardless of how many particles are in a unit, the
#               opposite of the raw-sum instability mechanism. Requires
#               tf_keras (already pinned to 2.15.* on this env from the
#               earlier dependency fix); untested on OSCAR before this.
#
# Both use the real package classes throughout (never the local Keras
# reimplementation) per standing instruction: real energyflow, not a
# workaround.
#
# Submit per (variant, seed):
#   sbatch submit_oscar_train_n42_variant.sh pfn 1
#   sbatch submit_oscar_train_n42_variant.sh efn 1

set -e
cd "$SLURM_SUBMIT_DIR"

VARIANT=$1
SEED=$2
if [ "$VARIANT" != "pfn" ] && [ "$VARIANT" != "efn" ]; then
    echo "usage: sbatch submit_oscar_train_n42_variant.sh {pfn|efn} <seed>"
    exit 1
fi
if [ -z "$SEED" ]; then
    echo "usage: sbatch submit_oscar_train_n42_variant.sh {pfn|efn} <seed>"
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
    LABEL="oscar_n42_energyflow_pfn_seed${SEED}"
else
    LATENT_SCALE="auto"
    LABEL="oscar_n42_energyflow_efn_seed${SEED}"
fi

apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" python -u pfn_libtest_train.py \
    --label "$LABEL" \
    --norm1-store "$NORM1_STORE" \
    --norm42-store "$NORM42_STORE" \
    --n-files 42 \
    --units-per-epoch 1000 \
    --val-units 300 \
    --batch-size 8 \
    --max-minutes 25 \
    --latent-scale "$LATENT_SCALE" \
    --phi-sizes 200,200,256 \
    --f-sizes 200,200,200 \
    --arch energyflow \
    --seed "$SEED"

echo ""
echo "=== history ==="
hist="pfn_results/${LABEL}/history.csv"
if [ -f "$hist" ]; then
    cat "$hist"
else
    echo "NO history.csv -- did not complete even one epoch"
fi
