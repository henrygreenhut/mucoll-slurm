#!/bin/bash
#SBATCH -J oscar_train_n420_variant
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 00:40:00
#SBATCH -o oscar_train_n420_variant_%x_%j.out
#SBATCH -e oscar_train_n420_variant_%x_%j.err

# OSCAR reproduction of the actual Perlmutter run that never got checked:
# submit_phi_half_long.slurm (commits d252f71/fed4e02/c96b236/dd05006/
# e3c54f2), the resumable shared-QOS follow-up to phi_half_scan_task.sh
# (816fb3c), itself a halved-Phi rerun after the original full-Phi
# arch_scan (job 56224614) hit the TF/XLA int32 overflow crash. Both use
# --arch energyflow throughout ("pfn and enf packages for everything"):
#
#   pfn -> latent-scale none (energyflow.archs.PFN, raw sum)   = halfphi_raw_large
#   efn -> latent-scale auto (energyflow.archs.EFN, normalized) = halfphi_scaled_large
#
# n_files=420, Phi=(100,100,128), F=(200,200,200), batch=4,
# units_per_epoch=500 -- matches submit_oscar_train.sh's existing
# oscar_raw_large exactly (that job's real provenance was misattributed to
# an earlier ad hoc debug scan; hyperparameters already line up with
# halfphi_raw_large). What's new here: the efn/scaled counterpart (never
# run on OSCAR before) and seed coverage for both (existing oscar_raw_large
# is seed=1 only, and sat at chance for its 5 completed epochs -- unclear
# if that's a real non-separation or just not enough epochs yet).
#
# --max-minutes 25 (not Perlmutter's up-to-7-hour shared-queue window):
# a first read on trend across seeds, not full convergence. At ~150s/epoch
# observed for this n_files=420 config, this budget covers ~8-10 epochs --
# resubmit the same label to keep going if a seed looks promising.
#
# Submit per (variant, seed):
#   sbatch submit_oscar_train_n420_variant.sh pfn 1
#   sbatch submit_oscar_train_n420_variant.sh efn 1

set -e
cd "$SLURM_SUBMIT_DIR"

VARIANT=$1
SEED=$2
if [ "$VARIANT" != "pfn" ] && [ "$VARIANT" != "efn" ]; then
    echo "usage: sbatch submit_oscar_train_n420_variant.sh {pfn|efn} <seed>"
    exit 1
fi
if [ -z "$SEED" ]; then
    echo "usage: sbatch submit_oscar_train_n420_variant.sh {pfn|efn} <seed>"
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

apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" python -u pfn_libtest_train.py \
    --label "$LABEL" \
    --norm1-store "$NORM1_STORE" \
    --norm42-store "$NORM42_STORE" \
    --n-files 420 \
    --units-per-epoch 500 \
    --batch-size 4 \
    --max-minutes 25 \
    --latent-scale "$LATENT_SCALE" \
    --phi-sizes 100,100,128 \
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
