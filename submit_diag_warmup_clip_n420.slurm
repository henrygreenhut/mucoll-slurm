#!/bin/bash
#SBATCH -J diag_warmup_clip_n420
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 00:25:00
#SBATCH --exclude=gpu3005
#SBATCH -o diag_warmup_clip_n420_%x_%j.out
#SBATCH -e diag_warmup_clip_n420_%x_%j.err

# n420-scale version of submit_diag_warmup_clip.slurm: the n42 version
# (all 4 modes) showed NO hang at all -- every mode, including "both",
# completed multiple epochs at normal speed. Since the actual hangs
# (jobs 4228090, 4228091) were both at n420 scale, n42 clearing doesn't
# resolve the question, just narrows it to something scale-dependent --
# most likely n420's much larger and more variable per-unit particle
# counts (bigger, more varied N per batch) interacting with XLA
# recompilation under the warmup schedule's ever-changing --iterations--
# dependent learning rate (see chat notes on this hypothesis). This
# script tests the same 4 modes at the scale that actually failed.
#
# --max-minutes 18 (vs n42's 8): n420 epochs take 150-253s (vs n42's
# 45-90s) in every prior healthy run, so 18 min leaves room for several
# epochs if healthy -- a hang will still reveal itself in a couple
# minutes past that per-epoch baseline, not require the original 40-
# minute wait to be sure.
#
# Same latent-scale none (raw/pfn, matching both jobs that hung) and
# --phi-sizes 100,100,128 (the halved-Phi n420 config) as every other
# n420 script -- isolates warmup/clip specifically, not the raw-vs-
# scaled-sum question or the Phi-width/int32-overflow question.
#
# MODE: warmup | clip | both | none -- same meaning as the n42 version.
#
# Submit one at a time:
#   sbatch submit_diag_warmup_clip_n420.slurm both
#   sbatch submit_diag_warmup_clip_n420.slurm warmup
#   sbatch submit_diag_warmup_clip_n420.slurm clip
#   sbatch submit_diag_warmup_clip_n420.slurm none

set -e
cd "$SLURM_SUBMIT_DIR"

MODE=$1
if [[ "$MODE" != "warmup" && "$MODE" != "clip" && "$MODE" != "both" && "$MODE" != "none" ]]; then
    echo "usage: sbatch submit_diag_warmup_clip_n420.slurm {warmup|clip|both|none}"
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

case "$MODE" in
    warmup) WARMUP_EPOCHS=1;   CLIPNORM=0 ;;
    clip)   WARMUP_EPOCHS=0;   CLIPNORM=1.0 ;;
    both)   WARMUP_EPOCHS=1;   CLIPNORM=1.0 ;;
    none)   WARMUP_EPOCHS=0;   CLIPNORM=0 ;;
esac
LABEL="diag_warmup_clip_n420_${MODE}"

apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" python -u pfn_libtest_train.py \
    --label "$LABEL" \
    --norm1-store "$NORM1_STORE" \
    --norm42-store "$NORM42_STORE" \
    --n-files 420 \
    --units-per-epoch 500 \
    --batch-size 4 \
    --max-minutes 18 \
    --latent-scale none \
    --phi-sizes 100,100,128 \
    --f-sizes 200,200,200 \
    --arch energyflow \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --clipnorm "$CLIPNORM" \
    --seed 1

echo ""
echo "=== history ==="
hist="pfn_results/${LABEL}/history.csv"
if [ -f "$hist" ]; then
    cat "$hist"
else
    echo "NO history.csv -- did not complete even one epoch"
fi
