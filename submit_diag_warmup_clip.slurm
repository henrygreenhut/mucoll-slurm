#!/bin/bash
#SBATCH -J diag_warmup_clip
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 00:15:00
#SBATCH --exclude=gpu3005
#SBATCH -o diag_warmup_clip_%x_%j.out
#SBATCH -e diag_warmup_clip_%x_%j.err

# --exclude=gpu3005: that node has hit CUDA_ERROR_ILLEGAL_ADDRESS on every
# job that's landed there so far (it's part of gpu-debug too, not just
# gpu) -- one of this script's own two failed attempts was exactly this.

# Isolates WHICH of --warmup-epochs / --clipnorm is responsible for the
# hang seen in oscar_train_n420_full (job 4228090) and oscar_train_n420_
# variant_long (job 4228091) -- both froze at 0% GPU utilization right
# after "Compiled cluster using XLA!", never completing even one epoch,
# and warmup+clipnorm together are the only thing those two configs
# shared (one used expanded features + _v2 stores, the other didn't; that
# turned out to be a separate, already-fixed memory issue on the first
# job, not the cause of the hang itself).
#
# Deliberately at n42 scale (45-90s/epoch observed, vs n420's 150-253s)
# and gpu-debug (faster queue turnaround, 30min cap is plenty): a hang
# here would announce itself within a couple minutes past the normal
# per-epoch time, not require a 40-minute wait to be sure. --max-minutes 8
# leaves room for several healthy epochs while keeping a stuck run's
# wasted time small.
#
# Same --latent-scale none (raw sum, pfn/PFN variant) as both jobs that
# hung -- isolates the warmup/clipnorm question without also reopening
# the separate raw-vs-scaled-sum question.
#
# MODE:
#   warmup -> --warmup-epochs 1, no clipnorm
#   clip   -> --clipnorm 1.0, no warmup
#   both   -> both (reproduces the exact hung configuration)
#   none   -> neither (control -- expected to behave like every prior
#             successful n42 raw run, i.e. NOT hang)
#
# Submit one at a time (still one GPU job at a time regardless of queue):
#   sbatch submit_diag_warmup_clip.slurm warmup
#   sbatch submit_diag_warmup_clip.slurm clip
#   sbatch submit_diag_warmup_clip.slurm both
#   sbatch submit_diag_warmup_clip.slurm none

set -e
cd "$SLURM_SUBMIT_DIR"

MODE=$1
if [[ "$MODE" != "warmup" && "$MODE" != "clip" && "$MODE" != "both" && "$MODE" != "none" ]]; then
    echo "usage: sbatch submit_diag_warmup_clip.slurm {warmup|clip|both|none}"
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
LABEL="diag_warmup_clip_${MODE}"

apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" python -u pfn_libtest_train.py \
    --label "$LABEL" \
    --norm1-store "$NORM1_STORE" \
    --norm42-store "$NORM42_STORE" \
    --n-files 42 \
    --units-per-epoch 1000 \
    --val-units 300 \
    --batch-size 8 \
    --max-minutes 8 \
    --latent-scale none \
    --phi-sizes 200,200,256 \
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
