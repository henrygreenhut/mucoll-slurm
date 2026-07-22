#!/bin/bash
#SBATCH -J oscar_train
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 00:30:00
#SBATCH -o oscar_train_%x_%j.out
#SBATCH -e oscar_train_%x_%j.err

# First OSCAR training run: same validated architecture as Perlmutter's
# halved-Phi debug scan (job 56272947) -- Phi=(100,100,128), F=(200,200,200),
# batch=4, raw sum -- to get a directly comparable result on new hardware
# (L40S, 48GB VRAM) and a new software stack (NGC TensorFlow container via
# Apptainer, not a native module). --max-minutes 20 on the gpu-debug
# partition: this is a smoke test (does the whole pipeline run end-to-end
# here at all), not a real training run yet -- move to the `gpu` partition
# with a much longer window once this confirms clean.
#
# apptainer exec --nv (not `run`): non-interactive, runs one command inside
# the container and returns -- the right mode for a batch job. $NGC_TENSORFLOW_CONTAINER
# is set by `module load ngc-tensorflow-container/...`, confirmed working
# with GPU detection + h5py already present, no extra installs needed.
#
# Store paths passed explicitly: pfn_libtest_train.py's defaults use
# $PSCRATCH (a Perlmutter-only env var), so on OSCAR both --norm1-store and
# --norm42-store MUST be given explicitly or it silently falls back to
# looking in the current directory.
#
# --mem=56g (was 32g): libtest_common.Store eagerly loads the FULL store
# into RAM on construction -- norm42 alone is ~29GB (matching Perlmutter's
# gen_norm42_MUPLUS.h5), plus ~0.7GB norm1, plus TF/Apptainer/Python
# overhead. 32g OOM-killed the job right after "loading stores", before
# training even started (confirmed via sacct: OUT_OF_MEMORY, 1:56 elapsed,
# SLURM's own "Detected 1 oom_kill event" message). 56g leaves real margin.

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
    --label oscar_raw_large \
    --norm1-store "$NORM1_STORE" \
    --norm42-store "$NORM42_STORE" \
    --n-files 420 \
    --units-per-epoch 500 \
    --batch-size 4 \
    --max-minutes 20 \
    --latent-scale none \
    --phi-sizes 100,100,128 \
    --f-sizes 200,200,200 \
    --arch energyflow

echo ""
echo "=== history ==="
hist="pfn_results/oscar_raw_large/history.csv"
if [ -f "$hist" ]; then
    cat "$hist"
else
    echo "NO history.csv -- did not complete even one epoch"
fi
