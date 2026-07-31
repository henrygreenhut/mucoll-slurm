#!/bin/bash
#SBATCH -J diag_gen_sizes
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 00:45:00
#SBATCH --exclude=gpu3005
#SBATCH -o diag_gen_sizes_%j.out
#SBATCH -e diag_gen_sizes_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -eq 0 ]; then
    set -- 840 1260
fi

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

NORM1_STORE="$HOME/mucoll/stores/gen_norm1_reconstructed_MUPLUS.h5"
NORM42_STORE="/oscar/scratch/$USER/mucoll/stores/gen_norm42_MUPLUS.h5"
OUTDIR="/oscar/scratch/$USER/mucoll/libtest/gen_size_diagnostics/$SLURM_JOB_ID"

for store in "$NORM1_STORE" "$NORM42_STORE"; do
    if [ ! -f "$store" ]; then
        echo "missing store: $store" >&2
        exit 1
    fi
done

for n_files in "$@"; do
    if (( n_files <= 0 || n_files % 42 != 0 )); then
        echo "N must be a positive multiple of 42: $n_files" >&2
        exit 2
    fi

    label="diag_n${n_files}_scaled_lr1e-4"
    echo "===== N=$n_files ====="

    apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" \
        python -u pfn_libtest_train.py \
        --label "$label" \
        --outdir "$OUTDIR" \
        --norm1-store "$NORM1_STORE" \
        --norm42-store "$NORM42_STORE" \
        --n-files "$n_files" \
        --units-per-epoch 4 \
        --val-units 4 \
        --batch-size 4 \
        --epochs 1 \
        --patience 1 \
        --latent-scale auto \
        --phi-sizes 100,100,128 \
        --f-sizes 200,200,200 \
        --arch energyflow \
        --features expanded \
        --lr 1e-4 \
        --warmup-epochs 1 \
        --decay-epochs 30 \
        --min-lr 1e-6 \
        --clipnorm 0 \
        --data-seed 1701 \
        --model-seed 1 \
        --select-metric loss \
        --min-delta 1e-4 \
        --progress-every 1 \
        --skip-evaluation
done

echo "GEN size diagnostics completed"
