#!/bin/bash
#SBATCH -J diag_n420_step
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 00:30:00
#SBATCH --exclude=gpu3005
#SBATCH -o diag_n420_step_%j.out
#SBATCH -e diag_n420_step_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

DATA_ROOT="/oscar/data/mleblan6/mucoll/$USER/mucoll"
NORM1_STORE="$HOME/mucoll/stores/gen_norm1_reconstructed_MUPLUS.h5"
NORM42_STORE="/oscar/scratch/$USER/mucoll/stores/gen_norm42_MUPLUS.h5"
NORM_STATS="$DATA_ROOT/mucoll-slurm/pfn_results/oscar_n420_halfphi_raw_seed1_w1_c0/norm_stats.json"

for path in "$NORM1_STORE" "$NORM42_STORE" "$NORM_STATS"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: required input not found: $path"
        exit 1
    fi
done

apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" \
    python -u diagnose_gen_n420_step.py \
        --norm1-store "$NORM1_STORE" \
        --norm42-store "$NORM42_STORE" \
        --norm-stats "$NORM_STATS" \
        --batch-sizes 1 2 4 \
        --steps 2
