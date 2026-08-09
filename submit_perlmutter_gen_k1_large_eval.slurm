#!/bin/bash
#SBATCH -J gen_k1_eval
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 08:00:00
#SBATCH -o perlmutter_gen_k1_large_eval_%j.out
#SBATCH -e perlmutter_gen_k1_large_eval_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load tensorflow/2.15.0

python -u pfn_libtest_evaluate.py \
    --label pm_n420_k1_vs_k1_synthetic_eval5000_cycleboot200_v1 \
    --source-label pm_n420_k1_vs_k1_synthetic_scaled_lr1e-4_decay80_mseed1_v1 \
    --point-units 5000 \
    --bootstrap-reps 200 \
    --bootstrap-units 100 \
    --seed 3727 \
    --batch-size 4 \
    --max-minutes 450
