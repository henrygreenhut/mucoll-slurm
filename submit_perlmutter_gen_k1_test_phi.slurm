#!/bin/bash
#SBATCH -J gen_k1_phi
#SBATCH -A m5197
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH --licenses=scratch
#SBATCH -o gen_k1_test_phi_%j.out
#SBATCH -e gen_k1_test_phi_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load tensorflow/2.15.0
python -u plot_gen_k1_test_phi.py
