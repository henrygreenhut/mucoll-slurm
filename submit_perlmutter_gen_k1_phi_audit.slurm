#!/bin/bash
#SBATCH -J gen_k1_phi_audit
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -o perlmutter_gen_k1_phi_audit_%j.out
#SBATCH -e perlmutter_gen_k1_phi_audit_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load tensorflow/2.15.0
python -u audit_gen_k1_phi.py
