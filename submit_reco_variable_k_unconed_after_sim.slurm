#!/bin/bash
#SBATCH -J recok_launch
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:10:00
#SBATCH --exclude=node2336
#SBATCH -o recok_unconed_launch_%j.out
#SBATCH -e recok_unconed_launch_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python3 submit_reco_variable_k_unconed.py
