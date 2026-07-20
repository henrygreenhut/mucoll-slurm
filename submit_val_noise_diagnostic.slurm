#!/bin/bash
#SBATCH -J val_noise
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 00:15:00
#SBATCH --output=val_noise_%x_%j.out
#SBATCH --error=val_noise_%x_%j.err

# Inference-only, ~a few minutes: measures val AUC's sampling noise for an
# existing checkpoint. Run via sbatch (not directly on the login node --
# `python` there resolves to system Python 2.7, which can't parse this
# file's f-strings; `module load tensorflow` inside a plain `ssh host "..."`
# command doesn't initialize the module system either, only inside a real
# job/login shell, which is what this script provides).
#
# Usage: sbatch --export=ALL,SOURCE_LABEL=A0_n420_rawsum_disjoint \
#            submit_val_noise_diagnostic.slurm

set -e
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

: "${SOURCE_LABEL:?set SOURCE_LABEL to an existing pfn_results run directory}"
REDRAWS=${REDRAWS:-30}

python val_noise_diagnostic.py --source-label "$SOURCE_LABEL" --redraws "$REDRAWS"
