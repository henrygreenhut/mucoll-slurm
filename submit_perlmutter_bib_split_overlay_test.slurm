#!/bin/bash
#SBATCH -J bib_split_overlay
#SBATCH -A m5197
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH --licenses=cfs,scratch
#SBATCH -o logs/bib_split_overlay_%j.out
#SBATCH -e logs/bib_split_overlay_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

IMAGE=docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.1
export MUCOLL_CALO_THRESHOLDS_DIR=${MUCOLL_CALO_THRESHOLDS_DIR:-$PSCRATCH/mucoll/maia_thresholds}

srun \
    --ntasks=1 \
    --cpus-per-task=8 \
    --cpu-bind=cores \
    shifter --image="$IMAGE" \
    bash "$SLURM_SUBMIT_DIR/run_perlmutter_bib_split_overlay_test.sh"
