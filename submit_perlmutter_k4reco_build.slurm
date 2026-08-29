#!/bin/bash
#SBATCH -J k4reco_build
#SBATCH -A m5197
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH --licenses=cfs,scratch
#SBATCH -o logs/k4reco_build_%j.out
#SBATCH -e logs/k4reco_build_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export K4RECO_SOURCE=${K4RECO_SOURCE:-$SLURM_SUBMIT_DIR/../k4Reco}
export K4RECO_BUILD=${K4RECO_BUILD:-$PSCRATCH/mucoll/k4Reco-overlay-entry-mix-build}
IMAGE=${MUCOLL_IMAGE:-docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.1}

srun \
    --ntasks=1 \
    --cpus-per-task=8 \
    --cpu-bind=cores \
    shifter --image="$IMAGE" \
    bash "$SLURM_SUBMIT_DIR/run_perlmutter_k4reco_build.sh"
