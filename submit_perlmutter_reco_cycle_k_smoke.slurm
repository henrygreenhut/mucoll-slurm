#!/bin/bash
#SBATCH -J cycle_k_smoke
#SBATCH -A m5197
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH --licenses=cfs,scratch
#SBATCH -o logs/cycle_k_smoke_%j.out
#SBATCH -e logs/cycle_k_smoke_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

IMAGE=docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.0
LOGS=$SLURM_SUBMIT_DIR/logs/cycle_k_smoke_$SLURM_JOB_ID
mkdir -p "$LOGS"

srun \
    --ntasks=4 \
    --cpus-per-task=4 \
    --cpu-bind=cores \
    --kill-on-bad-exit=0 \
    --output="$LOGS/rank_%t.out" \
    --error="$LOGS/rank_%t.err" \
    shifter --image="$IMAGE" bash "$SLURM_SUBMIT_DIR/run_perlmutter_reco_cycle_k_smoke.sh"
