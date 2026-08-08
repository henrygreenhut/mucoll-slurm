#!/bin/bash
#SBATCH -J cycle_k_gen
#SBATCH -A m5197
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t 06:00:00
#SBATCH --licenses=cfs,scratch
#SBATCH -o logs/cycle_k_gen_%j.out
#SBATCH -e logs/cycle_k_gen_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

IMAGE=docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.0
LOGS=$SLURM_SUBMIT_DIR/logs/cycle_k_gen_$SLURM_JOB_ID
mkdir -p "$LOGS"

srun \
    --ntasks="$SLURM_NTASKS" \
    --ntasks-per-node=128 \
    --cpus-per-task=2 \
    --cpu-bind=cores \
    --kill-on-bad-exit=0 \
    --output="$LOGS/rank_%t.out" \
    --error="$LOGS/rank_%t.err" \
    shifter --image="$IMAGE" bash "$SLURM_SUBMIT_DIR/run_perlmutter_reco_cycle_k_gen.sh"
