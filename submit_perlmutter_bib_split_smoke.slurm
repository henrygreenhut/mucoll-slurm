#!/bin/bash
#SBATCH -J bib_split_smoke
#SBATCH -A m5197
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH --licenses=cfs,scratch
#SBATCH -o logs/bib_split_smoke_%j.out
#SBATCH -e logs/bib_split_smoke_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

IMAGE=docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.0
LOGS=$SLURM_SUBMIT_DIR/logs/bib_split_smoke_$SLURM_JOB_ID
mkdir -p "$LOGS"

srun \
    --ntasks=2 \
    --cpus-per-task=8 \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    --output="$LOGS/rank_%t.out" \
    --error="$LOGS/rank_%t.err" \
    shifter --image="$IMAGE" bash -c '
        if [ "$SLURM_PROCID" -eq 0 ]; then polarity=MUPLUS; else polarity=MUMINUS; fi
        bash "$SLURM_SUBMIT_DIR/run_perlmutter_bib_split_smoke.sh" "$polarity"
    '
