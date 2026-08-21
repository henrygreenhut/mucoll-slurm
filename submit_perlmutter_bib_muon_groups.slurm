#!/bin/bash
#SBATCH -J bib_muon_groups
#SBATCH -A m5197
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 4
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=2
#SBATCH -t 24:00:00
#SBATCH --licenses=cfs,scratch
#SBATCH --requeue
#SBATCH -o logs/bib_muon_groups_%j.out
#SBATCH -e logs/bib_muon_groups_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

IMAGE=docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.0
export BIB_MUON_GROUPS=6666

srun \
    --ntasks="$SLURM_NTASKS" \
    --ntasks-per-node=96 \
    --cpus-per-task=2 \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    shifter --image="$IMAGE" \
    bash "$SLURM_SUBMIT_DIR/run_perlmutter_bib_muon_groups.sh"
