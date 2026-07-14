#!/bin/bash
#SBATCH -J libtest_bundle
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -t 00:30:00
#SBATCH --output=libtest_bundle_%x_%j.out
#SBATCH --error=libtest_bundle_%x_%j.err

# Pack four independent resumable libtest processes onto the four A100s for
# which a debug-QOS node is charged.  BUNDLE_CONFIG is a supervisor JSON plan;
# BUNDLE_LABELS is a comma-separated list in GPU/rank order.

set -e
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

: "${BUNDLE_CONFIG:?set BUNDLE_CONFIG}"
: "${BUNDLE_LABELS:?set BUNDLE_LABELS}"

echo "bundle config=$BUNDLE_CONFIG labels=$BUNDLE_LABELS"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
export SLURM_CPU_BIND=cores

srun --cpu-bind=cores python libtest_bundle_worker.py
