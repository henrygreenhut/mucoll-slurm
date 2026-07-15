#!/bin/bash
#SBATCH -J reco420_packed
#SBATCH -C cpu
#SBATCH --licenses=cfs,scratch
#SBATCH --output=logs/reco420_packed_%j.out
#SBATCH --error=logs/reco420_packed_%j.err

set -euo pipefail

MANIFEST=$(realpath "$1")
REPO=$(cd "$(dirname "$0")" && pwd)
TASKS_PER_NODE=${TASKS_PER_NODE:-64}
TASKS=$(wc -l < "$MANIFEST")
CAPACITY=$((SLURM_NNODES * TASKS_PER_NODE))

if [ "$TASKS" -lt 1 ] || [ "$TASKS" -gt "$CAPACITY" ]; then
    echo "ERROR: manifest has $TASKS tasks but allocation capacity is $CAPACITY" >&2
    exit 2
fi

mkdir -p "$REPO/logs/tasks_$SLURM_JOB_ID"
export CHAIN_WORK_BASE="$PSCRATCH/mucoll/libtest/work/reco420/$SLURM_JOB_ID"
mkdir -p "$CHAIN_WORK_BASE"

echo "Running $TASKS independent chains on $SLURM_NNODES CPU node(s)"
echo "Manifest: $MANIFEST"

srun --ntasks="$TASKS" \
    --ntasks-per-node="$TASKS_PER_NODE" \
    --cpus-per-task=4 \
    --cpu-bind=cores \
    --kill-on-bad-exit=0 \
    --output="$REPO/logs/tasks_$SLURM_JOB_ID/task_%t.out" \
    --error="$REPO/logs/tasks_$SLURM_JOB_ID/task_%t.err" \
    "$REPO/run_reco_libtest_task.sh" "$MANIFEST"
