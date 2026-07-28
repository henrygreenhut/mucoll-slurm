#!/bin/bash
#SBATCH -J reco_confirm_store
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --output=reco_confirm_store_%j.out
#SBATCH --error=reco_confirm_store_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ENV_DIR=/oscar/scratch/$USER/mucoll/envs/reco-store
N_FILES=${1:-420}
if (( N_FILES <= 0 || N_FILES % 42 != 0 )); then
    echo "N_FILES must be a positive multiple of 42" >&2
    exit 2
fi
BASE=/oscar/scratch/$USER/mucoll/libtest
if [ "$N_FILES" -eq 420 ]; then
    POOL_MANIFEST=$BASE/bib_pools_simple/manifest.json
else
    POOL_MANIFEST=$BASE/bib_pools_val25/manifest.json
fi
if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "ERROR: missing $ENV_DIR; run ./setup_oscar_reco_envs.sh first" >&2
    exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$ENV_DIR/bin/python" make_reco_libtest_confirmation_stores.py \
    --n-files "$N_FILES" \
    --pool-manifest "$POOL_MANIFEST"
