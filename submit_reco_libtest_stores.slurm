#!/bin/bash
#SBATCH -J reco420_stores
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --output=reco420_stores_%j.out
#SBATCH --error=reco420_stores_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ENV_DIR=/oscar/scratch/$USER/mucoll/envs/reco-store
RECO_DIR=${RECO_DIR:-/oscar/scratch/$USER/mucoll/libtest/reco_n420_pfn_trackfix}
STORE_DIR=${STORE_DIR:-/oscar/scratch/$USER/mucoll/libtest/reco_n420_pfn_stores_trackfix}

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "ERROR: missing $ENV_DIR; run ./setup_oscar_reco_envs.sh first"
    exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$ENV_DIR/bin/python" make_reco_libtest_stores.py \
    --reco-dir "$RECO_DIR" \
    --outdir "$STORE_DIR"
