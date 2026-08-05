#!/bin/bash
#SBATCH -J unconed_stores
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o reco_calo_unconed_stores_%j.out
#SBATCH -e reco_calo_unconed_stores_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

BASE=/oscar/scratch/$USER/mucoll/libtest
ENV_DIR=/oscar/scratch/$USER/mucoll/envs/reco-store
RECO_DIR=$BASE/reco_n420_calo_unconed
STORE_DIR=$BASE/reco_n420_pfn_stores_calo_unconed_val25
POOL_MANIFEST=$BASE/bib_pools_val25/manifest.json

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "missing environment: $ENV_DIR" >&2
    exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$ENV_DIR/bin/python" make_reco_libtest_stores.py \
    --n-files 420 \
    --reco-dir "$RECO_DIR" \
    --train-events 2000 \
    --val-events 2000 \
    --test-events 2000 \
    --pool-manifest "$POOL_MANIFEST" \
    --outdir "$STORE_DIR"
