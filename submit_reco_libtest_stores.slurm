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
STUDY=${1:-charged7}
BASE=/oscar/scratch/$USER/mucoll/libtest
case "$STUDY" in
    charged7)
        DEFAULT_RECO_DIR=$BASE/reco_n420_pfn_trackfix
        DEFAULT_STORE_DIR=$BASE/reco_n420_pfn_stores_trackfix
        DEFAULT_TRAIN_RECO_DIR=$DEFAULT_RECO_DIR
        DEFAULT_VAL_RECO_DIR=$DEFAULT_RECO_DIR
        DEFAULT_TEST_RECO_DIR=$DEFAULT_RECO_DIR
        DEFAULT_VAL_EVENTS=400
        DEFAULT_POOL_MANIFEST=$BASE/bib_pools_simple/manifest.json
        DEFAULT_TEST_POOL_MANIFEST=$DEFAULT_POOL_MANIFEST
        ;;
    val2000)
        DEFAULT_RECO_DIR=$BASE/reco_n420_pfn_trackfix
        DEFAULT_STORE_DIR=$BASE/reco_n420_pfn_stores_trackfix_val2000
        DEFAULT_TRAIN_RECO_DIR=$DEFAULT_RECO_DIR
        DEFAULT_VAL_RECO_DIR=$BASE/reco_n420_pfn_trackfix_val2000
        DEFAULT_TEST_RECO_DIR=$DEFAULT_RECO_DIR
        DEFAULT_VAL_EVENTS=2000
        DEFAULT_POOL_MANIFEST=$BASE/bib_pools_simple/manifest.json
        DEFAULT_TEST_POOL_MANIFEST=$DEFAULT_POOL_MANIFEST
        ;;
    val25)
        DEFAULT_RECO_DIR=$BASE/reco_n420_pfn_trackfix_val25
        DEFAULT_STORE_DIR=$BASE/reco_n420_pfn_stores_trackfix_val25
        DEFAULT_TRAIN_RECO_DIR=$DEFAULT_RECO_DIR
        DEFAULT_VAL_RECO_DIR=$DEFAULT_RECO_DIR
        DEFAULT_TEST_RECO_DIR=$BASE/reco_n420_pfn_trackfix
        DEFAULT_VAL_EVENTS=2000
        DEFAULT_POOL_MANIFEST=$BASE/bib_pools_val25/manifest.json
        DEFAULT_TEST_POOL_MANIFEST=$BASE/bib_pools_simple/manifest.json
        ;;
    *)
        echo "usage: sbatch $0 [charged7|val2000|val25]" >&2
        exit 2
        ;;
esac

RECO_DIR=${RECO_DIR:-$DEFAULT_RECO_DIR}
STORE_DIR=${STORE_DIR:-$DEFAULT_STORE_DIR}
TRAIN_RECO_DIR=${TRAIN_RECO_DIR:-$DEFAULT_TRAIN_RECO_DIR}
VAL_RECO_DIR=${VAL_RECO_DIR:-$DEFAULT_VAL_RECO_DIR}
TEST_RECO_DIR=${TEST_RECO_DIR:-$DEFAULT_TEST_RECO_DIR}
TRAIN_EVENTS=${TRAIN_EVENTS:-2000}
VAL_EVENTS=${VAL_EVENTS:-$DEFAULT_VAL_EVENTS}
TEST_EVENTS=${TEST_EVENTS:-800}
POOL_MANIFEST=${POOL_MANIFEST:-$DEFAULT_POOL_MANIFEST}
TEST_POOL_MANIFEST=${TEST_POOL_MANIFEST:-$DEFAULT_TEST_POOL_MANIFEST}

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "ERROR: missing $ENV_DIR; run ./setup_oscar_reco_envs.sh first"
    exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$ENV_DIR/bin/python" make_reco_libtest_stores.py \
    --reco-dir "$RECO_DIR" \
    --train-reco-dir "$TRAIN_RECO_DIR" \
    --val-reco-dir "$VAL_RECO_DIR" \
    --test-reco-dir "$TEST_RECO_DIR" \
    --train-events "$TRAIN_EVENTS" \
    --val-events "$VAL_EVENTS" \
    --test-events "$TEST_EVENTS" \
    --pool-manifest "$POOL_MANIFEST" \
    --test-pool-manifest "$TEST_POOL_MANIFEST" \
    --outdir "$STORE_DIR"
