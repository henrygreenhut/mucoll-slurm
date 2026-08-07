#!/bin/bash
#SBATCH -J reco_mult
#SBATCH -p debug
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH --exclude=node2336
#SBATCH --output=reco_mult_%j.out
#SBATCH --error=reco_mult_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -lt 2 ]; then
    echo "usage: sbatch $0 {coned|unconed} N [N ...]" >&2
    exit 2
fi

CONING=$1
shift
case "$CONING" in
    coned)
        STORE_TAG=trackfix_val25
        ;;
    unconed)
        STORE_TAG=calo_unconed_val25
        ;;
    *)
        echo "first argument must be coned or unconed" >&2
        exit 2
        ;;
esac

BASE=/oscar/scratch/$USER/mucoll/libtest
PYTHON=/oscar/scratch/$USER/mucoll/envs/reco-store/bin/python

for N in "$@"; do
    STORE_DIR="$BASE/reco_n${N}_pfn_stores_${STORE_TAG}"
    OUTDIR="$SLURM_SUBMIT_DIR/plots/reco_n${N}_calo_${CONING}/multiplicities"
    if [ ! -d "$STORE_DIR" ]; then
        echo "missing store directory: $STORE_DIR" >&2
        exit 1
    fi
    "$PYTHON" plot_reco_multiplicities.py \
        --store-dir "$STORE_DIR" \
        --outdir "$OUTDIR" \
        --n-files "$N" \
        --name "reco_n${N}_${CONING}_multiplicities"
done
