#!/bin/bash
#SBATCH -J reco_mult
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --output=reco_mult_%j.out
#SBATCH --error=reco_mult_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

STORE_DIR=/oscar/scratch/$USER/mucoll/libtest/reco_n420_pfn_stores_calo_unconed_val25
OUTDIR="$SLURM_SUBMIT_DIR/plots/reco_n420_calo_unconed/multiplicities"
PYTHON=/oscar/scratch/$USER/mucoll/envs/reco-store/bin/python

"$PYTHON" plot_reco_multiplicities.py \
    --store-dir "$STORE_DIR" \
    --outdir "$OUTDIR" \
    --n-files 420 \
    --name reco_n420_unconed_multiplicities
