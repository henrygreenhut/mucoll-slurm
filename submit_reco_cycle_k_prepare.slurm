#!/bin/bash
#SBATCH -J recok_cycle_prepare
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH --exclude=node2336
#SBATCH -o recok_cycle_prepare_%j.out
#SBATCH -e recok_cycle_prepare_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

BASE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_cycle_k
STORES=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores
POOL=/oscar/scratch/hgreenhu/mucoll/libtest/bib_pools_val25/manifest.json
PYTHON=/oscar/scratch/hgreenhu/mucoll/envs/reco-store/bin/python

mkdir -p "$BASE/pools" "$BASE/logs"

"$PYTHON" reco_cycle_k_library.py prepare \
    --source-pool-manifest "$POOL" \
    --muplus-bank "$STORES/gen_split_mothers_MUPLUS.h5" \
    --muminus-bank "$STORES/gen_split_mothers_MUMINUS.h5" \
    --output "$BASE/pools/manifest.json"
