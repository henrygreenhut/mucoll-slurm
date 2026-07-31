#!/bin/bash
#SBATCH -J recok_stores
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o recok_stores_%j.out
#SBATCH -e recok_stores_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ENV=/oscar/scratch/$USER/mucoll/envs/reco-store
RECO=/oscar/scratch/$USER/mucoll/libtest/reco_variable_k_n420
STORES=/oscar/scratch/$USER/mucoll/libtest/reco_variable_k_n420_stores
MANIFEST=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_variable_k/manifest/manifest.json

if [ ! -x "$ENV/bin/python" ]; then
    echo "missing RECO store environment: $ENV" >&2
    exit 1
fi

"$ENV/bin/python" make_reco_libtest_stores.py \
    --n-files 420 \
    --reco-dir "$RECO" \
    --outdir "$STORES" \
    --pool-manifest "$MANIFEST" \
    --samples k1 k5 k7 k10 k21 \
    --study-template 'reco_variable_n{n_files}_{sample}' \
    --train-events 2000 \
    --val-events 2000 \
    --test-events 2000
