#!/bin/bash
#SBATCH -J recok_prepare
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o recok_prepare_%j.out
#SBATCH -e recok_prepare_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

BASE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_variable_k
STORE_BASE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores

apptainer exec --bind /oscar:/oscar "$NGC_TENSORFLOW_CONTAINER" \
    python -u reco_variable_k_library.py prepare \
        --muplus-bank "$STORE_BASE/gen_split_mothers_MUPLUS.h5" \
        --muminus-bank "$STORE_BASE/gen_split_mothers_MUMINUS.h5" \
        --outdir "$BASE/manifest"
