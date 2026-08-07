#!/bin/bash
#SBATCH -J recok_prepare
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH --exclude=node2336
#SBATCH -o recok_prepare_%j.out
#SBATCH -e recok_prepare_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

BASE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_variable_k
STORE_BASE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores
CONTAINER=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif
MANIFEST_NAME=${1:-manifest}
if ! [[ "$MANIFEST_NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "invalid manifest name: $MANIFEST_NAME" >&2
    exit 2
fi

apptainer exec --bind /oscar:/oscar "$CONTAINER" bash -c "
    source /opt/setup_mucoll.sh
    python3 -u '$SLURM_SUBMIT_DIR/reco_variable_k_library.py' prepare \
        --muplus-bank "$STORE_BASE/gen_split_mothers_MUPLUS.h5" \
        --muminus-bank "$STORE_BASE/gen_split_mothers_MUMINUS.h5" \
        --outdir "$BASE/$MANIFEST_NAME"
"
