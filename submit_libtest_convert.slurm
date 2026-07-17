#!/bin/bash
#SBATCH -J libtest_convert
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 00:30:00
#SBATCH --output=libtest_convert_%j.out
#SBATCH --error=libtest_convert_%j.err

# Convert both GEN libraries (one polarity) into particle stores.
# Usage:  sbatch submit_libtest_convert.slurm
# Env overrides: POLARITY, NORM1_DIR, NORM42_DIR, STORE_DIR, WORKERS

set -e
cd "$SLURM_SUBMIT_DIR"
source config.sh

module load python
conda activate "$PSCRATCH/mucoll/envs/mucoll-inspect" 2>/dev/null \
    || source activate "$PSCRATCH/mucoll/envs/mucoll-inspect"

POLARITY=${POLARITY:-MUPLUS}
NORM1_DIR=${NORM1_DIR:-$DATA_GROUP_DIR/bib-v3p0-fmt2-norm1/GEN/$POLARITY}
NORM42_DIR=${NORM42_DIR:-$DATA_GROUP_DIR/bib-v3p0-fmt2-norm42-RandomRot/GEN/$POLARITY}
STORE_DIR=${STORE_DIR:-$PSCRATCH/mucoll/libtest/stores}
WORKERS=${WORKERS:-64}

mkdir -p "$STORE_DIR"
echo "norm1:  $NORM1_DIR"
echo "norm42: $NORM42_DIR"
echo "stores: $STORE_DIR"

python gen_libtest_make_store.py \
    --input-dir "$NORM1_DIR" \
    --output "$STORE_DIR/gen_norm1_${POLARITY}.h5" \
    --workers "$WORKERS"

python gen_libtest_make_store.py \
    --input-dir "$NORM42_DIR" \
    --output "$STORE_DIR/gen_norm42_${POLARITY}.h5" \
    --workers "$WORKERS"

echo "conversion complete"
