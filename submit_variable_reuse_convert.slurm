#!/bin/bash
#SBATCH -J vk_convert
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -A m5197
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 00:30:00
#SBATCH --licenses=cfs,scratch
#SBATCH --output=variable_k_convert_%j.out
#SBATCH --error=variable_k_convert_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source config.sh
module load python
conda activate "$PSCRATCH/mucoll/envs/mucoll-inspect" 2>/dev/null \
    || source activate "$PSCRATCH/mucoll/envs/mucoll-inspect"

POLARITY=${POLARITY:-MUPLUS}
INPUT_DIR=${INPUT_DIR:-$DATA_GROUP_DIR/bib-v3p0-fmt2-norm1-split-mother-norot/GEN/$POLARITY}
STORE_DIR=${STORE_DIR:-$PSCRATCH/mucoll/libtest/stores}
WORKERS=${WORKERS:-64}

mkdir -p "$STORE_DIR"
python3 gen_mother_make_store.py \
    --input-dir "$INPUT_DIR" \
    --output "$STORE_DIR/gen_split_mothers_${POLARITY}.h5" \
    --workers "$WORKERS" \
    --exclude-cycle 6291
