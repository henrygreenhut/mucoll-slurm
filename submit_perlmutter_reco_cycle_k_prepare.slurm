#!/bin/bash
#SBATCH -J cycle_k_prepare
#SBATCH -A m5197
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 00:30:00
#SBATCH --licenses=cfs,scratch
#SBATCH -o logs/cycle_k_prepare_%j.out
#SBATCH -e logs/cycle_k_prepare_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

BASE=${RECO_CYCLE_K_BASE:-$PSCRATCH/mucoll/reco_cycle_k}
DATA=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data
SOURCE_BANK=$PSCRATCH/mucoll/libtest/stores/gen_split_mothers_MUPLUS.h5
BANKS=$BASE/banks
SOURCE_SPLIT=$BASE/pools/source_split.json
MANIFEST=$BASE/pools/manifest.json

mkdir -p logs "$BANKS" "$BASE/pools"

module load python
conda activate "$PSCRATCH/mucoll/envs/mucoll-inspect" 2>/dev/null \
    || source activate "$PSCRATCH/mucoll/envs/mucoll-inspect"

plus=$BANKS/gen_split_mothers_MUPLUS.h5
minus=$BANKS/gen_split_mothers_MUMINUS.h5

if [ ! -s "$plus" ]; then
    temporary="$plus.partial.$SLURM_JOB_ID"
    cp "$SOURCE_BANK" "$temporary"
    python -c "import h5py; f=h5py.File('$temporary','r+'); f.attrs['polarity']='MUPLUS'; f.close()"
    mv "$temporary" "$plus"
fi

if [ ! -s "$minus" ]; then
    temporary="$minus.partial.$SLURM_JOB_ID"
    python -u gen_mother_make_store.py \
        --input-dir "$DATA/bib-v3p0-fmt2-norm1-split-mother-norot/GEN/MUMINUS" \
        --output "$temporary" \
        --workers 64 \
        --polarity MUMINUS \
        --exclude-cycle 6291
    mv "$temporary" "$minus"
fi

python reco_cycle_k_library.py make-source-split --output "$SOURCE_SPLIT"
python reco_cycle_k_library.py prepare \
    --source-pool-manifest "$SOURCE_SPLIT" \
    --muplus-bank "$plus" \
    --muminus-bank "$minus" \
    --output "$MANIFEST"

python reco_cycle_k_library.py inspect --manifest "$MANIFEST"
