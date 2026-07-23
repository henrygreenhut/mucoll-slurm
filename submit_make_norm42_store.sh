#!/bin/bash
#SBATCH -J make_norm42_store
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64g
#SBATCH -t 04:00:00
#SBATCH -o make_norm42_store_%x_%j.out
#SBATCH -e make_norm42_store_%x_%j.err

# Build the norm42 store on OSCAR using gen_libtest_make_store.py -- no
# deduplication needed here, since bib-v3p0-fmt2-norm42-RandomRot/GEN IS
# genuinely the 42x-cloned/rotated data (confirmed via
# inspect_gen_rotation.py earlier). Only the reconstruction (norm1) side
# needed the dedup workaround.
#
# _v2 suffix: gen_libtest_make_store.py now also reads "charge" (needed
# for --features expanded), so this store's schema differs from the
# original gen_norm42_${POLARITY}.h5 -- writing to a new path instead of
# overwriting it in place, since the n420 long-running jobs currently read
# that original file directly and an in-place overwrite could race with
# any of them starting fresh mid-rewrite. The old file stays valid for
# --features paper (Store only loads whichever RAW_KEYS a file actually
# has) -- rebuild is only required for --features expanded.
#
# Output goes to scratch, not home: expected size ~29GB (matching
# Perlmutter's gen_norm42_MUPLUS.h5), which would eat a large fraction of
# the 100GB home quota. Scratch's 30-day purge is access-time based, so
# it won't be at risk as long as we keep actively using it for training.
#
# Submit once per polarity:
#   sbatch submit_make_norm42_store.sh MUPLUS
#   sbatch submit_make_norm42_store.sh MUMINUS

set -e
cd "$SLURM_SUBMIT_DIR"

POLARITY=$1
if [ "$POLARITY" != "MUPLUS" ] && [ "$POLARITY" != "MUMINUS" ]; then
    echo "usage: sbatch submit_make_norm42_store.sh {MUPLUS|MUMINUS}"
    exit 1
fi

module load python/3.11.11-5e66
source ~/envs/mucoll/bin/activate

mkdir -p /oscar/scratch/$USER/mucoll/stores

python gen_libtest_make_store.py \
    --input-dir /oscar/data/mleblan6/mucoll/bib/bib-v3p0-fmt2-norm42-RandomRot/GEN/${POLARITY} \
    --output /oscar/scratch/$USER/mucoll/stores/gen_norm42_${POLARITY}_v2.h5 \
    --workers 16

echo "done -> /oscar/scratch/$USER/mucoll/stores/gen_norm42_${POLARITY}_v2.h5"
ls -lh /oscar/scratch/$USER/mucoll/stores/gen_norm42_${POLARITY}_v2.h5
