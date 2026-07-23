#!/bin/bash
#SBATCH -J reconstruct_unrotated
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64g
#SBATCH -t 04:00:00
#SBATCH -o reconstruct_unrotated_%x_%j.out
#SBATCH -e reconstruct_unrotated_%x_%j.err

# Full-library run of gen_libtest_reconstruct_unrotated.py: deduplicates
# the 42x-cloned/rotated GEN library back to a norm1-equivalent unrotated
# store. Validated on a 20-file sample first (job before this one) --
# clean 42-per-group after fixing a float32-precision bug in the dedup
# key computation (see git history). CPU-only, I/O-bound (uproot/awkward
# ROOT reads + numpy grouping), no GPU needed -- batch partition, not gpu.
#
# Output goes to $HOME, not /oscar/data/mleblan6/ or scratch: home has no
# purge risk (unlike scratch's 30-day access-time purge) and no permission
# uncertainty (some existing data-dir directories are group-read-only, not
# group-writable) -- and at ~1GB expected output size, its 100GB quota is
# a complete non-issue. Move into the group data directory later once
# write access there is confirmed, if that's preferred long-term.
#
# _v2 suffix: gen_libtest_reconstruct_unrotated.py now also reads
# "charge" (needed for --features expanded), so this store's schema
# differs from the original gen_norm1_reconstructed_${POLARITY}.h5 --
# writing to a new path instead of overwriting it in place, for the same
# reason as submit_make_norm42_store.sh's _v2 (avoid racing the
# currently-running n420 long jobs that read the original file). The old
# file stays valid for --features paper.
#
# Submit once per polarity:
#   sbatch submit_reconstruct_unrotated.sh MUPLUS
#   sbatch submit_reconstruct_unrotated.sh MUMINUS

set -e
cd "$SLURM_SUBMIT_DIR"

POLARITY=$1
if [ "$POLARITY" != "MUPLUS" ] && [ "$POLARITY" != "MUMINUS" ]; then
    echo "usage: sbatch submit_reconstruct_unrotated.sh {MUPLUS|MUMINUS}"
    exit 1
fi

module load python/3.11.11-5e66
source ~/envs/mucoll/bin/activate

mkdir -p ~/mucoll/stores

python gen_libtest_reconstruct_unrotated.py \
    --input-dir /oscar/data/mleblan6/mucoll/bib/bib-v3p0-fmt2-norm42-RandomRot/GEN/${POLARITY} \
    --output ~/mucoll/stores/gen_norm1_reconstructed_${POLARITY}_v2.h5 \
    --workers 16

echo "done -> ~/mucoll/stores/gen_norm1_reconstructed_${POLARITY}_v2.h5"
ls -lh ~/mucoll/stores/gen_norm1_reconstructed_${POLARITY}_v2.h5
