#!/bin/bash
#SBATCH -J mother_bank
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16g
#SBATCH -t 01:00:00
#SBATCH --exclude=node2336
#SBATCH -o mother_bank_%j.out
#SBATCH -e mother_bank_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

POLARITY=${1:-MUPLUS}
if [ "$POLARITY" != MUPLUS ] && [ "$POLARITY" != MUMINUS ]; then
    echo "usage: sbatch $0 {MUPLUS|MUMINUS}" >&2
    exit 2
fi

module load ngc-tensorflow-container/25.02-tf2-py3-azkb

FLUKA_DIR=/oscar/data/mleblan6/mucoll/bib/FLUKA
TASKLIST=/oscar/data/mleblan6/mucoll/bib/logs/gen/tasklist.txt
BENCHMARKS_DIR="$SLURM_SUBMIT_DIR/../mucoll-benchmarks"
DEFAULT_OUTPUT=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores/gen_split_mothers_${POLARITY}.h5
OUTPUT=${MOTHER_STORE_OUTPUT:-$DEFAULT_OUTPUT}
PARTIAL="${OUTPUT}.partial.${SLURM_JOB_ID}"

if [ -e "$OUTPUT" ]; then
    echo "ERROR: output already exists: $OUTPUT"
    echo "Move it aside explicitly before rebuilding."
    exit 1
fi

apptainer exec --bind /oscar:/oscar "$NGC_TENSORFLOW_CONTAINER" \
    python -u gen_fluka_make_mother_store.py \
        --fluka-dir "$FLUKA_DIR" \
        --tasklist "$TASKLIST" \
        --benchmarks-dir "$BENCHMARKS_DIR" \
        --output "$PARTIAL" \
        --polarity "$POLARITY" \
        --exclude-cycle 6291

mv "$PARTIAL" "$OUTPUT"
echo "published $OUTPUT"
