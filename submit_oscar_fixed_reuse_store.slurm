#!/bin/bash
#SBATCH -J fixed_reuse_store
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16g
#SBATCH -t 06:00:00
#SBATCH -o fixed_reuse_store_%j.out
#SBATCH -e fixed_reuse_store_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -ne 1 ]; then
    echo "usage: $0 {1|5|7}"
    exit 2
fi

K=$1
case "$K" in
    1|5|7) ;;
    *) echo "reuse factor must be 1, 5, or 7"; exit 2 ;;
esac

module load ngc-tensorflow-container/25.02-tf2-py3-azkb

STORE_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores
INPUT="$STORE_DIR/gen_split_mothers_MUPLUS.h5"
OUTPUT="$STORE_DIR/gen_fixed_k${K}_MUPLUS.h5"
PARTIAL="${OUTPUT}.partial.${SLURM_JOB_ID}"

if [ ! -f "$INPUT" ]; then
    echo "ERROR: missing $INPUT"
    exit 1
fi
if [ -e "$OUTPUT" ]; then
    echo "ERROR: output already exists: $OUTPUT"
    exit 1
fi

cleanup() {
    rm -f "$PARTIAL"
}
trap cleanup EXIT

apptainer exec "$NGC_TENSORFLOW_CONTAINER" \
    python -u gen_mother_make_fixed_reuse_store.py \
        --input "$INPUT" \
        --output "$PARTIAL" \
        --reuse-k "$K" \
        --seed 1701

mv "$PARTIAL" "$OUTPUT"
trap - EXIT
echo "published $OUTPUT"
