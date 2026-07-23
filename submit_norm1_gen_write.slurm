#!/bin/bash
#SBATCH -J norm1_gen_write
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16g
#SBATCH -t 01:00:00
#SBATCH --array=0-63
#SBATCH -o norm1_gen_write_%x_%A_%a.out
#SBATCH -e norm1_gen_write_%x_%A_%a.err

# 64-way sharded array job for gen_libtest_write_norm1_root.py (dedup norm42
# GEN -> real per-cycle norm1 EDM4hep GEN files). 64 shards, not one task
# per file: this account's normal-QOS cap on the batch partition is
# MaxTRESPU=cpu=64, so 64 concurrent 1-core tasks is the actual ceiling --
# more array indices would just queue behind that cap, not run any faster.
# At ~0.36 files/s/shard (single-file smoke-tested rate) x 64 shards, the
# full ~6666-file run should take on the order of minutes, not hours.
#
# --array=0-63 is fixed regardless of --num-shards below (kept equal by
# convention here) -- SLURM_ARRAY_TASK_ID directly becomes --shard-index.
#
# Submit once per polarity:
#   sbatch submit_norm1_gen_write.slurm MUPLUS
#   sbatch submit_norm1_gen_write.slurm MUMINUS

set -e
cd "$SLURM_SUBMIT_DIR"

POLARITY=$1
if [ "$POLARITY" != "MUPLUS" ] && [ "$POLARITY" != "MUMINUS" ]; then
    echo "usage: sbatch submit_norm1_gen_write.slurm {MUPLUS|MUMINUS}"
    exit 1
fi

NUM_SHARDS=64
CONTAINER=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif
INPUT_DIR=/oscar/data/mleblan6/mucoll/bib/bib-v3p0-fmt2-norm42-RandomRot/GEN/${POLARITY}
OUTPUT_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/bib_norm1_reconstructed/GEN/${POLARITY}

apptainer exec --bind /oscar:/oscar "$CONTAINER" bash -c "
source /opt/setup_mucoll.sh
python3 gen_libtest_write_norm1_root.py \
    --input-dir '$INPUT_DIR' \
    --output-dir '$OUTPUT_DIR' \
    --shard-index $SLURM_ARRAY_TASK_ID \
    --num-shards $NUM_SHARDS
"
