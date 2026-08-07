#!/bin/bash
#SBATCH -J recok_gen
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 08:00:00
#SBATCH --exclude=node2336
#SBATCH --array=0-63
#SBATCH -o recok_gen_%A_%a.out
#SBATCH -e recok_gen_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -ne 2 ]; then
    echo "usage: sbatch $0 {1-native|5|7|10|21} {MUPLUS|MUMINUS}" >&2
    exit 2
fi

TARGET=$1
POLARITY=$2
case "$TARGET" in
    1-native) K=1; SAMPLE=k1_native; MANIFEST_NAME=manifest_native1 ;;
    5|7|10|21) K=$TARGET; SAMPLE=k${K}; MANIFEST_NAME=manifest ;;
    *) echo "unsupported target=$TARGET" >&2; exit 2 ;;
esac
case "$POLARITY" in MUPLUS|MUMINUS) ;; *) echo "unsupported polarity" >&2; exit 2 ;; esac

NUM_SHARDS=${NUM_SHARDS:-64}
MAX_CHUNKS=${MAX_CHUNKS:-0}
BASE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_variable_k
STORE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores/gen_split_mothers_${POLARITY}.h5
BENCH=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/mucoll-benchmarks
CONTAINER=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif

for split in train val test; do
    output="$BASE/GEN/${SAMPLE}/${split}/${POLARITY}"
    apptainer exec --bind /oscar:/oscar "$CONTAINER" bash -c "
        source /opt/setup_mucoll.sh
        python3 '$SLURM_SUBMIT_DIR/reco_variable_k_library.py' write-gen \
            --bank '$STORE' \
            --manifest-dir '$BASE/$MANIFEST_NAME' \
            --benchmarks-dir '$BENCH' \
            --split '$split' \
            --reuse-k '$K' \
            --output-dir '$output' \
            --shard-index '$SLURM_ARRAY_TASK_ID' \
            --num-shards '$NUM_SHARDS' \
            --max-chunks '$MAX_CHUNKS'
    "
done
