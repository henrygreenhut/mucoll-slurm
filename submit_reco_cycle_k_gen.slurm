#!/bin/bash
#SBATCH -J recok_cycle_gen
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH --exclude=node2336
#SBATCH --array=0-63
#SBATCH -o recok_cycle_gen_%A_%a.out
#SBATCH -e recok_cycle_gen_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -ne 2 ]; then
    echo "usage: sbatch $0 {7|21} {MUPLUS|MUMINUS}" >&2
    exit 2
fi

K=$1
POLARITY=$2
case "$K" in 7|21) ;; *) echo "K must be 7 or 21" >&2; exit 2 ;; esac
case "$POLARITY" in MUPLUS|MUMINUS) ;; *) echo "invalid polarity" >&2; exit 2 ;; esac

NUM_SHARDS=${NUM_SHARDS:-64}
MAX_CYCLES=${MAX_CYCLES:-0}
BASE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_cycle_k
STORE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores/gen_split_mothers_${POLARITY}.h5
BENCH=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/mucoll-benchmarks
CONTAINER=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif

for split in train val test; do
    output="$BASE/GEN/k${K}/${split}/${POLARITY}"
    apptainer exec --bind /oscar:/oscar "$CONTAINER" bash -c "
        source /opt/setup_mucoll.sh
        python3 '$SLURM_SUBMIT_DIR/reco_cycle_k_library.py' write-gen \
            --bank '$STORE' \
            --manifest '$BASE/pools/manifest.json' \
            --benchmarks-dir '$BENCH' \
            --split '$split' \
            --reuse-k '$K' \
            --output-dir '$output' \
            --shard-index '$SLURM_ARRAY_TASK_ID' \
            --num-shards '$NUM_SHARDS' \
            --max-cycles '$MAX_CYCLES'
    "
done
