#!/bin/bash

set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
BASE=${RECO_CYCLE_K_BASE:-$PSCRATCH/mucoll/reco_cycle_k}
BENCH=${MUCOLL_BENCHMARKS_PATH:-$REPO/../mucoll-benchmarks}
MANIFEST=$BASE/pools/manifest.json
BANKS=$BASE/banks
RANK=${SLURM_PROCID:-0}
TASKS=${SLURM_NTASKS:-1}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

set +u
source /opt/setup_mucoll.sh
set -u

for split in train val test; do
    for polarity in MUPLUS MUMINUS; do
        for k in 7 21; do
            python3 -u "$REPO/reco_cycle_k_library.py" write-gen \
                --bank "$BANKS/gen_split_mothers_${polarity}.h5" \
                --manifest "$MANIFEST" \
                --benchmarks-dir "$BENCH" \
                --split "$split" \
                --reuse-k "$k" \
                --output-dir "$BASE/GEN/k${k}/${split}/${polarity}" \
                --shard-index "$RANK" \
                --num-shards "$TASKS" \
                --validate
        done
    done
done
