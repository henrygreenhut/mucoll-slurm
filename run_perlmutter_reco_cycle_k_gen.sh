#!/bin/bash

set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
BASE=${RECO_CYCLE_K_BASE:-$PSCRATCH/mucoll/reco_cycle_k}
BENCH=${MUCOLL_BENCHMARKS_PATH:-$REPO/../mucoll-benchmarks}
MANIFEST=$BASE/pools/manifest.json
BANKS=$BASE/banks
RANK=${SLURM_PROCID:-0}
TASKS=${SLURM_NTASKS:-1}
WORKDIR=$PSCRATCH/mucoll/reco_cycle_k/gen_work/${SLURM_JOB_ID}/rank_${RANK}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

set +u
source /opt/setup_mucoll.sh
set -u

write_gen_shard() {
    split=$1
    polarity=$2
    k=$3
    marker=$WORKDIR/${split}_${polarity}_k${k}.complete
    rm -f "$marker"

    set +e
    python3 -u "$REPO/reco_cycle_k_library.py" write-gen \
        --bank "$BANKS/gen_split_mothers_${polarity}.h5" \
        --manifest "$MANIFEST" \
        --benchmarks-dir "$BENCH" \
        --split "$split" \
        --reuse-k "$k" \
        --output-dir "$BASE/GEN/k${k}/${split}/${polarity}" \
        --shard-index "$RANK" \
        --num-shards "$TASKS" \
        --validate \
        --completion-marker "$marker"
    status=$?
    set -e

    if [ "$status" -eq 0 ]; then
        return
    fi
    if [ "$status" -eq 141 ] && [ -s "$marker" ]; then
        echo "rank $RANK: accepting SIGPIPE after validated GEN completion for $split $polarity k=$k" >&2
        return
    fi
    echo "rank $RANK: GEN writer failed with status $status for $split $polarity k=$k" >&2
    return "$status"
}

for split in train val test; do
    for polarity in MUPLUS MUMINUS; do
        for k in 7 21; do
            write_gen_shard "$split" "$polarity" "$k"
        done
    done
done
