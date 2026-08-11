#!/bin/bash

set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
BASE=${RECO_CYCLE_K_BASE:-$PSCRATCH/mucoll/reco_cycle_k}
BENCH=${MUCOLL_BENCHMARKS_PATH:-$REPO/../mucoll-benchmarks}
MANIFEST=$BASE/pools/manifest.json
RANK=${SLURM_PROCID:-0}
TASKS=${SLURM_NTASKS:-1}
WORKDIR=$PSCRATCH/mucoll/reco_cycle_k/work/${SLURM_JOB_ID}/rank_${RANK}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

set +u
source /opt/setup_mucoll.sh
set +o pipefail
source "$BENCH/setup_config.sh" "$BENCH" MAIA_v0
set -o pipefail
set -u
cd "$WORKDIR"

completed=0
while IFS=$'\t' read -r k split polarity cycle input output; do
    if [ -s "$output" ]; then
        completed=$((completed + 1))
        continue
    fi
    if [ ! -s "$input" ]; then
        echo "missing GEN input: $input" >&2
        exit 2
    fi

    mkdir -p "$(dirname "$output")"
    temporary="$WORKDIR/k${k}_${split}_${polarity}_${cycle}.partial.root"
    ddsim_log="$WORKDIR/k${k}_${split}_${polarity}_${cycle}.ddsim.log"
    rm -f "$temporary"
    rm -f "$ddsim_log"

    set +e
    ddsim \
        --steeringFile "$BENCH/simulation/steer_baseline.py" \
        --numberOfEvents 1 \
        --inputFiles "$input" \
        --outputFile "$temporary" >"$ddsim_log" 2>&1
    ddsim_status=$?
    set -e
    if [ "$ddsim_status" -ne 0 ]; then
        cat "$ddsim_log" >&2
        exit "$ddsim_status"
    fi

    if ! python3 -c "import uproot; f=uproot.open('$temporary'); assert f['events'].num_entries == 1; assert 'podio_metadata' in f"; then
        cat "$ddsim_log" >&2
        exit 1
    fi
    rm -f "$ddsim_log"
    mv "$temporary" "$output"
    completed=$((completed + 1))
    echo "rank $RANK/$TASKS completed $completed: k=$k $split $polarity cycle=$cycle"
done < <(
    python3 "$REPO/reco_cycle_k_perlmutter.py" items \
        --base "$BASE" \
        --manifest "$MANIFEST" \
        --rank "$RANK" \
        --tasks "$TASKS"
)

echo "rank $RANK/$TASKS finished $completed assigned files"
