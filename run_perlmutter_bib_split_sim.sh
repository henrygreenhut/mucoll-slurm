#!/bin/bash

set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
BENCH=${MUCOLL_BENCHMARKS_PATH:-$REPO/../mucoll-benchmarks}
OUTPUT=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data/bib-v3p0-fmt2-split-muon-v1
MANIFEST=$OUTPUT/manifests/partition.json
RANK=${SLURM_PROCID:-0}
TASKS=${SLURM_NTASKS:-1}
WORK=$PSCRATCH/mucoll/bib_split_muon_sim/$SLURM_JOB_ID/rank_$RANK

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

set +u
source /opt/setup_mucoll.sh
set +o pipefail
source "$BENCH/setup_config.sh" "$BENCH" MAIA_v0
set -o pipefail
set -u

mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

cycle_args=()
IFS=, read -ra cycles <<< "${BIB_SPLIT_CYCLES:-}"
for cycle in "${cycles[@]}"; do
    if [ -n "$cycle" ]; then
        cycle_args+=(--cycle "$cycle")
    fi
done

while IFS=$'\t' read -r component polarity cycle entries input output; do
    if [ -s "$output" ]; then
        continue
    fi
    if [ ! -s "$input" ]; then
        echo "missing GEN input: $input" >&2
        exit 2
    fi

    mkdir -p "$(dirname "$output")"
    temporary=$WORK/${component}_${polarity}_${cycle}.root
    partial=$(dirname "$output")/.${output##*/}.partial.$SLURM_JOB_ID.$RANK
    log=$WORK/${component}_${polarity}_${cycle}.log
    rm -f "$temporary" "$partial" "$log"

    if ! ddsim \
        --steeringFile "$BENCH/simulation/steer_baseline.py" \
        --numberOfEvents "$entries" \
        --inputFiles "$input" \
        --outputFile "$temporary" >"$log" 2>&1; then
        cat "$log" >&2
        exit 1
    fi

    python3 -c "import uproot; f=uproot.open('$temporary'); assert 'podio_metadata' in f; assert 'MCParticles' in f['events']; assert f['events'].num_entries == $entries"
    cp "$temporary" "$partial"
    mv "$partial" "$output"
    rm -f "$temporary" "$log"
    echo "$component $polarity cycle=$cycle events=$entries"
done < <(
    python3 "$REPO/bib_split_muon_production.py" items \
        --manifest "$MANIFEST" \
        --output-root "$OUTPUT" \
        --rank "$RANK" \
        --tasks "$TASKS" \
        "${cycle_args[@]}"
)
