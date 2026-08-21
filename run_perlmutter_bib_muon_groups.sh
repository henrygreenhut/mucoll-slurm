#!/bin/bash

set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
BENCH=${MUCOLL_BENCHMARKS_PATH:-$REPO/../mucoll-benchmarks}
OUTPUT=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data/bib-v3p0-fmt2-split-muon-v1
MANIFEST=$OUTPUT/manifests/partition.json
COMPONENT=decays-containing-muon-poisson-norot
NUMBER_GROUPS=${BIB_MUON_GROUPS:-6666}
RANK=${SLURM_PROCID:-0}
TASKS=${SLURM_NTASKS:-1}
WORK=$PSCRATCH/mucoll/bib_muon_groups/$SLURM_JOB_ID/rank_$RANK

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

for polarity in MUPLUS MUMINUS; do
    python3 -u "$REPO/bib_split_muon_gen.py" write-muon-groups \
        --manifest "$MANIFEST" \
        --output-root "$OUTPUT" \
        --polarity "$polarity" \
        --groups "$NUMBER_GROUPS" \
        --shard-index "$RANK" \
        --num-shards "$TASKS"

    for ((group=RANK; group<NUMBER_GROUPS; group+=TASKS)); do
        input=$OUTPUT/$COMPONENT/GEN/$polarity/bib_gen_$group.edm4hep.root
        output=$OUTPUT/$COMPONENT/SIM/$polarity/bib_sim_$group.edm4hep.root
        if [ -s "$output" ] && python3 -c "import uproot; f=uproot.open('$output'); assert 'podio_metadata' in f; assert f['events'].num_entries == 1"; then
            continue
        fi

        mkdir -p "$(dirname "$output")"
        temporary=$WORK/${polarity}_${group}.root
        partial=$(dirname "$output")/.${output##*/}.partial.$SLURM_JOB_ID.$RANK
        log=$WORK/${polarity}_${group}.log
        rm -f "$temporary" "$partial" "$log"

        if ! ddsim \
            --steeringFile "$BENCH/simulation/steer_baseline.py" \
            --numberOfEvents 1 \
            --inputFiles "$input" \
            --outputFile "$temporary" >"$log" 2>&1; then
            cat "$log" >&2
            exit 1
        fi

        python3 -c "import uproot; f=uproot.open('$temporary'); assert 'podio_metadata' in f; assert 'MCParticles' in f['events']; assert f['events'].num_entries == 1"
        cp "$temporary" "$partial"
        mv "$partial" "$output"
        rm -f "$temporary" "$log"
        echo "$polarity group=$group"
    done
done
