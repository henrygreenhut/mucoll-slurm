#!/bin/bash

set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
OUTPUT=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data/bib-v3p0-fmt2-split-muon-v1
MANIFEST=$OUTPUT/manifests/partition.json
RANK=${SLURM_PROCID:-0}
TASKS=${SLURM_NTASKS:-1}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

set +u
source /opt/setup_mucoll.sh
set -u

cycle_args=()
IFS=, read -ra cycles <<< "${BIB_SPLIT_CYCLES:-}"
for cycle in "${cycles[@]}"; do
    if [ -n "$cycle" ]; then
        cycle_args+=(--cycle "$cycle")
    fi
done

for polarity in MUPLUS MUMINUS; do
    python3 -u "$REPO/bib_split_muon_gen.py" write-gen \
        --manifest "$MANIFEST" \
        --output-root "$OUTPUT" \
        --polarity "$polarity" \
        --shard-index "$RANK" \
        --num-shards "$TASKS" \
        "${cycle_args[@]}"
done
