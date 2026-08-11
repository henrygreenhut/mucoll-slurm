#!/bin/bash

set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
BASE=${RECO_CYCLE_K_BASE:-$PSCRATCH/mucoll/reco_cycle_k}
BENCH=${MUCOLL_BENCHMARKS_PATH:-$REPO/../mucoll-benchmarks}
MANIFEST=$BASE/pools/manifest.json
RANK=${SLURM_PROCID:-0}

case "$RANK" in
    0) k=7; polarity=MUPLUS ;;
    1) k=7; polarity=MUMINUS ;;
    2) k=21; polarity=MUPLUS ;;
    3) k=21; polarity=MUMINUS ;;
    *) echo "smoke expects ranks 0 through 3" >&2; exit 2 ;;
esac

set +u
source /opt/setup_mucoll.sh
set -u

cycle=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['splits']['train']['cycles'][0])")
output_root=$BASE/smoke
gen_dir=$output_root/GEN/k${k}/train/${polarity}
sim_dir=$output_root/SIM/k${k}/train/${polarity}
input=$gen_dir/bib_gen_cycle_$(printf '%06d' "$cycle").edm4hep.root
output=$sim_dir/bib_sim_cycle_$(printf '%06d' "$cycle").edm4hep.root
workdir=$PSCRATCH/mucoll/reco_cycle_k/smoke_work/${SLURM_JOB_ID}/rank_${RANK}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$gen_dir" "$sim_dir" "$workdir"
trap 'rm -rf "$workdir"' EXIT

marker=$workdir/gen.complete
rm -f "$marker"
set +e
python3 -u "$REPO/reco_cycle_k_library.py" write-gen \
    --bank "$BASE/banks/gen_split_mothers_${polarity}.h5" \
    --manifest "$MANIFEST" \
    --benchmarks-dir "$BENCH" \
    --split train \
    --reuse-k "$k" \
    --output-dir "$gen_dir" \
    --max-cycles 1 \
    --validate \
    --completion-marker "$marker"
gen_status=$?
set -e
if [ "$gen_status" -ne 0 ]; then
    if [ "$gen_status" -eq 141 ] && [ -s "$marker" ]; then
        echo "rank $RANK: accepting SIGPIPE after validated GEN completion" >&2
    else
        echo "rank $RANK: GEN writer failed with status $gen_status" >&2
        exit "$gen_status"
    fi
fi

if [ ! -s "$output" ]; then
    temporary=$workdir/output.partial.root
    set +u
    source "$BENCH/setup_config.sh" "$BENCH" MAIA_v0
    set -u
    cd "$workdir"
    /usr/bin/time -v ddsim \
        --steeringFile "$BENCH/simulation/steer_baseline.py" \
        --numberOfEvents 1 \
        --inputFiles "$input" \
        --outputFile "$temporary"
    python3 -c "import uproot; f=uproot.open('$temporary'); assert f['events'].num_entries == 1; assert 'podio_metadata' in f"
    mv "$temporary" "$output"
fi

printf 'rank=%s k=%s polarity=%s cycle=%s GEN=%s SIM=%s\n' \
    "$RANK" "$k" "$polarity" "$cycle" \
    "$(du -h "$input" | cut -f1)" "$(du -h "$output" | cut -f1)"
