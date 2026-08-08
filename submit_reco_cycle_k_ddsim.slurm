#!/bin/bash
#SBATCH -J recok_cycle_sim
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 24:00:00
#SBATCH --exclude=node2336
#SBATCH --array=0-63
#SBATCH -o recok_cycle_sim_%A_%a.out
#SBATCH -e recok_cycle_sim_%A_%a.err

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
BENCH=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/mucoll-benchmarks
CONTAINER=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif
WORKDIR=$(mktemp -d "/oscar/scratch/$USER/recok_cycle_sim_k${K}_${POLARITY}_${SLURM_ARRAY_TASK_ID}.XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

for split in train val test; do
    gen_dir="$BASE/GEN/k${K}/${split}/${POLARITY}"
    sim_dir="$BASE/pools/k${K}/${split}/${POLARITY}"
    mkdir -p "$sim_dir"
    mapfile -t files < <(find "$gen_dir" -maxdepth 1 -type f -name '*.root' | sort -V)
    selected=()
    for ((index = SLURM_ARRAY_TASK_ID; index < ${#files[@]}; index += NUM_SHARDS)); do
        selected+=("${files[$index]}")
        if [ "$MAX_CYCLES" -gt 0 ] && [ "${#selected[@]}" -ge "$MAX_CYCLES" ]; then
            break
        fi
    done

    for input in "${selected[@]}"; do
        name=$(basename "$input" | sed 's/^bib_gen_/bib_sim_/')
        output="$sim_dir/$name"
        temporary="$WORKDIR/${split}_${name%.root}.partial.root"
        if [ -s "$output" ]; then
            continue
        fi
        rm -f "$temporary"
        apptainer exec --bind /oscar:/oscar "$CONTAINER" bash -c "
            source /opt/setup_mucoll.sh
            source '$BENCH/setup_config.sh' '$BENCH' MAIA_v0
            cd '$WORKDIR'
            ddsim --steeringFile '$BENCH/simulation/steer_baseline.py' \
                --numberOfEvents 1 \
                --inputFiles '$input' \
                --outputFile '$temporary'
            python3 -c \"import uproot; f=uproot.open('$temporary'); assert f['events'].num_entries == 1; assert 'podio_metadata' in f\"
        "
        mv "$temporary" "$output"
    done
done
