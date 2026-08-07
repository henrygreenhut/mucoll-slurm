#!/bin/bash
#SBATCH -J recok_ddsim
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 24:00:00
#SBATCH --exclude=node2336
#SBATCH --array=0-63
#SBATCH -o /oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_variable_k/logs/recok_ddsim_%A_%a.out
#SBATCH -e /oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_variable_k/logs/recok_ddsim_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -ne 2 ]; then
    echo "usage: sbatch $0 {1-native|5|7|10|21} {MUPLUS|MUMINUS}" >&2
    exit 2
fi

TARGET=$1
POLARITY=$2
case "$TARGET" in
    1-native) K=1; SAMPLE=k1_native ;;
    5|7|10|21) K=$TARGET; SAMPLE=k${K} ;;
    *) echo "unsupported target=$TARGET" >&2; exit 2 ;;
esac
case "$POLARITY" in MUPLUS|MUMINUS) ;; *) echo "unsupported polarity" >&2; exit 2 ;; esac

NUM_SHARDS=${NUM_SHARDS:-64}
MAX_CHUNKS=${MAX_CHUNKS:-0}
BASE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_variable_k
BENCH=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/mucoll-benchmarks
CONTAINER=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif
WORKDIR=$(mktemp -d "/oscar/scratch/$USER/recok_ddsim_k${K}_${POLARITY}_${SLURM_ARRAY_TASK_ID}.XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

LOOP="$WORKDIR/run.sh"
{
    echo "set -eo pipefail"
    echo "source /opt/setup_mucoll.sh"
    echo "source '$BENCH/setup_config.sh' '$BENCH' MAIA_v0"
    echo "cd '$WORKDIR'"
    for split in train val test; do
        gen_dir="$BASE/GEN/${SAMPLE}/${split}/${POLARITY}"
        sim_dir="$BASE/SIM/${SAMPLE}/${split}/${POLARITY}"
        mkdir -p "$sim_dir"
        mapfile -t files < <(find "$gen_dir" -maxdepth 1 -type f -name '*.root' | sort -V)
        selected=()
        for ((index = SLURM_ARRAY_TASK_ID; index < ${#files[@]}; index += NUM_SHARDS)); do
            selected+=("${files[$index]}")
            if [ "$MAX_CHUNKS" -gt 0 ] && [ "${#selected[@]}" -ge "$MAX_CHUNKS" ]; then
                break
            fi
        done
        for input in "${selected[@]}"; do
            name=$(basename "$input" | sed 's/^bib_gen_/bib_sim_/')
            output="$sim_dir/$name"
            temporary="$WORKDIR/${split}_${name%.root}.partial.root"
            if [ -s "$output" ]; then
                echo "echo 'skip $output'"
                continue
            fi
            echo "echo '--- k=$K $POLARITY $split $name ---'"
            echo "rm -f '$temporary'"
            echo "ddsim --steeringFile '$BENCH/simulation/steer_baseline.py' \\"
            echo "  --numberOfEvents 1 --inputFiles '$input' --outputFile '$temporary'"
            echo "python3 -c \"import uproot; f=uproot.open('$temporary'); assert 'events' in f; assert f['events'].num_entries == 1; assert 'podio_metadata' in f\""
            echo "mv '$temporary' '$output'"
        done
    done
} > "$LOOP"

apptainer exec --bind /oscar:/oscar "$CONTAINER" bash "$LOOP"
