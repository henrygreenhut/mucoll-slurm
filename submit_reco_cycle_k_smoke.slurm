#!/bin/bash
#SBATCH -J recok_cycle_smoke
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 00:45:00
#SBATCH --exclude=gpu3005
#SBATCH -o recok_cycle_smoke_%j.out
#SBATCH -e recok_cycle_smoke_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

BASE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_cycle_k
STORES=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores
BENCH=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/mucoll-benchmarks
CONTAINER=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif
OUT=/oscar/scratch/$USER/mucoll/libtest/reco_cycle_k_smoke
WORKDIR=$(mktemp -d "/oscar/scratch/$USER/recok_cycle_smoke.XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

cycle=$(python3 -c "import json; print(json.load(open('$BASE/pools/manifest.json'))['splits']['train']['cycles'][0])")

for polarity in MUPLUS MUMINUS; do
    for k in 7 21; do
        gen_dir="$OUT/GEN/k${k}/train/${polarity}"
        sim_dir="$OUT/SIM/k${k}/train/${polarity}"
        mkdir -p "$gen_dir" "$sim_dir"
        apptainer exec --bind /oscar:/oscar "$CONTAINER" bash -c "
            source /opt/setup_mucoll.sh
            python3 '$SLURM_SUBMIT_DIR/reco_cycle_k_library.py' write-gen \
                --bank '$STORES/gen_split_mothers_${polarity}.h5' \
                --manifest '$BASE/pools/manifest.json' \
                --benchmarks-dir '$BENCH' \
                --split train \
                --reuse-k '$k' \
                --output-dir '$gen_dir' \
                --max-cycles 1 \
                --validate
        "
        input="$gen_dir/bib_gen_cycle_$(printf '%06d' "$cycle").edm4hep.root"
        output="$sim_dir/bib_sim_cycle_$(printf '%06d' "$cycle").edm4hep.root"
        temporary="$WORKDIR/k${k}_${polarity}.partial.root"
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
        printf '%-8s k=%-2s cycle=%s GEN=%s SIM=%s\n' \
            "$polarity" "$k" "$cycle" \
            "$(du -h "$input" | cut -f1)" "$(du -h "$output" | cut -f1)"
    done
done
