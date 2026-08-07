#!/bin/bash
#SBATCH -J recok_smoke
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 08:00:00
#SBATCH -o recok_smoke_%j.out
#SBATCH -e recok_smoke_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

BASE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_variable_k
STORE_BASE=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores
BENCH=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/mucoll-benchmarks
CONTAINER=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif
OUT=/oscar/scratch/$USER/mucoll/libtest/reco_variable_k_native1_smoke
WORKDIR=$(mktemp -d "/oscar/scratch/$USER/recok_smoke.XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT
find "$OUT/GEN" -type f -name '*.partial.*' -delete 2>/dev/null || true

for polarity in MUPLUS MUMINUS; do
    for k in 1 7 21; do
        sample=k${k}
        [ "$k" -eq 1 ] && sample=k1_native
        gen_dir="$OUT/GEN/${sample}/$polarity"
        sim_dir="$OUT/SIM/${sample}/$polarity"
        mkdir -p "$gen_dir" "$sim_dir"
        apptainer exec --bind /oscar:/oscar "$CONTAINER" bash -c "
            source /opt/setup_mucoll.sh
            python3 '$SLURM_SUBMIT_DIR/reco_variable_k_library.py' write-gen \
                --bank '$STORE_BASE/gen_split_mothers_${polarity}.h5' \
                --manifest-dir '$BASE/manifest_native1' \
                --benchmarks-dir '$BENCH' \
                --split train \
                --reuse-k '$k' \
                --output-dir '$gen_dir' \
                --max-chunks 1 \
                --validate
        "
        input="$gen_dir/bib_gen_chunk_000000.edm4hep.root"
        output="$sim_dir/bib_sim_chunk_000000.edm4hep.root"
        temporary="$WORKDIR/k${k}_${polarity}.partial.root"
        if [ ! -s "$output" ]; then
            apptainer exec --bind /oscar:/oscar "$CONTAINER" bash -c "
                source /opt/setup_mucoll.sh
                source '$BENCH/setup_config.sh' '$BENCH' MAIA_v0
                cd '$WORKDIR'
                ddsim --steeringFile '$BENCH/simulation/steer_baseline.py' \
                    --numberOfEvents 1 \
                    --inputFiles '$input' \
                    --outputFile '$temporary'
                python3 -c \"import uproot; f=uproot.open('$temporary'); assert 'events' in f; assert f['events'].num_entries == 1; assert 'podio_metadata' in f\"
                mv '$temporary' '$output'
            "
        fi
        printf '%-8s %-7s GEN=%s SIM=%s\n' \
            "$polarity" "k=$k" "$(du -h "$input" | cut -f1)" "$(du -h "$output" | cut -f1)"
    done
done
