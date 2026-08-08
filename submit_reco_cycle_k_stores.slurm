#!/bin/bash
#SBATCH -J recok_cycle_stores
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH --exclude=node2336
#SBATCH -o recok_cycle_stores_%j.out
#SBATCH -e recok_cycle_stores_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

BASE=/oscar/scratch/$USER/mucoll/libtest
ENV=/oscar/scratch/$USER/mucoll/envs/reco-store
RECO=$BASE/reco_cycle_k_n420_unconed
STORES=$BASE/reco_cycle_k_n420_unconed_stores
SOURCE_STORES=$BASE/reco_n420_pfn_stores_calo_unconed_val25
MANIFEST=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_cycle_k/pools/manifest.json

"$ENV/bin/python" make_reco_libtest_stores.py \
    --n-files 420 \
    --reco-dir "$RECO" \
    --outdir "$STORES" \
    --pool-manifest "$MANIFEST" \
    --samples k7 k21 \
    --study-template 'reco_cycle_n{n_files}_{sample}' \
    --train-events 2000 \
    --val-events 2000 \
    --test-events 2000

for split in train val test; do
    source="$SOURCE_STORES/n420_U_${split}.h5"
    destination="$STORES/n420_k1_${split}.h5"
    if [ ! -s "$source" ]; then
        echo "missing K=1 store: $source" >&2
        exit 1
    fi
    if [ -L "$destination" ] && [ "$(readlink -f "$destination")" = "$(readlink -f "$source")" ]; then
        continue
    fi
    if [ -e "$destination" ] || [ -L "$destination" ]; then
        echo "refusing conflicting K=1 store: $destination" >&2
        exit 1
    fi
    ln -s "$source" "$destination"
done

echo "stores: $STORES"
