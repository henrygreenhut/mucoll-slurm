#!/bin/bash
#SBATCH -J reco_unconed
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 08:00:00
#SBATCH --exclude=node2336
#SBATCH --array=0-63
#SBATCH -o reco_calo_unconed_%A_%a.out
#SBATCH -e reco_calo_unconed_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source config.sh

MANIFEST=${1:?usage: sbatch $0 MANIFEST.tsv}
NUM_SHARDS=${NUM_SHARDS:?NUM_SHARDS was not exported}
TASKS=$(($(wc -l < "$MANIFEST") - 1))

MAIA="$MUCOLL_BENCHMARKS_PATH/configs/MAIAConfig/MAIAConfig"
grep -q -- "--disableCaloConing" "$MAIA/digi_args.py"
grep -Fq 'TrackCollections = ["SiTracks"]' "$MAIA/ParticleFlow/pandora.py"
grep -Fq 'RelTrackCollections = ["MergedTrackerHitsRelations"]' \
    "$MAIA/ParticleFlow/pandora.py"
grep -Fq 'MergeMCParticles = False' "$MAIA/Overlay/overlay_BIB.py"

REFERENCE_CONFIG=/oscar/scratch/$USER/mucoll/libtest/reco_n420_calo_unconed/reco_libtest_n420_U/train/job_0/config_sha256.txt
if [ ! -f "$REFERENCE_CONFIG" ]; then
    echo "missing N=420 configuration reference: $REFERENCE_CONFIG" >&2
    exit 1
fi
if ! sha256sum --check --status "$REFERENCE_CONFIG"; then
    echo "current MAIA configuration differs from the N=420 unconed production" >&2
    exit 1
fi

echo "shard $SLURM_ARRAY_TASK_ID/$NUM_SHARDS; $TASKS logical chunks"
for ((row = SLURM_ARRAY_TASK_ID + 1; row <= TASKS; row += NUM_SHARDS)); do
    apptainer exec --bind /oscar:/oscar "$IMAGE" \
        bash "$SLURM_SUBMIT_DIR/run_reco_calo_unconed_task.sh" \
        "$MANIFEST" "$row"
done
