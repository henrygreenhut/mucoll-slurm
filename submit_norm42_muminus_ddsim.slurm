#!/bin/bash
#SBATCH -J norm42_muminus_ddsim
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8g
#SBATCH -t 08:00:00
#SBATCH --array=0-63
#SBATCH -o norm42_muminus_ddsim_%x_%A_%a.out
#SBATCH -e norm42_muminus_ddsim_%x_%A_%a.err

# One-off gap-fill: bib-v3p0-fmt2-norm42-RandomRot has GEN/MUPLUS,
# GEN/MUMINUS, and SIM/MUPLUS (6666 files each) but SIM/MUMINUS is empty
# (0/6666) -- reco_libtest_prepare_pools.py requires both polarities present
# in both the norm1 and norm42 SIM libraries, so this blocks pool-building.
#
# Unlike submit_norm1_ddsim.slurm, no dedup/reconstruction step is needed
# here: norm42's GEN files are already proper multi-particle BIB events (not
# norm1's original degenerate 1-event-per-file GEN), so this is a plain
# ddsim pass over existing GEN/MUMINUS input. Same 64-way interleaved-shard
# pattern as submit_norm1_ddsim.slurm for the same reason: this account's
# normal-QOS MaxSubmitPU=1000 and the batch partition's MaxTRESPU=cpu=64
# ceiling make one-task-per-file (6666 tasks) both rejected outright and no
# faster than 64 shards looping sequentially anyway.
#
# Idempotent: skips any cycle whose output file already exists, so a
# resubmit after a TIMEOUT only redoes what's missing.
#
# Output goes under hgreenhu's own tree, NOT into the shared
# bib-v3p0-fmt2-norm42-RandomRot directory itself -- that's owned by
# mleblan6 (group ccv-mleblan6, r-x only, confirmed via getfacl: no write
# access for group members). reco_libtest_prepare_pools.py needs both
# polarities under one --norm42-sim base dir; bridge that with a small
# combined dir of symlinks (existing read-only MUPLUS + this job's own
# MUMINUS) built separately after this completes -- see chat.
#
# Submit once:
#   sbatch submit_norm42_muminus_ddsim.slurm

set -e
cd "$SLURM_SUBMIT_DIR"

NUM_SHARDS=64
BENCH=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/mucoll-benchmarks
CONTAINER=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif
GEN_DIR=/oscar/data/mleblan6/mucoll/bib/bib-v3p0-fmt2-norm42-RandomRot/GEN/MUMINUS
SIM_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/bib_norm42_muminus_sim/MUMINUS
mkdir -p "$SIM_DIR"

mapfile -t ALL_FILES < <(ls "$GEN_DIR"/*.edm4hep.root | sort -V)
N=${#ALL_FILES[@]}
if [ "$N" -eq 0 ]; then
    echo "ERROR: no *.edm4hep.root files in $GEN_DIR" >&2
    exit 1
fi

SHARD_FILES=()
for ((i = SLURM_ARRAY_TASK_ID; i < N; i += NUM_SHARDS)); do
    SHARD_FILES+=("${ALL_FILES[$i]}")
done
echo "shard $SLURM_ARRAY_TASK_ID/$NUM_SHARDS: ${#SHARD_FILES[@]} of $N files"

WORKDIR=$(mktemp -d "/oscar/scratch/$USER/norm42_muminus_ddsim_shard${SLURM_ARRAY_TASK_ID}.XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

LOOP_SCRIPT="$WORKDIR/run_shard.sh"
{
    echo "source /opt/setup_mucoll.sh"
    echo "source '$BENCH/setup_config.sh' '$BENCH' MAIA_v0"
    echo "cd '$WORKDIR'"
    for input_file in "${SHARD_FILES[@]}"; do
        cycle=$(basename "$input_file" | sed -E 's/bib_gen_([0-9]+)\.edm4hep\.root/\1/')
        output_file="$SIM_DIR/bib_sim_${cycle}.edm4hep.root"
        echo "if [ -f '$output_file' ]; then"
        echo "    echo 'skip (already exists): $output_file'"
        echo "else"
        echo "    echo '--- cycle $cycle ---'"
        echo "    ddsim --steeringFile '$BENCH/simulation/steer_baseline.py' \\"
        echo "        --numberOfEvents 1 \\"
        echo "        --inputFiles '$input_file' \\"
        echo "        --outputFile '$output_file'"
        echo "fi"
    done
} > "$LOOP_SCRIPT"

apptainer exec --bind /oscar:/oscar "$CONTAINER" bash "$LOOP_SCRIPT"

echo "shard $SLURM_ARRAY_TASK_ID done"
