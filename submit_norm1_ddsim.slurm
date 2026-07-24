#!/bin/bash
#SBATCH -J norm1_ddsim
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8g
#SBATCH -t 08:00:00
#SBATCH --array=0-63
#SBATCH -o norm1_ddsim_%x_%A_%a.out
#SBATCH -e norm1_ddsim_%x_%A_%a.err

# 64-way sharded array job (same pattern as submit_norm1_gen_write.slurm),
# NOT one task per cycle (%array=0-6665%64) as originally written -- that
# version was rejected by sbatch outright ("Invalid job array
# specification"): this account's normal-QOS MaxSubmitPU=1000 counts each
# array task as a separate submitted job, and 6666 > 1000. Since only 64
# can ever run concurrently anyway (MaxTRESPU=cpu=64), 64 shards each
# looping over ~104 cycles sequentially reaches the identical concurrency
# ceiling while staying at 64 submitted jobs, well under the 1000 cap.
#
# One apptainer exec + one setup_mucoll.sh/setup_config.sh source per
# TASK (not per file): the ~75s Geant4/geometry startup cost is inherent
# to each ddsim CLI invocation itself and unavoidable either way, but
# avoiding ~104 separate container-startup overheads per task (instead of
# just 1) is still worth the trivial restructuring.
#
# At ~114s/file (single-file smoke test), 104 files/shard sequentially
# implied =~3.2h/shard -- underestimated in practice: the original -t
# 04:00:00 wasn't enough (most shards TIMEOUT'd partway through on the
# real MUPLUS run, ending at 5530/6666 = 83% done), meaning per-file time
# varies more across the real cycle-to-cycle particle-count spread than
# the single cycle-0 smoke test suggested. Bumped to 8h with headroom.
#
# Idempotent/resumable: skips any cycle whose output file already exists,
# so a re-submission after a TIMEOUT or partial failure only redoes
# whatever's still missing -- just resubmit the same command again.
#
# Submit once per polarity, after submit_norm1_gen_write.slurm for that
# polarity has fully completed:
#   sbatch submit_norm1_ddsim.slurm MUPLUS
#   sbatch submit_norm1_ddsim.slurm MUMINUS

set -e
cd "$SLURM_SUBMIT_DIR"

POLARITY=$1
if [ "$POLARITY" != "MUPLUS" ] && [ "$POLARITY" != "MUMINUS" ]; then
    echo "usage: sbatch submit_norm1_ddsim.slurm {MUPLUS|MUMINUS}"
    exit 1
fi

NUM_SHARDS=64
BENCH=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/mucoll-benchmarks
CONTAINER=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif
GEN_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/bib_norm1_reconstructed/GEN/${POLARITY}
SIM_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/bib_norm1_reconstructed/SIM/${POLARITY}
mkdir -p "$SIM_DIR"

# sort -V (version sort): handles the embedded cycle numbers naturally
# (bib_gen_2 before bib_gen_1000), unlike a plain lexicographic sort.
mapfile -t ALL_FILES < <(ls "$GEN_DIR"/*.edm4hep.root | sort -V)
N=${#ALL_FILES[@]}
if [ "$N" -eq 0 ]; then
    echo "ERROR: no *.edm4hep.root files in $GEN_DIR" >&2
    echo "  (did submit_norm1_gen_write.slurm finish for $POLARITY?)" >&2
    exit 1
fi

# This shard's interleaved subset: indices shard_index, shard_index+64,
# shard_index+128, ... -- same load-balancing rationale as the GEN-write
# script's Python-side files[shard::num_shards].
SHARD_FILES=()
for ((i = SLURM_ARRAY_TASK_ID; i < N; i += NUM_SHARDS)); do
    SHARD_FILES+=("${ALL_FILES[$i]}")
done
echo "shard $SLURM_ARRAY_TASK_ID/$NUM_SHARDS: ${#SHARD_FILES[@]} of $N files"

WORKDIR=$(mktemp -d "/oscar/scratch/$USER/norm1_ddsim_${POLARITY}_shard${SLURM_ARRAY_TASK_ID}.XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

# Build the in-container loop as a single script so setup is sourced once
# and every file in this shard runs inside the same apptainer exec.
LOOP_SCRIPT="$WORKDIR/run_shard.sh"
{
    echo "source /opt/setup_mucoll.sh"
    echo "source '$BENCH/setup_config.sh' '$BENCH' MAIA_v0"
    echo "cd '$WORKDIR'"
    for input_file in "${SHARD_FILES[@]}"; do
        cycle=$(basename "$input_file" | sed -E 's/bib_gen_([0-9]+)\.edm4hep\.root/\1/')
        output_file="$SIM_DIR/bib_sim_${cycle}.edm4hep.root"
        # Idempotent: skip cycles already done (e.g. a prior partial run).
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
