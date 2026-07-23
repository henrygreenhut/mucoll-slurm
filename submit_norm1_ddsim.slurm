#!/bin/bash
#SBATCH -J norm1_ddsim
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8g
#SBATCH -t 00:30:00
#SBATCH --array=0-6665%64
#SBATCH -o norm1_ddsim_%x_%A_%a.out
#SBATCH -e norm1_ddsim_%x_%A_%a.err

# One ddsim invocation per reconstructed norm1 GEN cycle, SLURM-native
# throttle (%64) instead of manual sharding: unlike the GEN-write step,
# each task here is substantial (smoke test: 114s -- 75s Geant4/geometry
# startup + 39s/event), so per-task overhead is negligible relative to
# the work itself, and SLURM's own array-index-to-concurrency-limit
# mechanism is simpler than hand-rolling it. %64 matches this account's
# normal-QOS MaxTRESPU cpu cap on the batch partition (submit_norm1_gen_
# write.slurm's sharding hits the same ceiling from the other direction).
#
# At ~114s/task x 6666 tasks / 64 concurrent =~ 3.3h per polarity.
#
# --array=0-6665 assumes exactly 6666 reconstructed GEN files exist (true
# for the full run; will simply error out past the file-list length on a
# --max-files-limited test sample, so run submit_norm1_gen_write.slurm's
# FULL (non-sampled) output before this, not a partial one).
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

BENCH=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/mucoll-benchmarks
CONTAINER=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif
GEN_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/bib_norm1_reconstructed/GEN/${POLARITY}
SIM_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/bib_norm1_reconstructed/SIM/${POLARITY}
mkdir -p "$SIM_DIR"

# sort -V (version sort): handles the embedded cycle numbers naturally
# (bib_gen_2 before bib_gen_1000), unlike a plain lexicographic sort.
mapfile -t FILES < <(ls "$GEN_DIR"/*.edm4hep.root | sort -V)
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#FILES[@]}" ]; then
    echo "ERROR: array index $SLURM_ARRAY_TASK_ID >= ${#FILES[@]} files in $GEN_DIR" >&2
    echo "  (did submit_norm1_gen_write.slurm finish, and was it a full run, not --max-files-limited?)" >&2
    exit 1
fi
INPUT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
CYCLE=$(basename "$INPUT_FILE" | sed -E 's/bib_gen_([0-9]+)\.edm4hep\.root/\1/')
OUTPUT_FILE="$SIM_DIR/bib_sim_${CYCLE}.edm4hep.root"

echo "task $SLURM_ARRAY_TASK_ID: cycle $CYCLE"
echo "  in:  $INPUT_FILE"
echo "  out: $OUTPUT_FILE"

# Scratch workdir per task -- ddsim writes some auxiliary files alongside
# its output; keep tasks from stepping on each other.
WORKDIR=$(mktemp -d "/oscar/scratch/$USER/norm1_ddsim_${POLARITY}_${CYCLE}.XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

apptainer exec --bind /oscar:/oscar "$CONTAINER" bash -c "
source /opt/setup_mucoll.sh
source '$BENCH/setup_config.sh' '$BENCH' MAIA_v0
cd '$WORKDIR'
ddsim --steeringFile '$BENCH/simulation/steer_baseline.py' \
    --numberOfEvents 1 \
    --inputFiles '$INPUT_FILE' \
    --outputFile '$OUTPUT_FILE'
"

echo "done -> $OUTPUT_FILE"
ls -la "$OUTPUT_FILE"
