#!/bin/bash
# Digitize one prepared BIB-only SIM/COUNT pair. Submit as --array=0-1.
# Task 0 is SIM and task 1 is COUNT.
#SBATCH --job-name=count_v2_digi
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?set CT_DIR to the count_tracker directory}"
: "${EVENT_ROOT:?set EVENT_ROOT to the prepared v2.11 event directory}"
export IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"

case "${SLURM_ARRAY_TASK_ID:-}" in
    0) SAMPLE=SIM ;;
    1) SAMPLE=COUNT ;;
    *) echo "Submit this script with --array=0-1" >&2; exit 1 ;;
esac

INPUT_FILE="$EVENT_ROOT/$SAMPLE/input/input.edm4hep.root"
OUTPUT_DIR="$EVENT_ROOT/$SAMPLE/digi"
[ -f "$INPUT_FILE" ] || { echo "Missing input: $INPUT_FILE" >&2; exit 1; }
[ ! -e "$OUTPUT_DIR/digi_output.edm4hep.root" ] || {
    echo "$SAMPLE digitization already exists: $OUTPUT_DIR/digi_output.edm4hep.root"
    exit 0
}

IMAGE_V2_11="$IMAGE_V2_11" INPUT_FILE="$INPUT_FILE" OUTPUT_DIR="$OUTPUT_DIR" \
    NUM_EVENTS=1 bash "$CT_DIR/v2_11/run_reco_v2_11.sh" --stage digi
