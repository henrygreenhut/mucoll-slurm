#!/bin/bash
#SBATCH --job-name=count_v2_11
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
set -euo pipefail

: "${CT_DIR:?set CT_DIR to the count_tracker directory}"
: "${INPUT_EVENT:?set INPUT_EVENT to an existing paired event directory}"
: "${OUTPUT:?set OUTPUT to a new v2.11 event directory}"

export IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"

bash "$CT_DIR/v2_11/run_existing_pair_v2_11.sh" \
    --input-event "$INPUT_EVENT" \
    --output "$OUTPUT"
