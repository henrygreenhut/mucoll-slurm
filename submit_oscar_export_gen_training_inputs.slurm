#!/bin/bash
#SBATCH -J export_gen_inputs
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16g
#SBATCH -t 02:00:00
#SBATCH -o export_gen_inputs_%j.out
#SBATCH -e export_gen_inputs_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "usage: $0 RESULT_DIR OUTPUT [EPOCH] [EVENTS_PER_CLASS]"
    exit 2
fi

RESULT_DIR=$1
OUTPUT=$2
EPOCH=${3:-0}
EVENTS_PER_CLASS=${4:-1}

module load ngc-tensorflow-container/25.02-tf2-py3-azkb

apptainer exec "$NGC_TENSORFLOW_CONTAINER" \
    python -u export_gen_training_inputs.py \
        --result-dir "$RESULT_DIR" \
        --output "$OUTPUT" \
        --epoch "$EPOCH" \
        --events-per-class "$EVENTS_PER_CLASS"
