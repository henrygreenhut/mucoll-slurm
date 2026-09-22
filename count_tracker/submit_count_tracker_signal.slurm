#!/bin/bash
# Produce the shared neutrino signal file for a multi-event tracker cohort.
#SBATCH --job-name=count_signal
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail

: "${CT_DIR:?}"
: "${SIGNAL:?}"
: "${NEVENTS:?}"
: "${IMAGE:?}"
: "${BENCHMARK_DIR:?}"

SEED="${SEED:-260922}"

bash "$CT_DIR/count_tracker_signal.sh" \
    --nevents "$NEVENTS" --seed "$SEED" --outfile "$SIGNAL"
