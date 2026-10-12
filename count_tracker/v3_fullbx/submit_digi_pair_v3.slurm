#!/bin/bash
# Digitize a prepared full-BX SIM/COUNT pair with the standard v3 stack.
# Submit as an array 0-1: task 0 is SIM and task 1 is COUNT.
#SBATCH --job-name=count_v3_digi
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?set CT_DIR to count_tracker}"
: "${INPUT_ROOT:?set INPUT_ROOT to the prepared SIM/COUNT event directory}"
: "${EVENT_ROOT:?set EVENT_ROOT to a new v3 output directory}"
: "${IMAGE:?set the v3 image}"
: "${BENCHMARK_DIR:?set the pinned benchmark checkout}"

case "${SLURM_ARRAY_TASK_ID:-}" in
    0) SAMPLE=SIM ;;
    1) SAMPLE=COUNT ;;
    *) echo "Submit with --array=0-1" >&2; exit 1 ;;
esac

INPUT_FILE="$INPUT_ROOT/$SAMPLE/input/input.edm4hep.root"
OUTPUT_DIR="$EVENT_ROOT/$SAMPLE/digi"
[ -f "$INPUT_FILE" ] || { echo "Missing input: $INPUT_FILE" >&2; exit 1; }
if [ -f "$OUTPUT_DIR/digi_output.edm4hep.root" ]; then
    echo "$SAMPLE v3 digitization already exists"
    exit 0
fi

IMAGE="$IMAGE" BENCHMARK_DIR="$BENCHMARK_DIR" INPUT_FILE="$INPUT_FILE" \
    OUTPUT_DIR="$OUTPUT_DIR" NUM_EVENTS=1 \
    bash "$CT_DIR/v3_fullbx/run_reco_v3_theta.sh" --stage digi

cat > "$OUTPUT_DIR/run_metadata.txt" <<EOF
sample=$SAMPLE
input=$INPUT_FILE
image=$IMAGE
benchmark=$BENCHMARK_DIR
benchmark_commit=$(git -C "$BENCHMARK_DIR" rev-parse HEAD)
steering=$CT_DIR/v3_fullbx/tracker_reco_override_v3_theta.py
steering_sha256=$(sha256sum "$CT_DIR/v3_fullbx/tracker_reco_override_v3_theta.py" | cut -d' ' -f1)
configuration=standard_v3_tracker_digitization
EOF
