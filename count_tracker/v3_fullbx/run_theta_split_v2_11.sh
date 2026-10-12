#!/bin/bash
set -euo pipefail

IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
DIGI_FILE="${DIGI_FILE:?set DIGI_FILE}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
THETA_MIN="${THETA_MIN:?set THETA_MIN}"
THETA_MAX="${THETA_MAX:?set THETA_MAX}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for path in "$IMAGE_V2_11" "$DIGI_FILE" \
        "$SCRIPT_DIR/split_digi_by_theta_v2_11.py" \
        "$SCRIPT_DIR/validate_theta_split.py"; do
    [ -e "$path" ] || { echo "Missing required path: $path" >&2; exit 1; }
done

INPUT_DIR="$(cd "$(dirname "$DIGI_FILE")" && pwd)"
INPUT_NAME="$(basename "$DIGI_FILE")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
[ ! -e "$OUTPUT_DIR/split_output.edm4hep.root" ] || {
    echo "Refusing to overwrite $OUTPUT_DIR/split_output.edm4hep.root" >&2; exit 1;
}

apptainer exec \
    --pwd /work/output \
    --bind "$SCRIPT_DIR:/work/count-tracker-v3:ro,$INPUT_DIR:/work/input:ro,$OUTPUT_DIR:/work/output" \
    "$IMAGE_V2_11" bash -lc '
        set -eo pipefail
        source /opt/setup_mucoll.sh
        set -u
        export SPLIT_INPUT_FILE="/work/input/$1"
        export SPLIT_OUTPUT_FILE=/work/output/split_output.edm4hep.root
        export THETA_MIN="$2"
        export THETA_MAX="$3"
        set +e
        k4run /work/count-tracker-v3/split_digi_by_theta_v2_11.py
        status=$?
        set -e
        if [ "$status" -ne 0 ] && [ "$status" -ne 134 ]; then
            echo "theta splitter failed with exit status $status" >&2
            exit "$status"
        fi
        python3 /work/count-tracker-v3/validate_theta_split.py \
            --input "$SPLIT_OUTPUT_FILE"
        if [ "$status" -ne 0 ]; then
            echo "WARNING: accepting v2.11 cleanup abort after validating the split output" >&2
        fi
    ' _ "$INPUT_NAME" "$THETA_MIN" "$THETA_MAX"

echo "theta split: $OUTPUT_DIR/split_output.edm4hep.root"
