#!/bin/bash
set -euo pipefail

IMAGE="${IMAGE:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif}"
DIGI_FILE="${DIGI_FILE:?set DIGI_FILE}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
THETA_MIN="${THETA_MIN:?set THETA_MIN}"
THETA_MAX="${THETA_MAX:?set THETA_MAX}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for path in "$IMAGE" "$DIGI_FILE" \
        "$SCRIPT_DIR/split_digi_by_theta_v3.py" \
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
    "$IMAGE" bash -lc '
        set +u
        source /opt/setup_mucoll.sh 2>/dev/null || true
        set -eu
        python3 /work/count-tracker-v3/split_digi_by_theta_v3.py \
            --input "/work/input/$1" \
            --output /work/output/split_output.edm4hep.root \
            --theta-min "$2" --theta-max "$3"
        python3 /work/count-tracker-v3/validate_theta_split.py \
            --input /work/output/split_output.edm4hep.root \
            --theta-min "$2" --theta-max "$3"
    ' _ "$INPUT_NAME" "$THETA_MIN" "$THETA_MAX"

echo "theta split: $OUTPUT_DIR/split_output.edm4hep.root"
