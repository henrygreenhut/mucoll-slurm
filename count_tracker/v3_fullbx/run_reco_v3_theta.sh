#!/bin/bash
set -euo pipefail

IMAGE="${IMAGE:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif}"
BENCHMARK_DIR="${BENCHMARK_DIR:-}"
INPUT_FILE="${INPUT_FILE:-}"
DIGI_FILE="${DIGI_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
NUM_EVENTS="${NUM_EVENTS:-1}"
RUN_STAGE="${RUN_STAGE:-both}"
SPLIT_INPUT="${SPLIT_INPUT:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [ $# -gt 0 ]; do
    case "$1" in
        --stage) RUN_STAGE="$2"; shift 2 ;;
        --split-input) SPLIT_INPUT=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "$RUN_STAGE" in
    digi|reco) : ;;
    *) echo "--stage must be digi or reco" >&2; exit 1 ;;
esac
[ -n "$BENCHMARK_DIR" ] || { echo "Missing BENCHMARK_DIR" >&2; exit 1; }
[ -n "$OUTPUT_DIR" ] || { echo "Missing OUTPUT_DIR" >&2; exit 1; }

if [ "$RUN_STAGE" = digi ]; then
    [ -n "$INPUT_FILE" ] || { echo "Digitization needs INPUT_FILE" >&2; exit 1; }
    SOURCE_FILE="$INPUT_FILE"
    EXPECTED_OUTPUT=digi_output.edm4hep.root
    [ "$SPLIT_INPUT" = 0 ] || { echo "--split-input applies only to reco" >&2; exit 1; }
else
    [ -n "$DIGI_FILE" ] || { echo "Reconstruction needs DIGI_FILE" >&2; exit 1; }
    SOURCE_FILE="$DIGI_FILE"
    EXPECTED_OUTPUT=reco_output.edm4hep.root
    [ "$SPLIT_INPUT" = 1 ] || { echo "Regional full-BX reco needs --split-input" >&2; exit 1; }
fi

for path in "$IMAGE" "$BENCHMARK_DIR/setup_config.sh" "$SOURCE_FILE" \
        "$SCRIPT_DIR/tracker_reco_override_v3_theta.py"; do
    [ -e "$path" ] || { echo "Missing required path: $path" >&2; exit 1; }
done

SOURCE_DIR="$(cd "$(dirname "$SOURCE_FILE")" && pwd)"
SOURCE_NAME="$(basename "$SOURCE_FILE")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
[ ! -e "$OUTPUT_DIR/$EXPECTED_OUTPUT" ] || {
    echo "Refusing to overwrite $OUTPUT_DIR/$EXPECTED_OUTPUT" >&2; exit 1;
}

apptainer exec \
    --pwd /work/output \
    --bind "$SCRIPT_DIR:/work/count-tracker-v3:ro,$BENCHMARK_DIR:/work/mucoll-benchmarks:ro,$SOURCE_DIR:/work/input:ro,$OUTPUT_DIR:/work/output" \
    "$IMAGE" bash -lc '
        set -eo pipefail
        source /opt/setup_mucoll.sh 2>/dev/null || true
        STACK_SETUP="$(find /opt/spack/opt/spack -name setup.sh -path "*mucoll-stack*" 2>/dev/null | head -n1)"
        [ -n "$STACK_SETUP" ] && source "$STACK_SETUP"
        SETUP_CONFIG="$(mktemp)"
        sed "s/\r$//" /work/mucoll-benchmarks/setup_config.sh > "$SETUP_CONFIG"
        source "$SETUP_CONFIG" /work/mucoll-benchmarks MAIA_v0
        set -eu

        if [ "$3" = digi ]; then
            ln -sfn "/work/input/$1" /work/output/bib_flow.edm4hep.root
            k4run /work/count-tracker-v3/tracker_reco_override_v3_theta.py \
                --stage digi -n "$2"
        else
            ln -sfn "/work/input/$1" /work/output/digi_output.edm4hep.root
            k4run /work/count-tracker-v3/tracker_reco_override_v3_theta.py \
                --stage reco --split-input -n "$2"
        fi
    ' _ "$SOURCE_NAME" "$NUM_EVENTS" "$RUN_STAGE"

echo "$RUN_STAGE output: $OUTPUT_DIR/$EXPECTED_OUTPUT"
