#!/bin/bash
# Actual one-event v2.11 integration smoke test for an existing paired input.
set -euo pipefail

IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
CT_DIR="${CT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
V2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_EVENT=""
OUTPUT=""
SAMPLE="SIM"
HITS_PER_COLLECTION=128

while [ $# -gt 0 ]; do
    case "$1" in
        --input-event) INPUT_EVENT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --sample) SAMPLE="$2"; shift 2 ;;
        --hits-per-collection) HITS_PER_COLLECTION="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[ -n "$INPUT_EVENT" ] || { echo "Missing --input-event" >&2; exit 1; }
[ -n "$OUTPUT" ] || { echo "Missing --output" >&2; exit 1; }
case "$SAMPLE" in
    SIM|COUNT|both) : ;;
    *) echo "--sample must be SIM, COUNT, or both" >&2; exit 1 ;;
esac
case "$HITS_PER_COLLECTION" in
    ''|*[!0-9]*) echo "--hits-per-collection must be a positive integer" >&2; exit 1 ;;
    *) [ "$HITS_PER_COLLECTION" -gt 0 ] || {
        echo "--hits-per-collection must be positive" >&2; exit 1;
    } ;;
esac
for path in "$IMAGE_V2_11" "$V2_DIR/make_smoke_input_v2_11.py" \
            "$V2_DIR/validate_v2_11_output.py" "$V2_DIR/preflight_v2_11.sh" \
            "$V2_DIR/run_reco_v2_11.sh"; do
    [ -e "$path" ] || { echo "Missing required path: $path" >&2; exit 1; }
done

if [ "$SAMPLE" = both ]; then
    samples=(SIM COUNT)
else
    samples=("$SAMPLE")
fi
for sample in "${samples[@]}"; do
    input="$INPUT_EVENT/$sample/input/input.edm4hep.root"
    [ -f "$input" ] || { echo "Missing source input: $input" >&2; exit 1; }
done
[ ! -e "$OUTPUT" ] || { echo "Refusing to replace smoke output: $OUTPUT" >&2; exit 1; }
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
INPUT_EVENT="$(cd "$INPUT_EVENT" && pwd)"

run_v2_python() {
    apptainer exec --pwd "$CT_DIR" --bind /oscar:/oscar,"$CT_DIR:$CT_DIR:ro" \
        "$IMAGE_V2_11" bash -lc 'source /opt/setup_mucoll.sh; python3 "$@"' _ "$@"
}

for sample in "${samples[@]}"; do
    source_input="$INPUT_EVENT/$sample/input/input.edm4hep.root"
    smoke_input_dir="$OUTPUT/$sample/input"
    reco_dir="$OUTPUT/$sample/reco"

    echo "=== prepare reduced $sample input ==="
    run_v2_python "$V2_DIR/make_smoke_input_v2_11.py" \
        --input "$source_input" \
        --hits-per-collection "$HITS_PER_COLLECTION" \
        --output "$smoke_input_dir"
    run_v2_python "$V2_DIR/validate_v2_11_output.py" \
        --stage input --input "$smoke_input_dir/input.edm4hep.root" \
        > "$OUTPUT/$sample/input_validation.json"

    if [ "$sample" = "${samples[0]}" ]; then
        echo "=== read-only configuration preflight ==="
        IMAGE_V2_11="$IMAGE_V2_11" \
            INPUT_FILE="$smoke_input_dir/input.edm4hep.root" \
            bash "$V2_DIR/preflight_v2_11.sh"
    fi

    echo "=== digitize reduced $sample event ==="
    IMAGE_V2_11="$IMAGE_V2_11" INPUT_FILE="$smoke_input_dir/input.edm4hep.root" \
        OUTPUT_DIR="$reco_dir" NUM_EVENTS=1 \
        bash "$V2_DIR/run_reco_v2_11.sh" --stage digi
    run_v2_python "$V2_DIR/validate_v2_11_output.py" \
        --stage digi --input "$reco_dir/digi_output.edm4hep.root" \
        > "$OUTPUT/$sample/digi_validation.json"

    echo "=== reconstruct reduced $sample event ==="
    IMAGE_V2_11="$IMAGE_V2_11" DIGI_FILE="$reco_dir/digi_output.edm4hep.root" \
        OUTPUT_DIR="$reco_dir" NUM_EVENTS=1 \
        bash "$V2_DIR/run_reco_v2_11.sh" --stage reco
    run_v2_python "$V2_DIR/validate_v2_11_output.py" \
        --stage reco --input "$reco_dir/reco_output.edm4hep.root" \
        > "$OUTPUT/$sample/reco_validation.json"
done

echo "=== v2.11 integration smoke passed: $OUTPUT ==="
