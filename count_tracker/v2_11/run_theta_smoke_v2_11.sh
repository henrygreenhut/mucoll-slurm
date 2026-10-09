#!/bin/bash
# Exercise digitization plus all five paper-style theta CKF regions on a small input.
set -euo pipefail

IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
CT_DIR="${CT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
V2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_EVENT=""
OUTPUT=""
SAMPLE=SIM
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
case "$SAMPLE" in SIM|COUNT) : ;; *) echo "--sample must be SIM or COUNT" >&2; exit 1 ;; esac
case "$HITS_PER_COLLECTION" in
    ''|*[!0-9]*) echo "--hits-per-collection must be a positive integer" >&2; exit 1 ;;
    *) [ "$HITS_PER_COLLECTION" -gt 0 ] || {
        echo "--hits-per-collection must be positive" >&2; exit 1;
    } ;;
esac

SOURCE_INPUT="$INPUT_EVENT/$SAMPLE/input/input.edm4hep.root"
[ -f "$SOURCE_INPUT" ] || { echo "Missing source input: $SOURCE_INPUT" >&2; exit 1; }
[ ! -e "$OUTPUT" ] || { echo "Refusing to replace smoke output: $OUTPUT" >&2; exit 1; }
mkdir -p "$OUTPUT/$SAMPLE"
OUTPUT="$(cd "$OUTPUT" && pwd)"

run_v2_python() {
    apptainer exec --pwd "$CT_DIR" --bind /oscar:/oscar,"$CT_DIR:$CT_DIR:ro" \
        "$IMAGE_V2_11" bash -lc 'source /opt/setup_mucoll.sh; python3 "$@"' _ "$@"
}

INPUT_DIR="$OUTPUT/$SAMPLE/input"
DIGI_DIR="$OUTPUT/$SAMPLE/digi"
run_v2_python "$V2_DIR/make_smoke_input_v2_11.py" \
    --input "$SOURCE_INPUT" --hits-per-collection "$HITS_PER_COLLECTION" \
    --output "$INPUT_DIR"

IMAGE_V2_11="$IMAGE_V2_11" INPUT_FILE="$INPUT_DIR/input.edm4hep.root" \
    OUTPUT_DIR="$DIGI_DIR" NUM_EVENTS=1 \
    bash "$V2_DIR/run_reco_v2_11.sh" --stage digi

LOWER=(0 30 70 110 150)
UPPER=(30 70 110 150 180)
for region in 0 1 2 3 4; do
    min="${LOWER[$region]}"
    max="${UPPER[$region]}"
    tag="theta_$(printf '%03d' "$min")_$(printf '%03d' "$max")"
    echo "=== $SAMPLE regional smoke: [$min, $max) degrees ==="
    IMAGE_V2_11="$IMAGE_V2_11" DIGI_FILE="$DIGI_DIR/digi_output.edm4hep.root" \
        OUTPUT_DIR="$OUTPUT/$SAMPLE/theta_reco/$tag" NUM_EVENTS=1 \
        bash "$V2_DIR/run_reco_v2_11.sh" \
        --stage reco --theta-min "$min" --theta-max "$max"
done

run_v2_python "$V2_DIR/validate_theta_partition_v2_11.py" \
    --digi-file "$DIGI_DIR/digi_output.edm4hep.root" \
    --regional-root "$OUTPUT/$SAMPLE/theta_reco" \
    --output "$OUTPUT/$SAMPLE/theta_report.json"

echo "=== v2.11 regional integration smoke passed: $OUTPUT/$SAMPLE ==="
