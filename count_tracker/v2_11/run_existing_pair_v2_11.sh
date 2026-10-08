#!/bin/bash
# Reconstruct an existing, already assembled SIM/COUNT input pair with v2.11.
set -euo pipefail

IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
V2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_EVENT=""
OUTPUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --input-event) INPUT_EVENT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[ -n "$INPUT_EVENT" ] || { echo "Missing --input-event" >&2; exit 1; }
[ -n "$OUTPUT" ] || { echo "Missing --output" >&2; exit 1; }
for sample in SIM COUNT; do
    input="$INPUT_EVENT/$sample/input/input.edm4hep.root"
    [ -f "$input" ] || { echo "Missing input: $input" >&2; exit 1; }
done

mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
INPUT_EVENT="$(cd "$INPUT_EVENT" && pwd)"

sha256sum \
    "$INPUT_EVENT/SIM/input/input.edm4hep.root" \
    "$INPUT_EVENT/COUNT/input/input.edm4hep.root" \
    > "$OUTPUT/input_sha256.txt"

for sample in SIM COUNT; do
    echo "--- v2.11 reconstruct existing $sample input ---"
    IMAGE_V2_11="$IMAGE_V2_11" NUM_EVENTS=1 \
        INPUT_FILE="$INPUT_EVENT/$sample/input/input.edm4hep.root" \
        OUTPUT_DIR="$OUTPUT/$sample/reco" \
        bash "$V2_DIR/run_reco_v2_11.sh"
done

cat > "$OUTPUT/comparison_provenance.txt" <<EOF
purpose=v2.11 reconstruction of an existing paired input
source_event=$INPUT_EVENT
image=$IMAGE_V2_11
sim_input=$INPUT_EVENT/SIM/input/input.edm4hep.root
count_input=$INPUT_EVENT/COUNT/input/input.edm4hep.root
input_hashes=$OUTPUT/input_sha256.txt
EOF

echo "=== v2.11 pair done: $OUTPUT ==="
