#!/bin/bash
set -euo pipefail

IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
INPUT_FILE="${INPUT_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
NUM_EVENTS="${NUM_EVENTS:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for name in INPUT_FILE OUTPUT_DIR; do
    [ -n "${!name}" ] || { echo "Missing required environment: $name" >&2; exit 1; }
done
for path in "$IMAGE_V2_11" "$INPUT_FILE" "$SCRIPT_DIR/tracker_reco_override_v2_11.py"; do
    [ -e "$path" ] || { echo "Missing required path: $path" >&2; exit 1; }
done

INPUT_DIR="$(cd "$(dirname "$INPUT_FILE")" && pwd)"
INPUT_NAME="$(basename "$INPUT_FILE")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
for output in digi_output.edm4hep.root reco_output.edm4hep.root; do
    [ ! -e "$OUTPUT_DIR/$output" ] || {
        echo "Refusing to overwrite existing output: $OUTPUT_DIR/$output" >&2
        exit 1
    }
done

apptainer exec \
    --pwd /work/output \
    --bind "$SCRIPT_DIR:/work/count-tracker-v2:ro,$INPUT_DIR:/work/input:ro,$OUTPUT_DIR:/work/output" \
    "$IMAGE_V2_11" bash -lc '
        set -eo pipefail
        source /opt/setup_mucoll.sh
        set -u

        export MUCOLL_GEO="$(find /opt/spack/opt/spack -type f \
            -path "*/k4geo*/share/k4geo/MuColl/MAIA/compact/MAIA_v0/MAIA_v0.xml" \
            | head -n1)"
        ACTS_DATA="$(find /opt/spack/opt/spack -type d \
            -path "*/actstracking*/share/ACTSTracking/data" | head -n1)"
        export MUCOLL_TGEO="$ACTS_DATA/MAIA_v0.root"
        export MUCOLL_TGEO_DESC="$ACTS_DATA/MAIA_v0.json"
        export MUCOLL_MATMAP="$ACTS_DATA/MAIA_v0_material.json"
        for path in "$MUCOLL_GEO" "$MUCOLL_TGEO" "$MUCOLL_TGEO_DESC" "$MUCOLL_MATMAP"; do
            [ -f "$path" ] || { echo "Missing v2.11 detector asset: $path" >&2; exit 1; }
        done

        export V2_NUM_EVENTS="$3"
        export V2_INPUT_FILE="/work/input/$1"
        export V2_OUTPUT_FILE=/work/output/digi_output.edm4hep.root
        k4run /work/count-tracker-v2/tracker_reco_override_v2_11.py --stage digi

        export V2_INPUT_FILE=/work/output/digi_output.edm4hep.root
        export V2_OUTPUT_FILE=/work/output/reco_output.edm4hep.root
        k4run /work/count-tracker-v2/tracker_reco_override_v2_11.py --stage reco
    ' _ "$INPUT_NAME" "$OUTPUT_DIR" "$NUM_EVENTS"

echo "v2.11 digitization: $OUTPUT_DIR/digi_output.edm4hep.root"
echo "v2.11 reconstruction: $OUTPUT_DIR/reco_output.edm4hep.root"
