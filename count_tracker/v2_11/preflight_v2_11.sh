#!/bin/bash
# Read-only configuration checks for the v2.11 tracker workflow.
set -euo pipefail

IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
INPUT_FILE="${INPUT_FILE:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -f "$IMAGE_V2_11" ] || { echo "Missing v2.11 image: $IMAGE_V2_11" >&2; exit 1; }
[ -n "$INPUT_FILE" ] || { echo "Set INPUT_FILE to a v2.11-native EDM4hep input" >&2; exit 1; }
[ -f "$INPUT_FILE" ] || { echo "Missing input: $INPUT_FILE" >&2; exit 1; }

INPUT_DIR="$(cd "$(dirname "$INPUT_FILE")" && pwd)"
INPUT_NAME="$(basename "$INPUT_FILE")"

apptainer exec \
    --pwd /work/count-tracker-v2 \
    --bind "$SCRIPT_DIR:/work/count-tracker-v2:ro,$INPUT_DIR:/work/input:ro" \
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

        python3 /work/count-tracker-v2/validate_v2_11_output.py \
            --stage input --input "/work/input/$1" >/dev/null

        # Marlin can abort during cleanup after printing its registry in this
        # image, so use the emitted XML but do not interpret that cleanup as a
        # configuration failure.
        marlin_xml="$(Marlin -x 2>/dev/null || true)"
        for processor in AIDAProcessor InitializeDD4hep DDPlanarDigiProcessor \
                         ACTSSeededCKFTrackingProc ACTSDuplicateRemoval; do
            grep -q "type=\"$processor\"" <<< "$marlin_xml" || {
                echo "Missing Marlin processor: $processor" >&2
                exit 1
            }
        done

        export V2_INPUT_FILE="/work/input/$1"
        export V2_OUTPUT_FILE=/dev/null
        export V2_NUM_EVENTS=1

        export V2_STAGE=digi
        digi_config="$(k4run --dry-run /work/count-tracker-v2/tracker_reco_override_v2_11.py 2>&1)"
        grep -q -- "--> AIDAInputConverter --> InitializeDD4hep --> VXDBarrelDigitiser" \
            <<< "$digi_config" || {
                echo "Unexpected digitization algorithm order" >&2
                printf "%s\n" "$digi_config" >&2
                exit 1
            }

        export V2_STAGE=reco
        reco_config="$(k4run --dry-run /work/count-tracker-v2/tracker_reco_override_v2_11.py 2>&1)"
        grep -q -- "--> AIDAInputConverter --> InitializeDD4hep --> CKFTracking --> TrackDeduplication" \
            <<< "$reco_config" || {
                echo "Unexpected reconstruction algorithm order" >&2
                printf "%s\n" "$reco_config" >&2
                exit 1
            }

        echo "v2.11 preflight passed"
        echo "  input: /work/input/$1"
        echo "  digitization: AIDA -> DD4hep -> six tracker digitizers"
        echo "  reconstruction: AIDA -> DD4hep -> CKF -> duplicate removal"
    ' _ "$INPUT_NAME"
