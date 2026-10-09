#!/bin/bash
set -euo pipefail

IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
INPUT_FILE="${INPUT_FILE:-}"
DIGI_FILE="${DIGI_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
NUM_EVENTS="${NUM_EVENTS:-1}"
RUN_STAGE="${RUN_STAGE:-both}"
THETA_MIN="${THETA_MIN:-}"
THETA_MAX="${THETA_MAX:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [ $# -gt 0 ]; do
    case "$1" in
        --stage) RUN_STAGE="$2"; shift 2 ;;
        --theta-min) THETA_MIN="$2"; shift 2 ;;
        --theta-max) THETA_MAX="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "$RUN_STAGE" in
    digi|reco|both) : ;;
    *) echo "--stage must be digi, reco, or both" >&2; exit 1 ;;
esac
if [ -n "$THETA_MIN" ] || [ -n "$THETA_MAX" ]; then
    [ -n "$THETA_MIN" ] && [ -n "$THETA_MAX" ] || {
        echo "Set both --theta-min and --theta-max" >&2; exit 1;
    }
    [ "$RUN_STAGE" = reco ] || {
        echo "Theta bounds are supported only with --stage reco" >&2; exit 1;
    }
fi

[ -n "$OUTPUT_DIR" ] || { echo "Missing required environment: OUTPUT_DIR" >&2; exit 1; }
if [ "$RUN_STAGE" = reco ]; then
    DIGI_FILE="${DIGI_FILE:-$OUTPUT_DIR/digi_output.edm4hep.root}"
    SOURCE_FILE="$DIGI_FILE"
else
    [ -n "$INPUT_FILE" ] || { echo "Missing required environment: INPUT_FILE" >&2; exit 1; }
    SOURCE_FILE="$INPUT_FILE"
fi
for path in "$IMAGE_V2_11" "$SOURCE_FILE" \
        "$SCRIPT_DIR/tracker_reco_override_v2_11.py" \
        "$SCRIPT_DIR/validate_v2_11_output.py"; do
    [ -e "$path" ] || { echo "Missing required path: $path" >&2; exit 1; }
done

INPUT_DIR="$(cd "$(dirname "$SOURCE_FILE")" && pwd)"
INPUT_NAME="$(basename "$SOURCE_FILE")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
case "$RUN_STAGE" in
    digi) outputs=(digi_output.edm4hep.root) ;;
    reco) outputs=(reco_output.edm4hep.root) ;;
    both) outputs=(digi_output.edm4hep.root reco_output.edm4hep.root) ;;
esac
for output in "${outputs[@]}"; do
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

        run_stage() {
            stage="$1"
            set +e
            k4run /work/count-tracker-v2/tracker_reco_override_v2_11.py
            status=$?
            set -e

            # This v2.11 image can abort in allocator cleanup after Gaudi has
            # terminated and podio has finalized the file.  Preserve every
            # other failure.  For the cleanup abort, continue only if a new
            # reader independently verifies the complete stage schema.
            if [ "$status" -ne 0 ] && [ "$status" -ne 134 ]; then
                echo "v2.11 $stage failed with exit status $status" >&2
                return "$status"
            fi
            python3 /work/count-tracker-v2/validate_v2_11_output.py \
                --stage "$stage" --input "$V2_OUTPUT_FILE"
            if [ "$status" -ne 0 ]; then
                echo "WARNING: accepting v2.11 $stage cleanup abort (status $status); independently validated $V2_OUTPUT_FILE" >&2
            fi
        }

        export V2_NUM_EVENTS="$3"
        if [ "$4" = digi ] || [ "$4" = both ]; then
            export V2_STAGE=digi
            unset V2_THETA_MIN V2_THETA_MAX
            export V2_INPUT_FILE="/work/input/$1"
            export V2_OUTPUT_FILE=/work/output/digi_output.edm4hep.root
            run_stage digi
        fi

        if [ "$4" = reco ] || [ "$4" = both ]; then
            export V2_STAGE=reco
            if [ -n "$5" ]; then
                export V2_THETA_MIN="$5"
                export V2_THETA_MAX="$6"
            else
                unset V2_THETA_MIN V2_THETA_MAX
            fi
            if [ "$4" = reco ]; then
                export V2_INPUT_FILE="/work/input/$1"
            else
                export V2_INPUT_FILE=/work/output/digi_output.edm4hep.root
            fi
            export V2_OUTPUT_FILE=/work/output/reco_output.edm4hep.root
            run_stage reco
        fi
    ' _ "$INPUT_NAME" "$OUTPUT_DIR" "$NUM_EVENTS" "$RUN_STAGE" "$THETA_MIN" "$THETA_MAX"

if [ "$RUN_STAGE" = digi ] || [ "$RUN_STAGE" = both ]; then
    echo "v2.11 digitization: $OUTPUT_DIR/digi_output.edm4hep.root"
fi
if [ "$RUN_STAGE" = reco ] || [ "$RUN_STAGE" = both ]; then
    echo "v2.11 reconstruction: $OUTPUT_DIR/reco_output.edm4hep.root"
fi
