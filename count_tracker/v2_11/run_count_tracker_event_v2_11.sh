#!/bin/bash
# One-event SIM/COUNT comparison using the v2.11 digitization and CKF path.
set -euo pipefail

IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
GENBIB_DIR="${GENBIB_DIR:-}"
CT_DIR="${CT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
V2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSIGN_PYTHON="${ASSIGN_PYTHON:-python3}"

CONDITIONS=""; CONSTRUCTION=""; SPLIT=""; EVENT_ID=""
SIGNAL=""; SIGNAL_ENTRY=""; COUNT_SAMPLES=""; GEOMAP=""; ONLY=""; OUTPUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --conditions) CONDITIONS="$2"; shift 2 ;;
        --construction) CONSTRUCTION="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --event-id) EVENT_ID="$2"; shift 2 ;;
        --signal) SIGNAL="$2"; shift 2 ;;
        --signal-entry) SIGNAL_ENTRY="$2"; shift 2 ;;
        --count-samples) COUNT_SAMPLES="$2"; shift 2 ;;
        --geomap) GEOMAP="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for name in GENBIB_DIR CONDITIONS CONSTRUCTION SPLIT EVENT_ID SIGNAL SIGNAL_ENTRY OUTPUT; do
    [ -n "${!name}" ] || { echo "Missing required argument/environment: $name" >&2; exit 1; }
done
for path in "$IMAGE_V2_11" "$GENBIB_DIR/reco/assign_actual_cellid.py" \
            "$CT_DIR/count_tracker_input.py" "$V2_DIR/run_reco_v2_11.sh"; do
    [ -e "$path" ] || { echo "Missing required path: $path" >&2; exit 1; }
done

DO_SIM=1; DO_COUNT=1
case "$ONLY" in
    SIM) DO_COUNT=0 ;;
    COUNT) DO_SIM=0 ;;
    "") : ;;
    *) echo "--only must be SIM or COUNT" >&2; exit 1 ;;
esac

if [ "$DO_COUNT" -eq 1 ]; then
    [ -n "$COUNT_SAMPLES" ] || { echo "COUNT needs --count-samples" >&2; exit 1; }
    [ -n "$GEOMAP" ] || { echo "COUNT needs --geomap" >&2; exit 1; }
    COUNT_EVENT_DIR="$COUNT_SAMPLES/$SPLIT/$EVENT_ID"
    [ -d "$COUNT_EVENT_DIR" ] || { echo "Missing COUNT samples: $COUNT_EVENT_DIR" >&2; exit 1; }
    if [ ! -f "$GEOMAP" ]; then
        IMAGE_V2_11="$IMAGE_V2_11" bash "$V2_DIR/build_geomap_v2_11.sh" "$GEOMAP"
    fi
    "$ASSIGN_PYTHON" -c 'import numpy, scipy' >/dev/null 2>&1 || {
        echo "ASSIGN_PYTHON must provide NumPy and SciPy (use the count-hdf Python)" >&2
        exit 1
    }
fi

mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
echo "=== v2.11 event $CONSTRUCTION/$SPLIT/$EVENT_ID (SIM=$DO_SIM COUNT=$DO_COUNT) ==="

# CellID assignment is geometry dependent but not software-stack dependent once
# the v2.11 map exists.  Run it with count-hdf because the v2.11 image lacks
# SciPy; no v3 geometry or reconstruction code is used here.
if [ "$DO_COUNT" -eq 1 ]; then
    echo "--- assign v2.11 CellIDs (COUNT) ---"
    "$ASSIGN_PYTHON" "$GENBIB_DIR/reco/assign_actual_cellid.py" \
        --input-dir "$COUNT_EVENT_DIR" \
        --output-dir "$OUTPUT/COUNT/assigned" \
        --input-format 9col \
        --geomap-path "$GEOMAP"
fi

# Write both paired input records with the v2.11 podio/EDM4hep runtime.
export CTS_CT_DIR="$CT_DIR" CTS_CONDITIONS="$CONDITIONS"
export CTS_CONSTRUCTION="$CONSTRUCTION" CTS_SPLIT="$SPLIT" CTS_EVENT_ID="$EVENT_ID"
export CTS_SIGNAL="$SIGNAL" CTS_SIGNAL_ENTRY="$SIGNAL_ENTRY" CTS_OUTPUT="$OUTPUT"
export CTS_DO_SIM="$DO_SIM" CTS_DO_COUNT="$DO_COUNT"

apptainer exec --pwd /tmp --bind /oscar:/oscar,"$CT_DIR:$CT_DIR:ro" "$IMAGE_V2_11" bash -lc '
    set -eo pipefail
    source /opt/setup_mucoll.sh
    set -u
    if [ "$CTS_DO_COUNT" -eq 1 ]; then
        python3 "$CTS_CT_DIR/count_tracker_input.py" \
            --conditions "$CTS_CONDITIONS" --construction "$CTS_CONSTRUCTION" \
            --split "$CTS_SPLIT" --event-id "$CTS_EVENT_ID" \
            --sample COUNT --count-arrays "$CTS_OUTPUT/COUNT/assigned" \
            --signal "$CTS_SIGNAL" --signal-entry "$CTS_SIGNAL_ENTRY" \
            --output "$CTS_OUTPUT/COUNT/input"
    fi
    if [ "$CTS_DO_SIM" -eq 1 ]; then
        python3 "$CTS_CT_DIR/count_tracker_input.py" \
            --conditions "$CTS_CONDITIONS" --construction "$CTS_CONSTRUCTION" \
            --split "$CTS_SPLIT" --event-id "$CTS_EVENT_ID" \
            --sample SIM \
            --signal "$CTS_SIGNAL" --signal-entry "$CTS_SIGNAL_ENTRY" \
            --output "$CTS_OUTPUT/SIM/input"
    fi
'

run_one_reco() {
    local sample="$1"
    echo "--- v2.11 reconstruct $sample ---"
    IMAGE_V2_11="$IMAGE_V2_11" NUM_EVENTS=1 \
        INPUT_FILE="$OUTPUT/$sample/input/input.edm4hep.root" \
        OUTPUT_DIR="$OUTPUT/$sample/reco" \
        bash "$V2_DIR/run_reco_v2_11.sh"
}
[ "$DO_SIM" -eq 1 ] && run_one_reco SIM
[ "$DO_COUNT" -eq 1 ] && run_one_reco COUNT

echo "=== v2.11 done: $OUTPUT ==="
