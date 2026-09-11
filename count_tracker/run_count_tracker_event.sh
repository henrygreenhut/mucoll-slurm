#!/bin/bash
# =============================================================================
#  Per-event reconstruction for the SIM/COUNT tracker study.
#
#  For one prepared event, builds the SIM and/or COUNT tracker input and runs
#  the GenBIB tracker-only reconstruction (digi -> CKF -> SiTracks) on each:
#
#     COUNT:  assign_actual_cellid -> count_tracker_input --sample COUNT -> run_reco.sh
#     SIM:                            count_tracker_input --sample SIM   -> run_reco.sh
#
#  The two data-prep steps run in the v3.0 container; run_reco.sh (GenBIB,
#  UNMODIFIED) does its own container exec. Everything under /oscar is visible
#  via a single bind, so host paths are used directly inside the container.
#
#  Outputs:  <output>/SIM/reco/reco_output.edm4hep.root
#            <output>/COUNT/reco/reco_output.edm4hep.root
#
#  Env:
#    IMAGE          v3.0 apptainer SIF                         [default below]
#    BENCHMARK_DIR  GenBIB's mucoll-benchmarks (298d68ac)      [required]
#    GENBIB_DIR     GenBIB-ML checkout (assign_actual_cellid,  [required]
#                   reco/run_reco.sh)
#    CT_DIR         this count_tracker directory               [default: script dir]
#  Args:
#    --conditions DIR    prepared conditions directory         [required]
#    --construction C    norm1 | norm42 | training_domain      [required]
#    --split S           train | val | test                    [required]
#    --event-id ID       event id within the split             [required]
#    --signal FILE       neutrino signal SIM ROOT              [required]
#    --signal-entry N    entry index for this event            [required]
#    --count-samples DIR sampler output root (has <split>/<id>/tabddpm_*.npy)
#                                                              [required for COUNT]
#    --geomap FILE       assign_actual_cellid geometry .npz    [required for COUNT]
#    --only SIM|COUNT    run just one sample (default: both)
#    --output DIR        event output root                     [required]
# =============================================================================
set -euo pipefail

IMAGE="${IMAGE:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif}"
BENCHMARK_DIR="${BENCHMARK_DIR:-}"
GENBIB_DIR="${GENBIB_DIR:-}"
CT_DIR="${CT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

CONDITIONS="" ; CONSTRUCTION="" ; SPLIT="" ; EVENT_ID=""
SIGNAL="" ; SIGNAL_ENTRY="" ; COUNT_SAMPLES="" ; GEOMAP="" ; ONLY="" ; OUTPUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --conditions)    CONDITIONS="$2";    shift 2 ;;
        --construction)  CONSTRUCTION="$2";  shift 2 ;;
        --split)         SPLIT="$2";         shift 2 ;;
        --event-id)      EVENT_ID="$2";      shift 2 ;;
        --signal)        SIGNAL="$2";        shift 2 ;;
        --signal-entry)  SIGNAL_ENTRY="$2";  shift 2 ;;
        --count-samples) COUNT_SAMPLES="$2"; shift 2 ;;
        --geomap)        GEOMAP="$2";        shift 2 ;;
        --only)          ONLY="$2";          shift 2 ;;
        --output)        OUTPUT="$2";        shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

for name in BENCHMARK_DIR GENBIB_DIR CONDITIONS CONSTRUCTION SPLIT EVENT_ID SIGNAL SIGNAL_ENTRY OUTPUT; do
    [ -n "${!name}" ] || { echo "ERROR: missing required $name" >&2; exit 1; }
done
for path in "$IMAGE" "$BENCHMARK_DIR" "$GENBIB_DIR" "$CONDITIONS" "$SIGNAL" \
            "$CT_DIR/count_tracker_input.py" "$GENBIB_DIR/reco/run_reco.sh"; do
    [ -e "$path" ] || { echo "Missing required path: $path" >&2; exit 1; }
done

DO_SIM=1; DO_COUNT=1
case "$ONLY" in
    SIM)   DO_COUNT=0 ;;
    COUNT) DO_SIM=0 ;;
    "")    : ;;
    *) echo "ERROR: --only must be SIM or COUNT" >&2; exit 1 ;;
esac
if [ "$DO_COUNT" -eq 1 ]; then
    [ -n "$COUNT_SAMPLES" ] || { echo "ERROR: COUNT needs --count-samples" >&2; exit 1; }
    [ -n "$GEOMAP" ]        || { echo "ERROR: COUNT needs --geomap" >&2; exit 1; }
    COUNT_EVENT_DIR="$COUNT_SAMPLES/$SPLIT/$EVENT_ID"
    [ -d "$COUNT_EVENT_DIR" ] || { echo "Missing COUNT samples: $COUNT_EVENT_DIR" >&2; exit 1; }
fi

mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
GENBIB_RECO="$GENBIB_DIR/reco"

echo "=== event $CONSTRUCTION/$SPLIT/$EVENT_ID  (SIM=$DO_SIM COUNT=$DO_COUNT) ==="

# --- 1. Data prep inside the container (assign CellIDs + write inputs) --------
export CTS_CT_DIR="$CT_DIR" CTS_GENBIB_RECO="$GENBIB_RECO" CTS_CONDITIONS="$CONDITIONS"
export CTS_CONSTRUCTION="$CONSTRUCTION" CTS_SPLIT="$SPLIT" CTS_EVENT_ID="$EVENT_ID"
export CTS_SIGNAL="$SIGNAL" CTS_SIGNAL_ENTRY="$SIGNAL_ENTRY" CTS_OUTPUT="$OUTPUT"
export CTS_COUNT_EVENT_DIR="${COUNT_EVENT_DIR:-}" CTS_GEOMAP="${GEOMAP:-}"
export CTS_DO_SIM="$DO_SIM" CTS_DO_COUNT="$DO_COUNT" CTS_BENCHMARK="$BENCHMARK_DIR"

apptainer exec --bind /oscar:/oscar "$IMAGE" bash -lc '
    set +e +u +o pipefail
    source /opt/setup_mucoll.sh 2>/dev/null || true
    STACK_SETUP="$(find /opt/spack/opt/spack -name setup.sh -path "*mucoll-stack*" 2>/dev/null | head -n1)"
    [ -n "${STACK_SETUP}" ] && source "${STACK_SETUP}"
    SETUP_CONFIG="$(mktemp)"
    sed "s/\r$//" "${CTS_BENCHMARK}/setup_config.sh" > "${SETUP_CONFIG}"
    source "${SETUP_CONFIG}" "${CTS_BENCHMARK}" MAIA_v0
    set -euo pipefail

    if [ "${CTS_DO_COUNT}" -eq 1 ]; then
        echo "--- assign CellIDs (COUNT) ---"
        python3 "${CTS_GENBIB_RECO}/assign_actual_cellid.py" \
            --input-dir "${CTS_COUNT_EVENT_DIR}" \
            --output-dir "${CTS_OUTPUT}/COUNT/assigned" \
            --input-format 9col \
            --geomap-path "${CTS_GEOMAP}"

        echo "--- write COUNT input ---"
        python3 "${CTS_CT_DIR}/count_tracker_input.py" \
            --conditions "${CTS_CONDITIONS}" --construction "${CTS_CONSTRUCTION}" \
            --split "${CTS_SPLIT}" --event-id "${CTS_EVENT_ID}" \
            --sample COUNT --count-arrays "${CTS_OUTPUT}/COUNT/assigned" \
            --signal "${CTS_SIGNAL}" --signal-entry "${CTS_SIGNAL_ENTRY}" \
            --output "${CTS_OUTPUT}/COUNT/input"
    fi

    if [ "${CTS_DO_SIM}" -eq 1 ]; then
        echo "--- write SIM input ---"
        python3 "${CTS_CT_DIR}/count_tracker_input.py" \
            --conditions "${CTS_CONDITIONS}" --construction "${CTS_CONSTRUCTION}" \
            --split "${CTS_SPLIT}" --event-id "${CTS_EVENT_ID}" \
            --sample SIM \
            --signal "${CTS_SIGNAL}" --signal-entry "${CTS_SIGNAL_ENTRY}" \
            --output "${CTS_OUTPUT}/SIM/input"
    fi
'

# --- 2. Reconstruction (run_reco.sh does its own container exec) --------------
run_one_reco() {
    local sample="$1"
    echo "--- reconstruct ${sample} ---"
    IMAGE="$IMAGE" BENCHMARK_DIR="$BENCHMARK_DIR" NUM_EVENTS=1 \
        INPUT_FILE="$OUTPUT/${sample}/input/input.edm4hep.root" \
        OUTPUT_DIR="$OUTPUT/${sample}/reco" \
        bash "$GENBIB_RECO/run_reco.sh"
}
[ "$DO_SIM" -eq 1 ]   && run_one_reco SIM
[ "$DO_COUNT" -eq 1 ] && run_one_reco COUNT

echo "=== done: $OUTPUT ==="
if [ "$DO_SIM" -eq 1 ]; then
    echo "  SIM   -> $OUTPUT/SIM/reco/reco_output.edm4hep.root"
fi
if [ "$DO_COUNT" -eq 1 ]; then
    echo "  COUNT -> $OUTPUT/COUNT/reco/reco_output.edm4hep.root"
fi
