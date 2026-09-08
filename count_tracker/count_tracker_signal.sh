#!/bin/bash
# =============================================================================
#  Neutrino signal-SIM producer for the SIM/COUNT tracker study.
#
#  Runs only the GEN and SIM stages of the standard particle-gun chain
#  (pgun_edm4hep.py -> ddsim), producing one SIM file with NEVENTS entries: one
#  invisible PDG-14 neutrino per entry, each an independently directed event.
#  Event i of the study uses entry i (via count_tracker_input --signal-entry i),
#  shared between that event's SIM and COUNT members.
#
#  Reuses GenBIB's pinned mucoll-benchmarks (298d68ac) for geometry consistency
#  with the tracker reconstruction. Detector overlay of BIB/COUNT hits is NOT
#  done here -- count_tracker_input.py merges hits onto this record with no
#  overlay timing cut, identically for SIM and COUNT.
#
#  Env / args:
#    IMAGE          v3.0 apptainer SIF                      [default below]
#    BENCHMARK_DIR  GenBIB's mucoll-benchmarks (298d68ac)   [required]
#    --nevents N    number of neutrino entries              [default 10]
#    --seed S       base GEN seed                           [default 12345]
#    --pt V         neutrino pT [GeV]                        [default 100]
#    --theta-min V  min polar angle [deg]                    [default 10]
#    --theta-max V  max polar angle [deg]                    [default 170]
#    --outfile F    final SIM ROOT path                     [required]
# =============================================================================
set -euo pipefail

IMAGE="${IMAGE:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif}"
BENCHMARK_DIR="${BENCHMARK_DIR:-}"
NEVENTS=10
SEED=12345
PDG=14
PT=100
THETA_MIN=10
THETA_MAX=170
OUTFILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --nevents)   NEVENTS="$2";   shift 2 ;;
        --seed)      SEED="$2";      shift 2 ;;
        --pt)        PT="$2";        shift 2 ;;
        --theta-min) THETA_MIN="$2"; shift 2 ;;
        --theta-max) THETA_MAX="$2"; shift 2 ;;
        --outfile)   OUTFILE="$2";   shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$BENCHMARK_DIR" ] || [ -z "$OUTFILE" ]; then
    echo "ERROR: BENCHMARK_DIR env and --outfile are required." >&2
    exit 1
fi
for path in "$IMAGE" "$BENCHMARK_DIR"; do
    [ -e "$path" ] || { echo "Missing required path: $path" >&2; exit 1; }
done

OUT_DIR="$(cd "$(dirname "$OUTFILE")" 2>/dev/null && pwd || true)"
mkdir -p "$OUT_DIR"
OUT_NAME="$(basename "$OUTFILE")"

echo "=== neutrino signal SIM: $NEVENTS events, seed $SEED, PDG $PDG ==="
echo "benchmarks: $BENCHMARK_DIR"
echo "output:     $OUT_DIR/$OUT_NAME"

apptainer exec --bind /oscar:/oscar \
    --bind "${BENCHMARK_DIR}:/work/mucoll-benchmarks,${OUT_DIR}:/work/output" \
    "$IMAGE" bash -lc '
    set -euo pipefail
    set +e +u +o pipefail
    source /opt/setup_mucoll.sh 2>/dev/null || true
    STACK_SETUP="$(find /opt/spack/opt/spack -name setup.sh -path "*mucoll-stack*" 2>/dev/null | head -n1)"
    [ -n "${STACK_SETUP}" ] && source "${STACK_SETUP}"
    SETUP_CONFIG="$(mktemp)"
    sed "s/\r$//" /work/mucoll-benchmarks/setup_config.sh > "${SETUP_CONFIG}"
    source "${SETUP_CONFIG}" /work/mucoll-benchmarks MAIA_v0
    set -euo pipefail

    WORK="$(mktemp -d /work/output/.count_signal.XXXXXX)"
    trap "rm -rf ${WORK}" EXIT
    cd "${WORK}"

    echo "--- Generation (pgun PDG='"$PDG"') ---"
    python "/work/mucoll-benchmarks/generation/pgun/pgun_edm4hep.py" \
        -s '"$SEED"' -p 1 -e '"$NEVENTS"' --pdg '"$PDG"' --pt '"$PT"' \
        --theta '"$THETA_MIN"' '"$THETA_MAX"' -- gen.edm4hep.root

    echo "--- Simulation (ddsim) ---"
    ddsim --steeringFile "/work/mucoll-benchmarks/simulation/steer_baseline.py" \
        --numberOfEvents '"$NEVENTS"' \
        --inputFiles gen.edm4hep.root \
        --outputFile sim.partial.root

    # Publish atomically only after both stages succeed.
    mv sim.partial.root "/work/output/'"$OUT_NAME"'"
'

echo "Neutrino signal SIM: $OUT_DIR/$OUT_NAME"
