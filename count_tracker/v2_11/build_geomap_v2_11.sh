#!/bin/bash
set -euo pipefail

IMAGE="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
OUTPUT="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -n "$OUTPUT" ] || { echo "Usage: $0 OUTPUT.npz" >&2; exit 1; }
[ -f "$IMAGE" ] || { echo "Missing v2.11 image: $IMAGE" >&2; exit 1; }
mkdir -p "$(dirname "$OUTPUT")"
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT")" && pwd)"
OUTPUT_NAME="$(basename "$OUTPUT")"

apptainer exec \
    --bind "$SCRIPT_DIR:/work/count-tracker-v2:ro,$OUTPUT_DIR:/work/output" \
    "$IMAGE" bash -lc '
        set -euo pipefail
        source /opt/setup_mucoll.sh
        MUCOLL_GEO="$(find /opt/spack/opt/spack -type f \
            -path "*/k4geo*/share/k4geo/MuColl/MAIA/compact/MAIA_v0/MAIA_v0.xml" \
            | head -n1)"
        [ -n "$MUCOLL_GEO" ] || { echo "MAIA_v0 compact XML not found" >&2; exit 1; }
        export MUCOLL_GEO
        python3 /work/count-tracker-v2/build_sensor_geometry_v2_11.py \
            --output "/work/output/$1"
    ' _ "$OUTPUT_NAME"
