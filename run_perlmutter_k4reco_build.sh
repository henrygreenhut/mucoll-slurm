#!/bin/bash

set -euo pipefail

set +u
source /opt/setup_mucoll.sh
set -u

if [ ! -f "$K4RECO_SOURCE/k4Reco/Overlay/components/OverlayTimingRandomEntryMix.cpp" ]; then
    echo "OverlayTimingRandomEntryMix source not found under $K4RECO_SOURCE" >&2
    exit 1
fi

mkdir -p "$K4RECO_BUILD"
rm -f "$K4RECO_BUILD/overlay_entry_mix.complete"

cmake \
    -S "$K4RECO_SOURCE" \
    -B "$K4RECO_BUILD" \
    -DBUILD_TRACKING=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$K4RECO_BUILD" --parallel 8

export LD_LIBRARY_PATH="$K4RECO_BUILD:$K4RECO_BUILD/k4Reco:$K4RECO_BUILD/k4Reco/genConfDir/k4Reco:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$K4RECO_BUILD/k4Reco/genConfDir:${PYTHONPATH:-}"

python3 -c 'from Configurables import OverlayTimingRandomEntryMix'
touch "$K4RECO_BUILD/overlay_entry_mix.complete"

echo "k4Reco build ready: $K4RECO_BUILD"
