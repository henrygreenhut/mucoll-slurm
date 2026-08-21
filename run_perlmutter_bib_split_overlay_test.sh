#!/bin/bash

set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
BENCH=${MUCOLL_BENCHMARKS_PATH:-$REPO/../mucoll-benchmarks}
MAIA=$BENCH/configs/MAIAConfig
DATA=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data
SPLIT=$DATA/bib-v3p0-fmt2-split-muon-v1
LEGACY=$DATA/bib-v3p0-fmt2-norm42-RandomRot/SIM
SIGNAL_SIM=${SIGNAL_SIM:-/pscratch/sd/h/hgreen/mucoll/libtest/work/reco420_v2/56293200/mucoll_job_15.fx6hbR/sim_output.edm4hep.root}
EVENTS=${EVENTS:-3}
BIB_NUMBER=2
OUTPUT=${OUTPUT:-$PSCRATCH/mucoll/bib_split_muon_overlay_test/$SLURM_JOB_ID}
WORK=$(mktemp -d "$PSCRATCH/mucoll/bib_split_overlay_test.XXXXXX")

cleanup() {
    rm -rf "$WORK"
}
trap cleanup EXIT INT TERM

set +u
source /opt/setup_mucoll.sh
set +o pipefail
source "$BENCH/setup_config.sh" "$BENCH" MAIA_v0
set -o pipefail
set -u

threshold_name=ECAL_Thresholds_10TeV.root
threshold_file=${MUCOLL_CALO_THRESHOLDS_DIR:-}/$threshold_name
if [ ! -f "$threshold_file" ]; then
    threshold_file=$(find /opt -path "*/share/MyBIBUtils/data/$threshold_name" -print -quit)
fi
if [ ! -f "$threshold_file" ]; then
    echo "could not locate ECAL_Thresholds_10TeV.root" >&2
    exit 1
fi
export MUCOLL_CALO_THRESHOLDS_DIR=$(dirname "$threshold_file")

for path in \
    "$SIGNAL_SIM" \
    "$LEGACY/MUPLUS" \
    "$LEGACY/MUMINUS" \
    "$SPLIT/bulk-norm42/SIM/MUPLUS" \
    "$SPLIT/bulk-norm42/SIM/MUMINUS" \
    "$SPLIT/decays-containing-muon-poisson-norot/SIM/MUPLUS" \
    "$SPLIT/decays-containing-muon-poisson-norot/SIM/MUMINUS"
do
    if [ ! -e "$path" ]; then
        echo "missing input: $path" >&2
        exit 1
    fi
done

for polarity in MUPLUS MUMINUS; do
    count=$(find "$SPLIT/bulk-norm42/SIM/$polarity" -name 'bib_sim_*.edm4hep.root' -type f | wc -l)
    if [ "$count" -lt "$BIB_NUMBER" ]; then
        echo "need at least $BIB_NUMBER bulk files for $polarity; found $count" >&2
        exit 1
    fi
    count=$(find "$SPLIT/decays-containing-muon-poisson-norot/SIM/$polarity" -name 'bib_sim_*.edm4hep.root' -type f | wc -l)
    if [ "$count" -lt "$BIB_NUMBER" ]; then
        echo "need at least $BIB_NUMBER grouped muon files for $polarity; found $count" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT"
cd "$WORK"
cp -r "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/PandoraSettings" ./

run_digi() {
    label=$1
    shift
    echo "running DIGI: $label"
    k4run "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/digi_steer.py" \
        -n "$EVENTS" \
        --inputFiles "$SIGNAL_SIM" \
        --outputFile "$label.edm4hep.root" \
        --RandSeed 42 \
        --doOverlayFull \
        "$@"
}

run_digi legacy \
    --OverlayFullPathToMuPlus "$LEGACY/MUPLUS" \
    --OverlayFullPathToMuMinus "$LEGACY/MUMINUS" \
    --OverlayFullNumberBackground "$BIB_NUMBER"

run_digi split_bulk \
    --OverlayFullPathToMuPlus "$SPLIT/bulk-norm42/SIM/MUPLUS" \
    --OverlayFullPathToMuMinus "$SPLIT/bulk-norm42/SIM/MUMINUS" \
    --OverlayFullNumberBackground "$BIB_NUMBER"

run_digi split_muon \
    --OverlayFullPathToMuPlus "$SPLIT/bulk-norm42/SIM/MUPLUS" \
    --OverlayFullPathToMuMinus "$SPLIT/bulk-norm42/SIM/MUMINUS" \
    --OverlayFullNumberBackground "$BIB_NUMBER" \
    --OverlayFullUseMuonComponent \
    --OverlayFullMuonPathToMuPlus "$SPLIT/decays-containing-muon-poisson-norot/SIM/MUPLUS" \
    --OverlayFullMuonPathToMuMinus "$SPLIT/decays-containing-muon-poisson-norot/SIM/MUMINUS"

echo "running RECO: split_muon"
k4run "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/reco_steer.py" \
    -n "$EVENTS" \
    --inputFiles split_muon.edm4hep.root \
    --outputFile split_muon_reco.edm4hep.root

python3 - "$EVENTS" <<'PY'
import sys
import uproot

expected = int(sys.argv[1])
paths = (
    "legacy.edm4hep.root",
    "split_bulk.edm4hep.root",
    "split_muon.edm4hep.root",
    "split_muon_reco.edm4hep.root",
)

for path in paths:
    root = uproot.open(path)
    assert "events" in root, path
    assert "podio_metadata" in root, path
    entries = root["events"].num_entries
    assert entries == expected, (path, entries, expected)
    print(path, "events=", entries, "branches=", len(root["events"].keys()))
PY

{
    echo "events=$EVENTS"
    echo "bulk_files_per_polarity=$BIB_NUMBER"
    echo "muon_group_files_per_polarity=$BIB_NUMBER"
    echo "signal_sim=$SIGNAL_SIM"
    echo "maia_commit=$(git -C "$MAIA" rev-parse HEAD)"
    echo "maia_branch=$(git -C "$MAIA" branch --show-current)"
    echo "mucoll_slurm_commit=$(git -C "$REPO" rev-parse HEAD)"
} > test_context.txt

mv ./*.edm4hep.root test_context.txt "$OUTPUT/"
echo "BIB split-overlay integration test passed"
echo "outputs: $OUTPUT"
