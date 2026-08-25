#!/bin/bash

set -euo pipefail

MANIFEST=${1:?usage: $0 MANIFEST.tsv}
LINE_NUMBER=$((SLURM_PROCID + 1))
LINE=$(sed -n "${LINE_NUMBER}p" "$MANIFEST")
if [ -z "$LINE" ]; then
    echo "missing manifest row for rank $SLURM_PROCID" >&2
    exit 2
fi

IFS=$'\t' read -r JOB_ID EVENTS OUTPUT <<< "$LINE"
RECO_OUTPUT="$OUTPUT/reco_output_${JOB_ID}.edm4hep.root"
if [ -s "$RECO_OUTPUT" ] && [ -f "$OUTPUT/complete" ]; then
    echo "skip complete output: $OUTPUT"
    exit 0
fi

REPO=$(cd "$(dirname "$0")" && pwd)
BENCH=${MUCOLL_BENCHMARKS_PATH:-$REPO/../mucoll-benchmarks}
MAIA=$BENCH/configs/MAIAConfig
DATA=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data
MODE=${RECO_BIB_MODE:?RECO_BIB_MODE must be legacy or split_bh}
LEGACY=${LEGACY_BIB_ROOT:-$DATA/bib-v3p0-fmt2-norm42-RandomRot/SIM}
SPLIT=$DATA/bib-v3p0-fmt2-split-muon-v1
BULK=$SPLIT/bulk-norm42/SIM
BH=$SPLIT/decays-containing-muon-poisson-norot/SIM
BIB_NUMBER=10
TRACKING_THREADS=${TRACKING_THREADS:-3}
WORK=$(mktemp -d "$CHAIN_WORK_BASE/reco420_${MODE}_${JOB_ID}.XXXXXX")

cleanup() {
    status=$?
    if [ "$status" -eq 0 ]; then
        rm -rf "$WORK"
    else
        echo "task failed; workspace retained at $WORK" >&2
    fi
}
trap cleanup EXIT INT TERM

case "$MODE" in
    legacy)
        BIB_PATHS=("$LEGACY/MUPLUS" "$LEGACY/MUMINUS")
        ;;
    split_bh)
        BIB_PATHS=(
            "$BULK/MUPLUS" "$BULK/MUMINUS"
            "$BH/MUPLUS" "$BH/MUMINUS"
        )
        ;;
    *)
        echo "RECO_BIB_MODE must be legacy or split_bh" >&2
        exit 2
        ;;
esac

for path in "${BIB_PATHS[@]}"; do
    if [ ! -d "$path" ]; then
        echo "missing BIB directory: $path" >&2
        exit 1
    fi
done

set +u
source /opt/setup_mucoll.sh
set +o pipefail
source "$BENCH/setup_config.sh" "$BENCH" MAIA_v0
set -o pipefail
set -u

if [ "$(git -C "$MAIA" rev-parse HEAD)" != "$EXPECTED_MAIA_COMMIT" ]; then
    echo "MAIAConfig changed after submission" >&2
    exit 1
fi
if [ -n "$(git -C "$MAIA" status --porcelain)" ]; then
    echo "MAIAConfig must be clean" >&2
    git -C "$MAIA" status --short >&2
    exit 1
fi

grep -q -- '--OverlayBHMuonsSeparately' "$MAIA/MAIAConfig/digi_args.py"
grep -Fq 'TrackCollections = ["SiTracks"]' \
    "$MAIA/MAIAConfig/ParticleFlow/pandora.py"
grep -Fq 'RelTrackCollections = ["MergedTrackerHitsRelations"]' \
    "$MAIA/MAIAConfig/ParticleFlow/pandora.py"

export OMP_NUM_THREADS=$TRACKING_THREADS
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$OUTPUT"
cd "$WORK"
cp -r "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/PandoraSettings" ./

GEN_SEED=$((12345 + JOB_ID))
DIGI_SEED=$((42 + JOB_ID))

python3 "$BENCH/generation/pgun/pgun_edm4hep.py" \
    -s "$GEN_SEED" \
    -p 1 \
    -e "$EVENTS" \
    --pdg 14 \
    --pt 100 \
    --theta 10 170 \
    -- gen_output.edm4hep.root

ddsim \
    --steeringFile "$BENCH/simulation/steer_baseline.py" \
    --numberOfEvents "$EVENTS" \
    --inputFiles gen_output.edm4hep.root \
    --outputFile sim_output.edm4hep.root

OVERLAY_ARGS=(
    --OverlayFullNumberBackground "$BIB_NUMBER"
)
if [ "$MODE" = legacy ]; then
    OVERLAY_ARGS+=(
        --OverlayFullPathToMuPlus "$LEGACY/MUPLUS"
        --OverlayFullPathToMuMinus "$LEGACY/MUMINUS"
    )
else
    OVERLAY_ARGS+=(
        --OverlayFullPathToMuPlus "$BULK/MUPLUS"
        --OverlayFullPathToMuMinus "$BULK/MUMINUS"
        --OverlayBHMuonsSeparately
        --OverlayFullBHPathToMuPlus "$BH/MUPLUS"
        --OverlayFullBHPathToMuMinus "$BH/MUMINUS"
    )
fi

k4run "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/digi_steer.py" \
    -n "$EVENTS" \
    --inputFiles sim_output.edm4hep.root \
    --outputFile digi_output.edm4hep.root \
    --RandSeed "$DIGI_SEED" \
    --doOverlayFull \
    "${OVERLAY_ARGS[@]}"

k4run "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/reco_steer.py" \
    -n "$EVENTS" \
    --TrackingThreads "$TRACKING_THREADS" \
    --inputFiles digi_output.edm4hep.root \
    --outputFile reco_output.edm4hep.root

python3 - "$EVENTS" <<'PY'
import sys

import numpy as np
import uproot

expected = int(sys.argv[1])
for path in ("gen_output.edm4hep.root", "sim_output.edm4hep.root",
             "digi_output.edm4hep.root", "reco_output.edm4hep.root"):
    root = uproot.open(path)
    assert "events" in root, path
    assert "podio_metadata" in root, path
    assert root["events"].num_entries == expected, path

tree = uproot.open("reco_output.edm4hep.root")["events"]
pdg = tree["PandoraPFOs/PandoraPFOs.PDG"].array()
charge = tree["PandoraPFOs/PandoraPFOs.charge"].array()
tracks = tree["SiTracks_objIdx/SiTracks_objIdx.index"].array()
begin = tree["PandoraPFOs/PandoraPFOs.tracks_begin"].array()
end = tree["PandoraPFOs/PandoraPFOs.tracks_end"].array()
pfo_count = sum(len(event) for event in pdg)
charged_pfo_count = sum(
    int(np.count_nonzero(np.asarray(event))) for event in charge
)
track_count = sum(len(event) for event in tracks)
links = sum(int(np.sum(np.asarray(b) - np.asarray(a))) for a, b in zip(begin, end))
with open("validation.txt", "w") as output:
    output.write("events={}\n".format(expected))
    output.write("pfos={}\n".format(pfo_count))
    output.write("charged_pfos={}\n".format(charged_pfo_count))
    output.write("tracks={}\n".format(track_count))
    output.write("pfo_track_links={}\n".format(links))
print(
    "events=", expected,
    "pfos=", pfo_count,
    "charged_pfos=", charged_pfo_count,
    "tracks=", track_count,
    "pfo_track_links=", links,
)
PY

{
    echo "job_id=$JOB_ID"
    echo "events=$EVENTS"
    echo "particle_gun_pdg=14"
    echo "particle_gun_pt_gev=100"
    echo "particle_gun_theta_degrees=10,170"
    echo "bib_mode=$MODE"
    echo "n_norm1_file_equivalents=420"
    if [ "$MODE" = legacy ]; then
        echo "legacy_norm42_files_per_polarity=$BIB_NUMBER"
        echo "legacy_excluded_cycle=6291"
        echo "legacy_muplus=$LEGACY/MUPLUS"
        echo "legacy_muminus=$LEGACY/MUMINUS"
    else
        echo "bulk_norm42_files_per_polarity=$BIB_NUMBER"
        echo "bh_group_files_per_polarity=$BIB_NUMBER"
        echo "bulk_muplus=$BULK/MUPLUS"
        echo "bulk_muminus=$BULK/MUMINUS"
        echo "bh_muplus=$BH/MUPLUS"
        echo "bh_muminus=$BH/MUMINUS"
        echo "bh_bank_selection=complete_mother_muon_decays_containing_any_detector_bound_muon"
    fi
    echo "calorimeter_coning=disabled"
    echo "gen_seed=$GEN_SEED"
    echo "digi_seed=$DIGI_SEED"
    echo "tracking_threads=$TRACKING_THREADS"
    echo "image=$MUCOLL_IMAGE"
    echo "maia_commit=$EXPECTED_MAIA_COMMIT"
    echo "mucoll_slurm_commit=$(git -C "$REPO" rev-parse HEAD)"
} > metadata.txt

mv gen_output.edm4hep.root "$OUTPUT/gen_output_${JOB_ID}.edm4hep.root"
mv sim_output.edm4hep.root "$OUTPUT/sim_output_${JOB_ID}.edm4hep.root"
mv reco_output.edm4hep.root "$RECO_OUTPUT"
mv metadata.txt validation.txt "$OUTPUT/"
rm -f digi_output.edm4hep.root
touch "$OUTPUT/complete"
echo "published $OUTPUT"
