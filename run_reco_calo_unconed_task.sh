#!/bin/bash
set -euo pipefail

MANIFEST=${1:?usage: $0 MANIFEST.tsv ROW}
ROW=${2:?usage: $0 MANIFEST.tsv ROW}
LINE=$(sed -n "$((ROW + 1))p" "$MANIFEST")
if [ -z "$LINE" ]; then
    echo "missing manifest row $ROW" >&2
    exit 2
fi

IFS=$'\t' read -r SAMPLE SPLIT CHUNK JOB_ID NEVENTS SIGNAL_SIM OUTPUT \
    BIB_MUPLUS BIB_MUMINUS BIB_NUMBER DIGI_SEED <<< "$LINE"

RECO_OUTPUT="$OUTPUT/reco_output_${JOB_ID}.edm4hep.root"
if [ -s "$RECO_OUTPUT" ] && [ -f "$OUTPUT/complete" ]; then
    echo "skip complete chunk: $OUTPUT"
    exit 0
fi

for path in "$SIGNAL_SIM" "$BIB_MUPLUS" "$BIB_MUMINUS"; do
    if [ ! -e "$path" ]; then
        echo "missing input: $path" >&2
        exit 1
    fi
done

REPO=$(cd "$(dirname "$0")" && pwd)
source "$REPO/config.sh"
set +u
source "$REPO/scripts/setup.sh"
source "$MUCOLL_BENCHMARKS_PATH/setup_config.sh" \
    "$MUCOLL_BENCHMARKS_PATH" "$GEOM_NAME"
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

WORK_ROOT=${CHAIN_WORK_BASE:-/tmp}
WORKDIR=$(mktemp -d "$WORK_ROOT/reco_unconed_${SAMPLE}_${JOB_ID}.XXXXXX")
cleanup() {
    rm -rf "$WORKDIR"
}
trap cleanup EXIT INT TERM
cd "$WORKDIR"
cp -r "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/PandoraSettings" ./

echo "sample=$SAMPLE split=$SPLIT chunk=$CHUNK job_id=$JOB_ID events=$NEVENTS"
echo "signal_sim=$SIGNAL_SIM"
echo "bib_files_per_polarity=$BIB_NUMBER digi_seed=$DIGI_SEED"

DIGI_START=$(date +%s)
k4run "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/digi_steer.py" \
    -n "$NEVENTS" \
    --inputFiles "$SIGNAL_SIM" \
    --outputFile digi_output.edm4hep.root \
    --RandSeed "$DIGI_SEED" \
    --doOverlayFull \
    --OverlayFullPathToMuPlus "$BIB_MUPLUS" \
    --OverlayFullPathToMuMinus "$BIB_MUMINUS" \
    --OverlayFullNumberBackground "$BIB_NUMBER" \
    --disableCaloConing
DIGI_SECONDS=$(($(date +%s) - DIGI_START))

RECO_START=$(date +%s)
k4run "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/reco_steer.py" \
    -n "$NEVENTS" \
    --inputFiles digi_output.edm4hep.root \
    --outputFile reco_output.edm4hep.root
RECO_SECONDS=$(($(date +%s) - RECO_START))

test -s digi_output.edm4hep.root
test -s reco_output.edm4hep.root
mkdir -p "$OUTPUT"

{
    echo "sample=$SAMPLE"
    echo "split=$SPLIT"
    echo "chunk=$CHUNK"
    echo "job_id=$JOB_ID"
    echo "events=$NEVENTS"
    echo "signal_sim=$SIGNAL_SIM"
    echo "bib_muplus=$BIB_MUPLUS"
    echo "bib_muminus=$BIB_MUMINUS"
    echo "bib_files_per_polarity=$BIB_NUMBER"
    echo "digi_seed=$DIGI_SEED"
    echo "calo_coning=disabled"
    echo "merge_mc_particles=false"
    echo "pandora_tracks=SiTracks"
    echo "maia_commit=$(git -C "$MUCOLL_BENCHMARKS_PATH/configs/MAIAConfig" rev-parse HEAD 2>/dev/null || true)"
    echo "mucoll_slurm_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
} > metadata.txt
{
    echo "digi_seconds=$DIGI_SECONDS"
    echo "reco_seconds=$RECO_SECONDS"
} > timing.txt
sha256sum \
    "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/digi_args.py" \
    "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/digiAlgList.py" \
    "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/CaloDigi/calo_coning.py" \
    "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/ParticleFlow/pandora.py" \
    "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/Overlay/overlay_BIB.py" \
    > config_sha256.txt
POOL_ROOT=$(dirname "$(dirname "$(dirname "${BIB_MUPLUS%/}")")")
sha256sum "$SIGNAL_SIM" "$POOL_ROOT/manifest.json" > input_sha256.txt

mv digi_output.edm4hep.root \
    "$OUTPUT/digi_output_${JOB_ID}.edm4hep.root"
mv reco_output.edm4hep.root "$RECO_OUTPUT"
mv metadata.txt timing.txt config_sha256.txt input_sha256.txt "$OUTPUT/"
touch "$OUTPUT/complete"
echo "published $OUTPUT"

cd /
rm -rf "$WORKDIR"
trap - EXIT INT TERM
