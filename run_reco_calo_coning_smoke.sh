#!/bin/bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 {U|R} {on|off} RUN_ID" >&2
    exit 2
fi

SAMPLE=$1
CONING=$2
RUN_ID=$3
case "$SAMPLE:$CONING" in
    U:on|U:off|R:on|R:off) ;;
    *) echo "invalid condition: $SAMPLE $CONING" >&2; exit 2 ;;
esac

REPO=$(cd "$(dirname "$0")" && pwd)
source "$REPO/config.sh"
set +u
source "$REPO/scripts/setup.sh"
source "$MUCOLL_BENCHMARKS_PATH/setup_config.sh" \
    "$MUCOLL_BENCHMARKS_PATH" "$GEOM_NAME"
set -u

BASE=/oscar/scratch/$USER/mucoll/libtest
SIM_INPUT="$BASE/reco_n420_pfn_trackfix_val25/reco_libtest_n420_U/train/job_0/sim_output_0.edm4hep.root"
POOLS="$BASE/bib_pools_val25"
OUTPUT_ROOT=${CONING_SMOKE_OUT:-$BASE/reco_calo_coning_smoke}
OUTPUT="$OUTPUT_ROOT/run_$RUN_ID/$SAMPLE/$CONING"

if [ "$SAMPLE" = U ]; then
    LIBRARY=norm1
    BIB_NUMBER=420
else
    LIBRARY=norm42
    BIB_NUMBER=10
fi

BIB_MUPLUS="$POOLS/$LIBRARY/train/MUPLUS/"
BIB_MUMINUS="$POOLS/$LIBRARY/train/MUMINUS/"
DIGI_SEED=42

for path in "$SIM_INPUT" "$BIB_MUPLUS" "$BIB_MUMINUS"; do
    if [ ! -e "$path" ]; then
        echo "missing input: $path" >&2
        exit 1
    fi
done
if [ -s "$OUTPUT/reco_output.edm4hep.root" ]; then
    echo "skip existing output: $OUTPUT"
    exit 0
fi

WORK_ROOT=${CHAIN_WORK_BASE:-/tmp}
WORKDIR=$(mktemp -d "$WORK_ROOT/reco_coning_${SAMPLE}_${CONING}.XXXXXX")
cleanup() {
    rm -rf "$WORKDIR"
}
trap cleanup EXIT INT TERM
cd "$WORKDIR"
cp -r "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/PandoraSettings" ./

CALO_ARGS=()
if [ "$CONING" = off ]; then
    CALO_ARGS+=(--disableCaloConing)
fi

echo "sample=$SAMPLE coning=$CONING events=1 bib_files_per_polarity=$BIB_NUMBER digi_seed=$DIGI_SEED"
echo "signal_sim=$SIM_INPUT"
echo "bib_muplus=$BIB_MUPLUS"
echo "bib_muminus=$BIB_MUMINUS"

DIGI_START=$(date +%s)
k4run "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/digi_steer.py" \
    -n 1 \
    --inputFiles "$SIM_INPUT" \
    --outputFile digi_output.edm4hep.root \
    --RandSeed "$DIGI_SEED" \
    --doOverlayFull \
    --OverlayFullPathToMuPlus "$BIB_MUPLUS" \
    --OverlayFullPathToMuMinus "$BIB_MUMINUS" \
    --OverlayFullNumberBackground "$BIB_NUMBER" \
    "${CALO_ARGS[@]}"
echo "elapsed_seconds=$(($(date +%s) - DIGI_START))" > digi.time.log

RECO_START=$(date +%s)
k4run "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/reco_steer.py" \
    -n 1 \
    --inputFiles digi_output.edm4hep.root \
    --outputFile reco_output.edm4hep.root
echo "elapsed_seconds=$(($(date +%s) - RECO_START))" > reco.time.log

mkdir -p "$OUTPUT"
{
    echo "sample=$SAMPLE"
    echo "coning=$CONING"
    echo "events=1"
    echo "bib_library=$LIBRARY"
    echo "bib_files_per_polarity=$BIB_NUMBER"
    echo "digi_seed=$DIGI_SEED"
    echo "signal_sim=$SIM_INPUT"
    echo "bib_muplus=$BIB_MUPLUS"
    echo "bib_muminus=$BIB_MUMINUS"
    echo "maia_commit=$(git -C "$MUCOLL_BENCHMARKS_PATH/configs/MAIAConfig" rev-parse HEAD 2>/dev/null || true)"
    echo "mucoll_slurm_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
} > metadata.txt

sha256sum \
    "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/digi_args.py" \
    "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/digiAlgList.py" \
    "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/CaloDigi/calo_coning.py" \
    "$MUCOLL_CONFIG/$MUCOLL_CONFIG_NAME/ParticleFlow/pandora.py" \
    > config_sha256.txt

mv digi_output.edm4hep.root reco_output.edm4hep.root \
    digi.time.log reco.time.log metadata.txt config_sha256.txt "$OUTPUT/"
echo "published $OUTPUT"

cd /
rm -rf "$WORKDIR"
trap - EXIT INT TERM
