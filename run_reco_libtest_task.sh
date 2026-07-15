#!/bin/bash
set -euo pipefail

MANIFEST=$1
LINE_NUMBER=$((SLURM_PROCID + 1))
LINE=$(sed -n "${LINE_NUMBER}p" "$MANIFEST")
if [ -z "$LINE" ]; then
    echo "ERROR: no manifest row for Slurm rank $SLURM_PROCID" >&2
    exit 2
fi

IFS=$'\t' read -r SAMPLE SPLIT INDEX JOB_ID NEVENTS STUDY_NAME OUTPUT_BASE_DIR \
    BIB_MUPLUS BIB_MUMINUS BIB_NUMBER DIGI_SEED_OFFSET <<< "$LINE"

EXPECTED="$OUTPUT_BASE_DIR/$STUDY_NAME/job_$JOB_ID/reco_output_$JOB_ID.edm4hep.root"
if [ -s "$EXPECTED" ]; then
    echo "skip existing output: $EXPECTED"
    exit 0
fi

export BIB_MUPLUS BIB_MUMINUS BIB_NUMBER DIGI_SEED_OFFSET
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=$(cd "$(dirname "$0")" && pwd)
source "$REPO/config.sh"

echo "sample=$SAMPLE split=$SPLIT chunk=$INDEX job_id=$JOB_ID events=$NEVENTS"
shifter --image="$IMAGE" bash "$REPO/chains/run_chain_pgun.sh" \
    --job-id "$JOB_ID" \
    --nevents "$NEVENTS" \
    --outdir "$OUTPUT_BASE_DIR/$STUDY_NAME" \
    --pdg 14 \
    --pt 100 \
    --theta-min 10 \
    --theta-max 170 \
    --bib
