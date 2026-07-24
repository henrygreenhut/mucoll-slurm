#!/bin/bash
set -euo pipefail

# Runs one manifest row's GEN->SIM->DIGI->RECO chain. Ported from the
# Perlmutter srun-multiprog version (SLURM_PROCID indexing, shifter): OSCAR's
# batch partition caps this account at MaxTRESPU=cpu=64 total, so the packed
# job here is a 64-way array of shards, each looping sequentially over its
# assigned manifest lines and calling this script once per line with an
# explicit line number -- not one shifter/srun rank per manifest row.

MANIFEST=$1
LINE_NUMBER=$2
if [ -z "$MANIFEST" ] || [ -z "$LINE_NUMBER" ]; then
    echo "usage: run_reco_libtest_task.sh <manifest.tsv> <line_number>" >&2
    exit 1
fi
LINE=$(sed -n "${LINE_NUMBER}p" "$MANIFEST")
if [ -z "$LINE" ]; then
    echo "ERROR: no manifest row at line $LINE_NUMBER" >&2
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
apptainer exec --bind /oscar:/oscar "$IMAGE" bash "$REPO/chains/run_chain_pgun.sh" \
    --job-id "$JOB_ID" \
    --nevents "$NEVENTS" \
    --outdir "$OUTPUT_BASE_DIR/$STUDY_NAME" \
    --pdg 14 \
    --pt 100 \
    --theta-min 10 \
    --theta-max 170 \
    --bib
