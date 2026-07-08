#!/bin/bash
#SBATCH -J pgun_perlmutter
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --array=0-19
#SBATCH --output=pgun_perlmutter_%A_%a.out
#SBATCH --error=pgun_perlmutter_%A_%a.err

set -e

JOB_ID=${JOB_ID:-0}
NEVENTS=${NEVENTS:-10}
PDG=${PDG:-13}
PT=${PT:-10}
THETA_MIN=${THETA_MIN:-20}
THETA_MAX=${THETA_MAX:-22}
USE_BIB=${USE_BIB:-1}
STUDY_NAME="${STUDY_NAME:-muon_endcap_pt10_bib812_20evt}"

cd "$SLURM_SUBMIT_DIR"
source config.sh

abs_pdg="${PDG#-}"
particle_tag="pdg${PDG}"
case "$abs_pdg" in
    11) particle_tag="e${abs_pdg}" ;;
    13) particle_tag="mu${abs_pdg}" ;;
    14) particle_tag="nu${abs_pdg}" ;;
    211) particle_tag="pi${abs_pdg}" ;;
esac

bib_args=()
if [ "$USE_BIB" -eq 1 ]; then
    bib_tag="bib${BIB_NUMBER}"
    bib_args=(--bib)
else
    bib_tag="nobib"
fi

if [ -z "$STUDY_NAME" ]; then
    STUDY_NAME="${particle_tag}_pt${PT}_theta${THETA_MIN}-${THETA_MAX}_${bib_tag}_timing_check"
fi

mkdir -p "$OUTPUT_BASE_DIR"

shifter --image="$IMAGE" bash "$PWD/chains/run_chain_pgun.sh" \
    --job-id "$JOB_ID" \
    --nevents "$NEVENTS" \
    --outdir "$OUTPUT_BASE_DIR/$STUDY_NAME" \
    --pdg "$PDG" \
    --pt "$PT" \
    --theta-min "$THETA_MIN" \
    --theta-max "$THETA_MAX" \
    "${bib_args[@]}"
