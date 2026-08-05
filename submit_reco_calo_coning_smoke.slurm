#!/bin/bash
#SBATCH -J reco_coning_smoke
#SBATCH -p debug
#SBATCH --array=0-3
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o reco_coning_smoke_%A_%a.out
#SBATCH -e reco_coning_smoke_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source config.sh

case "$SLURM_ARRAY_TASK_ID" in
    0) SAMPLE=U; CONING=on ;;
    1) SAMPLE=U; CONING=off ;;
    2) SAMPLE=R; CONING=on ;;
    3) SAMPLE=R; CONING=off ;;
    *) echo "invalid array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

MAIA="$MUCOLL_BENCHMARKS_PATH/configs/MAIAConfig/MAIAConfig"
grep -q -- "--disableCaloConing" "$MAIA/digi_args.py"
grep -Fq 'TrackCollections = ["SiTracks"]' "$MAIA/ParticleFlow/pandora.py"

RUN_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
apptainer exec --bind /oscar:/oscar "$IMAGE" \
    bash "$SLURM_SUBMIT_DIR/run_reco_calo_coning_smoke.sh" \
    "$SAMPLE" "$CONING" "$RUN_ID"
