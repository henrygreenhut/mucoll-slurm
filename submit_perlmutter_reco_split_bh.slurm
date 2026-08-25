#!/bin/bash
#SBATCH -J reco420_split_bh
#SBATCH -A m5197
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=6
#SBATCH -t 04:00:00
#SBATCH --licenses=cfs,scratch
#SBATCH -o logs/reco420_split_bh_%j.out
#SBATCH -e logs/reco420_split_bh_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

MANIFEST=${1:?usage: sbatch $0 MANIFEST.tsv}
TASKS=$(wc -l < "$MANIFEST")
if [ "$TASKS" -lt 1 ] || [ "$TASKS" -gt 40 ]; then
    echo "manifest must contain between 1 and 40 tasks" >&2
    exit 2
fi

MUCOLL_IMAGE=${MUCOLL_IMAGE:-docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.1}
export MUCOLL_IMAGE
export MUCOLL_CALO_THRESHOLDS_DIR=${MUCOLL_CALO_THRESHOLDS_DIR:-$PSCRATCH/mucoll/maia_thresholds}
export CHAIN_WORK_BASE=${CHAIN_WORK_BASE:-$PSCRATCH/mucoll/reco_n420_${RECO_BIB_MODE}/work/$SLURM_JOB_ID}
export TRACKING_THREADS=3
mkdir -p logs "logs/reco420_${RECO_BIB_MODE}_$SLURM_JOB_ID" "$CHAIN_WORK_BASE"

if [ "$RECO_BIB_MODE" = legacy ]; then
    SOURCE=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data/bib-v3p0-fmt2-norm42-RandomRot/SIM
    export LEGACY_BIB_ROOT=$PSCRATCH/mucoll/bib-v3p0-fmt2-norm42-RandomRot-no6291/SIM
    for polarity in MUPLUS MUMINUS; do
        destination="$LEGACY_BIB_ROOT/$polarity"
        mkdir -p "$destination"
        while IFS= read -r -d '' source; do
            ln -sfn "$source" "$destination/${source##*/}"
        done < <(
            find "$SOURCE/$polarity" -maxdepth 1 -type f \
                -name 'bib_sim_*.edm4hep.root' \
                ! -name 'bib_sim_6291.edm4hep.root' -print0
        )
        count=$(find "$destination" -maxdepth 1 -type l \
            -name 'bib_sim_*.edm4hep.root' | wc -l)
        if [ "$count" -ne 6665 ]; then
            echo "$destination contains $count files; expected 6665" >&2
            exit 1
        fi
    done
fi

echo "tasks=$TASKS image=$MUCOLL_IMAGE maia=$EXPECTED_MAIA_COMMIT"
srun \
    --ntasks="$TASKS" \
    --ntasks-per-node=40 \
    --cpus-per-task=6 \
    --cpu-bind=cores \
    --kill-on-bad-exit=0 \
    --output="logs/reco420_${RECO_BIB_MODE}_$SLURM_JOB_ID/task_%t.out" \
    --error="logs/reco420_${RECO_BIB_MODE}_$SLURM_JOB_ID/task_%t.err" \
    shifter --image="$MUCOLL_IMAGE" \
    bash "$SLURM_SUBMIT_DIR/run_perlmutter_reco_split_bh_task.sh" "$MANIFEST"
