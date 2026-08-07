#!/bin/bash
#SBATCH -J reco_recipe
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --exclude=gpu3005
#SBATCH --output=reco_recipe_%j.out
#SBATCH --error=reco_recipe_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "usage: sbatch $0 {stabilized|stabilized_dropout} [charged7|val2000|val25|calo_unconed] [N_FILES]" >&2
    exit 2
fi
RECIPE=$1
STUDY=${2:-charged7}
N_FILES=${3:-420}
TEST_EVENTS=800
if (( N_FILES <= 0 || N_FILES % 42 != 0 )); then
    echo "N_FILES must be a positive multiple of 42" >&2
    exit 2
fi
case "$RECIPE" in
    stabilized|stabilized_dropout) ;;
    *)
        echo "unknown recipe: $RECIPE" >&2
        exit 2
        ;;
esac

case "$STUDY" in
    charged7)
        if [ "$N_FILES" -ne 420 ]; then
            echo "charged7 is the legacy N=420 study; use val25 for N=$N_FILES" >&2
            exit 2
        fi
        STORE_NAME=reco_n${N_FILES}_pfn_stores_trackfix
        DATASET_TAG=trackfix
        LABEL_PREFIX=reco_n${N_FILES}_trackfix_directlog_charged7
        VAL_EVENTS=400
        ;;
    val2000)
        if [ "$N_FILES" -ne 420 ]; then
            echo "val2000 is the legacy N=420 study; use val25 for N=$N_FILES" >&2
            exit 2
        fi
        STORE_NAME=reco_n${N_FILES}_pfn_stores_trackfix_val2000
        DATASET_TAG=trackfix_val2000
        LABEL_PREFIX=reco_n${N_FILES}_trackfix_val2000_directlog_charged7
        VAL_EVENTS=2000
        ;;
    val25)
        STORE_NAME=reco_n${N_FILES}_pfn_stores_trackfix_val25
        if [ "$N_FILES" -eq 420 ]; then
            DATASET_TAG=trackfix_val25
            LABEL_PREFIX=reco_n420_trackfix_val25_directlog_charged7
        else
            DATASET_TAG=n${N_FILES}_trackfix_val25
            LABEL_PREFIX=reco_n${N_FILES}_trackfix_val25_directlog_charged7
        fi
        VAL_EVENTS=2000
        ;;
    calo_unconed)
        if [ "$N_FILES" -ne 420 ]; then
            echo "calo_unconed is currently defined only for N=420" >&2
            exit 2
        fi
        STORE_NAME=reco_n420_pfn_stores_calo_unconed_val25
        DATASET_TAG=calo_unconed_val25
        LABEL_PREFIX=reco_n420_calo_unconed_val25_directlog_charged7
        VAL_EVENTS=2000
        TEST_EVENTS=2000
        ;;
    *)
        echo "unknown study: $STUDY" >&2
        exit 2
        ;;
esac

module load ngc-tensorflow-container/25.02-tf2-py3-azkb

TRAIN_SITE=/oscar/scratch/$USER/mucoll/envs/reco-train-site-v1
STORE_DIR=/oscar/scratch/$USER/mucoll/libtest/$STORE_NAME
RESULT_DIR="$SLURM_SUBMIT_DIR/reco_pfn_results"
if [ ! -d "$TRAIN_SITE/matplotlib" ]; then
    echo "ERROR: missing $TRAIN_SITE; run ./setup_oscar_reco_envs.sh first"
    exit 1
fi

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"
export PYTHONPATH="$TRAIN_SITE"
mkdir -p "$MPLCONFIGDIR"

run_training() {
    class_b=$1
    label=$2
    apptainer exec --nv --bind /oscar:/oscar \
        "$NGC_TENSORFLOW_CONTAINER" \
        python train_reco_libtest_pfn.py \
        --n-files "$N_FILES" \
        --store-dir "$STORE_DIR" \
        --class-a U \
        --class-b "$class_b" \
        --dataset-tag "$DATASET_TAG" \
        --require-clean-code \
        --require-pfo-track-links \
        --train-events 2000 \
        --val-events "$VAL_EVENTS" \
        --test-events "$TEST_EVENTS" \
        --recipe "$RECIPE" \
        --label "$label"
}

echo "RECO N=$N_FILES study=$STUDY dataset=$DATASET_TAG features=charged7 store=$STORE_DIR recipe=$RECIPE"
run_training R "${LABEL_PREFIX}_${RECIPE}_U_vs_R"
run_training null_b "${LABEL_PREFIX}_${RECIPE}_null"
