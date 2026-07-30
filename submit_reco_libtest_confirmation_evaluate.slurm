#!/bin/bash
#SBATCH -J reco_confirm_eval
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --exclude=gpu3005
#SBATCH --output=reco_confirm_eval_%j.out
#SBATCH --error=reco_confirm_eval_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

STUDY=${1:-charged7}
N_FILES=${2:-420}
STORE_SPLIT=confirmation
EVALUATION_NAME=confirmation
EVALUATION_ARGS=()
if (( N_FILES <= 0 || N_FILES % 42 != 0 )); then
    echo "N_FILES must be a positive multiple of 42" >&2
    exit 2
fi
case "$STUDY" in
    charged7)
        if [ "$N_FILES" -ne 420 ]; then
            echo "charged7 is the legacy N=420 study; use val25 for N=$N_FILES" >&2
            exit 2
        fi
        DATASET_TAG=trackfix
        CHECKPOINT_PREFIX=reco_n${N_FILES}_trackfix_directlog_charged7_stabilized_dropout
        OUTPUT_PREFIX=reco_n${N_FILES}_trackfix_directlog_charged7_stabilized_dropout_confirmation
        ;;
    val2000)
        if [ "$N_FILES" -ne 420 ]; then
            echo "val2000 is the legacy N=420 study; use val25 for N=$N_FILES" >&2
            exit 2
        fi
        DATASET_TAG=trackfix_val2000
        CHECKPOINT_PREFIX=reco_n${N_FILES}_trackfix_val2000_directlog_charged7_stabilized_dropout
        OUTPUT_PREFIX=reco_n${N_FILES}_trackfix_val2000_directlog_charged7_stabilized_dropout_confirmation
        ;;
    val25)
        if [ "$N_FILES" -eq 420 ]; then
            DATASET_TAG=trackfix_val25
        else
            DATASET_TAG=n${N_FILES}_trackfix_val25
        fi
        CHECKPOINT_PREFIX=reco_n${N_FILES}_trackfix_val25_directlog_charged7_stabilized_dropout
        OUTPUT_PREFIX=reco_n${N_FILES}_trackfix_val25_directlog_charged7_stabilized_dropout_confirmation
        ;;
    test2000)
        if [ "$N_FILES" -ne 1260 ]; then
            echo "test2000 currently identifies the N=1260 extension" >&2
            exit 2
        fi
        DATASET_TAG=n1260_trackfix_val25
        CHECKPOINT_PREFIX=reco_n1260_trackfix_val25_directlog_charged7_stabilized_dropout
        OUTPUT_PREFIX=reco_n1260_trackfix_val25_directlog_charged7_stabilized_dropout_test2000
        STORE_SPLIT=test
        EVALUATION_NAME=test2000
        EVALUATION_ARGS=(
            --evaluation-note
            "Additional pseudo-events from the same held-out test source-cycle pool"
            --allow-code-commit-mismatch
        )
        ;;
    *)
        echo "usage: sbatch $0 [charged7|val2000|val25|test2000] [N_FILES]" >&2
        exit 2
        ;;
esac

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

TRAIN_SITE=/oscar/scratch/$USER/mucoll/envs/reco-train-site-v1
if [ "$STUDY" = test2000 ]; then
    STORE_DIR=/oscar/scratch/$USER/mucoll/libtest/reco_n1260_pfn_stores_trackfix_val25_test2000
else
    STORE_DIR=/oscar/scratch/$USER/mucoll/libtest/reco_n${N_FILES}_confirmation_stores
fi
RESULT_DIR="$SLURM_SUBMIT_DIR/reco_pfn_results"
if [ ! -d "$TRAIN_SITE/matplotlib" ]; then
    echo "ERROR: missing $TRAIN_SITE; run ./setup_oscar_reco_envs.sh first" >&2
    exit 1
fi

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"
export PYTHONPATH="$TRAIN_SITE"
mkdir -p "$MPLCONFIGDIR"

evaluate() {
    class_b=$1
    checkpoint_label=$2
    output_label=$3
    apptainer exec --nv --bind /oscar:/oscar \
        "$NGC_TENSORFLOW_CONTAINER" \
        python evaluate_reco_libtest_confirmation.py \
        --n-files "$N_FILES" \
        --store-dir "$STORE_DIR" \
        --checkpoint-dir "$RESULT_DIR/$checkpoint_label" \
        --class-b "$class_b" \
        --expected-training-dataset-tag "$DATASET_TAG" \
        --store-split "$STORE_SPLIT" \
        --evaluation-name "$EVALUATION_NAME" \
        "${EVALUATION_ARGS[@]}" \
        --require-clean-code \
        --label "$output_label" \
        --outdir "$RESULT_DIR"
}

evaluate R \
    "${CHECKPOINT_PREFIX}_U_vs_R" \
    "${OUTPUT_PREFIX}_U_vs_R"
evaluate null_b \
    "${CHECKPOINT_PREFIX}_null" \
    "${OUTPUT_PREFIX}_null"
