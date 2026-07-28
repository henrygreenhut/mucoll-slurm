#!/bin/bash
#SBATCH -J reco420_confirm_eval
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --exclude=gpu3005
#SBATCH --output=reco420_confirm_eval_%j.out
#SBATCH --error=reco420_confirm_eval_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

TRAIN_SITE=/oscar/scratch/$USER/mucoll/envs/reco-train-site-v1
STORE_DIR=/oscar/scratch/$USER/mucoll/libtest/reco_n420_confirmation_stores
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
        --store-dir "$STORE_DIR" \
        --checkpoint-dir "$RESULT_DIR/$checkpoint_label" \
        --class-b "$class_b" \
        --expected-training-dataset-tag trackfix \
        --require-clean-code \
        --label "$output_label" \
        --outdir "$RESULT_DIR"
}

evaluate R \
    reco_n420_trackfix_directlog_stabilized_dropout_U_vs_R \
    reco_n420_trackfix_directlog_stabilized_dropout_confirmation_U_vs_R
evaluate null_b \
    reco_n420_trackfix_directlog_stabilized_dropout_null \
    reco_n420_trackfix_directlog_stabilized_dropout_confirmation_null
