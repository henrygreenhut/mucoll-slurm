#!/bin/bash
#SBATCH -J reco420_pfn
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 04:00:00
#SBATCH --exclude=gpu3005
#SBATCH --output=reco420_pfn_%j.out
#SBATCH --error=reco420_pfn_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

TRAIN_SITE=/oscar/scratch/$USER/mucoll/envs/reco-train-site-v1
STORE_DIR=/oscar/scratch/$USER/mucoll/libtest/reco_n420_pfn_stores_trackfix
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
    if [ -f "$RESULT_DIR/$label/summary.json" ]; then
        echo "complete -> $RESULT_DIR/$label/summary.json"
        return
    fi
    apptainer exec --nv --bind /oscar:/oscar \
        "$NGC_TENSORFLOW_CONTAINER" \
        python train_reco_libtest_pfn.py \
        --store-dir "$STORE_DIR" \
        --class-a U \
        --class-b "$class_b" \
        --dataset-tag trackfix \
        --require-clean-code \
        --require-pfo-track-links \
        --label "$label"
}

run_training R "reco_n420_trackfix_directlog_baseline_U_vs_R"
run_training null_b "reco_n420_trackfix_directlog_baseline_null"
