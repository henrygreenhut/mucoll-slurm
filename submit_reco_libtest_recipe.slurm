#!/bin/bash
#SBATCH -J reco420_recipe
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --exclude=gpu3005
#SBATCH --output=reco420_recipe_%j.out
#SBATCH --error=reco420_recipe_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -ne 1 ]; then
    echo "usage: sbatch $0 {stabilized|stabilized_dropout}" >&2
    exit 2
fi
RECIPE=$1
case "$RECIPE" in
    stabilized|stabilized_dropout) ;;
    *)
        echo "unknown recipe: $RECIPE" >&2
        exit 2
        ;;
esac

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

TRAIN_SITE=/oscar/scratch/$USER/mucoll/envs/reco-train-site-v1
STORE_DIR=/oscar/scratch/$USER/mucoll/libtest/reco_n420_pfn_stores_simple
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
        --recipe "$RECIPE" \
        --label "$label"
}

echo "RECO N=420 recipe: $RECIPE"
run_training R "reco_n420_directlog_${RECIPE}_U_vs_R"
run_training null_b "reco_n420_directlog_${RECIPE}_null"
