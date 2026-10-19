#!/bin/bash
#SBATCH --job-name=count_patch_pfn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu3005

set -euo pipefail

: "${CT_DIR:?export CT_DIR or pass it with sbatch --export}"
: "${PATCH_ROOT:?export PATCH_ROOT or pass it with sbatch --export}"
: "${PFN_OUT:?export PFN_OUT or pass it with sbatch --export}"

TRAIN_SITE="${TRAIN_SITE:-/oscar/scratch/$USER/mucoll/envs/reco-train-site-v1}"
CONSTRUCTION="${CONSTRUCTION:-norm42_fullbx_theta_phi72}"
LABEL="${LABEL:-fullbx_theta_phi72}"

for package in energyflow tf_keras; do
    [ -d "$TRAIN_SITE/$package" ] || {
        echo "Missing $TRAIN_SITE/$package; recreate the established RECO PFN environment" >&2
        exit 1
    }
done
for split in train val test; do
    for sample in SIM COUNT; do
        [ -f "$PATCH_ROOT/${CONSTRUCTION}_${sample}_${split}.npz" ] || {
            echo "Missing angular patch store for $sample/$split" >&2
            exit 1
        }
    done
done

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj
export PYTHONPATH="$TRAIN_SITE"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR" "$PFN_OUT"

apptainer exec --nv --bind /oscar:/oscar \
    "$NGC_TENSORFLOW_CONTAINER" \
    python -u "$CT_DIR/count_tracker_train.py" \
        --store-dir "$PATCH_ROOT" \
        --construction "$CONSTRUCTION" \
        --unit event \
        --sample-a SIM --sample-b COUNT \
        --label "$LABEL" \
        --outdir "$PFN_OUT" \
        --recipe fixed \
        --epochs 150 \
        --batch-size 32 \
        --patience 20

echo "PFN result -> $PFN_OUT/$LABEL"
