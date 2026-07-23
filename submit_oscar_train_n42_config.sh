#!/bin/bash
#SBATCH -J oscar_train_n42_config
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C l40s
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH -t 00:40:00
#SBATCH -o oscar_train_n42_config_%x_%j.out
#SBATCH -e oscar_train_n42_config_%x_%j.err

# Old vs new validation/feature configuration, side by side, at the n42
# scale (--arch local, matching the existing seed 1/2/3 baseline already
# collected). Two axes changed together under "new":
#
#   val_units:     300 (old, the original default) -> 1000 (new). SE of
#                  val_auc near AUC=0.5 (SE = sqrt((2n+1)/(12n^2))) drops
#                  from ~0.024 to ~0.013 -- less noisy early-stopping
#                  decisions, at ~3.3x the per-epoch validation-side cost.
#   select_metric: auc (old, fixed --min-delta=1e-4 threshold -- already
#                  known to sit BELOW the val_auc noise floor at n=300,
#                  a real soft spot) -> loss (new, self-calibrating:
#                  improved iff val_loss drops by more than
#                  --min-delta-sigma SEMs of the per-unit val loss, so it
#                  auto-adjusts to whatever --val-units is in effect
#                  instead of needing a hand-derived threshold).
#   features:      paper (old, 9 features, momentum + PDG only) ->
#                  expanded (new, +loge/asinh_t/asinh_vz/asinh_vr/charge,
#                  14 features -- requires the _v2 stores, since "charge"
#                  isn't in the original gen_norm42_MUPLUS.h5 /
#                  gen_norm1_reconstructed_MUPLUS.h5; rebuild via
#                  submit_make_norm42_store.sh / submit_reconstruct_unrotated.sh
#                  first).
#
# Submit either config, any seed:
#   sbatch submit_oscar_train_n42_config.sh old 1
#   sbatch submit_oscar_train_n42_config.sh new 1

set -e
cd "$SLURM_SUBMIT_DIR"

CONFIG=$1
SEED=$2
if [ "$CONFIG" != "old" ] && [ "$CONFIG" != "new" ]; then
    echo "usage: sbatch submit_oscar_train_n42_config.sh {old|new} <seed>"
    exit 1
fi
if [ -z "$SEED" ]; then
    echo "usage: sbatch submit_oscar_train_n42_config.sh {old|new} <seed>"
    exit 1
fi

module load ngc-tensorflow-container/25.02-tf2-py3-j4zj

if [ "$CONFIG" = "old" ]; then
    NORM1_STORE="$HOME/mucoll/stores/gen_norm1_reconstructed_MUPLUS.h5"
    NORM42_STORE="/oscar/scratch/$USER/mucoll/stores/gen_norm42_MUPLUS.h5"
    VAL_UNITS=300
    SELECT_METRIC=auc
    FEATURES=paper
    LABEL="oscar_n42_config_old_seed${SEED}"
else
    NORM1_STORE="$HOME/mucoll/stores/gen_norm1_reconstructed_MUPLUS_v2.h5"
    NORM42_STORE="/oscar/scratch/$USER/mucoll/stores/gen_norm42_MUPLUS_v2.h5"
    VAL_UNITS=1000
    SELECT_METRIC=loss
    FEATURES=expanded
    LABEL="oscar_n42_config_new_seed${SEED}"
fi

if [ ! -f "$NORM1_STORE" ]; then
    echo "ERROR: norm1 store not found at $NORM1_STORE"
    [ "$CONFIG" = "new" ] && echo "  (run: sbatch submit_reconstruct_unrotated.sh MUPLUS)"
    exit 1
fi
if [ ! -f "$NORM42_STORE" ]; then
    echo "ERROR: norm42 store not found at $NORM42_STORE"
    [ "$CONFIG" = "new" ] && echo "  (run: sbatch submit_make_norm42_store.sh MUPLUS)"
    exit 1
fi

apptainer exec --nv "$NGC_TENSORFLOW_CONTAINER" python -u pfn_libtest_train.py \
    --label "$LABEL" \
    --norm1-store "$NORM1_STORE" \
    --norm42-store "$NORM42_STORE" \
    --n-files 42 \
    --units-per-epoch 1000 \
    --val-units "$VAL_UNITS" \
    --select-metric "$SELECT_METRIC" \
    --features "$FEATURES" \
    --batch-size 8 \
    --max-minutes 25 \
    --latent-scale none \
    --phi-sizes 200,200,256 \
    --f-sizes 200,200,200 \
    --arch local \
    --seed "$SEED"

echo ""
echo "=== history ==="
hist="pfn_results/${LABEL}/history.csv"
if [ -f "$hist" ]; then
    cat "$hist"
else
    echo "NO history.csv -- did not complete even one epoch"
fi
