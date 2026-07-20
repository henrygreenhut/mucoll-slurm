#!/bin/bash
# One architecture-scan config per GPU: SLURM_PROCID (0..3) picks the config.
# Tier A (debug-screenable, comparable per-epoch cost): configs #1-3 from the
# N=420 breakdown, plus a null test at #1's exact settings (raw sum, large
# Phi/F, batch=4) -- never validated at batch=4 before, same cost as #1/#3,
# so it rides along for free as a pipeline sanity check.
#
# Config #4 (2x units-per-epoch + 2x val-units) is NOT here: at ~829s/epoch
# it wouldn't clear one full epoch in a debug window, and its whole
# hypothesis ("does more resources help") only shows up after many epochs.
# It's a full shared-queue convergence run, only worth launching once #1
# looks like a healthy baseline worth improving on.
#
# --max-minutes triggers a checkpoint-and-return BEFORE the evaluation
# block, so this screen has zero eval cost -- purely epoch-by-epoch
# train_loss/val_auc trajectories in each rank's history.csv.
#
# --arch energyflow: all four configs run through the actual
# energyflow.archs.PFN (raw sum, latent-scale=none) or energyflow.archs.EFN
# with z_i=latent_scale weighting (scaled sum, latent-scale=auto), both
# verified bitwise-equivalent to the local build by
# pfn_arch_equivalence_check.py -- official-package provenance throughout.
set -e

LABELS=(scan_raw_large    scan_raw_small     scan_scaled_large  scan_null_large)
LATENT=(none               none               auto               none)
PHI=(200,200,256           100,100,128        200,200,256        200,200,256)
F=(200,200,200             100,100,100        200,200,200        200,200,200)
BATCH=(4                   8                  4                  4)
EXTRA=(""                  ""                 ""                 "--null-test")

LABEL=${LABELS[$SLURM_PROCID]}
OUT="arch_scan_${LABEL}_${SLURM_JOB_ID}.out"

echo "rank $SLURM_PROCID -> $LABEL (latent-scale=${LATENT[$SLURM_PROCID]}," \
     "phi=${PHI[$SLURM_PROCID]}, f=${F[$SLURM_PROCID]}," \
     "batch=${BATCH[$SLURM_PROCID]}) -> $OUT"
python -u pfn_libtest_train.py \
    --label "$LABEL" \
    --n-files "${N_FILES:-420}" \
    --units-per-epoch "${UNITS_PER_EPOCH:-500}" \
    --batch-size "${BATCH[$SLURM_PROCID]}" \
    --max-minutes "${MAX_MINUTES:-22}" \
    --latent-scale "${LATENT[$SLURM_PROCID]}" \
    --phi-sizes "${PHI[$SLURM_PROCID]}" \
    --f-sizes "${F[$SLURM_PROCID]}" \
    --arch energyflow \
    ${EXTRA[$SLURM_PROCID]} \
    > "$OUT" 2>&1
echo "rank $SLURM_PROCID ($LABEL) done (or checkpointed at --max-minutes)"
