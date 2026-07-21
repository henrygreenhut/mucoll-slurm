#!/bin/bash
# Longer, shared-queue continuation of the two halved-Phi configs that
# showed real learning in the debug scan (job 56272947): raw_large and
# scaled_large. SAME labels as that scan, so this RESUMES from their
# existing epoch-3 checkpoints rather than starting over. raw_small
# (capacity-starved, flat at chance) and null_large (already validated)
# are deliberately excluded.
#
# --max-minutes leaves a buffer under the 7-hour (420 min) SLURM wall
# time for store loading + TF/CUDA startup + the final checkpoint write.
# Resubmit the same sbatch script to continue past this window -- the
# trainer's checkpoint/resume logic picks up automatically from the
# same --label.
set -e

LABELS=(halfphi_raw_large    halfphi_scaled_large)
LATENT=(none                  auto)
PHI=(100,100,128              100,100,128)
F=(200,200,200                200,200,200)
BATCH=(4                      4)

LABEL=${LABELS[$SLURM_PROCID]}
OUT="phi_half_long_${LABEL}_${SLURM_JOB_ID}.out"

echo "rank $SLURM_PROCID -> $LABEL (latent-scale=${LATENT[$SLURM_PROCID]}," \
     "phi=${PHI[$SLURM_PROCID]}, f=${F[$SLURM_PROCID]}," \
     "batch=${BATCH[$SLURM_PROCID]}) -> $OUT"
python -u pfn_libtest_train.py \
    --label "$LABEL" \
    --n-files "${N_FILES:-420}" \
    --units-per-epoch "${UNITS_PER_EPOCH:-500}" \
    --batch-size "${BATCH[$SLURM_PROCID]}" \
    --max-minutes "${MAX_MINUTES:-400}" \
    --latent-scale "${LATENT[$SLURM_PROCID]}" \
    --phi-sizes "${PHI[$SLURM_PROCID]}" \
    --f-sizes "${F[$SLURM_PROCID]}" \
    --arch energyflow \
    > "$OUT" 2>&1
echo "rank $SLURM_PROCID ($LABEL) done (or checkpointed at --max-minutes)"
