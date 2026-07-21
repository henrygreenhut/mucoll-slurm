#!/bin/bash
# Re-run of the original arch_scan debug screen (arch_scan_task.sh) with
# every config's Phi network HALVED, batch size UNCHANGED per config. Tests
# whether the large Phi (200,200,256) is actually worth its cost against a
# half-width Phi at the SAME batch size -- not a batch-size tradeoff, a
# pure "do we need this much per-particle capacity" question.
#
# F sizes are untouched (F only ever sees the pooled (batch, width) tensor,
# not (batch, N, width), so it isn't part of the int32 overflow story and
# isn't what this test is about).
#
# Side effect, not the point of this test: halving the widest Phi layer
# halves batch*width, which DOUBLES the int32-safe N ceiling (was N <
# 2,097,152 at batch=4/width=256; now N < 4,194,304 at batch=4/width=128).
# That's real headroom against the crash, but not guaranteed -- norm42
# unit N was observed up to ~2.07M in just 30 draws, so don't assume this
# alone rules out a crash over a full debug window.
#
# New labels (halfphi_* not scan_*) so this doesn't try to resume from the
# original (crashed, full-width) checkpoints. No --jit -- proven a dead
# end (4x+ slower, 3/4 ranks never finished an epoch in 29 minutes).
set -e

LABELS=(halfphi_raw_large  halfphi_raw_small   halfphi_scaled_large  halfphi_null_large)
LATENT=(none                none                 auto                   none)
PHI=(100,100,128            50,50,64             100,100,128            100,100,128)
F=(200,200,200              100,100,100          200,200,200            200,200,200)
BATCH=(4                    8                    4                      4)
EXTRA=(""                   ""                   ""                     "--null-test")

LABEL=${LABELS[$SLURM_PROCID]}
OUT="phi_half_scan_${LABEL}_${SLURM_JOB_ID}.out"

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
