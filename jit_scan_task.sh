#!/bin/bash
# XLA JIT test: the SAME 4 configs as arch_scan_task.sh (3 of which crashed
# on the TF/XLA int32 overflow bug), now with --jit added. Tests whether
# XLA-compiled kernels sidestep the legacy GPU kernel's int32 launch-config
# overflow (different codegen path -- see libtest_common.build_pfn* for the
# jit_compile wiring). New labels (jit_* not scan_*) so this doesn't try to
# resume from the old crashed (non-JIT) checkpoints.
#
# Watch per-epoch seconds in each rank's .out: our particle count N varies
# every batch (padded to the largest unit in that batch), and XLA compiles
# kernels keyed to shape -- if N changes enough to force frequent
# recompilation, per-epoch time could balloon even if the crash goes away.
set -e

LABELS=(jit_raw_large    jit_raw_small     jit_scaled_large  jit_null_large)
LATENT=(none              none              auto              none)
PHI=(200,200,256          100,100,128       200,200,256       200,200,256)
F=(200,200,200            100,100,100       200,200,200       200,200,200)
BATCH=(4                  8                 4                 4)
EXTRA=(""                 ""                ""                "--null-test")

LABEL=${LABELS[$SLURM_PROCID]}
OUT="jit_scan_${LABEL}_${SLURM_JOB_ID}.out"

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
    --jit \
    ${EXTRA[$SLURM_PROCID]} \
    > "$OUT" 2>&1
echo "rank $SLURM_PROCID ($LABEL) done (or checkpointed at --max-minutes)"
