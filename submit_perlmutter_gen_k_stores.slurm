#!/bin/bash
#SBATCH -J gen_k_stores
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A m5197
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -o perlmutter_gen_k_stores_%j.out
#SBATCH -e perlmutter_gen_k_stores_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load python
conda activate "$PSCRATCH/mucoll/envs/mucoll-inspect" 2>/dev/null \
    || source activate "$PSCRATCH/mucoll/envs/mucoll-inspect"

STORE_DIR="$PSCRATCH/mucoll/libtest/stores"
INPUT="$STORE_DIR/gen_split_mothers_MUPLUS.h5"

if [ ! -s "$INPUT" ]; then
    echo "missing mother bank: $INPUT" >&2
    exit 1
fi

build_store() {
    local K=$1
    local OUTPUT="$STORE_DIR/gen_fixed_k${K}_MUPLUS.h5"
    local PARTIAL="${OUTPUT}.partial.${SLURM_JOB_ID}"
    if [ -s "$OUTPUT" ]; then
        echo "k=$K already exists: $OUTPUT"
        return
    fi
    if python -u gen_mother_make_fixed_reuse_store.py \
        --input "$INPUT" \
        --output "$PARTIAL" \
        --reuse-k "$K" \
        --seed 1701 > "perlmutter_gen_k${K}_store_${SLURM_JOB_ID}.log" 2>&1
    then
        mv "$PARTIAL" "$OUTPUT"
        echo "k=$K complete: $OUTPUT"
    else
        rm -f "$PARTIAL"
        echo "k=$K failed; see perlmutter_gen_k${K}_store_${SLURM_JOB_ID}.log" >&2
        return 1
    fi
}

PIDS=()
for K in 5 42; do
    build_store "$K" &
    PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || STATUS=1
done
exit "$STATUS"
