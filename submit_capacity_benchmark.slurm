#!/bin/bash
#SBATCH -J cap_bench
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -t 00:30:00
#SBATCH --output=cap_bench_%x_%j.out
#SBATCH --error=cap_bench_%x_%j.err

# n=42/126/210/420 capacity-benchmark sweep, one N per GPU on a single
# exclusive debug node. Billing is per-node regardless of GPU utilization,
# so running all 4 concurrently costs the same as running one -- see
# cap_bench_task.sh for the per-rank N assignment.
#
# Each rank writes its own cap_bench_<label>_<jobid>.out; after the job
# finishes:  cat cap_bench_n*_<jobid>.out

set -e
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

export REAL_STORE=${REAL_STORE:-$PSCRATCH/mucoll/libtest/stores/gen_norm1_MUPLUS.h5}
# --kill-on-bad-exit=0: a single rank hitting a hard GPU abort (SIGABRT --
# not a catchable Python exception, e.g. from a wildly oversized allocation
# attempt) must not take the other three, still-succeeding ranks down with
# it. Matches the proven pattern in submit_reco_libtest_packed.slurm.
srun --cpu-bind=cores --kill-on-bad-exit=0 bash cap_bench_task.sh

echo ""
echo "=== GPU binding verification ==="
cat "cap_bench_gpuid_${SLURM_JOB_ID}_rank"*.txt
N_UUID=$(grep -h -v "^rank=" "cap_bench_gpuid_${SLURM_JOB_ID}_rank"*.txt | cut -d, -f1 | sort -u | wc -l)
echo "distinct physical GPUs used: $N_UUID (expect 4)"
if [ "$N_UUID" -ne 4 ]; then
    echo "FAIL: ranks did not each get a distinct GPU -- --gpu-bind mis-configured;" >&2
    echo "      the timing/memory results above are not trustworthy, do not use them." >&2
    exit 1
fi
echo "PASS: all 4 ranks confirmed on distinct physical GPUs."
