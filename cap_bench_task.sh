#!/bin/bash
# One task per GPU: SLURM_PROCID (0..3) picks this rank's N-sweep point.
# --gpu-bind=single:1 (set on the parent srun) is SUPPOSED to give each rank
# its own GPU -- verified below rather than trusted. Note: TF/CUDA device
# names inside a CUDA_VISIBLE_DEVICES-restricted process always read back as
# GPU:0 regardless of which physical card was assigned, so that check would
# be uninformative; nvidia-smi's UUID is the one identifier that survives
# the remapping and genuinely differs per physical GPU.
set -e

N_VALUES=(125000 375000 625000 1255800)
LABELS=(n42 n126 n210 n420)
N=${N_VALUES[$SLURM_PROCID]}
LABEL=${LABELS[$SLURM_PROCID]}
OUT="cap_bench_${LABEL}_${SLURM_JOB_ID}.out"
IDFILE="cap_bench_gpuid_${SLURM_JOB_ID}_rank${SLURM_PROCID}.txt"

{
    echo "rank=$SLURM_PROCID host=$(hostname) SLURM_LOCALID=$SLURM_LOCALID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    nvidia-smi --query-gpu=uuid,pci.bus_id,name --format=csv,noheader
} > "$IDFILE"
cat "$IDFILE"

echo "rank $SLURM_PROCID -> $LABEL (N=$N) -> $OUT"
# -u: belt-and-suspenders alongside the script's own flush=True print --
# guarantees no already-succeeded result is lost to output buffering if a
# later, larger batch size hard-aborts the process.
python -u pfn_capacity_benchmark.py \
    --n-list "$N" \
    --batch-sizes 1,2,4,8 \
    --real-store "$REAL_STORE" \
    > "$OUT" 2>&1
echo "rank $SLURM_PROCID ($LABEL) done"
