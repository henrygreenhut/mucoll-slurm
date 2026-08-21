#!/bin/bash
#SBATCH -J bib_split_sim
#SBATCH -A m5197
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 4
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=2
#SBATCH -t 24:00:00
#SBATCH --licenses=cfs,scratch
#SBATCH --requeue
#SBATCH -o logs/bib_split_sim_%j.out
#SBATCH -e logs/bib_split_sim_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

IMAGE=docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.0
OUTPUT=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data/bib-v3p0-fmt2-split-muon-v1
MANIFEST=$OUTPUT/manifests/partition.json
LOGS=$SLURM_SUBMIT_DIR/logs/bib_split_sim_$SLURM_JOB_ID
mkdir -p "$LOGS"

srun \
    --ntasks="$SLURM_NTASKS" \
    --ntasks-per-node=96 \
    --cpus-per-task=2 \
    --cpu-bind=cores \
    --kill-on-bad-exit=0 \
    --output="$LOGS/rank_%t.out" \
    --error="$LOGS/rank_%t.err" \
    shifter --image="$IMAGE" bash "$SLURM_SUBMIT_DIR/run_perlmutter_bib_split_sim.sh"

cycle_args=()
IFS=, read -ra cycles <<< "${BIB_SPLIT_CYCLES:-}"
for cycle in "${cycles[@]}"; do
    if [ -n "$cycle" ]; then
        cycle_args+=(--cycle "$cycle")
    fi
done

python3 "$SLURM_SUBMIT_DIR/bib_split_muon_production.py" status \
    --manifest "$MANIFEST" \
    --output-root "$OUTPUT" \
    "${cycle_args[@]}" \
    --require-complete
