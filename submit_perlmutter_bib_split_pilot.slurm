#!/bin/bash
#SBATCH -J bib_split_pilot
#SBATCH -A m5197
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH -t 00:30:00
#SBATCH --licenses=cfs,scratch
#SBATCH -o logs/bib_split_pilot_%j.out
#SBATCH -e logs/bib_split_pilot_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

IMAGE=docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.0
OUTPUT=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data/bib-v3p0-fmt2-split-muon-v1
MANIFEST=$OUTPUT/manifests/partition.json
LOGS=$SLURM_SUBMIT_DIR/logs/bib_split_pilot_$SLURM_JOB_ID
export BIB_SPLIT_CYCLES=${BIB_SPLIT_CYCLES:-2320,4152}

mkdir -p "$LOGS"

srun \
    --ntasks=4 \
    --cpus-per-task=2 \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    --output="$LOGS/gen_rank_%t.out" \
    --error="$LOGS/gen_rank_%t.err" \
    shifter --image="$IMAGE" bash "$SLURM_SUBMIT_DIR/run_perlmutter_bib_split_gen.sh"

srun \
    --ntasks=4 \
    --cpus-per-task=2 \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    --output="$LOGS/sim_rank_%t.out" \
    --error="$LOGS/sim_rank_%t.err" \
    shifter --image="$IMAGE" bash "$SLURM_SUBMIT_DIR/run_perlmutter_bib_split_sim.sh"

cycle_args=()
IFS=, read -ra cycles <<< "$BIB_SPLIT_CYCLES"
for cycle in "${cycles[@]}"; do
    cycle_args+=(--cycle "$cycle")
done

python3 "$SLURM_SUBMIT_DIR/bib_split_muon_production.py" status \
    --manifest "$MANIFEST" \
    --output-root "$OUTPUT" \
    "${cycle_args[@]}" \
    --require-complete
